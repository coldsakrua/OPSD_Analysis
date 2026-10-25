import os
import json
from datetime import datetime

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore, LanguageModel, LMStyle
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)


def main():
    args = get_args()

    # Support local model paths by creating a LanguageModel if not found in store
    if args.model in LanguageModelStore:
        model = LanguageModelStore[args.model]
        # Allow --model_style to override even registered models
        if hasattr(args, 'model_style') and args.model_style:
            try:
                override_style = LMStyle(args.model_style)
                print(f"Overriding model style from {model.model_style} to {override_style} (via --model_style)")
                model = LanguageModel(
                    model_name=model.model_name,
                    model_repr=model.model_repr,
                    model_style=override_style,
                    release_date=model.release_date,
                    link=model.link,
                )
            except ValueError:
                print(f"WARNING: Unknown model style '{args.model_style}', ignoring. Valid options: {[s.value for s in LMStyle]}")
    else:
        # Assume it's a local path, try to infer model style from path
        model_name = args.model
        # Extract model representation from path (last directory name)
        model_repr = os.path.basename(os.path.normpath(args.model))
        
        # If --model_style is explicitly set, use it directly
        if hasattr(args, 'model_style') and args.model_style:
            try:
                model_style = LMStyle(args.model_style)
                print(f"Using explicitly set model style: {model_style}")
            except ValueError:
                print(f"WARNING: Unknown model style '{args.model_style}'. Valid options: {[s.value for s in LMStyle]}")
                model_style = LMStyle.GenericBase
        else:
            # Try to infer model style from path/name
            model_name_lower = model_name.lower()
            model_repr_lower = model_repr.lower()
            
            if "AZR" in model_repr or "Absolute_Zero" in model_repr or "azr" in model_repr_lower:
                model_style = LMStyle.AZR
            elif "instruct" in model_repr_lower and ("step" in model_repr_lower or "ckpt" in model_repr_lower or "checkpoint" in model_repr_lower):
                # RL-trained models converted to HF format (e.g., "7b_step_80_instruct")
                # These use the AZR prompt template
                model_style = LMStyle.AZR
            elif "qwen" in model_repr_lower or "qwen" in model_name_lower:
                if "coder" in model_repr_lower or "coder" in model_name_lower:
                    model_style = LMStyle.Qwen
                else:
                    model_style = LMStyle.Qwen
            elif "Coder" in model_repr:
                model_style = LMStyle.CodeQwenInstruct
            else:
                model_style = LMStyle.GenericBase  # Default fallback
            
            print(f"Auto-inferred model style: {model_style} (from model_repr='{model_repr}')")
        
        model = LanguageModel(
            model_name=model_name,
            model_repr=model_repr,
            model_style=model_style,
            release_date=datetime(2025, 1, 1),
            link=None
        )
    
    # Set environment variables for prompt formatting (especially for Qwen models)
    if hasattr(args, 'local_model_path') and args.local_model_path:
        os.environ['LOCAL_MODEL_PATH'] = args.local_model_path
    os.environ['MODEL_PATH'] = args.model
    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"]
            # and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    # for i in range(len(combined_results)):
    #     for j in range(len(combined_results[i][1])):
    #         if "def solve()" in combined_results[i][1][j]:
    #             from lcb_runner.utils.extraction_utils import extract_code, LMStyle

    #             combined_results[i][1][j] = extract_code(
    #                 combined_results[i][0][j], LMStyle.Gemini
    #             )
    #             if "\nsolve()" not in combined_results[i][1][j]:
    #                 combined_results[i][1][j] += "\n\nsolve()"

    #                 # combined_results[i][1][j] += "\n\nsolve()"
    #                 print(combined_results[i][1][j])

    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()
