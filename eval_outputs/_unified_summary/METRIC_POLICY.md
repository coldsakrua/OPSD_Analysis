# Metric policy (no MATH500)

- **Primary (结论表默认)**: `pass@1` = `avg1_pct` in `*.metrics.json` (also `pass_at_k["1"].pct`).
- **Secondary**: `average_correct_pct` (mean accuracy over n=8 samples per problem; often labeled Avg@8 / avg16 in logs).
- **Protocol**: AIME24 / AIME25 / AIME26 / HMMT25, 30 problems × n=8.
- **Inclusion**: `num_problems == 30` and `partial_only == false` only.
- **MATH500**: intentionally excluded from this conclusion pipeline.
- Rebuild: `python scripts/analysis/rebuild_opsd_unified_summary.py`
