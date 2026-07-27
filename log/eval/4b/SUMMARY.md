# Qwen3-4B AIME24 / AIME25 / AIME26 评测汇总

- **Checkpoint**: `checkpoint-100`
- **采样**: `n=8`（Pass@1 / Pass@8）
- **Avg@8**: 日志中的 `Avg16(overall correctness)`（实际 `n=8`）
- **Length**: Avg output tokens
- **评测模式**: Think = `enable_thinking=1`；NoThink = `enable_thinking=0`
- **Teacher Privilege**: 完整轨迹 = teacher 看到 solution；仅答案 = teacher 只看到最终 answer

> 全部方法的 AIME24/25/26 Think + NoThink 评测已完成。

---

## 方法配置

| 方法 | Student Think | Teacher Think | 训练数据 | Teacher Privilege | LR |
|------|:-------------:|:-------------:|----------|:-----------------:|-----|
| Qwen3-4B (base) | — | — | — | — | — |
| snt_tt | 否 | 是 | DAPO-Math-17k | 仅答案 | 2e-6 |
| snt_tt | 否 | 是 | DAPO-Math-17k | 仅答案 | 5e-6 |
| snt_tt | 否 | 是 | DAPO-Math-17k | 仅答案 | 1e-6 |
| snt_tt_ot | 否 | 是 | OpenThoughts | 完整轨迹 | 1e-6 |
| snt_tt_ota | 否 | 是 | OpenThoughts | 仅答案 | 1e-6 |
| ot_think | 是 | 是 | OpenThoughts | 仅答案 | 5e-6 |
| ot_think_sol | 是 | 是 | OpenThoughts | 完整轨迹 | 5e-6 |
| ot_think | 是 | 是 | OpenThoughts | 仅答案 | 1e-6 |
| ot_think_sol | 是 | 是 | OpenThoughts | 完整轨迹 | 1e-6 |

**说明**

- `snt_tt` = student no-think / teacher think
- DAPO 的 `snt_tt` parquet 中 `solution` 列实际是短答案，故 Teacher Privilege 记为「仅答案」
- `snt_tt_ot` / `ot_think_sol`：完整轨迹；`snt_tt_ota` / `ot_think`：仅答案

---

## Think 评测（enable_thinking=1）

| 方法 | 训练数据 | Privilege | LR | A24 Pass@1 | A24 Pass@8 | A24 Avg@8 | A24 Length | A25 Pass@1 | A25 Pass@8 | A25 Avg@8 | A25 Length | A26 Pass@1 | A26 Pass@8 | A26 Avg@8 | A26 Length |
|------|----------|-----------|-----|-----------:|-----------:|----------:|-----------:|-----------:|-----------:|----------:|-----------:|-----------:|-----------:|----------:|-----------:|
| Qwen3-4B (base) | — | — | — | 66.67 | 90.00 | 73.33 | 15223.9 | 63.33 | 80.00 | 65.00 | 18263.2 | 66.67 | 80.00 | 65.00 | 16727.8 |
| snt_tt | DAPO | 仅答案 | 2e-6 | 63.33 | 80.00 | 65.00 | 14705.4 | 63.33 | 86.67 | 62.08 | 16620.1 | 66.67 | 80.00 | 60.42 | 15871.6 |
| snt_tt | DAPO | 仅答案 | 5e-6 | 33.33 | 63.33 | 37.08 | 20957.9 | 30.00 | 60.00 | 36.25 | 20998.6 | 36.67 | 60.00 | 31.25 | 21701.8 |
| snt_tt | DAPO | 仅答案 | 1e-6 | 76.67 | 86.67 | 72.50 | 14303.3 | 56.67 | 76.67 | 61.25 | 17454.6 | 70.00 | 83.33 | 62.08 | 15983.1 |
| snt_tt_ot | OpenThoughts | 完整轨迹 | 1e-6 | 66.67 | 86.67 | 72.92 | 14968.7 | 56.67 | 80.00 | 60.00 | 17745.1 | 63.33 | 80.00 | 59.58 | 16358.5 |
| snt_tt_ota | OpenThoughts | 仅答案 | 1e-6 | 70.00 | 86.67 | 67.92 | 14481.8 | 46.67 | 80.00 | 59.58 | 17170.4 | 66.67 | 83.33 | 60.83 | 15238.1 |
| ot_think | OpenThoughts | 仅答案 | 5e-6 | 40.00 | 63.33 | 40.00 | 26119.9 | 40.00 | 63.33 | 42.92 | 25720.9 | 43.33 | 66.67 | 40.83 | 25856.7 |
| ot_think_sol | OpenThoughts | 完整轨迹 | 5e-6 | 70.00 | 76.67 | 67.50 | 15895.4 | 53.33 | 76.67 | 49.58 | 20021.4 | 63.33 | 80.00 | 55.42 | 18593.2 |
| ot_think | OpenThoughts | 仅答案 | 1e-6 | 33.33 | 63.33 | 36.67 | 28192.1 | 33.33 | 53.33 | 37.50 | 28138.4 | 26.67 | 66.67 | 35.83 | 28565.4 |
| ot_think_sol | OpenThoughts | 完整轨迹 | 1e-6 | 63.33 | 86.67 | 68.75 | 17079.3 | 63.33 | 83.33 | 61.67 | 19340.2 | 70.00 | 86.67 | 66.67 | 18213.7 |

---

## NoThink 评测（enable_thinking=0）

| 方法 | 训练数据 | Privilege | LR | A24 Pass@1 | A24 Pass@8 | A24 Avg@8 | A24 Length | A25 Pass@1 | A25 Pass@8 | A25 Avg@8 | A25 Length | A26 Pass@1 | A26 Pass@8 | A26 Avg@8 | A26 Length |
|------|----------|-----------|-----|-----------:|-----------:|----------:|-----------:|-----------:|-----------:|----------:|-----------:|-----------:|-----------:|----------:|-----------:|
| Qwen3-4B (base) | — | — | — | 30.00 | 50.00 | 25.00 | 8239.1 | 23.33 | 36.67 | 21.67 | 5726.9 | 10.00 | 43.33 | 19.58 | 7309.2 |
| snt_tt | DAPO | 仅答案 | 2e-6 | 23.33 | 50.00 | 27.08 | 19608.1 | 20.00 | 40.00 | 21.67 | 19456.0 | 13.33 | 36.67 | 17.92 | 19410.8 |
| snt_tt | DAPO | 仅答案 | 5e-6 | 20.00 | 46.67 | 22.92 | 26594.7 | 40.00 | 50.00 | 29.58 | 24841.7 | 20.00 | 43.33 | 20.00 | 26135.3 |
| snt_tt | DAPO | 仅答案 | 1e-6 | 23.33 | 46.67 | 20.42 | 13479.9 | 20.00 | 30.00 | 17.92 | 10355.1 | 13.33 | 36.67 | 20.83 | 10100.2 |
| snt_tt_ot | OpenThoughts | 完整轨迹 | 1e-6 | 16.67 | 46.67 | 22.50 | 10943.7 | 33.33 | 36.67 | 29.17 | 9458.9 | 36.67 | 46.67 | 32.50 | 8114.9 |
| snt_tt_ota | OpenThoughts | 仅答案 | 1e-6 | 33.33 | 63.33 | 34.17 | 8635.5 | 26.67 | 53.33 | 29.58 | 6611.9 | 16.67 | 46.67 | 21.25 | 9025.7 |
| ot_think | OpenThoughts | 仅答案 | 5e-6 | 16.67 | 56.67 | 27.50 | 24505.8 | 23.33 | 43.33 | 26.25 | 23854.1 | 20.00 | 36.67 | 20.42 | 25863.1 |
| ot_think_sol | OpenThoughts | 完整轨迹 | 5e-6 | 30.00 | 46.67 | 26.25 | 6355.2 | 10.00 | 30.00 | 16.25 | 4611.7 | 13.33 | 26.67 | 12.50 | 6938.3 |
| ot_think | OpenThoughts | 仅答案 | 1e-6 | 26.67 | 50.00 | 28.33 | 19573.7 | 26.67 | 36.67 | 20.83 | 21177.3 | 16.67 | 40.00 | 16.67 | 21302.2 |
| ot_think_sol | OpenThoughts | 完整轨迹 | 1e-6 | 23.33 | 46.67 | 21.67 | 10387.6 | 30.00 | 36.67 | 22.50 | 8479.5 | 16.67 | 50.00 | 20.83 | 7675.8 |

---

## 训练 / 评测对应关系（简表）

| 方法 | 训练输出目录 | 评测输出目录 |
|------|--------------|--------------|
| Qwen3-4B (base) | `/labShare/.../model/qwen3-4b` | `eval_outputs/qwen3-4b/` |
| snt_tt @2e-6 | `outputs/opsd_student_nothink_teacher_think_4b/2941631` | `eval_outputs/opsd_s_nt_t_th_4b_ckpt100/` / `snt_tt_2e_6_ckpt100/` |
| snt_tt @5e-6 | `outputs/opsd_student_nothink_teacher_think_4b/2941958` | `eval_outputs/opsd_s_nt_t_th_4b_2941958_ckpt100/` / `snt_tt_5e_6_ckpt100/` |
| snt_tt @1e-6 | `outputs/snt_tt_1e_6/2944387` | `eval_outputs/snt_tt_1e_6_ckpt100/` |
| snt_tt_ot @1e-6 | `outputs/snt_tt_1e_6_openthoughts/2947311` | `eval_outputs/snt_tt_1e6_ot_ckpt100/` |
| snt_tt_ota @1e-6 | `outputs/snt_tt_1e_6_openthoughts_answer/2948221` | `eval_outputs/snt_tt_1e6_ota_ckpt100/` |
| ot_think @5e-6 | `outputs/opsd_think_4b_ot/2951325` | `eval_outputs/ot_think_ckpt100/` |
| ot_think_sol @5e-6 | `outputs/opsd_think_solution_4b_ot/2951070` | `eval_outputs/ot_think_sol_ckpt100/` |
| ot_think @1e-6 | `outputs/opsd_think_4b_ot/2952332` | `eval_outputs/ot_think_1e6_ckpt100/` |
| ot_think_sol @1e-6 | `outputs/opsd_think_solution_4b_ot/2952336` | `eval_outputs/ot_think_sol_1e6_ckpt100/` |

日志目录：`OPSD_Analysis/log/eval/4b/aime{24,25,26}/`（文件名含 `_5e6_` / `_1e6_`）
