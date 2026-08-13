# OPSD unified conclusions (pass@1 primary)

Primary metric: **pass@1** (`avg1_pct`). Secondary: `average_correct_pct`.
Only complete AIME/HMMT runs (n=30, non-partial) are included. MATH500 excluded by request.

## Nothink pass@1 mean over AIME24/25/26 + HMMT25

| Family | Run | NT mean pass@1 | Δ vs base |
|---|---|---:|---:|
| Qwen3.5 | Qwen3.5-4B base | 50.84 | — |
| Qwen3.5 | snt_tt FT (OT) | 65.00 | +14.16 |
| Qwen3.5 | snt_tnt FT (OT) | 56.67 | +5.83 |
| Qwen3.5 | st_tt FT (OT) | 36.67 | -14.17 |
| Qwen3.5 | st_tnt FT (OT) | 10.00 | -40.84 |
| Qwen3.5 | snt_tnt otas | 69.17 | +18.33 |
| Qwen3.5 | snt_tnt ios | 66.67 | +15.83 |
| Qwen3.5 | snt_tnt LoRA | 45.83 | -5.01 |
| Qwen3.5 | snt_tt LoRA | 20.00 | -30.84 |
| Qwen3.5 | st_tt LoRA | — | — |
| Qwen3-4B | Qwen3-4B base | 18.33 | — |
| Qwen3-4B | snt_tt FT (OT) | 27.50 | +9.17 |
| Qwen3-4B | snt_tnt FT (OT) | 26.67 | +8.34 |
| Qwen3-4B | st_tnt FT (OT) | 16.66 | -1.67 |
| Qwen3-4B | snt_tt LoRA | 18.33 | +0.00 |
| Qwen3-4B | snt_tnt LoRA | 17.50 | -0.83 |
| Qwen3-4B | st_tt LoRA | — | — |
| Instruct | 4B-Instruct base | 51.67 | — |
| Instruct | snt_tt LoRA | 55.00 | +3.33 |
| OLMo | OLMo3-7B-IT base | 39.17 | — |
| OLMo | snt_tnt FT | 18.34 | -20.83 |
| OLMo | snt_tnt LoRA | 41.66 | +2.49 |
| 1.7B | Qwen3-1.7B base | 7.50 | — |
| 1.7B | snt_tt FT | 11.67 | +4.17 |
| 1.7B | snt_tnt FT | 7.50 | +0.00 |
| 1.7B | snt_tt LoRA | 10.83 | +3.33 |
| 1.7B | st_tt LoRA | — | — |

## Per-dataset pass@1 (nothink)

| Family | Run | A24 | A25 | A26 | H25 |
|---|---|---:|---:|---:|---:|
| Qwen3.5 | Qwen3.5-4B base | 63.33 | 46.67 | 56.67 | 36.67 |
| Qwen3.5 | snt_tt FT (OT) | 73.33 | 60.00 | 73.33 | 53.33 |
| Qwen3.5 | snt_tnt FT (OT) | 56.67 | 50.00 | 70.00 | 50.00 |
| Qwen3.5 | st_tt FT (OT) | 33.33 | 46.67 | 46.67 | 20.00 |
| Qwen3.5 | st_tnt FT (OT) | 6.67 | 16.67 | 13.33 | 3.33 |
| Qwen3.5 | snt_tnt otas | 76.67 | 70.00 | 76.67 | 53.33 |
| Qwen3.5 | snt_tnt ios | 80.00 | 70.00 | 70.00 | 46.67 |
| Qwen3.5 | snt_tnt LoRA | 53.33 | 50.00 | 53.33 | 26.67 |
| Qwen3.5 | snt_tt LoRA | 23.33 | 23.33 | 23.33 | 10.00 |
| Qwen3.5 | st_tt LoRA | — | — | — | — |
| Qwen3-4B | Qwen3-4B base | 30.00 | 23.33 | 10.00 | 10.00 |
| Qwen3-4B | snt_tt FT (OT) | 16.67 | 33.33 | 36.67 | 23.33 |
| Qwen3-4B | snt_tnt FT (OT) | 30.00 | 26.67 | 30.00 | 20.00 |
| Qwen3-4B | st_tnt FT (OT) | 23.33 | 20.00 | 13.33 | 10.00 |
| Qwen3-4B | snt_tt LoRA | 23.33 | 23.33 | 16.67 | 10.00 |
| Qwen3-4B | snt_tnt LoRA | 16.67 | 23.33 | 16.67 | 13.33 |
| Qwen3-4B | st_tt LoRA | — | — | — | — |
| Instruct | 4B-Instruct base | 66.67 | 46.67 | 56.67 | 36.67 |
| Instruct | snt_tt LoRA | 50.00 | 63.33 | 66.67 | 40.00 |
| OLMo | OLMo3-7B-IT base | 46.67 | 36.67 | 46.67 | 26.67 |
| OLMo | snt_tnt FT | 26.67 | 26.67 | 6.67 | 13.33 |
| OLMo | snt_tnt LoRA | 53.33 | 40.00 | 50.00 | 23.33 |
| 1.7B | Qwen3-1.7B base | 10.00 | 10.00 | 10.00 | 0.00 |
| 1.7B | snt_tt FT | 13.33 | 10.00 | 16.67 | 6.67 |
| 1.7B | snt_tnt FT | 13.33 | 6.67 | 3.33 | 6.67 |
| 1.7B | snt_tt LoRA | 23.33 | 6.67 | 10.00 | 3.33 |
| 1.7B | st_tt LoRA | — | — | — | — |

## Pending / partial (excluded)

- `snt_tnt_ota_ckpt100`: hmmt25/nothink n=16
- `snt_tnt_otas_ckpt100`: hmmt25/nothink n=16
- `snt_tt_acc70_1p7b_ckpt100`: aime24/think n=16, aime25/think n=16, aime26/think n=16, hmmt25/think n=16
- `snt_tt_acc70_4b_ckpt100`: aime24/nothink n=8, aime24/think n=8, aime25/nothink n=24, aime25/think n=8, aime26/nothink n=8, aime26/think n=8
- `st_tt_1e6_ot_1p7b_ckpt100`: hmmt25/think n=8
