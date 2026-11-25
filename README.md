```bash
# Updated a strong Dolma 3 baseline

We have added **Dolma 3 as a strong recent baseline**, using its [official released quality-classifier weights](https://huggingface.co/allenai/dolma3-fasttext-quality-classifier). The 1.7B target models use the same candidate corpus, domain-token quotas, initialization, 8K context, global batch and 100B training schedule. We directly apply the released classifier weights without additional fitting.

The [Olmo 3 report, Section 3.4.1](https://arxiv.org/html/2512.13961v2#S3.SS4.SSS1.Px4) describes supervised fastText training with source-derived quality labels: OpenHermes-2.5 and ELI5 provide positive examples, supplemented by UltraChat-200k and WildChat-1M; negatives are 30GB sampled from DCLM-RefinedWeb. This source-based weak supervision gives Dolma 3 the benefit of dedicated quality-supervised classifier training. **AttentionInfluence is training-free at the data-selection stage:** it uses a frozen selector without fitting an additional quality classifier or requiring target-corpus quality labels. This training-free construction of the selection signal is a central contribution of our work.

**All four 1.7B target-model runs have completed 100B training tokens.** Both tables below use their final checkpoints at **step 95,368**.

Edu denotes the FineWeb-Edu classifier. **Our method (CoT-proxy)** is AttentionInfluence with the CoT-crucial proxy.

| Task | Random | Edu | Our method (CoT-proxy) | Dolma 3 |
|---|---:|---:|---:|---:|
| Step | 95,368 | 95,368 | 95,368 | 95,368 |
| Training tokens (B) | 100.00 | 100.00 | 100.00 | 100.00 |
| CommonSenseQA | 18.51 | 18.18 | 19.00 | 18.43 |
| PIQA | 74.86 | 73.45 | 74.43 | 74.27 |
| TriviaQA | 29.79 | 30.90 | 31.40 | 31.87 |
| WinoGrande | 59.04 | 57.30 | 60.46 | 58.09 |
| HellaSwag | 60.41 | 57.02 | 59.18 | 60.21 |
| ARC-Easy (acc) | 73.91 | 74.96 | 73.74 | 75.76 |
| ARC-Challenge (acc) | 41.21 | 43.43 | 40.70 | 42.75 |
| GSM8K | 3.64 | 2.50 | 4.09 | 3.71 |
| MATH | 1.12 | 2.26 | 2.32 | 2.00 |
| HumanEval | 7.93 | 7.04 | 8.23 | 5.21 |
| SVAMP | 10.60 | 12.40 | 19.40 | 18.00 |
| Mean | 34.64 | 34.49 | 35.72 | 35.48 |
| Mean Δ vs 100B Random (pp) | — | -0.14 | +1.09 | +0.84 |

The rBridge comparison uses the same final checkpoints; lower is better. Each aggregate weights its tasks equally: six tasks for 0-shot and two for 5-shot. Deltas use the 100B Random result.

| Data selection | 0-shot mean ↓ | Δ vs Random | 5-shot mean ↓ | Δ vs Random |
|---|---:|---:|---:|---:|
| Random | 0.63355 | — | 0.51385 | — |
| FineWeb-Edu classifier | 0.61061 | -0.02294 | 0.50222 | -0.01164 |
| Our method (CoT-proxy) | 0.60597 | -0.02758 | 0.49568 | -0.01818 |
| Dolma 3 classifier | 0.58792 | -0.04563 | 0.48602 | -0.02784 |

Dolma 3 remains a strong baseline. At 100B tokens, our method achieves an 11-task mean of 35.72% versus 35.48% for Dolma 3 (+0.24 percentage points), and scores higher on GSM8K, MATH, HumanEval and SVAMP than all three baselines. Dolma 3 obtains lower aggregate rBridge in both 0-shot (0.58792 vs. 0.60597) and 5-shot (0.48602 vs. 0.49568); both methods improve over Random in both conditions. These comparisons distinguish task performance from prediction of fixed reference reasoning.
---
# Reproduction
Please refer to the Supplementary Material for the latest code to reproduce our experiments.

For evaluation, please refer to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness).
```
