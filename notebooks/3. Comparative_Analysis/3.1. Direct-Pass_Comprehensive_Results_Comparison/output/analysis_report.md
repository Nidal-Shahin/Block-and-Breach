# LLM Security Evaluation: Comprehensive Analysis Report
Generated: 2026-05-09 21:28:42

## 1. Executive Summary
This report evaluates 7 defense mechanisms across 2 target LLMs. 
We utilize the **SUE Score (Security-Utility-Efficiency)** to identify defenses that provide robust 
safety without compromising system latency.

### 🏆 Top Performers
* **Best Production Defense (SUE Score):** X-Guard on Qwen2.5-7B-Instruct (96.39)
* **Highest Throughput:** X-Guard (77.00 it/s)
* **Maximum Security (Lowest ASR):** Llama-Guard-3 on Qwen2.5-7B-Instruct (ASR: 1.60%)

## 2. Security-Utility-Efficiency (SUE) Rankings
The following table ranks defenses by their SUE score (Security + Utility - Latency Penalty):

| Rank | Defense | Model | SUE Score | ASR | TPR | Throughput |
|------|---------|-------|-----------|-----|-----|------------|
| 1 | X-Guard | Qwen2.5-7B-Instruct | 96.39 | 4.00% | 96.80% | 77.00 it/s |
| 2 | Llama-Guard-3 | Qwen2.5-7B-Instruct | 96.38 | 1.60% | 96.00% | 2.39 it/s |
| 3 | Llama-Guard-3 | Llama-3-8B-Lexi-Uncensored | 91.78 | 2.00% | 87.20% | 2.39 it/s |
| 4 | X-Guard | Llama-3-8B-Lexi-Uncensored | 89.39 | 9.20% | 88.00% | 77.00 it/s |
| 5 | ShieldGemma-9B | Qwen2.5-7B-Instruct | 82.89 | 24.40% | 91.20% | 3.77 it/s |
| 6 | Baseline | Qwen2.5-7B-Instruct | 82.40 | 34.00% | 98.80% | 100.00 it/s |
| 7 | WildGuard-7B | Qwen2.5-7B-Instruct | 79.59 | 28.80% | 94.40% | 0.62 it/s |
| 8 | ShieldGemma-9B | Llama-3-8B-Lexi-Uncensored | 76.09 | 30.40% | 83.60% | 3.77 it/s |
| 9 | Llama-Guard-4 | Qwen2.5-7B-Instruct | 74.91 | 1.60% | 92.00% | 0.10 it/s |
| 10 | ShieldGemma-2B | Qwen2.5-7B-Instruct | 74.14 | 22.40% | 71.20% | 7.09 it/s |

## 3. Key Efficiency Insights
* **Latency Cost:** The fastest defense (X-Guard) is 781.8x faster than the slowest defense.
* **Trade-off Observation:** High-security models often incur a significant "Time Tax." The SUE score identifies **X-Guard** as the optimal balance for real-time applications.

## 4. Conclusion
For deployments where user experience (latency) is as critical as safety, **X-Guard** is the recommended defense mechanism. If maximum security is required regardless of cost, **Llama-Guard-3** remains the strongest barrier.

---
*End of Report*
