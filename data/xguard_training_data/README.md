# Disclaimer & Acknowledgments – X‑Guard Training Dataset

This dataset (`processed_data.csv`) is a **re‑aggregated and pre‑processed** compilation of prompt data sourced from several independent open‑source corpora. It has been created for the sole purpose of training the **X‑Guard** pre‑generation harmful‑content classifier. The dataset is **not** a standalone product; it is a derivative work that combines contributions from the original datasets listed below.

## Dataset Lineage & Composition

The training data has been aggregated from the following sources:

| Source | Description | License / Terms |
|--------|-------------|------------------|
| **WildJailbreak** | Large‑scale synthetic safety dataset containing vanilla (direct harmful) and adversarial (complex jailbreak) prompt‑response pairs; covers 13 risk categories. | Open‑source (ODC‑BY) |
| **GenTelBench‑v1** | Benchmark dataset designed to evaluate generalized safety and jailbreak detection; helps guardrail classifiers identify a broad spectrum of unsafe generation requests. | Provided for research and evaluation purposes under the GenTelLab terms. |
| **HarmBench Prompt Injection** | Independent set of harmful prompts tailored for injection attacks, verified to contain entirely unique prompt text. | MIT License |
| **JailbreakBench (JBB‑Behaviors)** | Standardised single‑turn benchmark comprising 100 distinct prohibited behaviours and 100 benign controls. | MIT License |
| **AdvBench** | Foundational baseline dataset of 500 harmful behaviours (instructions) used to measure base‑level refusal robustness. | MIT License |

Each source was originally formatted with varying column names and label encodings. All entries have been unified into a common schema that includes the raw prompt, a binary label (0 = benign, 1 = harmful), and metadata (e.g., intent, encoding type). The dataset has been balanced to contain approximately equal numbers of harmful and benign examples.

## License & Redistribution

Because this dataset is a **derivative work** of the sources listed above, it may be used only in ways that respect the original licenses. The dataset:

- **May** be used for research, safety training, and evaluation purposes as described in the X‑Guard project.
- **May not** be redistributed in any form without appropriate attribution to all original sources.
- **Must not** be used for any malicious, harmful, or illegal activities (see Safety Warning below).

The original creators bear no responsibility for any misuse or unintended consequences arising from the use of this re‑aggregated dataset. Users are responsible for complying with all applicable laws and the terms of the original licenses.

## Acknowledgements

We gratefully acknowledge the authors and contributors of the following works, which made this dataset possible:

- **WildJailbreak**: Liwei Jiang et al., “WildTeaming at Scale: From In‑the‑Wild Jailbreaks to (Adversarially) Safer Language Models”, arXiv:2406.18510 (2024).
- **GenTelBench‑v1**: GenTelLab Research Group, “GenTelBench‑v1: Generalized Jailbreaking and Safety Benchmark”, 2024. Available at [https://huggingface.co/datasets/GenTelLab/gentelbench-v1](https://huggingface.co/datasets/GenTelLab/gentelbench-v1).
- **HarmBench Prompt Injection**: M. Yehia, “HarmBench Prompt Injection Dataset”, GitHub, 2024. Available at [https://github.com/YEHIA060606/harmbench-prompt-injection](https://github.com/YEHIA060606/harmbench-prompt-injection).
- **JailbreakBench (JBB‑Behaviors)**: Patrick Chao et al., “JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models”, NeurIPS Datasets and Benchmarks Track (2024). Available at [https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors).
- **AdvBench**: Andy Zou et al., “Universal and Transferable Adversarial Attacks on Aligned Language Models”, arXiv:2307.15043 (2023).

## ⚠️ Safety & Usage Warning ⚠️

**This dataset contains examples of harmful, offensive, dangerous, or otherwise unsafe text.** It includes prompts that describe illegal activities, violence, self‑harm, harassment, and other harmful content. **Do not** use this dataset to generate or propagate such content.

**Intended Use**: This dataset is provided **only** for training and evaluating the X‑Guard harmful‑content classifier – a protective system designed to **detect and block** harmful prompts before they reach a language model. **Any other use is strictly prohibited.**

**Ethical Guidelines**:
- You **must not** use this data to train, fine‑tune, or otherwise improve any language model’s ability to generate harmful or unsafe responses.
- You **must not** use this data to create adversarial attacks or bypass content safety mechanisms.
- You **must not** use this data to harass, threaten, or cause harm to any person or group.
- You **must not** share or redistribute this dataset in any form without the explicit disclaimers and safety warnings provided here.

**Liability**: The authors of the original datasets, the creators of this aggregated dataset, and the X‑Guard team **disclaim any liability** for misuse, harm, or damage caused by the use (or misuse) of this dataset. By accessing or using this dataset, you agree to these terms.

---

*For any questions regarding attribution or licensing, please contact the X‑Guard team.  
This README is provided for informational purposes and does not constitute legal advice.*