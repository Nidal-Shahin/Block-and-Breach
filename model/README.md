---
license: apache-2.0
tags:
- security
- safety
- adversarial
- RoBERTa
- text-classification
language:
- en
pipeline_tag: text-classification
extra_gated_heading: Access X-Guard Model
extra_gated_prompt: >
  This model is released under the **Apache 2.0 License**.

  Access is granted on an individual basis. Please provide your details below.

  We collect this information to understand the usage of this research artifact.
  Your information will not be shared with third parties.
extra_gated_button_content: Submit Access Request
extra_gated_fields:
  Name: text
  Affilation / Organization: text
  Email: text
  Reason for Use: text
datasets:
- allenai/wildjailbreak
- GenTelLab/gentelbench-v1
- JailbreakBench/JBB-Behaviors
- walledai/AdvBench
metrics:
- name: Accuracy
  type: accuracy
  value: 0.9930
- name: Benign Precision
  type: precision
  value: 0.9914
- name: Benign Recall
  type: recall
  value: 0.9944
- name: Benign F1
  type: f1
  value: 0.9929
- name: Harmful Precision
  type: precision
  value: 0.9946
- name: Harmful Recall
  type: recall
  value: 0.9917
- name: Harmful F1
  type: f1
  value: 0.9932
base_model:
- FacebookAI/roberta-base
library_name: transformers
---

# Model Card for X-Guard

## Model Description

**X-Guard** is a compact and high-throughput harmful-content classifier designed for pre-generation filtering in Large Language Models (LLMs). Built on a **RoBERTa-base** encoder, it is fine-tuned to detect harmful prompts and adversarial jailbreak attempts. The model uses a **min-max adversarial fine-tuning loop (FGM)** combined with **Explainable AI (xAI) regularization (LIG)** to ensure robustness and transparency. It is a core component of the **Block and Breach** framework.

- **Model Type:** Transformer-based text classification (Encoder-only)
- **Base Model:** [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) (MIT License)
- **Language(s):** English
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Parameters:** 125 Million
- **Disk Size:** ~500 MB (FP16)

## Intended Uses & Limitations

### Direct Use
X-Guard is intended to be used as a pre-generation filter for LLMs. Given an input prompt, it outputs a binary classification:
- **LABEL_1:** Harmful
- **LABEL_0:** Benign

### Out-of-Scope Use
- Do not use as a standalone content moderator without human oversight.
- Not designed for detecting non-textual harm (e.g., images, audio).
- Not intended for real-time systems without evaluating latency on target hardware.

### Limitations
- The model was trained and evaluated on an English-only dataset.
- Performance on prompts outside the distribution of the training data (e.g., extremely long prompts) may degrade.
- The current version supports a maximum sequence length of 256 tokens.

## Training Details

### Training Data
X-Guard was fine-tuned on a stratified 25% subsample of a curated dataset compiled from five open-source corpora: WildJailbreak, GenTelBench-v1, HarmBench Prompt Injection, JailbreakBench, and AdvBench. The total training set used was approximately 85k prompts after stratification.

### Training Procedure
The model was fine-tuned for 3 epochs using the AdamW optimizer with a learning rate of 1e-5, a batch size of 16, and a fixed sequence length of 256. A two-stage stratified random split (70/15/15) was applied to the training data, with the final test set held out for final evaluation.

- **Adversarial Training:** Fast Gradient Method (FGM) with a perturbation radius ε = 0.5.
- **xAI Regularization:** Layer Integrated Gradients (LIG) with penalty weight λ = 0.5, applied every 40 steps.
- **Gradient Clipping:** 1.0

## Evaluation Metrics

On the held-out test set, X-Guard achieved the following standalone performance:
- **Overall Accuracy:** 99.30%
- **Harmful Class:** Precision 99.46%, Recall 99.17%, F1 99.32%
- **Benign Class:** Precision 99.14%, Recall 99.44%, F1 99.29%

## Citation

If you use X-Guard in your research, please cite the associated paper:

```bibtex
@misc{xguard2026,
	author       = { Nidal Shahin and Abdelrahman Alsheyab and Mohammad Alkhasawneh and Ahmad Bataineh },
	title        = { X-Guard-Bench (Revision 7e9f62b) },
	year         = 2026,
	url          = { https://huggingface.co/Nid4l/X-Guard-Bench },
	doi          = { 10.57967/hf/9143 },
	publisher    = { Hugging Face }
}
```
Also cite:

```bibtex
@article{DBLP:journals/corr/abs-1907-11692,
  author    = {Yinhan Liu and
               Myle Ott and
               Naman Goyal and
               Jingfei Du and
               Mandar Joshi and
               Danqi Chen and
               Omer Levy and
               Mike Lewis and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
  journal   = {CoRR},
  volume    = {abs/1907.11692},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11692},
  archivePrefix = {arXiv},
  eprint    = {1907.11692},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-11692.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```