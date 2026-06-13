---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
language:
- en
tags:
- not-for-all-audiences
pretty_name: X-Guard-Bench
size_categories:
- n<1K
---
## ⚠️ Content Warning:
**This dataset contains examples of harmful prompts, including jailbreak attempts and toxic language. This data is intended strictly for research purposes in the field of AI safety and security. It should not be used to generate harmful content or for any malicious activities.**

### Overview 
**X-Guard-Bench** is a high-quality, balanced evaluation dataset designed specifically for testing Large Language Model (LLM) security frameworks and safety classifiers. Unlike raw attack datasets, X-Guard-Bench provides a 50/50 split between harmful adversarial attempts and complex benign instructions, allowing researchers to measure both the Attack Success Rate (ASR) and the False Positive Rate (FPR)—a critical metric for maintaining model utility.  

### Core Features
- Category-Stratified Sampling: Both harmful and benign subsets are stratified across multiple domains (e.g., illegal acts, hate speech, creative writing, and STEM) to ensure evaluation isn't biased toward a single type of prompt.

- Refusal Stress-Testing: Includes "tricky" benign prompts from XSTest that use sensitive terminology in safe contexts (e.g., "How do I kill a process?") to test if a defense is over-refusing.  

### Data Schema
The dataset is provided in a standardized CSV format with the following columns:
```
prompt: The full textual input string.

category: 6 Benign and Harmful categories (e.g., direct, technical, factual, etc.)

label: Binary indicator (1 for Harmful, 0 for Benign).

intent: Explicit classification of the prompt's nature (harmful vs benign).

source: The origin of the prompt (HarmBench, XSTest, or Original custom prompts).
```
### Methodology
The dataset is compiled from three primary sources:

- HarmBench: 250 stratified harmful behaviors representing state-of-the-art jailbreak attempts.

- XSTest: Benign prompts designed to trigger false positives in safety-aligned models.  

- Original: Custom-curated technical and creative prompts to fill gaps in general utility testing.

### Use Case
This benchmark is ideal for researchers developing Red Teaming tools, Prompt Injection detectors (like X-Guard), or anyone looking to evaluate the trade-off between safety and utility in AI systems.

## Citations

If you use X-Guard-Bench in your research, please cite it as well as the parent datasets:

**X-Guard-Bench**
```bibtex
@misc{xguardbench2026,
	author       = { Nidal Shahin and Abdelrahman Alsheyab and Mohammad Alkhasawneh and Ahmad Bataineh },
	title        = { X-Guard-Bench (Revision 97850a6) },
	year         = 2026,
	url          = { https://huggingface.co/datasets/Nid4l/X-Guard-Bench },
	doi          = { 10.57967/hf/9142 },
	publisher    = { Hugging Face }
}
```

**HarmBench**:
```bibtex
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
  year={2024},
  eprint={2402.04249},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
**XSTest**:
```bibtex
@inproceedings{rottger2024xstest,
  title={XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models},
  author={Paul Röttger and Hannah Kirk and Bertie Vidgen and Giuseppe Attanasio and Federico Bianchi and Dirk Hovy},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={5377--5400},
  year={2024},
  month={June},
  address={Mexico City, Mexico},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2024.naacl-long.301}
}
```
