# Block and Breach: LLM Security Research Framework

A comprehensive research framework for evaluating Large Language Model (LLM) security through systematic empirical validation of adversarial attack vectors, external defensive classifier layers, and mechanistic explainability telemetry. This architecture establishes a technical pipeline designed to analyze and optimize the security-utility-efficiency (SUE) trade-offs inherent in production-grade and uncensored language models.

---

## 0. License

**Code** (all `.py`, `.ipynb`, and other source files) is licensed under the [MIT License](LICENSE).

**Dataset** (`processed_data.csv`, `eval_500.csv` and any other data files) is licensed separately under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](DATA_LICENSE).  
This means you may use the dataset for non‑commercial purposes only, and any derivative work must give credit and share under the same terms.

---

## 1. System Architecture Overview

The framework evaluates safety alignment vulnerability and external classifier defense robustness across eight interconnected technical flows:


```

[ Raw Evaluation Corpus (eval_500.csv) ]
│
├───► [ Direct-Pass Sub-Pipeline ] ───► [ Defense Filter Layer ] ─────►  [ Target LLM Ingestion ]
│                                                │                                   │
│                                                ▼                                   ▼
└───► [ Iterative Refinement ] ────────► [ Refusal Intercept ]           [ Raw Text Generation ]
                 │                                                                   │
                 ▼                                                                   ▼
    [ Mutation Engine (LLM Gen) ]                                       [ Attention Capture (xAI) ]
                 │                                                                   │
                 ▼                                                                   ▼
    [ Adversarial Verification ]                                         [ External Judge Model ]
                 │                                                                   │
                 ▼                                                                   ▼
   [ Success Prompt Extraction ] ──► [ Comparative Robustness ] ──────►  [ Cross-Axis Analytics ]

```

1. **Dataset Flow:** Traces inputs from raw datasets (`processed_data.csv` for classifier tuning; `eval_500.csv` stratified across explicit safety risk vectors for evaluations) through downstream partitions.
2. **Prompt Flow:** Routes static language strings directly (Direct-Pass) or wraps inputs within conditional mutation blocks (Iterative Refinement) depending on the active test paradigm.
3. **Attack Generation Flow:** Implements zero-obfuscation direct prompts or orchestrates multi-agent adversarial mutation routines using a high-capacity generator (`llama-3.3-70b-versatile` via Groq API) coupled with character-level and semantic transformation primitives.
4. **Defense Layer Flow:** Coordinates evaluation across six upstream guard systems: our regularized **X-Guard** module, the Llama-Guard ecosystem, the ShieldGemma framework, and the instruction-guided WildGuard architecture.
5. **Response Generation Flow:** Conducts deterministic execution (`do_sample=False`) on target backends (4-bit quantized `Llama-3-8B-Lexi-Uncensored` and `Qwen2.5-7B-Instruct`) to extract reproducible token paths and intermediate tensors.
6. **Response Evaluation Flow:** Assigns generations into mutually exclusive safety-utility matrices using high-fidelity judges (`DeepSeek-V4`) or audits breach depth via Bypass Severity Taxonomy (BST) and Explicit Harm Binary (EHB) rules.
7. **Metric Generation Flow:** Aggregates categorical outcomes into a multi-axis statistical matrix computing Attack Success/Block Rates (ASR/ABR), Task Performance Retention (TPR), Over-Refusal Rates (ORR), and composite SUE indices.
8. **XAI Analysis Flow:** Captures mechanistic telemetry using `Captum LayerIntegratedGradients` on defense classifiers alongside target model generation-step attention maps to expose global token-importance rankings and structural vulnerabilities.

---

## 2. Directory Structure Matrix


```

.
├── data/
│   ├── evaluation_data/                        # Central evaluation prompt collections
│   │   ├── original_source/
│   │   │   ├── HarmBench/                      # Standardized adversarial benchmarks
│   │   │   └── XSTest/                         # Calibration sets for exaggerated over-refusal
│   │   └── eval_500.csv                        # Main 500-prompt evaluation dataset
│   └── xguard_training_data/                   # Upstream classifier datasets
│       ├── original_source/                    # URLs to source text collections
│       └── processed_data.csv.url              # Primary multi-source fine-tuning corpus
│
├── model/                                      # Model artifacts, checkpoints, and references
├── paper/                                      # Paper sources and compiled manuscript files
└── src/
├── 0. X-Guard_Classifier/                      # Defense Fine-tuning & Introspection
│   ├── 0.1. Adversarial-Tuning_with_XAI-Regularization/
│   └── 0.2. X-Guard_XAI_Analysis/              # Generates global token importance & positional smoothing maps
├── 1. Direct-Pass_Evaluation/                  # Static Evaluation Pipeline
│   ├── 1.1. Llama-3-8B-Lexi_Evaluation/        # Direct-pass baselines and guards on Llama-3
│   └── 1.2. Qwen2.5-7B-Instruct_Evaluation/    # Direct-pass baselines and guards on Qwen2.5
├── 2. Iterative-Refinement_Evaluation/         # Evolutionary Attack Pipeline
│   ├── 2.1. Llama-3-8B-Lexi_Evaluation/        # Groq-mutated prompt iterations & filters on Llama-3
│   └── 2.2. Qwen2.5-7B-Instruct_Evaluation/    # Groq-mutated prompt iterations & filters on Qwen2.5
└── 3. Comparative_Analysis/                    # Cross-Axis Aggregation and Metrics Synthesis
├── 3.1. Direct-Pass_Comprehensive_Results_Comparison/
└── 3.2. Iterative-Refinement_Comprehensive_Results_Comparison/

```

---

## 3. Experimental Pipeline & Usage Lifecycle

To execute or reproduce the experiments, navigate to the relevant sub-directories within `src/` and execute the pipeline phases sequentially:

### Phase 0: Defense Preparation & X-Guard Training
* **Component:** `src/0. X-Guard_Classifier/0.1. Adversarial-Tuning_with_XAI-Regularization/`
* **Operation:** Process the underlying merged dataset (`processed_data.csv`) through an adversarial min-max optimization loop. This couples functional cross-entropy objectives with `Captum`-driven explainable AI (xAI) token regularization to generate robust model checkpoints (`xguard_best`, `xguard_core`) and telemetry logs (`xai_observations.jsonl`).
* **Introspection:** Run `0.2. X-Guard_XAI_Analysis` to compute positional intensity overlays and isolate attack indicator tokens from the training logs.

### Phase 1: Static Direct-Pass Benchmarking
* **Component:** `src/1. Direct-Pass_Evaluation/`
* **Operation:** 1. Execute the baseline notebooks within the `Llama` and `Qwen` sub-directories to ingest `eval_500.csv`, record raw text outputs, and dump attention tensor fields.
  2. Route these baseline configurations through the respective classifier notebooks (X-Guard, Llama-Guard-3-8B, Llama-Guard-4-12B, ShieldGemma-2B, ShieldGemma-9B, and WildGuard-7B) to cache classification decisions and establish token filtering boundaries.
  3. Responses are evaluated via a `DeepSeek-V4` judge to construct a multi-axis performance layout.

### Phase 2: Evolutionary Iterative Refinement Attacks
* **Component:** `src/2. Iterative-Refinement_Evaluation/`
* **Operation:** 1. Deploy the multi-agent adversarial loop against the target models. Unaligned prompts are processed via the `llama-3.3-70b-versatile` attacker engine using synonym mapping, character obfuscation, and attention evasion.
  2. Filter mutations locally through `distilroberta-base-rejection-v1` with a conservative fallback to Gemini clusters (`gemma-4-31b-it`, `gemini-3.5-flash`) for indeterminate outputs.
  3. Extract successful bypass vectors and parse them against BST and EHB taxonomies to derive accurate ground-truth safety states.
  4. Evaluate all six external guard networks against these successful adversarial branches to measure downstream detection accuracy under refined stress.

### Phase 3: Analytical Aggregation and Optimization Modeling
* **Component:** `src/3. Comparative_Analysis/`
* **Operation:** Execute the cross-experiment consolidation modules. The analysis scripts ingest intermediate outputs across both attack branches to export combined evaluation dataframes (`comprehensive_metrics.csv`), defensive charts (`ABR_BST_Comparison.png`), and multidimensional SUE trade-off scatter metrics.

---

## 4. Hardware and Computational Environment

* **Target Compute Platform:** Kaggle Environment optimized for execution on dual NVIDIA T4 (16GB) GPU clusters.
* **Memory Management Strategy:** Implements 4-bit model quantization via `BitsAndBytes` layers, coupled with selective token caching and gradient checkpoint routines to accommodate LLM generation and evaluation workloads.
* **Determinism Protocol:** All inference nodes enforce `do_sample=False` to ensure reproducible attention attribution layers and consistent text synthesis across independent pipeline runs.

---

## 5. System Dependencies

Deploy the core framework and environment prerequisites via:

```bash
pip install -U bitsandbytes>=0.46.1
pip install transformers torch
pip install pandas numpy seaborn matplotlib
pip install scikit-learn
pip install captum  # Mandatory for mechanistic XAI and regularized X-Guard loops

```

## 6. Authors
*Jordan University of Science and Technology* 
 1. Prof. Ahmad Bataineh, asbataineh@just.edu.jo
 2. Abdelrahman Alsheyab, arahmadalsheyab22@cit.just.edu.jo
 3. Mohamamd Alkhasawneh, myalkhasawneh22@cit.just.edu.jo
 4. Nidal Shahin, nkhameedshahin22@cit.just.edu.jo
