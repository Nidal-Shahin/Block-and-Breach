# Block and Breach: LLM Security Research Framework

A comprehensive research framework for evaluating Large Language Model (LLM) security through systematic testing of attacks and defenses to optimize the security-utility tradeoff.

## Overview

This project conducts empirical research on LLM security by systematically testing various attack vectors against multiple target models with and without defensive mechanisms. Our goal is to identify the optimal balance between robust security protection and maintaining useful model functionality.

## Research Methodology

### Target Models

- **Llama-3-8B-Lexi-Uncensored**: Uncensored variant for baseline vulnerability assessment
- **Qwen2.5-7B-Instruct**: Production-grade model with built-in safety mechanisms
- **Strategic Model Selection**: Focus on models representing different safety paradigms

### Attack Vectors

- **Direct-Pass Attacks**: Baseline prompt injection without obfuscation
- **Future Attack Methods**: Framework designed for extensibility to additional attack types

### Defense Mechanisms

- **Baseline**: No additional defense (model's native safety only)
- **X-Guard**: External safety filtering system with configurable risk thresholds
  - **Training**: 425K+ prompts with adversarial tuning and XAI regularization
  - **Architecture**: RoBERTa-base classifier with FGM adversarial training
  - **Explainability**: Integrated Gradients for token-level attribution analysis
  - **Threshold**: Configurable risk scoring (default: 0.5)
- **Llama-Guard-3**: Meta's safety classifier for harmful content detection
- **Llama-Guard-4**: Latest generation safety classifier with enhanced capabilities
- **ShieldGemma-2B**: Google's compact safety model for efficient filtering
- **ShieldGemma-9B**: Large-scale safety model with comprehensive coverage
- **WildGuard-7B**: Advanced safety classifier with robust detection capabilities

### Defense Evalauation Diagram
![Defense Evaluation Diagram](https://lh3.googleusercontent.com/u/0/d/1BTKqvLdT9SrJpf7Vl1cloabgNBaxKL3M)

## Project Structure

```
├── data/
│   ├── evaluation_data/
│   │   ├── eval_500.csv          # Standardized evaluation dataset (500 prompts)
│   │   ├── link.txt              # Kaggle dataset reference
│   │   └── original_source/      # Source datasets (HarmBench, XSTest)
│   └── xguard_training_data/
│       ├── processed_data.csv.url # Kaggle dataset URL for large-scale training data
│       ├── link.txt               # Kaggle dataset reference
│       └── original_source/       # Original training datasets
├── notebooks/
│   ├── 0. X-Guard_Classifier/
│   │   ├── 0.1. Adversarial-Tuning_with_XAI-Regularization/
│   │   └── 0.2. XAI-Analysis/
│   ├── 1. Llama-3-8B-Lexi-Uncensored/
│   │   ├── 1.1.1. Llama_Direct-Pass_Baseline/
│   │   ├── 1.1.2. Llama_Direct-Pass_X-Guard/
│   │   ├── 1.1.3. Llama_Direct-Pass_Llama-Guard-3/
│   │   ├── 1.1.4. Llama_Direct-Pass_Llama-Guard-4/
│   │   ├── 1.1.5. Llama_Direct-Pass_ShieldGemma-2B/
│   │   ├── 1.1.6. Llama_Direct-Pass_ShieldGemma-9B/
│   │   └── 1.1.7. Llama_Direct-Pass_WildGuard-7B/
│   ├── 2. Qwen2.5-7B-Instruct/
│   │   ├── 2.1.1. Qwen_Direct-Pass_Baseline/
│   │   ├── 2.1.2. Qwen_Direct-Pass_X-Guard/
│   │   ├── 2.1.3. Qwen_Direct-Pass_Llama-Guard-3/
│   │   ├── 2.1.4. Qwen_Direct-Pass_Llama-Guard-4/
│   │   ├── 2.1.5. Qwen_Direct-Pass_ShieldGemma-2B/
│   │   ├── 2.1.6. Qwen_Direct-Pass_ShieldGemma-9B/
│   │   └── 2.1.7. Qwen_Direct-Pass_WildGuard-7B/
│   └── 3. Comparative_Analysis/
│       └── 3.1. Comprehensive_Results_Comparison/
│           ├── comprehensive-comparative-analysis.ipynb  # Cross-experiment analysis with SUE scoring
│           ├── output/  # Generated analysis results and visualizations
│           └── link.url  # Kaggle dataset reference
└── README.md
```

### Experimental Pipeline

Each experiment follows a standardized three-phase pipeline:

1. **Response Generation**: Target model processes prompts with/without defenses
2. **Response Evaluation**: External judge models assess response safety and utility
3. **Results Analysis**: Comprehensive metrics and visualizations

### Comparative Analysis Framework

- **Comprehensive Results Comparison**: Cross-experiment analysis across all 14 model-defense combinations
- **SUE Score (Security-Utility-Efficiency)**: Unified metric combining security effectiveness, utility preservation, and computational efficiency
- **Performance Benchmarking**: Latency and throughput analysis for production deployment considerations
- **Statistical Significance Testing**: Rigorous evaluation of defense performance improvements

### Key Metrics

- **Attack Success Rate (ASR)**: Percentage of harmful prompts that successfully elicit harmful responses
- **True Positive Rate (TPR)**: Defense effectiveness in blocking attacks (Task Performance Retention)
- **False Positive Rate (FPR)**: Over-blocking of benign requests (utility impact)
- **SUE Score (Security-Utility-Efficiency)**: Unified metric combining ASR, TPR, and computational efficiency
- **Attention-based Explainability**: Token-level influence analysis for interpretability
- **XAI Attribution Scores**: Integrated Gradients for understanding model decision patterns
- **Adversarial Robustness**: FGM-based perturbation resistance measurement
- **Throughput**: Requests processed per second (production deployment consideration)

## Data Schema

### Evaluation Dataset

Standardized CSV format with columns:

- `prompt`: Input text
- `category`: Attack classification (harmful_technical, harmful_social, benign_definitional, etc.)
- `label`: Binary harmful/benign classification
- `intent`: Primary intent (harmful/benign)
- `source`: Original dataset (HarmBench, XSTest, Original)

### Results Format

Each experiment generates:

- CSV files with model responses and metadata
- JSON files with attention-based explainability data
- Comprehensive evaluation metrics and visualizations

## Current Findings

### X-Guard Performance

- **High Effectiveness**: Demonstrates strong blocking of harmful requests with risk scores >0.99
- **Training Scale**: 425K+ prompts with 25% stratified subsampling for efficient training
- **Adversarial Robustness**: FGM perturbation with ε=0.5 for enhanced security
- **XAI Integration**: LayerIntegratedGradients for token-level attribution analysis
- **Utility Preservation**: Maintains acceptable false positive rates on benign requests

### Model Comparisons

- **Uncensored vs. Censored**: Baseline vulnerability assessment across safety paradigms
- **Defense Integration**: Comparative analysis of X-Guard vs. native model safety
- **Attention Patterns**: Token-level insights into model decision-making processes
- **Quantization Impact**: 4-bit NF4 quantization for efficient deployment

### Comparative Analysis Results

- **Best Overall Defense (SUE Score)**: X-Guard on Qwen2.5-7B-Instruct (96.39)
- **Highest Throughput**: X-Guard (77.00 it/s) - 781.8x faster than slowest defense
- **Maximum Security**: Llama-Guard-3 on Qwen2.5-7B-Instruct (ASR: 1.60%)
- **Optimal Balance**: X-Guard provides best security-utility-efficiency tradeoff for production deployment

### Technical Implementation

- **Memory Optimization**: Gradient checkpointing and explicit tensor management
- **Multi-GPU Support**: DataParallel training for dual T4 16GB configurations
- **Robust Training**: NaN/Inf gradient detection and recovery mechanisms
- **Performance Benchmarking**: Comprehensive latency and throughput analysis across all defenses

## Research Contributions

1. **Systematic Framework**: Reproducible methodology for LLM security evaluation
2. **Comprehensive Metrics**: Balanced security-utility assessment approach
3. **Explainability Integration**: Attention-based analysis for interpretability
4. **Extensible Design**: Framework supports additional models, attacks, and defenses

## Future Directions

- **Additional Attack Vectors**: Jailbreaking, prompt obfuscation, multi-turn attacks
- **Advanced Defenses**: White-box monitoring, adversarial training, prompt filtering
- **Model Scaling**: Evaluation across different model sizes and architectures
- **Real-world Deployment**: Production environment testing and optimization

## Usage

### Running Experiments

1. Navigate to the appropriate notebook directory
2. Execute Response Generation notebook
3. Run Response Evaluation with external judge models
4. Analyze results in Results notebook

### Data Requirements

- Hugging Face token for model access
- Sufficient computational resources (GPU recommended)
- External judge model access for evaluation phase

### Dependencies

```bash
pip install -U bitsandbytes>=0.46.1
pip install transformers torch
pip install pandas numpy seaborn matplotlib
pip install scikit-learn
pip install captum  # For XAI analysis
```

### Environment Setup

- **Platform**: Designed for Kaggle notebooks with GPU support
- **Authentication**: Uses `kaggle_secrets` for HF_TOKEN management
- **Hardware**: Optimized for dual T4 16GB GPUs with 4-bit quantization
- **Memory**: Implements gradient checkpointing and explicit memory management

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite appropriately.
