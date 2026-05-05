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
- **Future Defenses**: Framework supports integration of additional defensive approaches

## Project Structure

```
├── data/
│   ├── evaluation_data/
│   │   ├── eval_500.csv          # Standardized evaluation dataset (500 prompts)
│   │   └── original_source/      # Source datasets (HarmBench, XSTest)
│   └── xguard_training_data/
│       ├── processed_data.csv    # Large-scale training data for X-Guard
│       └── original_source/      # Original training datasets
├── notebooks/
│   ├── 1.1.1. Llama_Direct-Pass_Baseline/
│   ├── 1.1.2. Llama_Direct-Pass_X-Guard/
│   ├── 2.1.1. Qwen_Direct-Pass_Baseline/
│   └── 2.1.2. Qwen_Direct-Pass_X-Guard/
└── README.md
```

### Experimental Pipeline

Each experiment follows a standardized three-phase pipeline:

1. **Response Generation**: Target model processes prompts with/without defenses
2. **Response Evaluation**: External judge models assess response safety and utility
3. **Results Analysis**: Comprehensive metrics and visualizations

### Key Metrics

- **Attack Success Rate (ASR)**: Percentage of harmful prompts that successfully elicit harmful responses
- **True Positive Rate (TPR)**: Defense effectiveness in blocking attacks
- **False Positive Rate (FPR)**: Over-blocking of benign requests (utility impact)
- **Attention-based Explainability**: Token-level influence analysis for interpretability

## Data Schema

### Evaluation Dataset
Standardized CSV format with columns:
- `prompt`: Input text
- `category`: Attack classification (harmful_technical, harmful_social, benign_definitional, etc.)
- `label`: Binary harmful/benign classification
- `intent`: Primary intent (harmful/benign)
- `source`: Original dataset (HarmBench, XSTest)

### Results Format
Each experiment generates:
- CSV files with model responses and metadata
- JSON files with attention-based explainability data
- Comprehensive evaluation metrics and visualizations

## Current Findings

### X-Guard Performance
- **High Effectiveness**: Demonstrates strong blocking of harmful requests
- **Risk Scoring**: Configurable threshold system (default: 0.5)
- **Utility Preservation**: Maintains acceptable false positive rates

### Model Comparisons
- **Uncensored vs. Censored**: Baseline vulnerability assessment
- **Defense Integration**: Comparative analysis of defense mechanisms
- **Attention Patterns**: Token-level insights into model decision-making

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

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite appropriately.
