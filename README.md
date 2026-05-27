# WildLLM

ICWSM 2026 Paper: [WildLLM: Uncovering Hidden Wildlife Trafficking on Social Media with Augmented and Fine-Tuned LLMs](https://ojs.aaai.org/index.php/ICWSM/article/view/42629)

Wildlife trafficking is a serious global issue that increasingly uses online platforms for illegal trade. The dataset aims to help determine and prevent such activities through automated content analysis. 

Our research focused on using machine learning and large language models to identify wildlife trafficking patterns in online content, contributing to conservation efforts and the protection of endangered species. 

## Dataset

This dataset is designed to help the community prevent wildlife trafficking by tracking and analyzing content behavior patterns. Our research focuses on identifying and combating illegal wildlife trade through content analysis.

## Dataset Overview

- **Total Samples**: 2,805 data points
- **Source**: Based on the original dataset from [WLT-OSN](https://github.com/GMouYes/WLT-OSN/tree/main)
- **Purpose**: Wildlife trafficking prevention through behavioral content analysis
- **Data Balance**: Original imbalanced dataset has been adjusted to meet research requirements

## Files Description

### Core Dataset Files
| File | Description |
|------|-------------|
| `train.csv` | Training dataset |
| `val.csv` | Validation dataset |
| `test.csv` | Testing dataset |

### Augmented Dataset Files
| File | Description |
|------|-------------|
| `Train_multimodel_20_shot.csv` | Augmented dataset using **three different Large Language Models** combined with `train.csv` |
| `Train_llama_20_shot.csv` | Augmented dataset using **LLaMA 3.18B-Instruct model only** combined with `train.csv` |

## Code

We appreciate your interest in our research. We have shared our code in a single file `WildLlama.py`.

You can set hyperparameters and base configuration at the `#Base configuration` section.

### Getting Started

1. **Download the code**: `WildLlama.py`
2. **Configure parameters**: Modify the `#Base configuration` section according to your requirements
3. **Run the model**: Execute the script with your desired settings

## Data Augmentation

We have enhanced the original dataset using advanced language models to improve the training data quality and quantity:

- **Multi-Model Augmentation**: Utilized three different LLMs for diverse data generation
- **LLaMA-Specific Augmentation**: Focused augmentation using LLaMA 3.18B-Instruct model
- **20-Shot Learning**: Applied 20-shot learning techniques for data augmentation

## Data Source

Original dataset collected from previous research work:
- **Repository**: [GMouYes/WLT-OSN](https://github.com/GMouYes/WLT-OSN/tree/main)
- **Acknowledgment**: We acknowledge the original authors for their valuable contribution to wildlife trafficking detection research

## Usage

This dataset is shared with the community to support research and development efforts in:
- Wildlife trafficking detection
- Illegal wildlife trade prevention
- Content behavior analysis
- Machine learning model training for conservation efforts

## Data Balance

⚠️ **Note**: The original dataset was highly imbalanced. We have adjusted the class distribution ratios according to our specific research requirements to ensure more effective model training.

## Contributing

We encourage the community to use this data for good deeds and wildlife conservation efforts. If you use this dataset in your research, please consider citing the original work and contributing back to the community.

---

*This dataset is provided to support wildlife conservation efforts and combat illegal wildlife trafficking through advanced content analysis techniques.*
