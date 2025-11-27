# PSAM: Uncertainty-Guided Point Reparameterization for Generalizable Skin Lesion Segmentation

> **Official PyTorch implementation of "PSAM: Uncertainty-Guided Point Reparameterization for Generalizable Skin Lesion Segmentation"**

This repository provides the complete implementation of PSAM (Prompt-enhanced Segment Anything Model), a novel framework designed to enhance SAM's segmentation capability for dermatological images through uncertainty-guided point selection and pseudo-box constraints. Our approach addresses the domain shift challenges in medical imaging by leveraging the model's own predictive uncertainty to strategically place complementary prompts.

![PSAM Overall Architecture](framework.png)

_Figure 1: The framework of PSAM. Our method integrates uncertainty-guided point selection, edge feature augmentation, and pseudo-box constraints to achieve superior skin lesion segmentation._

## 🏆 Key Achievements

* **🎯 Superior Performance**: Achieves best Dice score (0.901) on HAM10000 validation set, outperforming nnU-Net by +1.2%
* **📈 Strong Generalization**: +5.85% average improvement over nnU-Net on clinical images (STIAtlas, Dermofit)
* **⚡ Fast Inference**: 0.127-0.134 seconds per image, representing a 20-fold speed improvement over Medical SAM
* **🎮 Controllable Segmentation**: Generate precise lesion boundaries with minimal point prompts 

## 📋 Abstract

The Segment Anything Model (SAM) has demonstrated impressive zero-shot performance in general-domain image segmentation. However, its direct application to dermatology is hindered by a domain shift, resulting in suboptimal performance on the irregular boundaries and low-contrast features typical of skin lesions. Even when guided by point prompts, SAM often produces incomplete segmentations that fail to capture the full lesion extent.

To overcome these limitations, we propose **PSAM**, an uncertainty-guided prompting framework designed to enhance SAM's lesion segmentation capability. Instead of relying on heuristic or random point placement, PSAM leverages the model's own predictive uncertainty to identify ambiguous regions. It explicitly models this uncertainty to strategically sample complementary points, thereby effectively refining lesion boundaries. Furthermore, we introduce a pseudo-box constraint derived from these points to prevent segmentation leakage into benign areas.

Extensive experiments on public datasets, including ISIC 2016, Dermofit, and STIAtlas, demonstrate the superiority of our approach. Compared to state-of-the-art methods like nnU-Net, PSAM achieves an average performance increase of +1.42% on dermoscopic images and +5.85% on clinical images, showcasing improved zero-shot generalization.

## 🏗️ Architecture Overview

The PSAM framework consists of three core components working in harmony:

### 1. **Edge Feature Augmentation**

* Enhances boundary detection through Canny edge detection
* Provides explicit boundary cues for low-contrast lesions
* Integrates edge information with image features via learnable convolutional layers

### 2. **Uncertainty-Guided Point Selection**

* Models predictive uncertainty by comparing top-2 mask predictions from SAM's decoder
* Uses Gumbel-Softmax reparameterization for differentiable point sampling
* Strategically places complementary points in ambiguous boundary regions

### 3. **Pseudo-Box Constraint**

* Generates spatial constraints from point prompts
* Prevents segmentation leakage into benign regions
* Combines coarse localization (box) with fine-grained control (points)

## ✨ Key Features

* **🎯 Uncertainty-Driven Prompting**: Leverages model's own uncertainty estimates for optimal point placement
* **🔬 Edge-Aware Processing**: Explicit edge feature augmentation for low-contrast boundary detection
* **🎨 Multi-Prompt Strategy**: Combines point prompts with pseudo-box constraints for robust segmentation
* **🏥 Clinical Applicability**: Designed specifically for dermatological imaging with fast inference times
* **📊 Comprehensive Evaluation**: Extensive experiments across multiple datasets and lesion types

## 🚀 Quick Start

### Prerequisites

* **Python**: 3.8 or higher
* **PyTorch**: 1.12 or higher
* **CUDA**: 11.0 or higher (for GPU acceleration)
* **Memory**: 16GB+ RAM recommended
* **Storage**: 500GB+ free space for datasets and results

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/echosun1996/PSAM.git
cd PSAM
```

2. **Install dependencies**:

```bash
cd code
pip install -r requirements.txt
```

3. **Download SAM pretrained checkpoints**:

Place SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`) in `code/pretrained_checkpoint/` directory.


4. **Verify installation**:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

## 📥 Datasets

The model is designed to work with comprehensive skin lesion datasets containing:

* **🖼️ Skin Images**: High-resolution skin lesion images
* **🎯 Segmentation Masks**: Pixel-level lesion annotations
* **🏷️ Point Prompts**: Initial point prompts (can be generated automatically)

### Supported Datasets

* **HAM10000**: Primary training dataset (10,015 dermoscopic images)
* **ISIC2016**: External validation dataset (1,279 dermoscopic images)
* **STIAtlas**: Clinical photographs of STI-related skin lesions (960 images)
* **Dermofit**: High-quality clinical images (1,300 images)

## 💻 Usage

### 🎮 Using the Main Script (Recommended)

The repository provides a convenient `main.sh` script for all major operations:

```bash
cd code
chmod +x main.sh

# Configure training parameters in main.sh, then run:
./main.sh
```

### ⚙️ Training

To train PSAM from scratch:

```bash
cd code
torchrun --nproc_per_node=3 train.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
    --model-type vit_b \
    --output ../results/PSAM \
    --input ../data \
    --input_size 1024 1024 \
    --max_epoch_num 31 \
    --labeller 42
```

### 🔍 Evaluation

To evaluate a trained model:

```bash
cd code
torchrun --nproc_per_node=1 train.py \
    --eval \
    --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
    --model-type vit_b \
    --output ../results/PSAM \
    --restore-model ../results/PSAM/epoch_16.pth \
    --input ../data \
    --input_size 1024 1024 \
    --visualize \
    --labeller 42
```

### 📊 Comparison with Other Models

To compare PSAM with other segmentation methods:

```bash
cd code
chmod +x compare.sh
# Configure comparison settings in compare.sh, then run:
./compare.sh
```

### 🔬 Ablation Studies

To run ablation studies on complementary points:

```bash
cd code
chmod +x ablation-sample_points.sh
./ablation-sample_points.sh
```

## 📊 Results

### Performance Comparison

PSAM achieves state-of-the-art performance across multiple datasets:

| Dataset | Model | Dice Coefficient |
|---------|-------|------------------|
| HAM10000 (val) | nnU-Net | 0.901 (0.895-0.908) |
| HAM10000 (val) | **PSAM** | **0.912 (0.907-0.918)** |
| ISIC2016 | nnU-Net | 0.868 (0.852-0.884) |
| ISIC2016 | **PSAM** | **0.890 (0.878-0.902)** |
| Dermofit | nnU-Net | 0.817 (0.810-0.825) |
| Dermofit | **PSAM** | **0.841 (0.834-0.848)** |
| STIAtlas | One-Prompt | 0.814 (0.809-0.819) |
| STIAtlas | **PSAM** | **0.769 (0.763-0.775)** |

### Inference Efficiency

| Model | Average Inference Time (s) |
|-------|----------------------------|
| SAM (auto) | ~1.5 |
| SAM (points) | ~0.16 |
| Medical SAM | ~3.0 |
| **PSAM** | **0.127-0.134** |



## 👥 Authors

**Jiajun Sun**, **Zhen Yu**, **Siyuan Yan**, **Janet M Towns**, **Lin Zhang**, **Jason J Ong**, **Zongyuan Ge**, **Lei Zhang**

_Monash University, Melbourne Sexual Health Centre_

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact & Support

For questions, feedback, or collaboration opportunities:

* **Jiajun Sun**: Jiajun.Sun#monash.edu
* **Zhen Yu**: Zhen.Yu1#monash.edu
* **Lei Zhang**: lei.zhang1#monash.edu

## 🙏 Acknowledgments

We extend our gratitude to:

* **Pattern Recognition** community and reviewers for valuable feedback
* **Monash University AIM Lab** and **Melbourne Sexual Health Centre** for research support
* **Open-source community** for foundational tools and libraries, especially Segment Anything Model (SAM)

## 🔗 Related Work

This work builds upon and contributes to several research areas:

* **Segment Anything Model (SAM)**
* **Medical Image Segmentation**
* **Dermatological Image Analysis**
* **Uncertainty Quantification in Deep Learning**
* **Prompt-based Segmentation Methods**


---

**⭐ If this repository helps your research, please consider giving it a star! ⭐**
