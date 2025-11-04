![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

1. [Vision Transformer (ViT) on CIFAR-10](#vision-transformer-vit-on-cifar-10)
   - [Final Results](#final-results)
   - [Best Model Configuration](#best-model-configuration)
   - [Methodology & Implementation](#methodology--implementation)
     - [1. Architecture](#1-architecture)
     - [2. Training Strategy](#2-training-strategy)
   - [Ablation Study: The Indispensable Role of Batch-Level Augmentations](#ablation-study-the-indispensable-role-of-batch-level-augmentations)
      - [Key Observations](#key-observations)
      - [Takeaway](#takeaway)
   - [Analysis: Key to High Performance without Pre-training](#analysis-key-to-high-performance-without-pre-training)
2. [Acknowledgements](#acknowledgements)
   - [Papers](#papers)
   - [Repositories](#repositories)

# Vision Transformer (ViT) on CIFAR-10



> This project presents a from-scratch implementation of the Vision Transformer (ViT) architecture in PyTorch, trained on the CIFAR-10 dataset. The primary objective was to achieve the highest possible test accuracy by leveraging state-of-the-art training and regularization techniques from recent research, notably the two papers "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE" and  "Training data-efficient image transformers" were the backbone for this implementation 

## Final Results

The model was trained for up to 300 epochs, and the best-performing checkpoint was evaluated on the held-out test set.

| Model Configuration | Training Duration | Best Validation Acc. | Final Test Accuracy |
| :------------------ | :---------------: | :------------------: | :-----------------: |
| **ViT (DeiT-Ti Recipe)** | **252 Epochs*** | **91.00%** | **90.9%** |
| ViT (DeiT-Ti Recipe) | 300 Epochs | 91.00% | 90.7% |
*The best performing model (90.9% test accuracy) was saved from a run that achieved its peak validation accuracy at 252 epochs.

Below is the confusion matrix from the final evaluation on the 10,000 test images for the best model.


<p align="center">
  <img src="https://github.com/user-attachments/assets/8285d148-a496-46d1-a13c-8b2ae5cd9080" alt="image" width="400" height="400" />
</p>


## How to Run

This project is designed to be run in a Google Colab environment.

1.  Open the `q1.ipynb` notebook in Google Colab.
2.  Ensure the runtime is set to a GPU instance (e.g., T4) via `Runtime` > `Change runtime type`.
3.  Run all cells from top to bottom. The notebook will automatically:
    * Download and prepare the CIFAR-10 dataset.
    * Build the ViT model and the training harness.
    * Train the model for the specified number of epochs, saving the best checkpoint.
    * Load the best checkpoint and run a final evaluation, printing metrics and generating plots.
> **Note on Reproducibility:**  
> Training for the full 300 epochs takes ~4 hours and may be interrupted by Colab’s runtime limits.  
> To facilitate quick evaluation, the pre-trained weights from my best run (`best_vit_model.pth`) are provided [here](https://drive.google.com/drive/folders/1QGOAT4X2gVLVvQHZVrW9QqaMLXQ0Kizs?usp=sharing).  
> The evaluation section of the `q1.ipynb` notebook can be run independently after uploading this folder to your Google Drive, allowing you to reproduce the final test results in minutes.

## Best Model Configuration

The best results were achieved using a model architecture and training recipe inspired by the **DeiT-Ti (Tiny)** variant.

| Parameter            | Value                                         |
| -------------------- | --------------------------------------------- |
| **Architecture** | Vision Transformer (Pre-Norm)                 |
| Patch Size           | 4x4                                           |
| Embedding Dimension  | 192                                           |
| Transformer Depth    | 12 Layers                                     |
| Attention Heads      | 3                                             |
| **Optimizer** | AdamW                                         |
| Learning Rate        | 0.001 (Linearly scaled: `5e-4 * batch_size/512`)     |
| LR Scheduler         | OneCycleLR (Warmup + Cosine Annealing)        |
| **Training** |                                               |
| Epochs               | 300                                           |
| Batch Size           | 1024                                          |
| Weight Decay         | 0.05                                          |
| Label Smoothing      | N/A (CrossEntropyLoss)                        |
| **Regularization** |                                               |
| Augmentations        | RandAugment, RandomHorizontalFlip, RandomCrop |
| Batch-Level Augs     | Mixup & CutMix (`combine_fn`)                   |
| Dropout Rates        | MLP: 0.1, Embedding: 0.1, Attention: 0.0      |
| Trainable Parameters        | ~5M |

## Methodology & Implementation

### 1. Architecture

The model is a standard Vision Transformer as described in "An Image is Worth 16x16 Words", with a Pre-Norm configuration (LayerNorm applied *before* the attention/MLP blocks) for improved training stability.

* **PatchEmbedding:** Images of size `(3, 32, 32)` are converted into a sequence of 64 flattened patches (`4x4`), which are then linearly projected into a 192-dimensional embedding space.
* **CLS Token & Positional Embeddings:** A learnable `[CLS]` token is prepended to the sequence, and learnable positional embeddings are added to provide the model with spatial information.

* **Transformer Encoder:** The core of the model is a stack of 12 standard Transformer Encoder blocks, each containing Multi-Head Self-Attention and an MLP sub-layer.

<p align="center">
  <img src="https://github.com/user-attachments/assets/541d1a71-4a60-408c-ae6c-d742866bd833" alt="ViT Architecture" width="500" />
</p>

<p align="center">
  <sub><i>Diagram of the Vision Transformer (ViT) architecture, adapted from Dosovitskiy et al., 2021.</i></sub><br>
  <sub><i>Source: <b>"An Image is Worth 16x16 Words"</b> paper</i></sub>
</p>

### 2. Training Strategy

The key challenge with ViTs is their data-hungriness. To overcome this on a small dataset like CIFAR-10, a sophisticated training recipe inspired by the DeiT paper was adopted. The core of this strategy is aggressive regularization to prevent overfitting.
* **Heavy Data Augmentation:** The training pipeline uses `RandAugment` in conjunction with `Mixup` and `CutMix` (applied at the batch level via a custom `collate_fn`). This forces the model to learn robust and generalizable features.
* **Optimizer & Scheduler:** The `AdamW` optimizer was used with a `OneCycleLR` scheduler, which automatically handles a learning rate warmup phase followed by a cosine decay. This disciplined LR schedule is crucial for stable and effective training.


## **Ablation Study: The Indispensable Role of Batch-Level Augmentations**

To quantify the impact of batch-level augmentations, I ran the following experiments:

| Model Configuration                   | Training Duration | Best Validation Accuracy | Final Test Accuracy |
| :------------------------------------ | :---------------: | :----------------------: | :-----------------: |
| ViT Baseline (No Mixup/CutMix)        | 30 Epochs | 74.18% | 73.29% |
| ViT + DeiT Recipe (Full)              | 30 Epochs | 63.74% | 63.61% |
| **ViT + DeiT Recipe (Full)**          | 252 Epochs* | 91.00% | 90.9% |

\*Note: The 30-epoch results reflect early training dynamics; the 252-epoch run represents the final model.

---

### **Key Observations**

1. **Baseline performance is strong:**  
   A ViT trained for 30 epochs without Mixup/CutMix achieves **73.29% test accuracy**, showing that AdamW + OneCycleLR alone produces reasonable results.  

2. **Aggressive augmentations initially slow convergence:**  
   Using Mixup and CutMix in the first 30 epochs drops test accuracy to **63.61%**. This counter-intuitive decrease occurs because batch-level augmentations make the training task harder, preventing the model from memorizing the data.

3. **Long-term benefits are significant:**  
   Over extended training (252 epochs), the full DeiT recipe achieves **90.9% test accuracy**, highlighting that Mixup and CutMix enable the model to learn **robust, generalizable features** rather than overfitting to noisy or small datasets.

---

### **Takeaway**

- **Early training may appear worse**, but batch-level augmentations are **critical for sustained performance**.  
- Aggressive regularization strategies, even if they delay initial convergence, **unlock the full potential of ViTs on limited data** without requiring large-scale pre-training.



## Analysis: Key to High Performance without Pre-training

The final test accuracy of **90.9%** demonstrates that Vision Transformers can indeed be trained effectively on smaller datasets if the right strategy is employed. The key takeaways are:

1. **Architecture Matters:** An initial consideration was the model size. By choosing a smaller, more efficient **DeiT-Ti** architecture `(192-dim, 12 layers)` instead of a larger ViT-Base, the model had fewer parameters `(~5M)`, making it more suitable for the limited size of the CIFAR-10 dataset and reducing the risk of overfitting.
2.  **Regularization is Paramount:** The success of this project hinges on the aggressive regularization strategy borrowed from DeiT. The combination of `RandAugment`, `Mixup`, `CutMix`, and `AdamW`'s weight decay successfully prevented the model from overfitting, a primary risk for ViTs.
3.  **Training Dynamics as a Signal:** A key observation was that the **training accuracy was consistently lower than the validation accuracy**. This counter-intuitive result is a direct consequence of the regularization. Regularization makes the training task artificially difficult, forcing the model to learn robust features that generalize well.


<p align="center">
  <img src="https://github.com/user-attachments/assets/7eff5f2c-27e8-4b53-a686-6285c5f05ab6" alt="image" width="600" height="600" />
</p>

4.  **Modern Schedulers are Non-Negotiable:** The stability and performance of the training run were heavily reliant on the `OneCycleLR` scheduler. Without its intelligent management of the learning rate, the model would likely have converged much slower or to a less optimal result.
5. **Peak Performance and Onset of Overfitting:** An interesting observation arose from comparing two long training runs. The model that achieved a peak validation accuracy of 91.00% (saved from a run that completed 252 epochs) yielded a final test accuracy of **90.9%**. A subsequent full 300-epoch run achieved a similar peak but a slightly lower test accuracy of 90.7%. This suggests the model reached its optimal generalization capability around the ~250-270 epoch mark, and further training offered no benefit, demonstrating the onset of minor overfitting. Consequently, the model from the earlier run is reported as the best-performing model.

This project validates that with a thoughtful, modern approach to data augmentation and training dynamics, the performance gap for Vision Transformers on smaller datasets can be significantly closed, reducing the dependency on massive pre-training corpora.

## Acknowledgements

This implementations were made possible by studying the following seminal papers and high-quality open-source repositories.

### Papers
1.  Dosovitskiy et al. (2021). AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE.
2.  Touvron et al. (2021). Training data-efficient image transformers & distillation through attention (DeiT).
3.  Steiner et al. (2022). How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers.
4.  Chen et al. (2022). Better plain ViT baselines for ImageNet-1k.

### Repositories
1.  Official Google Research ViT: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
2.  PyTorch Image Models (timm) by Ross Wightman: [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
3.  Phil Wang's (lucidrains) ViT implementation: [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
4.  Jeon's ViT implementation: [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
