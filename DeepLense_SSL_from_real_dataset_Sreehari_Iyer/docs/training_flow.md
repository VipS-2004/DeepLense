# Training Flow Overview (DeepLense SSL)

This document explains the end-to-end training pipeline used in the DeepLense self-supervised learning (SSL) project.
It is intended for new contributors who want to understand how data flows from input to evaluation.

---

## 1. Dataset Loading

The dataset consists of real gravitational lensing images divided into lens and non-lens categories.
Images are loaded from the local `input/` directory using predefined train, validation, and test splits stored in `indices.pkl`.

The `indices.pkl` file ensures that all experiments use consistent data splits for fair comparison.

---

## 2. Self-Supervised Pretraining

During SSL pretraining, the Vision Transformer (ViT) backbone is trained without using labels.
Instead, different augmented views of the same image are generated and used for representation learning.

Depending on the configuration, one of the following SSL methods is used:
- SimSiam
- DINO
- iBOT

The goal of this stage is to learn meaningful visual representations from unlabeled data.

---

## 3. Linear Probing / Fine-Tuning

After SSL pretraining:
- The ViT backbone is frozen
- A lightweight linear classifier is added on top

This classifier is trained using a limited fraction of labeled data to evaluate the quality of the learned representations.

---

## 4. Evaluation

The trained model is evaluated on a held-out test set.
Performance metrics such as accuracy and AUC are reported to compare:
- SSL-pretrained models
- Fully supervised baselines

---

## 5. Outputs and Checkpoints

During training and evaluation, the following outputs are generated:
- Model checkpoints
- Training logs
- Evaluation metrics

These outputs can be reused for downstream experiments or comparisons.
