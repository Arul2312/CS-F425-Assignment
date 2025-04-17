# Kolmogorov–Arnold Networks (KAN) - Reimplementation

This repository contains a custom implementation of the **Kolmogorov–Arnold Networks (KAN)**, based on the paper _["Kolmogorov–Arnold Networks" (arXiv:2401.01416)](https://arxiv.org/abs/2401.01416)_. The KAN model introduces a novel neural network architecture that replaces traditional neuron-based activations with **grid-based spline interpolations**, enabling better interpretability, efficiency, and flexibility.

> 📌 This project was done as part of the CS F425 - Deep Learning course project.

---

## 📁 Repository Structure

```
CS-F425-Assignment/
├── kan.py              # Core implementation of the KAN architecture
├── spline_utils.py     # Utility functions for spline activations
├── trainer.py          # Training and evaluation routines
├── datasets.py         # Data preprocessing and loading
├── configs/            # Config files for running different experiments
├── experiments/        # Sample experiments and results
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Features

- ✅ **KAN Layers** with B-spline interpolation for neuron-free computation
- ✅ **Weight pruning** and **activation regularization** support
- ✅ Configurable architecture via external config files
- ✅ Easy-to-run experiments on synthetic and real-world datasets
- ✅ Clean, modular codebase for extensibility

---

Authors

Naman Bhatia
Arul Bhardwaj
Medhansh Kumar
Tanay Kulkarni 
Shrey Gupta

---


## 📚 Reference

Original paper:  
[Kolmogorov–Arnold Networks (arXiv:2401.01416)](https://arxiv.org/abs/2401.01416)  
Authors: Hongyang Xue, Yubei Chen, Jianbo Yang, and Joan Bruna

https://github.com/KindXiaoming/pykan



---

## ⭐ Acknowledgements

- The original KAN authors for their insightful research
- Course instructors **Prof Tanmay Tulsidas Verlekar** (https://www.bits-pilani.ac.in/goa/tanmay-tulsidas-verlekar) and TAs for guidance on the project
- Open-source contributors whose work inspired this reimplementation

---


