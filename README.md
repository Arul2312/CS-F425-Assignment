# Kolmogorov–Arnold Networks (KAN) - Reimplementation

This repository contains a custom implementation of the **Kolmogorov–Arnold Networks (KAN)**, based on the paper _["Kolmogorov–Arnold Networks" (arXiv:2401.01416)](https://arxiv.org/abs/2401.01416)_. The KAN model introduces a novel neural network architecture that replaces traditional neuron-based activations with **grid-based spline interpolations**, enabling better interpretability, efficiency, and flexibility.

> 📌 This project was done as part of the CS F425 - Advanced Machine Learning course assignment.

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

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/Arul2312/CS-F425-Assignment.git
cd CS-F425-Assignment
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 📊 Usage

### Training a KAN Model

```bash
python trainer.py --config configs/kan_config.yaml
```

Modify the config file to change model parameters, dataset paths, training settings, etc.

---

## 🧪 Experiments

The repository includes sample experiments comparing KANs to traditional MLPs on:

- Function regression tasks
- Toy classification datasets
- Learning simple mathematical operations

Results show promising interpretability and parameter efficiency for KANs.

---

## 📚 Reference

Original paper:  
[Kolmogorov–Arnold Networks (arXiv:2401.01416)](https://arxiv.org/abs/2401.01416)  
Authors: Hongyang Xue, Yubei Chen, Jianbo Yang, and Joan Bruna

---

## 👨‍💻 Author

**Arul R**  
BITS Pilani, Goa Campus  
Course: CS F425 - Advanced Machine Learning  
Contact: [arulr2312@gmail.com](mailto:arulr2312@gmail.com)

---

## ⭐ Acknowledgements

- The original KAN authors for their insightful research
- Course instructors and TAs for guidance on the project
- Open-source contributors whose work inspired this reimplementation

---

## 📜 License

This project is for educational and research purposes only. For official implementations or commercial usage, please refer to the license terms of the original KAN paper.
