# Physics-Informed Neural Networks for Boundary Value Problems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains PyTorch/TensorFlow implementations of Physics-Informed Neural Networks (PINNs) for solving Boundary Value Problems (BVPs), along with a comparative analysis of several Hyperparameter Optimization (HPO) strategies.

The primary focus is on providing a framework for both standard PINN implementations and efficient HPO techniques for scientific machine learning applications.

![Helmholtz Solution Figure](figures/helmholtz_solution.pdf)
*Example: Solution of the 2D Helmholtz equation using a Hard-Constrained PINN.*

---

## ✨ Key Features

* **PINN Implementations:** Includes both **soft-constrained (SCPINN)** and **hard-constrained (HCPINN)** models that enforce physical laws by satisfying boundary conditions exactly.
* **Hyperparameter Optimization (HPO):** A comparative study of four HPO strategies:
    * Grid Search 
    * Random Search
    * Surrogate-Based Approaches (Bayesian Optimization)
    * Our novel **Mixed Grid-Random Search** strategy.
* **Published Research:** This repository contains the official code for our paper on an efficient HPO method.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/rodrigorezende1/Physics-Informed-Neural-Networks.git](https://github.com/rodrigorezende1/Physics-Informed-Neural-Networks.git)
cd Physics-Informed Neural Networks for Boundary Value Problems
```

### 2. Set Up a Virtual Environment
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Usage

### Running a PINN Model
To run a specific model, navigate to its directory and execute the script. For example:
```bash
cd models/helmholtz_2d_square_cylinder/
python HCPINN.py
```
For a more detailed, step-by-step guide, please see the tutorial in `notebooks/tutorial_running_a_pinn.ipynb`. [under construction...]

### Running an HPO Benchmark
To compare the different HPO strategies, you can run the benchmark script:
```bash
cd hpo/
python benchmark_all.py
```

---

## 📄 Our Research: Mixed Grid-Random Search

The HPO strategy implemented in `hpo/half_grid_random.py` is based on our published research, which demonstrates a more efficient approach to hyperparameter tuning for complex models.

> **Title:** An Efficient Architecture Selection Approach for PINNs Applied to Electromagnetic Problems
> **Publication:** IEEE Transactions on Magnetics, 2025
> **Link:** [https://ieeexplore.ieee.org/document/11184611](https://ieeexplore.ieee.org/document/11184611)

If you use this specific HPO strategy in your research, please cite our paper.

---

## ✍️ How to Cite

If you use the code from this repository in your work, please cite it as follows:
```bibtex
@software{Silva_Rezende_PINNs_2025,
  author = {Rodrigo Silva Rezende},
  title = {{Physics-Informed Neural Networks for Boundary Value Problems}},
  url = {[https://github.com/rodrigorezende1/Physics-Informed-Neural-Networks.git](https://github.com/rodrigorezende1/Physics-Informed-Neural-Networks.git)},
  year = {2025}
}
```
And for our HPO paper:
```bibtex
@inproceedings{rezende2025mixed,
  title={An Efficient Architecture Selection Approach for PINNs Applied to Electromagnetic Problems},
  author={Rezende, Rodrigo and, Piwonski, Albert, and Schuhmann, Rolf},
  booktitle={IEEE Transactions on Magnetics, 2025},
  year={2025},
  ...
}
```

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.