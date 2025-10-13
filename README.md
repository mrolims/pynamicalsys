# pynamicalsys: A Python toolkit for the analysis of dynamical systems

[![Documentation Status](https://readthedocs.org/projects/pynamicalsys/badge/?version=latest)](https://pynamicalsys.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/pynamicalsys.svg)](https://pypi.org/project/pynamicalsys/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

**pynamicalsys** is designed to provide a fast, flexible, and user-friendly environment for analyzing **nonlinear dynamical systems**. It is intended for students, researchers, educators, and enthusiasts who want to explore the world of chaos and dynamical systems. Beyond standard tools like trajectory generation and Lyapunov exponents calculation, **pynamicalsys** includes advanced features such as

- **Bifurcation diagrams**, **Poincaré sections**, and **stroboscopic maps** for analyzing system trajectories.
- **Symplectic integrators** for analyzing Hamiltonian systems.
- **Linear dependence index** for chaos detection.
- **Recurrence plots** and recurrence time statistics.
- Chaos indicators based on **weighted Birkhoff averages**.
- Statistical measures of **diffusion and transport** in dynamical systems.
- Computation of **periodic orbits**, their **stability** and their **manifolds**.
- Basin metric for **quantifying** the structure of **basins of attraction**.
- **Plot styling** for consistent and customizable visualizations.

**pynamicalsys** is built on top of NumPy and Numba, ensuring high performance and efficiency. Thanks to Numba accelerated computation, **pynamicalsys** offers speedups up to **130x** compared to the original Python implementation of the algorithms. This makes it suitable for large-scale simulations and analyses.

## Related Publication

The methods and applications implemented in **pynamicalsys** are presented and discussed in our research article:

M. Rolim Sales et al., *pynamicalsys: A Python toolkit for the analysis of dynamical systems*, [**Chaos, Solitons and Fractals** 201, 117269 (2025)](https://doi.org/10.1016/j.chaos.2025.117269).

You can reproduce the numerical experiments, figures, and performance benchmarks presented in the paper using the companion Jupyter notebooks:

- [Reproducibility Notebook](https://github.com/mrolims/pynamicalsys/blob/main/paper/paper.ipynb)  
- [Benchmarks](https://github.com/mrolims/pynamicalsys/blob/main/paper/benchmarks.ipynb)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Install via PyPI

To install the latest stable release, run in your command line:

```bash
pip install pynamicalsys
```

> **Note:** On **Windows**, it is **strongly recommended** to use [Anaconda](https://www.anaconda.com). It simplifies dependency management and avoids potential issues with scientific libraries during installation. Be sure to run the command from the **Anaconda Prompt**, not from Command Prompt or PowerShell, to ensure the correct environment is activated.

### Upgrade via PyPI

To upgrade your current version of **pynamicalsys** to the latest stable release, run in your command line:

```bash
pip install pynamicalsys --upgrade
```

### Install from source

If you want to install the development version from the source repository, clone the repo and install with:

```bash
git clone https://github.com/mrolims/pynamicalsys.git
cd pynamicalsys
pip install .
```

### Verifying the installation

After installation, you can verify it by running Python and importing the package:

```python
>>> import pynamicalsys
>>> pynamicalsys.__version__
```

### Troubleshooting

If you encounter any issues, make sure you have the latest version of pip:

```bash
pip install --upgrade pip build
```

## Documentation

For detailed instructions, examples, and API references, please visit the full documentation page:

[pynamicalsys.readthedocs.io](https://pynamicalsys.readthedocs.io/)

The documentation includes:

- **Getting Started** — installation and quick-start examples.
- **Tutorial** — step-by-step explanations of key concepts and pratical demonstrations.
- **API Reference** — detailed descriptions of all functions, parameters, and return values.

## Citation

If you use **pynamicalsys** in your work, please cite our [research paper](https://doi.org/10.1016/j.chaos.2025.117269). Below are recommended citation formats for different use cases.

### BibTeX

```bibtex
@article{pynamicalsys,
  title={pynamicalsys: A Python toolkit for the analysis of dynamical systems},
  author={Matheus Rolim Sales and Leonardo Costa de Souza and Daniel Borin and Michele Mugnaine and José Danilo Szezech and Ricardo Luiz Viana and Iberê Luiz Caldas and Edson Denis Leonel and Chris G. Antonopoulos},
  journal={Chaos, Solitons & Fractals},
  volume={201},
  pages={117269},
  year={2025},
  doi={https://doi.org/10.1016/j.chaos.2025.117269},
  url={https://www.sciencedirect.com/science/article/pii/S0960077925012822},
}
```

### API Style

```text
Rolim Sales, M., Costa de Souza, L., Borin, D., Mugnaine, M., Szezech Jr., J. D., Viana, R. L., Caldas, I. L., Leonel, E. D., & Antonopoulos, C. G. (2025). pynamicalsys: A Python toolkit for the analysis of dynamical systems. Chaos, Solitons & Fractals, 201, 117269. https://doi.org/10.1016/j.chaos.2025.117269
```

### Short citation style

```text
Matheus Rolim Sales et al., "pynamicalsys: A Python toolkit for the analysis of dynamical systems," Chaos, Solitons & Fractals, vol. 201, p. 117269, 2025. https://doi.org/10.1016/j.chaos.2025.117269
```

## Contributing

We welcome contributions from the community! To get started, please see our [Contributing Guidelines](https://pynamicalsys.readthedocs.io/en/latest/contributing.html).

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](./LICENSE) file for details.

## Acknowledgments

This project was financed, in part, by the São Paulo Research Foundation (FAPESP, Brazil), under process numbers 2023/08698-9 and 2024/09208-8.

## Disclaimer

As opiniões, hipóteses e conclusões ou recomendações expressas neste material são de responsabilidade do(s) autor(es) e não necessariamente refletem a visão da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP, Brasil).

The opinions, hypotheses, and conclusions or recommendations expressed in this material are the sole responsibility of the author(s) and do not necessarily reflect the views of the São Paulo Research Foundation (FAPESP, Brazil).
