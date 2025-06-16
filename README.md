# pynamicalsys: A Python toolkit for the analysis of dynamical systems

[![Documentation Status](https://readthedocs.org/projects/pynamicalsys/badge/?version=latest)](https://pynamicalsys.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/pynamicalsys.svg)](https://pypi.org/project/pynamicalsys/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

**pynamicalsys** is designed to provide a fast, flexible, and user-friendly environment for analyzing **nonlinear dynamical systems**. It is designed for students, researchers, educators, and enthusiasts who want to explore the world of chaos and dynamical systems. Beyond standard tools like trajectory generation and Lyapunov exponents calculation, pynamicalsys includes advanced features such as

-   **Linear dependence index** for chaos detection.
-   **Recurrence plots** and recurrence time statistics.
-   Chaos indicators based on **weighted Birkhoff averages**.
-   Statistical measures of **diffusion and transport** in dynamical systems.
-   Computation of **periodic orbits**, their **stability** and their **manifolds**.
-   Basin metric for **quantifying** the structure of **basins of attraction**.
-   **Plot styling** for consistent and customizable visualizations.

pynamicalsys is built on top of NumPy and Numba, ensuring high performance and efficiency. Thanks to Numba accelerated computation, pynamicalsys offers speedups up to **130x** compared to the original Python implementation of the algorithms. This makes it suitable for large-scale simulations and analyses.

## Installation

### Prerequisites

-   Python 3.8 or higher
-   pip (Python package installer)

### Install via PyPI

To install the latest stable release, run:

```bash
$ pip install pynamicalsys
```

> **Note:** On **Windows**, it is **strongly recommended** to use [Anaconda](https://www.anaconda.com). It simplifies dependency management and avoids potential issues with scientific libraries during installation. Be sure to run the command from the **Anaconda Prompt**, not from Command Prompt or PowerShell, to ensure the correct environment is activated.


### Install from source

If you want to install the development version from the source repository, clone the repo and install with:

```bash
$ git clone https://github.com/mrolims/pynamicalsys.git
$ cd pynamicalsys
$ pip install .
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
$ pip install --upgrade pip build
```

## Citation

Currently, our research paper is under review, but in the mean time, if you use **pynamicalsys** in your work, you can cite the arXiv version:

## Contributing

We welcome contributions from the community! To get started, please see our Contributing Guidelines.

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](./LICENSE) file for details.

## Acknowledgments

This project was financed, in part, by the São Paulo Research Foundation (FAPESP, Brazil), under process numbers 2023/08698-9 and 2024/09208-8.

## Disclaimer

As opiniões, hipóteses e conclusões ou recomendações expressas neste material são de responsabilidade do(s) autor(es) e não necessariamente refletem a visão da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP, Brasil).

The opinions, hypotheses, and conclusions or recommendations expressed in this material are the sole responsibility of the author(s) and do not necessarily reflect the views of the São Paulo Research Foundation (FAPESP, Brazil).
