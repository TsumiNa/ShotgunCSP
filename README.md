
# Shotgun CSP

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/TsumiNa/ShotgunCSP)

**Shotgun CSP** is a Python package designed to solve the crystal structure prediction (CSP) problem using a non-iterative, single-shot screening framework. This method leverages a large library of virtually created crystal structures and employs a machine-learning energy predictor for efficient and accurate predictions.

## Features

- Non-iterative, single-shot screening framework for CSP
- Transfer learning for accurate energy prediction
- Generative models based on element substitution (**ShotgunCSP-GT**) and symmetry-restricted structure generation (**ShotgunCSP-GW**)
- High prediction accuracy with reduced computational intensity

## Installation

Before install  **shotgun-csp**, you have to install **PyTorch (^2.0.0)** first. Please follow the [official installation guidance](https://pytorch.org/get-started/locally/).

To install **shotgun-csp**, you can use [Poetry](https://python-poetry.org/) and [PyPI](https://pypi.org/):

```bash
# Install poetry if you haven't already
pip install poetry

# Clone the repository
git clone https://github.com/yourusername/shotgun-csp.git
cd shotgun-csp

# Install dependencies and package
poetry install
```

Alternatively, you can directly install the package from PyPI:

```bash
pip install shotgun-csp
```

## Usage

Here is a simple example of how to use **shotgun-csp**:

```python
from shotgun_csp.generator import TemplateSelector
from shotgun_csp.io import VASPInputGenerator

# Example usage
selector = TemplateSelector()
templates = selector.from(<pymatgen structures>, filter='DBSCAN')

vasp_inputs = VASPInputGenerator(templates)
```

<!-- Please refer to the [documentation](https://yourdocumentationlink) for more detailed instructions and advanced usage. -->

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve **Shotgun CSP**. Please fork the repository and submit your pull requests.

## Acknowledgements

We would like to thank all contributors and the scientific community for their valuable input and support.

## Contact

For any inquiries or issues, please [open an issue](https://github.com/TsumiNa/ShutgunCSP/issues/new/choose).
