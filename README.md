
# ShotgunCSP

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/TsumiNa/ShotgunCSP?quickstart=1)

**ShotgunCSP** is a Python package designed to solve the crystal structure prediction (CSP) problem using a non-iterative, single-shot screening framework. This method leverages a large library of virtually created crystal structures and employs a machine-learning energy predictor for efficient and accurate predictions.

## Features

- Non-iterative, single-shot screening framework for CSP
- Transfer learning for accurate energy prediction
- Generative models based on element substitution (**ShotgunCSP-GT**) and symmetry-restricted structure generation (**ShotgunCSP-GW**)
- High prediction accuracy with reduced computational intensity

## Installation

Before installing  **shotgun-csp**, you have to install **PyTorch (^2.0.0)** first. Please follow the [official installation guidance](https://pytorch.org/get-started/locally/).

To install **shotgun-csp**, you can use [Poetry](https://python-poetry.org/):
<!-- To install **shotgun-csp**, you can use [Poetry](https://python-poetry.org/) and [PyPI](https://pypi.org/): -->

```bash
# Clone the repository
git clone https://github.com/tsumina/shotgun-csp.git
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
from shotgun_csp.utils import VASPInputGenerator

# Select templates
selector = TemplateSelector(target=<composition>, volume=<predicted volume>)
templates = selector(<pymatgen structures>, filter=<structure filter (optional)>)

# Generate VASP input
generator = VASPInputGenerator(save_to='/path/to/save')
generator.static_input(<pymatgen structure>)  # static calculation
generator.relax_input(<pymatgen structure>)  # relax calculation

```

See [example](examples/CSP_with_ShotgunCSP-GT.ipynb) for details.

<!-- Please refer to the [documentation](https://yourdocumentationlink) for more detailed instructions and advanced usage. -->

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve **ShotgunCSP**. Please fork the repository and submit your pull requests.

## Acknowledgements

We would like to thank all contributors and the scientific community for their valuable input and support.

## Contact

For any inquiries or issues, please [open an issue](https://github.com/TsumiNa/ShutgunCSP/issues/new/choose).
