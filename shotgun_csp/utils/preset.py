# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import os

import pandas as pd

from .toolbox import Singleton


class _Preset(metaclass=Singleton):
    def __init__(self):
        # Get the directory of the current script
        self._current_dir = os.path.dirname(os.path.abspath(__file__))

    @property
    def elements_completed(self) -> pd.DataFrame:
        # Construct the path to the CSV file
        csv_path = os.path.join(self._current_dir, "elements_completed.csv")
        return pd.read_csv(csv_path, index_col=0)

    @property
    def atom_init(self) -> pd.DataFrame:
        # Construct the path to the CSV file
        csv_path = os.path.join(self._current_dir, "atom_init.csv")
        return pd.read_csv(csv_path, index_col=0)

    @property
    def covalent_radius(
        self,
    ) -> dict:
        return {
            "H": 0.26,
            "He": 0.28,
            "Li": 1.21,
            "Be": 0.93,
            "B": 0.81,
            "C": 0.68,
            "N": 0.7,
            "O": 0.64,
            "F": 0.54,
            "Ne": 0.58,
            "Na": 1.59,
            "Mg": 1.34,
            "Al": 1.18,
            "Si": 1.09,
            "P": 1.04,
            "S": 1.02,
            "Cl": 0.98,
            "Ar": 0.96,
            "K": 1.91,
            "Ca": 1.86,
            "Sc": 1.63,
            "Ti": 1.52,
            "V": 1.45,
            "Cr": 1.34,
            "Mn": 1.34,
            "Fe": 1.29,
            "Co": 1.23,
            "Ni": 1.2,
            "Cu": 1.28,
            "Zn": 1.18,
            "Ga": 1.19,
            "Ge": 1.18,
            "As": 1.15,
            "Se": 1.16,
            "Br": 1.17,
            "Kr": 1.12,
            "Rb": 2.11,
            "Sr": 1.85,
            "Y": 1.83,
            "Zr": 1.68,
            "Nb": 1.58,
            "Mo": 1.49,
            "Tc": 1.4,
            "Ru": 1.39,
            "Rh": 1.35,
            "Pd": 1.33,
            "Ag": 1.4,
            "Cd": 1.35,
            "In": 1.37,
            "Sn": 1.35,
            "Sb": 1.34,
            "Te": 1.34,
            "I": 1.36,
            "Xe": 1.31,
            "Cs": 2.33,
            "Ba": 2.04,
            "La": 1.99,
            "Ce": 1.95,
            "Pr": 1.96,
            "Nd": 1.95,
            "Pm": 1.99,
            "Sm": 1.9,
            "Eu": 1.93,
            "Gd": 1.9,
            "Tb": 1.89,
            "Dy": 1.85,
            "Ho": 1.85,
            "Er": 1.83,
            "Tm": 1.8,
            "Yb": 1.79,
            "Lu": 1.65,
            "Hf": 1.81,
            "Ta": 1.62,
            "W": 1.55,
            "Re": 1.44,
            "Os": 1.4,
            "Ir": 1.35,
            "Pt": 1.31,
            "Au": 1.33,
            "Hg": 1.27,
            "Tl": 1.38,
            "Pb": 1.41,
            "Bi": 1.44,
            "Po": 1.36,
            "At": 1.5,
            "Rn": 1.5,
            "Fr": 2.6,
            "Ra": 2.19,
            "Ac": 2.15,
            "Th": 2.0,
            "Pa": 2.0,
            "U": 1.89,
            "Np": 1.89,
            "Pu": 1.86,
            "Am": 1.74,
            "Cm": 1.66,
        }.copy()


preset = _Preset()
