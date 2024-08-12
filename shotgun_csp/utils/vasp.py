# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Callable, Union

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import VaspInput
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet


class VASPSetting(BaseModel):
    """
    VASP setting model.
    """

    potcar_functional: str = Field("PBE_54", description="POTCAR functional")
    potcar_settings: dict = Field({}, description="POTCAR settings")
    incar_settings: dict = Field({}, description="INCAR settings for static calculation")
    kpoints_settings: dict = Field({}, description="KPOINTS settings")


class VASPInputGenerator(object):
    """
    Generate VASP input files from pymatgen structure. This generator internally use MPRelaxSet and MPStaticSet settings.
    You can customize the settings by passing the parameters.
    """

    # default settings for static calculation
    static_preset = VASPSetting(
        potcar_settings={"W": "W"},
        static_incar_settings={
            "ALGO": "Fast",
            "EDIFF": 1e-4,
            "EDIFFG": 1e-3,
            "ENCUT": 520,
            "IBRION": -1,
            "ISIF": 3,
            "ISMEAR": -5,
            "ISPIN": 2,
            "LCHARG": False,
            "LREAL": "Auto",
            "LWAVE": False,
            "NCORE": 4,
            "NELM": 100,
            "NSW": 0,
            "POTIM": 0.3,
            "PREC": "Accurate",
            "SIGMA": 0.05,
            # "PSTRESS": 500,  # for 50GPa
        },
    )

    # default settings for relaxation calculation
    relax_presets = [
        VASPSetting(
            potcar_settings={"W": "W"},
            incar_settings={
                "ALGO": "Fast",
                "EDIFF": 1e-2,
                "EDIFFG": 1e-1,
                "IBRION": 2,
                "ISIF": 4,
                "ISMEAR": 1,
                "ISPIN": 2,
                "LCHARG": False,
                "LREAL": "Auto",
                "LWAVE": False,
                "NCORE": 4,
                "NELM": 100,
                "NSW": 90,
                "POTIM": 0.02,
                "PREC": "LOW",
                "SIGMA": 0.15,
                # "PSTRESS": 500,  # for 50GPa
            },
        ),
        VASPSetting(
            incar_settings={
                "ALGO": "Fast",
                "EDIFF": 1e-3,
                "EDIFFG": 1e-2,
                "ENCUT": 520,
                "IBRION": 1,
                "ISIF": 4,
                "ISMEAR": 1,
                "ISPIN": 2,
                "LCHARG": False,
                "LREAL": "Auto",
                "LWAVE": False,
                "NCORE": 4,
                "NELM": 100,
                "NSW": 80,
                "POTIM": 0.3,
                "PREC": "Normal",
                "SIGMA": 0.12,
                # "PSTRESS": 500,  # for 50GPa
            },
        ),
        VASPSetting(
            incar_settings={
                "ALGO": "Fast",
                "EDIFF": 1e-3,
                "EDIFFG": 1e-2,
                "ENCUT": 520,
                "IBRION": 2,
                "ISIF": 3,
                "ISMEAR": 1,
                "ISPIN": 2,
                "LCHARG": False,
                "LREAL": "Auto",
                "LWAVE": False,
                "NCORE": 4,
                "NELM": 100,
                "NSW": 90,
                "POTIM": 0.02,
                "PREC": "Normal",
                "SIGMA": 0.1,
                # "PSTRESS": 500,  # for 50GPa
            },
        ),
        VASPSetting(
            incar_settings={
                "ALGO": "Fast",
                "EDIFF": 1e-4,
                "EDIFFG": 1e-3,
                "ENCUT": 520,
                "IBRION": 1,
                "ISIF": 3,
                "ISMEAR": 1,
                "ISPIN": 2,
                "LCHARG": False,
                "LREAL": "Auto",
                "LWAVE": False,
                "NCORE": 4,
                "NELM": 100,
                "NSW": 80,
                "POTIM": 0.3,
                "PREC": "Accurate",
                "SIGMA": 0.1,
                # "PSTRESS": 500,  # for 50GPa
            },
        ),
        VASPSetting(
            kpoints_settings={"reciprocal_density": 1000},
            incar_settings={
                "ALGO": "Fast",
                "EDIFF": 1e-4,
                "EDIFFG": 1e-3,
                "ENCUT": 520,
                "IBRION": -1,
                "ISIF": 3,
                "ISMEAR": -5,
                "ISPIN": 2,
                "LCHARG": False,
                "LREAL": "Auto",
                "LWAVE": False,
                "NCORE": 4,
                "NELM": 100,
                "NSW": 0,
                "POTIM": 0.3,
                "PREC": "Accurate",
                "SIGMA": 0.05,
                # "PSTRESS": 500,  # for 50GPa
            },
        ),
    ]

    def __init__(
        self,
        save_to: Union[str, Path],
        *,
        static_settings: Union[VASPSetting, None] = None,
        relax_settings: Union[list[VASPSetting], None] = None,
    ) -> None:
        """
        Initialize VASP input generator.

        Parameters
        ----------
        save_to
            Directory to save the input files.
        static_settings
            Static calculation settings.
        relax_settings
            Relaxation calculation settings

        Returns
        -------
        None
        """
        self._save_to = Path(save_to) if isinstance(save_to, str) else save_to
        self._static_settings = static_settings or self.static_preset
        self._relax_settings = relax_settings or self.relax_presets

    @property
    def save_to(self):
        return self._save_to

    @property
    def static_settings(self):
        return self._static_settings

    @property
    def relax_settings(self):
        return self._relax_settings

    @classmethod
    def half_kpts(vasp_input: VaspInput) -> VaspInput:
        """
        Generate half kpoints from the input file.

        Parameters
        ----------
        vasp_input
            VaspInput object.

        Returns
        -------
        VaspInput
            VaspInput object with half kpoints.
        """
        vasp_input = vasp_input.copy()
        kpts = vasp_input["KPOINTS"]
        # lattice vectors with length < 8 will get >1 KPOINT
        kpts.kpts = np.round(np.maximum(np.array(kpts.kpts) / 2, 1)).astype(int).tolist()

        return vasp_input

    def static_input(
        self,
        structure: Structure,
        *,
        path_prefix: Union[str, None] = None,
        path_suffix: Union[str, None] = None,
        custom_input: Union[None, Callable[[VaspInput], VaspInput]] = None,
        no_write=False,
    ) -> Union[None, VaspInput]:
        formula = structure.composition.reduced_formula
        input_file = MPStaticSet(
            structure,
            user_potcar_functional=self._static_settings.potcar_functional,
            user_incar_settings=self._static_settings.incar_settings,
        )

        if custom_input:
            input_file = custom_input(input_file)
        if not no_write:
            input_file.write_input(
                self._save_to
                / f"{path_prefix + '_' if path_prefix else '' }{formula}{ '_' + path_suffix if path_suffix else '' }"
                / "static"
            )
        else:
            return input_file

    def relax_input(
        self,
        structure: Structure,
        *,
        path_prefix: Union[str, None] = None,
        path_suffix: Union[str, None] = None,
        custom_input: Union[None, list[Callable[[VaspInput], VaspInput]]] = None,
        no_write=False,
    ) -> Union[None, list[VaspInput]]:
        """
        Generate relaxation input files.

        Parameters
        ----------
        structure
            Input structure.
        custom_input
            Custom input function for each relaxation calculation.
        no_write
            If True, return the input files without writing.

        Returns
        -------
        list (optional)
            List of VaspInput objects.
        """

        input_files = []
        formula = structure.composition.reduced_formula
        for setting in self._relax_settings:
            input_file = MPRelaxSet(
                structure,
                user_potcar_functional=setting.potcar_functional,
                user_incar_settings=setting.incar_settings,
            )

            if custom_input:
                input_file = custom_input(input_file)
            input_files.append(input_file)
        if not no_write:
            for i, input_file in enumerate(input_files):
                input_file.write_input(
                    self._save_to
                    / f"{path_prefix + '_' if path_prefix else '' }{formula}{ '_' + path_suffix if path_suffix else '' }"
                    / f"relax_{i}"
                )
        else:
            return input_files
