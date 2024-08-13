# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from itertools import product
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Composition, Structure
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from shotgun_csp.utils import preset

from .filter import StructureFilter
from .toolbox import convert_struct


class TemplateSelector:
    def __init__(
        self,
        target: Union[str, Composition],
        volume: float,
        *,
        volume_perturbation: float = 0.1,
        structure_matcher_params: dict = dict(ltol=0.05, angle_tol=3),
    ):
        """
        This class selects the template structures that match the target composition and volume.

        Parameters
        ----------
        target: Union[str, Composition]
            The target composition.
        volume: float
            The target volume.
        volume_variance: float
            The variance of the volume. The default is 0.05.
        structure_matcher_params: dict
            The parameters for the StructureMatcher. The default is dict(ltol=0.05, angle_tol=3).
        """
        self._target = Composition(target) if isinstance(target, str) else target
        self._full_formula = self._target.formula.replace(" ", "")
        self._reduced_formula = self._target.reduced_formula
        self._reduced_composition = self._target.reduced_composition
        self._volume = volume
        self._volume_perturbation = volume_perturbation
        self._ratio = tuple(sorted([v for v in self._target.values()]))
        self._structure_matcher_params = structure_matcher_params.copy()

        # Generate the distance matrix
        _ = preset.elements_completed
        self._dis_matrix = pd.DataFrame(cdist(_, _, "seuclidean"), index=_.index, columns=_.index)

    def _check_dis(self, structure: Structure, *, dtol: float = 0.15):
        """
        Check the distance between the atoms in the structure.

        Parameters
        ----------
        structure: Structure
            The structure to check.
        dtol: float
            The tolerance of the distance. The default is 0.15.
        """
        covalent_radius = preset.covalent_radius
        sites = structure.sites
        distance_matrix = structure.distance_matrix
        checker = []

        # Check the distance between the atoms
        for s1, s2 in product(sites, sites):
            checker.append((covalent_radius[str(s1.specie)] + covalent_radius[str(s2.specie)]) * (1 - dtol))
        checker = np.asarray(checker)
        checker = checker.reshape(int(np.sqrt(checker.size)), -1)
        np.fill_diagonal(checker, 0.0)

        return np.all((distance_matrix - checker) >= 0)

    def _gen_replace_rule(self, template: Composition, target: Composition) -> dict:
        def _extract(comp: Composition) -> list:
            ret = defaultdict(list)
            for k, v in comp.items():
                ret[v].append(k.name)

            return [ret[k] for k in sorted(ret.keys())]

        tpl_pattern = _extract(template)
        tar_pattern = _extract(target)

        ret = {}
        for tpl_e, tar_e in zip(tpl_pattern, tar_pattern):
            if len(template) == 1:
                ret[tpl_e[0]] = tar_e[0]
            else:
                dists = sorted(
                    [(self._dis_matrix.loc[e1, e2], e1, e2) for e1, e2 in product(tpl_e, tar_e)], key=lambda s: s[0]
                )
                while len(dists) > 0:
                    _, e1, e2 = dists.pop(0)
                    ret[e1] = e2
                    dists = [(d, e1_, e2_) for d, e1_, e2_ in dists if e1_ != e1 and e2_ != e2]

        return ret

    def __call__(
        self,
        structures: list[Structure],
        n_structures: int,
        *,
        filter: Union[None, StructureFilter] = None,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        """
        Select the template structures that match the target composition and volume.

        Parameters
        ----------
        structures: list[Structure]
            The list of structures to select the templates.
        n_structures: int
            The number of structures to generate.
        filter: Union[None, StructureFilter]
            The filter to apply to the structures. The default is None.
        n_jobs: int
            The number of jobs to run in parallel. The default is -1.
        verbose: bool
            The flag to show the progress. The default is False.

        Returns
        -------
        pd.DataFrame
            The DataFrame of the selected structures.
        """

        def _info(info: str):
            if verbose:
                print(info)

        def _gen(
            structure: Structure,
            rule: dict,
            volume: float,
            gid: int,
            *,
            dtol: float = 0.15,
        ) -> Union[None, tuple[Structure, int, str, int]]:
            structure = structure.copy()
            structure.replace_species(rule)
            structure.scale_lattice(volume)

            if self._check_dis(structure, dtol=dtol):
                return (structure, gid) + structure.get_space_group_info()

        # Step 1: Select structures by the full formula
        # Select the structure that matches the ratio of the full formula
        # and has a different reduced composition from the target composition.
        templates = [
            structure
            for structure in structures
            if tuple(sorted([v for v in structure.composition.values()])) == self._ratio
            and structure.composition.reduced_composition != self._reduced_composition
        ]
        _info(f"Selected {len(templates)} structures by the full formula.")

        # Step 2: Convert the structures to primitive structures
        templates = [
            convert_struct(
                structure,
            )
            for structure in templates
        ]

        # Step 3: Filter the structures
        # Filter the structures if filter is given.
        if filter is not None:
            templates = filter(self._target, templates)
            _info(f"Selected {len(templates)} structures after filtering.")

        # Step 4: Remove the almost identical structures
        # Remove the structures that are almost identical to the target structure using StructureMatcher
        templates = [(template,) + template.get_space_group_info() for template in templates]
        templates = pd.DataFrame(templates, columns=["structure", "space_group", "space_group_number"])

        # Group the structures by the space group number
        ret = []
        matcher = StructureMatcher(**self._structure_matcher_params)
        for spg, data in templates.groupby("space_group_number"):
            _info(f"for spg: {spg}, size: {data.shape[0]}")
            group_structures = matcher.group_structures(data.structure, anonymous=True)
            for group in group_structures:
                structure = group[0]
                ret.append((structure, structure.composition, spg))

        _info(f"Selected {len(ret)} structures after removing almost identical structures.")
        templates = pd.DataFrame(ret, columns=["structure", "composition", "space_group_number"])

        # Step 5: Generate virtual structures
        # Generate virtual structures by replacing the elements in the template structure.
        # The replacement is based on the distance matrix.
        ret = []
        for gid, row in tqdm(list(templates.iterrows()), desc=self._full_formula):
            rule = self._gen_replace_rule(row.composition, self._target)

            ret += Parallel(n_jobs=n_jobs)(
                delayed(_gen)(
                    row.structure,
                    rule,
                    volume,
                    gid,
                )
                for volume in np.random.normal(
                    self._volume, self._volume * self._volume_perturbation, size=n_structures
                )
            )

        ret = [t for t in ret if t is not None]
        templates = pd.DataFrame(ret, columns=["structure", "structure_group", "space_group", "space_group_num"])

        return templates
