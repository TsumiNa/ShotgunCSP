# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod
from typing import Union

from pymatgen.core.structure import Composition, Structure
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from shotgun_csp.descriptor import Compositions


class StructureFilter(ABC):
    """
    The base class for structure filters.

    Args:
        verbose: Whether to print the filtering results.
    """

    def __init__(self, *, verbose: bool = False):
        self._verbose = verbose

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        self._verbose = verbose

    def _info(self, info: str):
        if self._verbose:
            print(info)

    @abstractmethod
    def __call__(self, target: Union[str, Composition, dict], structures: list[Structure]) -> list[Structure]:
        """Filter the structures."""


class DBSCANFilter(StructureFilter):
    """Filter the structures using DBSCAN clustering."""

    def __init__(self, eps: float = 9, min_samples: int = 10, *, n_jobs=-1, verbose: bool = False):
        """
        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
            verbose: Whether to print the clustering results.
        """
        super().__init__(verbose=verbose)
        self._eps = eps
        self._min_samples = min_samples
        self._n_jobs = n_jobs

    def __call__(self, target: Composition, structures: list[Structure]) -> list[Structure]:
        compositions = [structure.composition for structure in structures]
        compositions.append(target)

        # Calculate the compositional descriptor
        desc_cal = Compositions(n_jobs=self._n_jobs)
        desc = desc_cal.fit_transform(compositions)
        X = StandardScaler().fit_transform(desc)

        # Perform DBSCAN clustering
        db = DBSCAN(
            eps=self._eps,
            min_samples=self._min_samples,
            n_jobs=self._n_jobs,
        ).fit(X)
        labels = db.labels_

        # Get the number of clusters and noise points
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # Print the clustering results
        self._info("  Target class number: %d" % labels[-1])
        self._info("  Estimated number of clusters: %d" % n_clusters_)
        self._info("  Estimated number of noise points: %d" % n_noise_)

        target_label = labels[-1]
        template_labels = labels[:-1]

        # Filter the structures
        return [structures[i] for i, label in enumerate(template_labels) if label == target_label]
