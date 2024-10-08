# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union

import numpy as np
import pandas as pd

from shotgun_csp.descriptor.base import BaseCompositionFeaturizer, BaseDescriptor

__all__ = [
    "Compositions",
    "Counting",
    "WeightedAverage",
    "WeightedSum",
    "WeightedVariance",
    "MaxPooling",
    "MinPooling",
]


class Counting(BaseCompositionFeaturizer):
    def __init__(self, *, one_hot_vec=False, n_jobs=-1, on_errors="raise", return_type="any", target_col=None):
        """

        Parameters
        ----------
        one_hot_vec : bool
            Set ``true`` to using one-hot-vector encoding.
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.one_hot_vec = one_hot_vec
        self._elems = self.elements.index.tolist()
        self.__authors__ = ["TsumiNa"]

    def mix_function(self, elems, nums):
        vec = np.zeros(len(self._elems), dtype=np.int)
        for i, e in enumerate(elems):
            if self.one_hot_vec:
                vec[self._elems.index(e)] = 1
            else:
                vec[self._elems.index(e)] = nums[i]

        return vec

    @property
    def feature_labels(self):
        return self._elems


class WeightedAverage(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
    n_jobs: int
        The number of jobs to run in parallel for both fit and predict.
        Set -1 to use all cpu cores (default).
        Inputs ``X`` will be split into some blocks then run on each cpu cores.
    on_errors: string
        How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
        When 'nan', return a column with ``np.nan``.
        The length of column corresponding to the number of feature labs.
        When 'keep', return a column with exception objects.
        The default is 'raise' which will raise up the exception.
    return_type: str
        Specific the return type.
        Can be ``any``, ``array`` and ``df``.
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = self.elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ["ave:" + s for s in self.elements]


class WeightedSum(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
    n_jobs: int
        The number of jobs to run in parallel for both fit and predict.
        Set -1 to use all cpu cores (default).
        Inputs ``X`` will be split into some blocks then run on each cpu cores.
    on_errors: string
        How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
        When 'nan', return a column with ``np.nan``.
        The length of column corresponding to the number of feature labs.
        When 'keep', return a column with exception objects.
        The default is 'raise' which will raise up the exception.
    return_type: str
        Specific the return type.
        Can be ``any``, ``array`` and ``df``.
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = self.elements.loc[elems, :].values
        w_ = np.array(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ["sum:" + s for s in self.elements]


class WeightedVariance(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
    n_jobs: int
        The number of jobs to run in parallel for both fit and predict.
        Set -1 to use all cpu cores (default).
        Inputs ``X`` will be split into some blocks then run on each cpu cores.
    on_errors: string
        How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
        When 'nan', return a column with ``np.nan``.
        The length of column corresponding to the number of feature labs.
        When 'keep', return a column with exception objects.
        The default is 'raise' which will raise up the exception.
    return_type: str
        Specific the return type.
        Can be ``any``, ``array`` and ``df``.
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = self.elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        mean_ = w_.dot(elems_)
        var_ = elems_ - mean_
        return w_.dot(var_**2)

    @property
    def feature_labels(self):
        return ["var:" + s for s in self.elements]


class MaxPooling(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
    n_jobs: int
        The number of jobs to run in parallel for both fit and predict.
        Set -1 to use all cpu cores (default).
        Inputs ``X`` will be split into some blocks then run on each cpu cores.
    on_errors: string
        How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
        When 'nan', return a column with ``np.nan``.
        The length of column corresponding to the number of feature labs.
        When 'keep', return a column with exception objects.
        The default is 'raise' which will raise up the exception.
    return_type: str
        Specific the return type.
        Can be ``any``, ``array`` and ``df``.
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.max().values

    @property
    def feature_labels(self):
        return ["max:" + s for s in self.elements]


class MinPooling(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
    n_jobs: int
        The number of jobs to run in parallel for both fit and predict.
        Set -1 to use all cpu cores (default).
        Inputs ``X`` will be split into some blocks then run on each cpu cores.
    on_errors: string
        How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
        When 'nan', return a column with ``np.nan``.
        The length of column corresponding to the number of feature labs.
        When 'keep', return a column with exception objects.
        The default is 'raise' which will raise up the exception.
    return_type: str
        Specific the return type.
        Can be ``any``, ``array`` and ``df``.
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.min().values

    @property
    def feature_labels(self):
        return ["min:" + s for s in self.elements]


class Compositions(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    _classic = ["WeightedAverage", "WeightedSum", "WeightedVariance", "MaxPooling", "MinPooling"]

    def __init__(
        self,
        *,
        elemental_info: Union[pd.DataFrame, None] = None,
        n_jobs: int = -1,
        featurizers: Union[None, List[str]] = None,
        on_errors: str = "nan",
    ):
        """

        Parameters
        ----------
        elemental_info
            Elemental level information for each element. For example, the ``atomic number``,
            ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        featurizers: Union[str, List[str]]
            Name of featurizers that will be used.
            Set to `classic` to be compatible with the old version.
            This is equal to set ``featurizers=['WeightedAverage', 'WeightedSum',
            'WeightedVariance', 'MaxPooling', 'MinPooling']``.
            Default is 'all'.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'nan' which will raise up the exception.
        """

        super().__init__(featurizers=featurizers or self._classic)

        self.composition = Counting(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = WeightedAverage(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = WeightedSum(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = WeightedVariance(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = MaxPooling(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = MinPooling(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
