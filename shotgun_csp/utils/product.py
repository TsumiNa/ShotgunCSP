# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from numpy import product


class Product(object):
    """
    A class to represent a Cartesian product of input iterables.

    Attributes:
    -----------
    paras : tuple
        The input iterables repeated as specified.
    lens : list
        The lengths of the input iterables.
    size : int
        The total number of elements in the Cartesian product.
    acc_list : list
        The accumulated list used for indexing.
    """

    def __init__(self, *paras, repeat=1):
        """
        Initializes the Product object with input iterables and repeat count.

        Parameters:
        -----------
        *paras : tuple
            The input iterables.
        repeat : int, optional
            The number of times to repeat the input iterables (default is 1).

        Raises:
        -------
        ValueError
            If repeat is not an integer.
        """
        if not isinstance(repeat, int):
            raise ValueError("repeat must be int but got {}".format(type(repeat)))
        lens = [len(p) for p in paras]
        if repeat > 1:
            lens = lens * repeat
        size = product(lens)
        acc_list = [np.floor_divide(size, lens[0])]
        for len_ in lens[1:]:
            acc_list.append(np.floor_divide(acc_list[-1], len_))

        self.paras = paras * repeat if repeat > 1 else paras
        self.lens = lens
        self.size = size
        self.acc_list = acc_list

    def __getitem__(self, index):
        """
        Returns the element at the specified index in the Cartesian product.

        Parameters:
        -----------
        index : int
            The index of the element to retrieve.

        Returns:
        --------
        tuple
            The element at the specified index.

        Raises:
        -------
        IndexError
            If the index is out of range.
        """
        if index > self.size - 1:
            raise IndexError
        ret = [s - 1 for s in self.lens]  # from len to index
        remainder = index + 1
        for i, acc in enumerate(self.acc_list):
            quotient, remainder = np.divmod(remainder, acc)
            if remainder == 0:
                ret[i] = quotient - 1
                break
            ret[i] = quotient

        return tuple(self.paras[i][j] for i, j in enumerate(ret))

    def __len__(self):
        """
        Returns the total number of elements in the Cartesian product.

        Returns:
        --------
        int
            The total number of elements.
        """
        return self.size
