# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import re
from contextlib import contextmanager
from os import getenv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Structure
from ruamel.yaml import YAML
from xenonpy._conf import __cfg_root__, __db_version__, __github_username__

__all__ = ["camel_to_snake", "get_data_loc", "get_dataset_url", "get_sha256", "absolute_path", "set_env", "config"]


@contextmanager
def set_env(**kwargs):
    """
    Set temp environment variable with ``with`` statement.

    Examples
    --------
    >>> import os
    >>> with set_env(test='test env'):
    >>>    print(os.getenv('test'))
    test env
    >>> print(os.getenv('test'))
    None

    Parameters
    ----------
    kwargs: dict[str]
        Dict with string value.
    """
    import os

    tmp = dict()
    for k, v in kwargs.items():
        tmp[k] = os.getenv(k)
        os.environ[k] = v
    yield
    for k, v in tmp.items():
        if not v:
            del os.environ[k]
        else:
            os.environ[k] = v


def config(key=None, **key_vals):
    """
    Return config value with key or all config.

    Parameters
    ----------
    key: str
        Keys of config item.
    key_vals: dict
        Set item's value by key.
    Returns
    -------
    str
        The value corresponding to the key.
    """
    yaml = YAML(typ="safe")
    yaml.indent(mapping=2, sequence=4, offset=2)
    dir_ = Path(__cfg_root__)
    cfg_file = dir_ / "conf.yml"

    # from user local
    with open(str(cfg_file), "r") as f:
        conf = yaml.load(f)

    value = None

    # getter
    if key:
        if key in conf:
            value = conf[key]
        else:
            tmp = Path(__file__).parents[1] / "conf.yml"
            with open(str(tmp)) as f:
                conf_ = yaml.load(f)

            if key not in conf_:
                raise RuntimeError("No item(s) named %s in configurations" % key)

            value = conf_[key]

    # setter
    if key_vals:
        for key, v in key_vals.items():
            conf[key] = v
        with open(str(cfg_file), "w") as f:
            yaml.dump(conf, f)

    return value


def camel_to_snake(text):
    str1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", str1).lower()


def get_dataset_url(name, version=__db_version__):
    """
    Return url with the given file name.

    Args
    ----
    name: str
        binary file name.
    version: str
        The version of repository.
        See Also: https://github.com/yoshida-lab/dataset/releases

    Return
    ------
    str
        binary file url.
    """
    return "https://github.com/{0}/dataset/releases/download/v{1}/{2}.pd.xz".format(__github_username__, version, name)


def get_data_loc(name):
    """Return user data location"""

    scheme = ("userdata", "usermodel")
    if name not in scheme:
        raise ValueError("{} not in {}".format(name, scheme))
    if getenv(name):
        return str(Path(getenv(name)).expanduser())
    return str(Path(config(name)).expanduser())


def absolute_path(path, ignore_err=True):
    """
    Resolve path when path include ``~``, ``parent/here``.

    Parameters
    ----------
    path: str
        Path to expand.
    ignore_err: bool
        FileNotFoundError is raised when set to False.
        When True, the path will be created.
    Returns
    -------
    str
        Expanded path.
    """
    from sys import version_info

    if isinstance(path, str):
        path = Path(path)

    if version_info[1] == 5:
        if ignore_err:
            path.expanduser().mkdir(parents=True, exist_ok=True)
        return str(path.expanduser().resolve())
    return str(path.expanduser().resolve(not ignore_err))


def get_sha256(fname):
    """
    Calculate file's sha256 value

    Parameters
    ----------
    fname: str
        File name.

    Returns
    -------
    str
        sha256 value.
    """
    from hashlib import sha256

    hasher = sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_dissimilarity(anchor_structure: Structure, *structure: Structure, n_jobs=10):
    """Calculate the dissimilarity between the anchor structure and the other structures.

    Args:
        anchor_structure: The anchor structure.
        structure: The other structures.
        n_jobs: The number of jobs to run in parallel. Default is 10.

    Returns:
        np.array: The dissimilarity between the anchor structure and the other structures.
    """
    ssf = SiteStatsFingerprint(
        CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
        stats=("mean", "std_dev", "minimum", "maximum"),
    )
    v_anchor = np.array(ssf.featurize(anchor_structure))
    tmp = Parallel(n_jobs=n_jobs)(delayed(ssf.featurize)(s) for s in structure)
    return [np.linalg.norm(np.array(s) - v_anchor) for s in tmp]


def convert_struct(structure: Structure, volume=None) -> Structure:
    """
    Convert the structure to a primitive structure and adjust the volume of the unit cell.

    Args:
        structure (Structure): The input structure.
        volume (float): The volume of the unit cell. Default is None.

    Returns:
        Structure: The modified structure.
    """
    structure = structure.get_primitive_structure()
    if volume is not None:
        structure.scale_lattice(volume)
    return structure
