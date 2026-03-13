# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the allex_rl_dexblind package."""

import os
import re

from setuptools import find_packages, setup

# Project root (where this setup.py lives)
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML = os.path.join(PROJECT_ROOT, "config", "extension.toml")
with open(EXTENSION_TOML) as f:
    content = f.read()
_version = re.search(r'version\s*=\s*"([^"]+)"', content).group(1) if re.search(r'version\s*=', content) else "0.1.0"
_description = re.search(r'description\s*=\s*"([^"]+)"', content).group(1) if re.search(r'description\s*=', content) else "ALLEX + Dexblind RL with Newton physics"
_repo = re.search(r'repository\s*=\s*"([^"]+)"', content).group(1) if re.search(r'repository\s*=', content) else ""

INSTALL_REQUIRES = [
    "psutil",
]

setup(
    name="allex_rl_dexblind",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    author="IRIM",
    maintainer="IRIM",
    url=_repo,
    version=_version,
    description=_description,
    keywords=["extension", "isaaclab", "newton", "allex", "dexblind"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
