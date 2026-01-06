"""
@Author: Conghao Wong
@Date: 2025-12-02 10:01:49
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-06 15:43:08
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import EncoreArgs
from .model import Encore, EncoreModel

qpid.register(
    enc=[Encore, EncoreModel],
)

qpid.register_args(EncoreArgs, 'Encore Args')
