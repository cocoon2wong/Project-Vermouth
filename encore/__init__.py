"""
@Author: Conghao Wong
@Date: 2025-12-02 10:01:49
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-16 21:08:27
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import EncoreArgs
from .minimumModel import MinimumEncore, MinimumEncoreModel
from .model import Encore, EncoreModel

qpid.register(
    enc=[Encore, EncoreModel],
    minienc=[MinimumEncore, MinimumEncoreModel],
)

qpid.register_args(EncoreArgs, 'Encore Args')
