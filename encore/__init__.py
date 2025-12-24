"""
@Author: Conghao Wong
@Date: 2025-12-02 10:01:49
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-24 11:26:30
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import EncoreArgs
from .model import Encore, EncoreModel

qpid.register(
    ver=[Encore, EncoreModel],  # Support previously trained models
    enc=[Encore, EncoreModel],
)

qpid.register_args(EncoreArgs, 'Encore Args')
