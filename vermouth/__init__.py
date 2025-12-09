"""
@Author: Conghao Wong
@Date: 2025-12-02 10:01:49
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-02 11:16:12
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import VermouthArgs
from .model import Vermouth, VermouthModel

qpid.register(ver=[Vermouth, VermouthModel])
qpid.register_args(VermouthArgs, 'Vermouth Args')
