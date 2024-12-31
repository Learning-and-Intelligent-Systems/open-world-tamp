# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os.path as osp

from .imdb import imdb
from .ocid_object import OCIDObject
from .osd_object import OSDObject
from .tabletop_object import TableTopObject

ROOT_DIR = osp.join(osp.dirname(__file__), "..", "..")
