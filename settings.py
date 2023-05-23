import logging
import numpy as np
from torch import Tensor


# Data settings
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SCALE_SIZE = 448 #原本的size为512
CROP_SIZE=572
CROP_NUM = 0
IGNORE_LABEL = 255

# Model definition
N_CLASSES = 20
STRIDE = 8
BN_MOM = 3e-4
EM_MOM = 0.9
STAGE_NUM = 3

# Training settings
BATCH_SIZE = 8
ITER_MAX = 30000
ITER_SAVE = 2000
ITER_OUT=10
EPOCHS=50

LR_DECAY = 10
LR = 0.01
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

DEVICE = 0
DEVICES = [0, 1]

