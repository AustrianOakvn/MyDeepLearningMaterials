import os.path as osp
import random
import itertools
from math import sqrt
import pandas as pd
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)