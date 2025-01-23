import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax

from colabdesign import clear_mem, mk_afdesign_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.utils import copy_dict
