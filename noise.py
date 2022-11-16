from pathlib import Path
from typing import Union, Tuple, List, Optional

import matplotlib.pyplot as plt
from matplotlib import axes, cm, colors, cycler, gridspec
import numpy as np
import scipy.signal as scipy_signal
from tqdm.notebook import tqdm

from mcbj import TracePair
import utils