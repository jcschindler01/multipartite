"""
Hello! I am the comment heading the top level __init__.py in skeleton.
"""

from importlib.metadata import version
__version__ = version("multipartite")

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("classic")

from multipartite import mptools
from multipartite import blochvec
from multipartite.mptools import *