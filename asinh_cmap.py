import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from gzbuilder_analysis.rendering import asinh_stretch

__x_in = np.linspace(0, 1, 512)
__x_out = asinh_stretch(__x_in)
__x_out /= __x_out.max()
__newcolors = cm.get_cmap('gray', 2048)(__x_out)
asinh_cmap = ListedColormap(__newcolors)

__newcolors_r = cm.get_cmap('gray_r', 2048)(__x_out)
asinh_cmap_r = ListedColormap(__newcolors_r)
