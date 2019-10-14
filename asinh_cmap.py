import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from gzbuilder_analysis.rendering import asinh_stretch

__x_in = np.linspace(0, 1, 512)
__x_out = asinh_stretch(__x_in)
__x_out /= __x_out.max()
__newcolors = cm.get_cmap('gray', 1024)(__x_out)
asinh_colormap = ListedColormap(__newcolors)

if __name__ == '__main__':
    import pandas as pd
    fm = pd.read_pickle('lib/fitting_metadata.pkl')
    data = np.nanprod((
        fm.loc[20902040].galaxy_data,
        fm.loc[20902040].pixel_mask
    ), axis=0)

    """
    helper function to plot two colormaps
    """
    cms = (cm.gray, asinh_colormap)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True)
        fig.colorbar(psm, ax=ax)
    plt.show()
