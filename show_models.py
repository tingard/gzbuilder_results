import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.affinity import scale, translate
from descartes import PolygonPatch
from tqdm import tqdm
import gzbuilder_analysis.aggregation as aggregation
from asinh_cmap import asinh_cmap

loc = os.path.abspath(os.path.dirname(__file__))
lib = join(loc, 'lib')

models = pd.read_pickle(join(lib, 'models2.pickle'))
fitting_metadata = pd.read_pickle(join(lib, 'fitting_metadata.pkl'))

types = (
    'best_individual', 'tuned_best_individual',
    'aggregate',  'tuned_aggregate'
)
names = [t.replace('_', ' ').capitalize() for t in types]
with tqdm(models.index) as bar:
    for subject_id in bar:
        data = fitting_metadata.loc[subject_id]['galaxy_data']

        # functions for plotting
        def transform_patch(p):
            corrected_patch = scale(
                translate(p, xoff=-data.shape[1]/2, yoff=-data.shape[0]/2),
                xfact=0.396,
                yfact=0.396,
                origin=(0, 0),
            )
            # display patch at 3*Re
            return scale(corrected_patch, 3, 3)

        def transform_arm(arm):
            return (arm - np.array(data.shape) / 2) * 0.396

        extent = (np.array([[-1, -1], [1, 1]]) * data.shape).T.ravel() / 2 * 0.396
        imshow_kwargs = dict(
            cmap=asinh_cmap, origin='lower',
            extent=extent
        )

        fig, ax = plt.subplots(
            ncols=2, nrows=2, sharex=True, sharey=True,
            figsize=(12, 12), dpi=100
        )
        ax = ax.ravel()
        for i, type in enumerate(types):
            ax[i].imshow(data, **imshow_kwargs)
            try:
                model = models.loc[subject_id][type]
                disk = aggregation.make_ellipse(model['disk'])
                bulge = aggregation.make_ellipse(model['bulge'])
                bar = aggregation.make_box(model['bar'])
            except TypeError:
                continue
            for j, geom in enumerate((disk, bulge, bar)):
                if geom is not None:
                    p = PolygonPatch(
                        transform_patch(geom), fc='none', ec='C{}'.format(j),
                        zorder=2, lw=1
                    )
                    p2 = PolygonPatch(
                        transform_patch(geom), fc='C{}'.format(j), ec='none',
                        alpha=0.1, zorder=2, lw=2
                    )
                    ax[i].add_patch(p)
                    ax[i].add_patch(p2)
            for points, params in model['spiral']:
                ax[i].plot(*transform_arm(points).T, c='r', linewidth=2, alpha=0.6)
            ax[i].set_title(type.replace('_', ' ').capitalize())
        ax[2].set_xlabel('Arcseconds from galaxy centre')
        ax[3].set_ylabel('Arcseconds from galaxy centre')
        ax[0].set_ylabel('Arcseconds from galaxy centre')
        ax[2].set_ylabel('Arcseconds from galaxy centre')
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plt.tight_layout()
        plt.savefig(f'tuning_plots/{subject_id}.png', bbox_inches='tight')
        plt.close()
