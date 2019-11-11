import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import gzbuilder_analysis.rendering.cuda as rg
except ModuleNotFoundError:
    import gzbuilder_analysis.rendering as rg
from lib.galaxy_utilities import get_diff_data
from asinh_cmap import asinh_cmap

fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl').loc[20902040]
models = pd.read_pickle('lib/models.pickle').loc[20902040]
multiplier = get_diff_data(20902040)['multiplier']
data = fitting_metadata['galaxy_data']
psf = fitting_metadata['psf']
sigma_image = fitting_metadata['sigma_image']

extent = (np.array([[-1, -1],[1, 1]]) * data.shape).T.ravel() / 2 * 0.396
imshow_kw = dict(extent=extent, origin='lower')

r_bi = rg.calculate_model(models['tuned_best_individual'], data.shape, psf)
r_agg = rg.calculate_model(models['tuned_aggregate'], data.shape, psf)
d_bi = data - r_bi
d_agg = data - r_agg
cq_bi = models['tuned_best_individual_chisq']
cq_agg = models['tuned_aggregate_chisq']

r_lims = dict(vmin=0, vmax=max(r_bi.max(), r_agg.max()))
_l = max(np.abs(d_bi).max(), np.abs(d_bi).max())
d_lims = dict(vmin=-_l, vmax=_l)

fig, ax = plt.subplots(figsize=(9, 8), dpi=100, ncols=2, nrows=2, sharex=True, sharey=True)
title_bi = 'Tuned Best Individual\n'+ r'$\chi_\nu^2 = ' + f'{cq_bi:.2f}$'
title_agg = 'Tuned Aggregate\n'+ r'$\chi_\nu^2 = ' + f'{cq_agg:.2f}$'
ax[0][0].set_title(title_bi)
ax[0][1].set_title(title_agg)

ax[0][0].imshow(r_bi, cmap=asinh_cmap, **r_lims, **imshow_kw)
rendered_model = ax[0][1].imshow(r_agg, cmap=asinh_cmap, **r_lims, **imshow_kw)
r_c = plt.colorbar(rendered_model, ax=ax[0], shrink=0.95)

ax[1][0].imshow(d_bi, cmap='seismic', **d_lims, **imshow_kw)
diff_im = ax[1][1].imshow(d_agg, cmap='seismic', **d_lims, **imshow_kw)
d_c = plt.colorbar(diff_im, ax=ax[1], shrink=0.95)

ax[1][0].set_xlabel('Arcseconds from galaxy centre')
ax[1][1].set_xlabel('Arcseconds from galaxy centre')
ax[0][0].set_ylabel('Arcseconds from galaxy centre')
ax[1][0].set_ylabel('Arcseconds from galaxy centre')
r_c.ax.set_ylabel('nMgy')
d_c.ax.set_ylabel('nMgy')
plt.savefig('method-paper-plots/bi_vs_agg_comparison.pdf', bbox_inches='tight')
