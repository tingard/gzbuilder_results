import os
from tqdm import tqdm
import numpy as onp
import pandas as pd
import jax.numpy as np
from jax import ops
from jax.config import config
from scipy.optimize import minimize
from copy import deepcopy
from gzbuilder_analysis.fitting.reparametrization import from_reparametrization
from gzbuilder_analysis.fitting.optimizer import Optimizer, render_comps
from gzbuilder_analysis.fitting.misc import psf_conv, get_luminosity_keys, \
    remove_zero_brightness_components, lower_spiral_indices, correct_spirals, \
    correct_axratio


config.update("jax_enable_x64", True)
fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')


# define two handy functions to read results back from the GPU for scipy's
# LBFGS-b
def __f(p, optimizer, keys):
    return onp.array(optimizer(p, keys).block_until_ready(), dtype=np.float64)


def __j(p, optimizer, keys):
    return onp.array(optimizer.jac(p, keys).block_until_ready(), dtype=np.float64)


def __bar_incrementer(bar):
    def f(*a, **k):
        bar.update(1)
    return f


SCALE_FACTOR = 2

base_models = pd.read_pickle(
    'affirmation_subjects_results/base_models.pkl.gz'
)
aggregation_results = pd.read_pickle(
    'affirmation_subjects_results/agg_results.pkl.gz'
)
agg_fit_metadata = pd.read_pickle(
    'affirmation_subjects_results/affirmation_metadata.pkl.gz'
)


# def plot_fit_result(
#     name, target, mask, multiplier, sigma,
#     op, agg_res,
#     starting_model, final_model, final_gal
# ):
#     f, ax = plt.subplots(nrows=2, ncols=3, figsize=(15*1.2, 8*1.2), dpi=80)
#     plt.subplots_adjust(wspace=0, hspace=0.1)
#     s = AsinhStretch()
#     masked_target = ops.index_update(target, mask, np.nan)
#     lm = s([
#         min(
#             np.nanmin(masked_target),
#             final_gal.min()
#         ),
#         max(
#             np.nanmax(masked_target),
#             final_gal.max()
#         )
#     ])
#     ax[0, 0].imshow(
#         s(masked_target),
#         vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
#     )
#     ax[0, 1].imshow(
#         s(final_gal),
#         vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
#     )
#     d = ops.index_update((final_gal - target)/sigma, mask, np.nan)
#     l2 = np.nanmax(np.abs(d))
#     c = ax[0][2].imshow(d, vmin=-l2, vmax=l2, cmap='seismic', origin='lower')
#     cbar = plt.colorbar(c, ax=ax, shrink=0.475, anchor=(0, 1))
#     cbar.ax.set_ylabel(r'Residual, units of $\sigma$')
#     ax[1, 0].imshow(
#         s(masked_target),
#         vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
#     )
#     ax[1, 1].imshow(
#         s(masked_target),
#         vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
#     )
#
#     initial_disk = scale(
#         ag.make_ellipse(starting_model['disk']), SCALE_FACTOR, SCALE_FACTOR
#     )
#     final_disk = scale(
#         ag.make_ellipse(final_model['disk']), SCALE_FACTOR, SCALE_FACTOR
#     )
#     ax[1, 0].add_patch(PolygonPatch(initial_disk, ec='C0', fc='none'))
#     ax[1, 1].add_patch(PolygonPatch(final_disk, ec='C0', fc='none'))
#     if starting_model['bulge'] is not None:
#         initial_bulge = scale(
#             ag.make_ellipse(starting_model['bulge']),
#             SCALE_FACTOR, SCALE_FACTOR
#         )
#         ax[1, 0].add_patch(PolygonPatch(initial_bulge, ec='C1', fc='none'))
#     if final_model['bulge'] is not None:
#         final_bulge = scale(
#             ag.make_ellipse(final_model['bulge']),
#             SCALE_FACTOR, SCALE_FACTOR
#         )
#         ax[1, 1].add_patch(PolygonPatch(final_bulge, ec='C1', fc='none'))
#     if starting_model['bar'] is not None:
#         initial_bar = scale(
#             ag.make_box(starting_model['bar']),
#             SCALE_FACTOR, SCALE_FACTOR
#         )
#         ax[1, 0].add_patch(PolygonPatch(initial_bar, ec='C2', fc='none'))
#     if final_model['bar'] is not None:
#         final_bar = scale(
#             ag.make_box(final_model['bar']),
#             SCALE_FACTOR, SCALE_FACTOR
#         )
#         ax[1, 1].add_patch(PolygonPatch(final_bar, ec='C2', fc='none'))
#     for i in range(op.n_spirals):
#         ax[1, 0].plot(*fit.log_spiral(
#             t_min=starting_model['spiral']['t_min.{}'.format(i)],
#             t_max=starting_model['spiral']['t_max.{}'.format(i)],
#             A=starting_model['spiral']['A.{}'.format(i)],
#             phi=starting_model['spiral']['phi.{}'.format(i)],
#             **starting_model['disk']
#         ).T, 'r')
#     for a in agg_res.spiral_arms:
#         ax[1, 0].plot(*a.reprojected_log_spiral.T, 'r')
#     delta_roll = (
#         final_model['disk']['roll']
#         - starting_model['disk']['roll']
#     )
#     for i in range(op.n_spirals):
#         try:
#             ax[1][1].plot(*fit.log_spiral(
#                 t_min=final_model['spiral']['t_min.{}'.format(i)],
#                 t_max=final_model['spiral']['t_max.{}'.format(i)],
#                 A=final_model['spiral']['A.{}'.format(i)],
#                 phi=final_model['spiral']['phi.{}'.format(i)],
#                 delta_roll=delta_roll,
#                 **final_model['disk']
#             ).T, 'r')
#         except KeyError:
#             # some spirals may have been removed if their intensity was set to 0
#             pass
#     ax[0, 0].set_title('Galaxy Image')
#     ax[0, 1].set_title('Model after fitting')
#     ax[0, 2].set_title('Residuals')
#     ax[1, 0].set_title('Raw Aggregate model overlaid on galaxy')
#     ax[1, 1].set_title('Fit model overlaid on galaxy')
#     for a in ax.ravel():
#         a.axis('off')
#         a.set_xlim(0, target.shape[1])
#         a.set_ylim(0, target.shape[0])
#     os.makedirs('affirmation_subjects_results/fitting_plots', exist_ok=True)
#     plt.savefig('affirmation_subjects_results/fitting_plots/{}.pdf'.format(name), bbox_inches='tight')
#     plt.close()


def do_subject(subject_id):
    fm = agg_fit_metadata.loc[subject_id]
    name = base_models.loc[subject_id]['name']
    agg_res = aggregation_results.loc[subject_id]
    starting_model = agg_res.model

    o = Optimizer(
        agg_res,
        *fm[['psf', 'galaxy_data', 'sigma_image']],
        oversample_n=5
    )
    # define the parameters controlling only the brightness of components, and
    # fit them first
    L_keys = get_luminosity_keys(o.model)

    # perform the first fit
    with tqdm(desc='Fitting brightness', leave=False) as bar:
        res = minimize(
            __f,
            onp.array([o.model_[k] for k in L_keys]),
            jac=__j,
            args=(o, L_keys),
            callback=__bar_incrementer(bar),
            bounds=onp.array([o.lims_[k] for k in L_keys]),
        )

    # update the optimizer with the new parameters
    for k, v in zip(L_keys, res['x']):
        o[k] = v

    # perform the full fit
    with tqdm(desc='Fitting everything', leave=False) as bar:
        res_full = minimize(
            __f,
            onp.array([o.model_[k] for k in o.keys]),
            jac=__j,
            args=(o, o.keys),
            callback=__bar_incrementer(bar),
            bounds=onp.array([o.lims_[k0][k1] for k0, k1 in o.keys]),
            options=dict(maxiter=10000)
        )

    final_model = pd.Series({
        **deepcopy(o.model_),
        **{k: v for k, v in zip(o.keys, res_full['x'])}
    })

    # correct the parameters of spirals in this model for the new disk,
    # allowing rendering of the model without needing the rotation of the disk
    # before fitting
    final_model = correct_spirals(final_model, o.base_roll)

    # fix component axis ratios (if > 1, flip major and minor axis)
    final_model = correct_axratio(final_model)

    # remove components with zero brightness
    final_model = remove_zero_brightness_components(final_model)

    # lower the indices of spirals where possible
    final_model = lower_spiral_indices(final_model)

    comps = o.render_comps(final_model.to_dict(), correct_spirals=False)

    d = ops.index_update(
        psf_conv(sum(comps.values()), o.psf) - o.target,
        o.mask,
        np.nan
    )
    chisq = float(np.sum((d[~o.mask] / o.sigma[~o.mask])**2) / (~o.mask).sum())
    disk_spiral_L = (
        final_model[('disk', 'L')]
        + (comps['spiral'].sum() if 'spiral' in comps else 0)
    )
    # fractions were originally parametrized vs the disk and spirals (bulge
    # had no knowledge of bar and vice versa)
    bulge_frac = final_model.get(('bulge', 'frac'), 0)
    bar_frac = final_model.get(('bar', 'frac'), 0)

    bulge_L = bulge_frac * disk_spiral_L / (1 - bulge_frac)
    bar_L = bar_frac * disk_spiral_L / (1 - bar_frac)
    gal_L = disk_spiral_L + bulge_L + bar_L

    bulge_frac = bulge_L / (disk_spiral_L + bulge_L + bar_L)
    bar_frac = bar_L / (disk_spiral_L + bulge_L + bar_L)

    deparametrized_model = from_reparametrization(final_model, o)
    os.makedirs('affirmation_subjects_results/tuning_results', exist_ok=True)
    pd.to_pickle(
        dict(
            base_model=starting_model,
            fit_model=final_model,
            deparametrized=deparametrized_model,
            res=res_full,
            chisq=chisq,
            comps=comps,
            r_band_luminosity=float(gal_L),
            bulge_frac=float(bulge_frac),
            bar_frac=float(bar_frac),
            keys=o.keys,
        ),
        'affirmation_subjects_results/tuning_results/{}.pickle.gz'.format(name)
    )


def main(subject_ids, check=True):
    with tqdm(subject_ids, desc='Iterating over subjects') as bar0:
        for subject_id in bar0:
            do_subject(subject_id)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            'Fit Aggregate model and best individual'
            ' model for a galaxy builder subject'
        )
    )
    parser.add_argument(
        '--subjects',
        metavar='subject_ids', type=int, nargs='+',
        help='Subject ids to work on (otherwise will run all un-fit subjects)')
    args = parser.parse_args()
    main(args.subjects or agg_fit_metadata.index)
