from jax.config import config
import jax
import os
import re
from time import sleep
import matplotlib.pyplot as plt
from shapely.affinity import scale
from descartes import PolygonPatch
from astropy.visualization import AsinhStretch
from tqdm import tqdm
import numpy as onp
import jax.numpy as np
from jax import ops
import pandas as pd
from scipy.optimize import minimize
import gzbuilder_analysis.fitting.jax as fit
import gzbuilder_analysis.aggregation as ag
import gzbuilder_analysis.parsing as pg
print('JAX version:', jax.__version__)
assert float(jax.__version__.split('.')[-1]) > 54
config.update("jax_enable_x64", True)

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


def plot_fit_result(
    name, target, mask, multiplier, sigma,
    op, agg_res,
    starting_model, final_model, final_gal
):
    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(15*1.2, 8*1.2), dpi=80)
    plt.subplots_adjust(wspace=0, hspace=0.1)
    s = AsinhStretch()
    masked_target = ops.index_update(target, mask, np.nan)
    lm = s([
        min(
            np.nanmin(masked_target),
            final_gal.min()
        ),
        max(
            np.nanmax(masked_target),
            final_gal.max()
        )
    ])
    ax[0, 0].imshow(
        s(masked_target),
        vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
    )
    ax[0, 1].imshow(
        s(final_gal),
        vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
    )
    d = ops.index_update((final_gal - target)/sigma, mask, np.nan)
    l2 = np.nanmax(np.abs(d))
    c = ax[0][2].imshow(d, vmin=-l2, vmax=l2, cmap='seismic', origin='lower')
    cbar = plt.colorbar(c, ax=ax, shrink=0.475, anchor=(0, 1))
    cbar.ax.set_ylabel(r'Residual, units of $\sigma$')
    ax[1, 0].imshow(
        s(masked_target),
        vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
    )
    ax[1, 1].imshow(
        s(masked_target),
        vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
    )

    initial_disk = scale(
        ag.make_ellipse(starting_model['disk']), SCALE_FACTOR, SCALE_FACTOR
    )
    final_disk = scale(
        ag.make_ellipse(final_model['disk']), SCALE_FACTOR, SCALE_FACTOR
    )
    ax[1, 0].add_patch(PolygonPatch(initial_disk, ec='C0', fc='none'))
    ax[1, 1].add_patch(PolygonPatch(final_disk, ec='C0', fc='none'))
    if starting_model['bulge'] is not None:
        initial_bulge = scale(
            ag.make_ellipse(starting_model['bulge']),
            SCALE_FACTOR, SCALE_FACTOR
        )
        ax[1, 0].add_patch(PolygonPatch(initial_bulge, ec='C1', fc='none'))
    if final_model['bulge'] is not None:
        final_bulge = scale(
            ag.make_ellipse(final_model['bulge']),
            SCALE_FACTOR, SCALE_FACTOR
        )
        ax[1, 1].add_patch(PolygonPatch(final_bulge, ec='C1', fc='none'))
    if starting_model['bar'] is not None:
        initial_bar = scale(
            ag.make_box(starting_model['bar']),
            SCALE_FACTOR, SCALE_FACTOR
        )
        ax[1, 0].add_patch(PolygonPatch(initial_bar, ec='C2', fc='none'))
    if final_model['bar'] is not None:
        final_bar = scale(
            ag.make_box(final_model['bar']),
            SCALE_FACTOR, SCALE_FACTOR
        )
        ax[1, 1].add_patch(PolygonPatch(final_bar, ec='C2', fc='none'))
    for i in range(op.n_spirals):
        ax[1, 0].plot(*fit.log_spiral(
            t_min=starting_model['spiral']['t_min.{}'.format(i)],
            t_max=starting_model['spiral']['t_max.{}'.format(i)],
            A=starting_model['spiral']['A.{}'.format(i)],
            phi=starting_model['spiral']['phi.{}'.format(i)],
            **starting_model['disk']
        ).T, 'r')
    for a in agg_res.spiral_arms:
        ax[1, 0].plot(*a.reprojected_log_spiral.T, 'r')
    delta_roll = (
        final_model['disk']['roll']
        - starting_model['disk']['roll']
    )
    for i in range(op.n_spirals):
        try:
            ax[1][1].plot(*fit.log_spiral(
                t_min=final_model['spiral']['t_min.{}'.format(i)],
                t_max=final_model['spiral']['t_max.{}'.format(i)],
                A=final_model['spiral']['A.{}'.format(i)],
                phi=final_model['spiral']['phi.{}'.format(i)],
                delta_roll=delta_roll,
                **final_model['disk']
            ).T, 'r')
        except KeyError:
            # some spirals may have been removed if their intensity was set to 0
            pass
    ax[0, 0].set_title('Galaxy Image')
    ax[0, 1].set_title('Model after fitting')
    ax[0, 2].set_title('Residuals')
    ax[1, 0].set_title('Raw Aggregate model overlaid on galaxy')
    ax[1, 1].set_title('Fit model overlaid on galaxy')
    for a in ax.ravel():
        a.axis('off')
        a.set_xlim(0, target.shape[1])
        a.set_ylim(0, target.shape[0])
    os.makedirs('affirmation_subjects_results/fitting_plots', exist_ok=True)
    plt.savefig('affirmation_subjects_results/fitting_plots/{}.pdf'.format(name), bbox_inches='tight')
    plt.close()


def do_subject(subject_id):
    fm = agg_fit_metadata.loc[subject_id]
    target = np.asarray(fm['galaxy_data'].data)
    mask = np.asarray(fm['galaxy_data'].mask)
    sigma = np.asarray(fm['sigma_image'].data)
    multiplier = fm.multiplier
    name =  base_models.loc[subject_id]['name']
    agg_res = aggregation_results.loc[subject_id]
    starting_model = agg_res.model

    op = fit.Optimizer(agg_res, fm)

    op.lims['disk']['Re'] = [0.01, target.shape[0] / 2]
    op.reset_keys()

    op.set_keys(
        [('disk', 'L')] + [
            (k, 'frac') for k in ('bulge', 'bar')
            if agg_res.model[k] is not None
        ] + [
            ('spiral', 'I.{}'.format(i))
            for i in range(op.n_spirals)
        ]
    )

    res0 = minimize(
        lambda p: onp.float64(op(p).block_until_ready()),
        onp.array(op.p0, dtype=onp.float64),
        method='L-BFGS-B',
        jac=lambda p: onp.array(op.jacobian(p).block_until_ready()),
        bounds=op.limits,
        options=dict(maxiter=10000),
    )
    # update the Optimizer with the new parameters
    for i, (k0, k1) in enumerate(op.keys):
        op.model[k0][k1] = res0['x'][i]

    # reset so we can fit the entire model
    op.reset_keys()
    desc = 'Fitting {}, {} parameters, {} pixels'.format(
        subject_id, len(op.keys), fm['galaxy_data'].size
    )
    with tqdm(desc=desc, leave=False) as pbar:
        def callback(*args, **kwargs):
            pbar.update(1)
            pass
        # L-BFGS-B needs numpy float64 arrays, so we need to convert
        # from JAX DeviceArrays, which adds a slight overhead
        res = minimize(
            lambda p: onp.float64(op(p).block_until_ready()),
            onp.array(op.p0, dtype=onp.float64),
            method='L-BFGS-B',
            jac=lambda p: onp.array(
                op.jacobian(p).block_until_ready()
            ),
            bounds=op.limits,
            callback=callback,
            options=dict(maxiter=10000),
        )

    final_gal = fit.create_model(
        res['x'], op.keys, op.n_spirals, op.model, op.psf, op.target, 5
    )

    new_params = fit.to_dict(res['x'], op.keys)
    final_model = pg.sanitize_model(fit.remove_invisible_components(
        fit.from_reparametrization({
            k: {**op.model[k], **new_params.get(k, {})}
            for k in op.model
        })
    ))
    plot_fit_result(
        name, target, mask, multiplier, sigma,
        op, agg_res,
        starting_model, final_model, final_gal
    )

    chisq = np.sum(
        (((target - final_gal)/sigma)**2)[~mask]
    ) / (~mask).astype(int).sum()

    output = dict(
        res=res,
        final_model=final_model,
        chisq=chisq,
    )
    os.makedirs('affirmation_subjects_results/tuning_results', exist_ok=True)
    pd.to_pickle(
        output,
        'affirmation_subjects_results/tuning_results/{}.pickle.gz'.format(
            name
        )
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
