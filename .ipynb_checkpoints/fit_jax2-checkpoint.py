import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from shapely.affinity import scale
from descartes import PolygonPatch
from astropy.visualization import AsinhStretch
from tqdm import tqdm
from time import sleep
import jax.numpy as np
from jax import ops
import gzbuilder_analysis.aggregation as ag
import gzbuilder_analysis.parsing as pg
from gzbuilder_analysis.fitting.optimizer import Optimizer, get_p, get_spirals
from jax.config import config
config.update("jax_enable_x64", True)


SCALE_FACTOR = 2

agg_res = pd.read_pickle('output_files/aggregation_results/20902040.pkl.gz')
fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')


def get_finished_subjects():
    try:
        return {
            int(g.group(1))
            for g in (
                re.match(r'([0-9]+).pickle.gz', f)
                for f in os.listdir('output_files/tuning_results')
            )
            if g is not None
        }
    except OSError:
        return {}


def plot_fit_result(
    subject_id, galaxy_data, sigma_image, psf,
    op, agg_res,
    starting_model, final_model
):
    fit_model_render = op.render(final_model)
    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(15*1.2, 8*1.2), dpi=80)
    plt.subplots_adjust(wspace=0, hspace=0.1)
    s = AsinhStretch()
    lm = s([
        min(
            np.nanmin(galaxy_data),
            fit_model_render.min()
        ),
        max(
            np.nanmax(galaxy_data),
            fit_model_render.max()
        )
    ])
    ax[0, 0].imshow(
        s(galaxy_data),
        vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
    )
    ax[0, 1].imshow(
        s(fit_model_render),
        vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
    )
    d = (galaxy_data - np.array(fit_model_render)) / sigma_image
    l2 = np.nanmax(np.abs(d))
    c = ax[0][2].imshow(d, vmin=-l2, vmax=l2, cmap='seismic', origin='lower')
    cbar = plt.colorbar(c, ax=ax, shrink=0.475, anchor=(0, 1))
    cbar.ax.set_ylabel(r'Residual, units of $\sigma$')
    ax[1, 0].imshow(
        s(galaxy_data),
        vmin=lm[0], vmax=lm[1], cmap='gray_r', origin='lower'
    )
    ax[1, 1].imshow(
        s(galaxy_data),
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
    for a in agg_res.spiral_arms:
        ax[1, 0].plot(*a.reprojected_log_spiral.T, 'r')
    delta_roll = (
        final_model['disk']['roll']
        - starting_model['disk']['roll']
    )
    spiral_arms = get_spirals(final_model, op.n_spirals, starting_model['disk']['roll'])
    for arm in spiral_arms:
        ax[1, 1].plot(*arm.T, 'r')
    ax[0, 0].set_title('Galaxy Image')
    ax[0, 1].set_title('Model after fitting')
    ax[0, 2].set_title('Residuals')
    ax[1, 0].set_title('Raw Aggregate model overlaid on galaxy')
    ax[1, 1].set_title('Fit model overlaid on galaxy')
    for a in ax.ravel():
        a.axis('off')
        a.set_xlim(0, galaxy_data.shape[1])
        a.set_ylim(0, galaxy_data.shape[0])
    os.makedirs('fitting_plots', exist_ok=True)
    plt.savefig('fitting_plots/{}.pdf'.format(subject_id), bbox_inches='tight')
    plt.close()


def do_subject(subject_id):
    fm = fitting_metadata.loc[20902040]
    o = Optimizer(agg_res, **fm.to_dict(), oversample_n=5)
    keys = [(k0, k1) for k0 in o.model for k1 in o.model[k0]]
    L_keys = (
        [('disk', 'L'), ('bulge', 'frac'), ('bar', 'frac')]
        + [('spiral', 'I.{}'.format(i)) for i in range(len(o.model['spiral']) // 6)]
    )
    L_keys = [i for i in L_keys if bool(o.model[i[0]]) and i[1] in o.model[i[0]]]
    fit_result0 = o.do_fit(options=dict(maxiter=500), keys=L_keys, desc='Fitting Brightness')
    keys1 = [(k0, k1) for k0 in o.model for k1 in o.model[k0]]
    p1 = get_p(fit_result0['fit_model'], keys)
    fit_result1 = o.do_fit(keys=keys1, p0=p1, options=dict(maxiter=1000))
    final_params = pd.DataFrame(fit_result1['fit_model']).unstack().dropna()

    plot_fit_result(
        subject_id, fm.galaxy_data, fm.sigma_image, fm.psf,
        o, agg_res,
        agg_res.model, final_params['fit_galaxy_uncorrected']
    )
    os.makedirs('output_files/tuning_results', exist_ok=True)
    pd.to_pickle(
        dict(final_params=final_params, **fit_result1),
        'output_files/tuning_results/{}.pickle.gz'.format(subject_id)
    )


def main(subject_ids, check=True):
    with tqdm(subject_ids, desc='Iterating over subjects') as bar0:
        for subject_id in bar0:
            if (not check) or subject_id not in get_finished_subjects():
                do_subject(subject_id)
            else:
                sleep(0.05)


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
    main(args.subjects or fitting_metadata.index, check=False)
