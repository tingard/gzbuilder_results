import jax
from jax.config import config
import os
from tqdm import tqdm
import numpy as onp
import jax.numpy as np
import pandas as pd
from scipy.optimize import minimize
import gzbuilder_analysis.fitting.jax as fit
from gzbuilder_analysis.rendering.jax.sersic import sersic_I


assert float(jax.__version__.split('.')[-1]) > 54
config.update("jax_enable_x64", True)

fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')

with tqdm(fitting_metadata.index, desc='Iterating over subjects') as bar0:
    for subject_id in bar0:
        fm = fitting_metadata.loc[subject_id]
        psf = fm['psf']
        target = np.asarray(fm['galaxy_data'].data)
        mask = np.asarray(fm['galaxy_data'].mask)
        sigma = np.asarray(fm['sigma_image'].data)

        agg_res = pd.read_pickle(
            f'output_files/aggregation_results/{subject_id}.pkl.gz'
        )

        op = fit.Optimizer(agg_res, fm)

        # Increase the disk radius by a correction factor of two, and limit
        # the maximum disk effective radius to be half the image size
        op.model['disk']['Re'] *= 2
        op.model['bulge']['scale'] /= 2
        op.model['bar']['scale'] /= 2
        op.lims['disk']['Re'] = [0.01, target.shape[0] / 2]
        op.reset_keys()

        op.set_keys([
            ('disk', 'Re'), ('disk', 'L'),
            ('bulge', 'scale'), ('bulge', 'frac'),
            ('bar', 'scale'), ('bar', 'frac')
        ])

        res = minimize(
            lambda p: onp.float64(op(p).block_until_ready()),
            onp.array(op.p0, dtype=onp.float64),
            method='L-BFGS-B',
            jac=lambda p: onp.array(op.jacobian(p).block_until_ready()),
            bounds=op.limits,
            options=dict(disp=True, maxiter=10000),
        )
        for i, (k0, k1) in enumerate(op.keys):
            op.model[k0][k1] = res['x'][i]

        # fit the entire model
        op.reset_keys()

        with tqdm(desc='Fitting', leave=False) as pbar:
            def callback(*args, **kwargs):
                pbar.update(1)
                pass
            # L-BFGS-B needs numpy float64 arrays, so we need to convert from JAX
            # DeviceArrays, which adds a slight overhead
            res = minimize(
                lambda p: onp.float64(op(p).block_until_ready()),
                onp.array(op.p0, dtype=onp.float64),
                method='L-BFGS-B',
                jac=lambda p: onp.array(op.jacobian(p).block_until_ready()),
                bounds=op.limits,
                callback=callback,
                options=dict(disp=True, maxiter=10000),
            )

        final_p = pd.Series(res['x'], index=pd.MultiIndex.from_tuples(op.keys))
        final_p.loc[('disk', 'I')] = sersic_I(final_p.disk.L, final_p.disk.Re, 1.0)
        if 'bulge' in final_p:
            bulge_l = final_p.disk.L * final_p.bulge.frac / (1 - final_p.bulge.frac)
            bulge_re = final_p.bulge.scale * final_p.disk.Re
            final_p.loc[('bulge', 'Re')] = bulge_re
            final_p.loc[('bulge', 'I')] = sersic_I(bulge_l, bulge_re, final_p.bulge.n)
            final_p.loc[('bulge', 'mux')] = final_p.loc[('centre', 'mux')]
            final_p.loc[('bulge', 'muy')] = final_p.loc[('centre', 'muy')]
        if 'bar' in final_p:
            bar_l = final_p.disk.L * final_p.bar.frac / (1 - final_p.bar.frac)
            bar_re = final_p.bar.scale * final_p.disk.Re
            final_p.loc[('bar', 'Re')] = bar_re
            final_p.loc[('bar', 'I')] = sersic_I(bar_l, bar_re, final_p.bar.n)
            final_p.loc[('bar', 'mux')] = final_p.loc[('centre', 'mux')]
            final_p.loc[('bulge', 'muy')] = final_p.loc[('centre', 'muy')]
        final_p = final_p.sort_index()

        final_gal = fit.create_model(
            res['x'], op.keys, op.n_spirals, op.model, op.psf, op.target, 5
        )
        chisq = np.sum(
            (((target - final_gal)/sigma)**2)[~mask]
        ) / (~mask).astype(int).sum()

        output = dict(
            res=res,
            final_params=final_p,
            chisq=chisq,
        )
        os.makedirs('output_files/tuning_results', exist_ok=True)
        pd.to_pickle(output, f'output_files/tuning_results/{subject_id}.pickle.gz')
