import os
import json
from time import sleep
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, basinhopping
from scipy.special import gamma
from tqdm import tqdm
try:
    import cupy as _p
    from cupy import asnumpy
    from cupyx.scipy.ndimage.filters import convolve as cuda_conv
    from gzbuilder_analysis.rendering.cuda.sersic import sersic2d

    def convolve(render, psf, **kwargs):
        return cuda_conv(render, psf, mode='mirror')
except ModuleNotFoundError:
    _p = np
    asnumpy = np.asarray
    from scipy.signal import convolve2d
    from gzbuilder_analysis.rendering.sersic import sersic2d

    def convolve(render, psf, **kwargs):
        return convolve2d(render, psf, mode='same', boundary='symm')

from gzbuilder_analysis.rendering.sersic import _b

warnings.simplefilter('ignore', RuntimeWarning)


def sersic_ltot(I, Re, n, gamma=gamma):
    return (
        2 * np.pi * I * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * gamma(2 * n)
    )


def sersic_I(L, Re, n, gamma=gamma):
    return L / (
        2 * np.pi * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * gamma(2 * n)
    )


def gen_grid(shape, oversample_n):
    x = _p.linspace(
        0.5 / oversample_n - 0.5,
        shape[1] - 0.5 - 0.5 / oversample_n,
        shape[1] * oversample_n
    )
    y = _p.linspace(
        0.5 / oversample_n - 0.5,
        shape[0] - 0.5 - 0.5 / oversample_n,
        shape[0] * oversample_n
    )
    return _p.meshgrid(x, y)


def bulge_disk_render(
    cx, cy,
    mux=0, muy=0, Re=1, q=1, I=1, roll=0,
    bulge_dx=0, bulge_dy=0, bulge_scale=0.1, bulge_q=1, bulge_roll=0,
    bulge_frac=0.1, bulge_n=1
):
    if I == 0 or Re == 0:
        disk = _p.zeros(cx.shape)
        bulge = _p.zeros(cx.shape)
    else:
        #      sersic2d(x,  y,  mux, muy, roll, Re, q, c, I, n)
        disk = sersic2d(cx, cy, mux, muy, roll, Re, q, 2, I, 1)
        if bulge_scale == 0 or bulge_frac == 0:
            bulge = _p.zeros(cx.shape)
        else:
            disk_l = sersic_ltot(I, Re, 1)
            comp_l = disk_l * bulge_frac / (1 - bulge_frac)
            bulge_I = sersic_I(comp_l, bulge_scale * Re, bulge_n)
            bulge = sersic2d(
                cx, cy,
                mux + bulge_dx, muy + bulge_dy,
                bulge_roll, bulge_scale * Re,
                bulge_q, 2, bulge_I, bulge_n
            )
    return (disk + bulge)


def downsample(render, oversample_n, size):
    return render.reshape(
        size[0], oversample_n, size[1], oversample_n
    ).mean(3).mean(1)


fm = pd.read_pickle('lib/fitting_metadata.pkl')

lims_df = pd.DataFrame(dict(
    mux=[-np.inf, np.inf],
    muy=[-np.inf, np.inf],
    Re=[0, np.inf],
    q=[0.2, 1],
    I=[0, np.inf],
    roll=[-np.inf, np.inf],
    bulge_dx=[-np.inf, np.inf],
    bulge_dy=[-np.inf, np.inf],
    bulge_scale=[0, 1],
    bulge_q=[0.4, 1],
    bulge_roll=[-np.inf, np.inf],
    bulge_frac=[0, 0.95],
    bulge_n=[0.6, 8],
), index=('lower', 'upper')).T


class BasinhoppingBounds(object):
    def __init__(self, lims):
        self.lims = lims

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = np.all(x <= self.lims['upper'].values)
        tmin = np.all(x >= self.lims['lower'].values)
        return tmax and tmin


lims = BasinhoppingBounds(lims_df)

with tqdm(fm.index, desc='Fitting subjects') as bar:
    for subject_id in bar:
        bar.set_description(f'Fitting subjects (minimize)    ')
        # subject_id = 21686598
        if not os.path.isfile(f'2comp_fits_nb4/minima/{subject_id}.csv'):
            target = fm.loc[subject_id]['galaxy_data']
            cp_mask = _p.asarray(target.mask)
            cp_target = _p.asarray(target.data)
            cp_psf = _p.asarray(fm['psf'][subject_id])
            cp_sigma = _p.asarray(fm['sigma_image'][subject_id].data)

            p0 = pd.Series(dict(
                mux=target.shape[1] / 2,
                muy=target.shape[1] / 2,
                Re=20,
                q=1,
                I=0.8,
                roll=0,
                # bulge_dx=0,
                # bulge_dy=0,
                bulge_scale=0.2,
                bulge_q=1,
                bulge_roll=0,
                bulge_frac=0.2,
                # bulge_n=2,
            ))

            oversample_n = 5
            cx, cy = gen_grid(target.shape, oversample_n)
            ndof = len(target.compressed())

            def _f(p):
                kw = {k: v for k, v in zip(p0.index, p)}
                kw.setdefault('bulge_n', 4)
                render = bulge_disk_render(cx, cy, **kw)
                downsampled_render = downsample(render, oversample_n, size=target.shape)
                psf_conv_render = convolve(downsampled_render, cp_psf)
                diff = psf_conv_render[~cp_mask] - cp_target[~cp_mask]
                chisq = asnumpy(_p.sum((diff / cp_sigma[~cp_mask])**2) / ndof)
                return chisq

            gradient_descent_res = minimize(
                _f,
                p0,
                bounds=lims_df.reindex(p0.index).values,
            )
            p_gd = pd.Series(gradient_descent_res['x'], index=p0.index)

            bar.set_description(f'Fitting subjects (basinhopping)')

            minima = pd.DataFrame(columns=(*p0.index, 'chisq', 'accepted'))

            def save_minima(x, f, accepted):
                minimum = pd.Series(x, index=p0.index)
                minimum['chisq'] = f
                minimum['accepted'] = accepted
                minima.loc[len(minima)] = minimum

            lims = BasinhoppingBounds(lims_df.reindex(p0.index))
            minimizer_kwargs = dict(
                method='L-BFGS-B',
                bounds=lims_df.reindex(p0.index).values
            )

            basinhopping_res = basinhopping(
                _f, p_gd.reindex_like(p0),
                minimizer_kwargs=minimizer_kwargs,
                niter=10, accept_test=lims, T=1, callback=save_minima
            )
            p_gd['chisq'] = gradient_descent_res['fun']
            p_global = pd.Series(basinhopping_res['x'], index=p0.index)
            p_global['chisq'] = basinhopping_res['fun']

            comparison_df = pd.concat((
                p_gd.rename('Gradient Descent'),
                p_global.rename('Basinhopping')
            ), axis=1).T

            os.makedirs('2comp_fits_nb4/best_results', exist_ok=True)
            comparison_df.to_csv(f'2comp_fits_nb4/best_results/{subject_id}.csv')

            # save the local minima
            os.makedirs('2comp_fits_nb4/minima', exist_ok=True)
            minima.sort_values(by='chisq')\
                .to_csv(f'2comp_fits_nb4/minima/{subject_id}.csv')
        else:
            sleep(0.05)
