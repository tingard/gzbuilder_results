import pandas as pd
import os
import sys
import pickle
from time import time
import numpy as np
import pandas as pd
import gzbuilder_analysis.parsing as pg
import gzbuilder_analysis.aggregation as ag
import gzbuilder_analysis.config as cfg
import gzbuilder_analysis.fitting as fg
# from shapely.affinity import scale
# from descartes import PolygonPatch
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PIL import Image
from lib.galaxy_utilities import get_diff_data

loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')
sid_list = np.loadtxt(
    os.path.join(lib, 'subject-id-list.csv'),
    dtype='u8'
)

subject_id = 20902040

fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')
models = pd.read_pickle('lib/models.pickle').loc[subject_id]

agg_model_dict = models['aggregate']

fm = fitting_metadata.loc[subject_id]
data = fm['galaxy_data']
sigma_image = fm['sigma_image']
psf = fm['psf']
multiplier = get_diff_data(subject_id)['multiplier']

i = 0

def fit_model(model_obj, params=cfg.FIT_PARAMS, progress=True, **kwargs):
    os.makedirs('animations/fit', exist_ok=True)

    tuples = [(k, v) for k in params.keys() for v in params[k] if k is not 'spiral']
    tuples += [
        (f'spiral{i}', v)
        for i in range(len(model_obj.spiral_distances))
        for v in params['spiral']
    ]
    p0 = model_obj.params[tuples].dropna()
    if len(p0) == 0:
        print('No parameters to optimize')
        return {}, model_obj.to_dict()
    bounds = [cfg.PARAM_BOUNDS[param] for param in p0.index.levels[1][p0.index.codes[1]]]

    def _func(p):
        new_params = pd.Series(p, index=p0.index)
        r = model_obj.render(params=new_params)
        cq = fg.chisq(r, model_obj.data, model_obj.sigma_image)
        if np.isnan(cq):
            return 1E5
        return cq

    print(f'Optimizing {len(p0)} parameters')
    print(f'Original chisq: {_func(p0.values):.4f}')
    print()

    def render_frame(*args):
        global i
        new_params = pd.Series(args[0], index=p0.index)
        r = model_obj.render(params=new_params)
        plt.figure(figsize=(8, 6), dpi=150)
        diff = (data - r)*multiplier
        l = 0.45
        plt.imshow(diff, vmin=-l, vmax=l, cmap='seismic')
        plt.axis('off')
        plt.title(r'$\chi_\nu^2='+f'{_func(args[0]):.4f}$')
        c = plt.colorbar(shrink=0.95)
        # c.ax.set_yticklabels([f'{str(i):<3s}' for i in c.ax.get_yticks()])
        plt.savefig(f'animations/fit/{i:03d}.png', bbox_inches='tight')
        plt.close()
        i += 1

    with tqdm(desc='Fitting model', leave=True) as pbar:
        pbar.set_description(f'chisq={_func(p0):.4f}')

        def update_bar(*args):
            pbar.update(1)
            pbar.set_description(f'chisq={_func(args[0]):.4f}')
            render_frame(*args)

        res = minimize(_func, p0, bounds=bounds, callback=update_bar, **kwargs)
    print(f'Final chisq: {res["fun"]:.4f}')
    # the fitting process allows parameters to vary outside conventional bounds
    # (roll > 2*pi, axis ratio > 1 etc...). We fix this before exiting
    model_obj.sanitize()

    # obtain a model dict to return:
    final_params = model_obj.params.copy()
    final_params[p0.index] = res['x']
    # we use the model object rather than `from_pandas` in order to recover the
    # original spiral arm points, which would otherwise have been lost
    tuned_model = model_obj.to_dict(params=final_params)
    return res, tuned_model



# now to tune the model, this can take some time
agg_model = fg.Model(
    agg_model_dict,
    data,
    psf=psf,
    sigma_image=sigma_image
)
# tune the brightness to provide a good starting point
params_to_fit = {
    'disk':   ('I', 'Re'),
    'bulge':  ('I', 'Re'),
    'bar':    ('I', 'Re'),
    'spiral': ('I', 'spread', 'falloff'),
}
res, partially_tuned_model_dict = fit_model(
    agg_model,
    params=params_to_fit,
    options=dict(maxiter=50),
)

# allow roll and position to vary
params_to_fit2 = {
    'disk':   ('mux', 'muy', 'roll', 'I', 'Re', 'q'),
    'bulge':  ('mux', 'muy', 'roll', 'I', 'Re', 'q', 'n'),
    'bar':    ('mux', 'muy', 'roll', 'I', 'Re', 'q', 'n', 'c'),
    'spiral': ('I', 'spread', 'falloff'),
}
res, tuned_model_dict = fit_model(
    agg_model,
    params=params_to_fit2,
    options=dict(maxiter=500)
)
