import os
import sys
from time import time
import numpy as np
import pandas as pd
import gzbuilder_analysis.parsing as pg
try:
    from gzbuilder_analysis.rendering.cuda import calculate_model
except ModuleNotFoundError:
    from gzbuilder_analysis.rendering import calculate_model
import gzbuilder_analysis.fitting as fg
import gzbuilder_analysis.config as cfg
from gzbuilder_analysis.fitting import Model, fit_model
from PIL import Image
import argparse

loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')
sid_list = np.loadtxt(
    os.path.join(lib, 'subject-id-list.csv'),
    dtype='u8'
)

parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--index', '-i', metavar='N', default=-1, type=int,
                    help='Subject id index (from liv/subject-id-list.csv)')
parser.add_argument('--subject', '-s', metavar='N', default=-1, type=int,
                    help='Subject id')
parser.add_argument('--output', '-O', metavar='/path/to/output',
                    default='output_files/tuned_models',
                    help='Where to save the output tuned model')
parser.add_argument('--progress', '-P', action='store_true',
                    help='Whether to use a progress bar')

args = parser.parse_args()
if args.index == -1 and args.subject == -1:
    print('Invalid arguments')
    parser.print_help()
    sys.exit(0)
elif args.subject >= 0:
    subject_id = args.subject
else:
    subject_id = sid_list[args.index]

print(f'Working on {subject_id}')

im = np.array(Image.open('lib/subject_data/{}/image.png'.format(subject_id)))[::-1]
classifications = pd.read_csv('lib/galaxy-builder-classifications.csv', index_col=0)
fm = pd.read_pickle('lib/fitting_metadata.pkl').loc[subject_id]
data = fm['galaxy_data']
sigma = fm['sigma_image']
psf = fm['psf']
gal = pd.read_csv('lib/gal-metadata.csv', index_col=0).loc[subject_id]
c = classifications.query('subject_ids == {}'.format(subject_id))

zoo_models = c.apply(
    pg.parse_classification,
    axis=1,
    # parse_classification kwargs
    image_size=np.array(im.shape),
    size_diff=im.shape[0] / data.shape[0],
)

scaled_models = zoo_models.apply(
    pg.scale_model,
    args=(fm['size_diff'],),
)

models = scaled_models.apply(
    pg.reproject_model,
    wcs_in=fm['montage_wcs'], wcs_out=fm['original_wcs']
)

print('Rendering models')
t0 = time()
rendered = models.apply(
    calculate_model,
    image_size=data.shape,
    psf=psf,
    oversample_n=3
)
print('\tElapsed time: {:.2f}'.format(time() - t0))
print('Calucating losses')
t0 = time()
chisq = rendered.apply(fg.chisq, args=(data, sigma))
print('\tElapsed time: {:.2f}'.format(time() - t0))

# save the results
models_output = os.path.join(lib, 'volunteer_models')
os.makedirs(models_output, exist_ok=True)
pd.concat(
    (
        models.rename('model'),
        chisq.rename('chisq'),
        rendered.rename('rendered')
    ),
    axis=1
).to_pickle(
    os.path.join(models_output, f'{subject_id}.pickle')
)


# now to tune the model, this can take some time
print('Tuning model')
idx_best = chisq.idxmin()
best_model = Model(
    models.loc[idx_best],
    data,
    psf=psf,
    sigma_image=sigma
)
# tune the brightness to provide a good starting point
params_to_fit = {
    'disk':   ('I', 'Re'),
    'bulge':  ('I', 'Re'),
    'bar':    ('I', 'Re'),
    'spiral': ('I', 'spread', 'falloff'),
}
res, partially_tuned_model_dict = fg.fit_model(
    best_model,
    params=params_to_fit,
    options=dict(maxiter=50)
)

# allow roll and position to vary
params_to_fit2 = {
    'disk':   ('mux', 'muy', 'roll', 'I', 'Re', 'q'),
    'bulge':  ('mux', 'muy', 'roll', 'I', 'Re', 'q', 'n'),
    'bar':    ('mux', 'muy', 'roll', 'I', 'Re', 'q', 'n', 'c'),
    'spiral': ('I', 'spread', 'falloff'),
}
res, tuned_model_dict = fg.fit_model(
    best_model,
    params=params_to_fit2,
    options=dict(maxiter=500)
)

os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, f'bi/{subject_id}.json'), 'w') as f:
    f.write(pg.make_json(tuned_model))
