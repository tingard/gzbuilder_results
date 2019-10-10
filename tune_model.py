import os
import sys
import time
import json
import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
import gzbuilder_analysis.parsing as parsing
import gzbuilder_analysis.fitting as fitting
from gzbuilder_analysis.config import SLIDER_FITTING_TEMPLATE
# ## Optimization
#
# We perform fitting on the models created earlier. Parameters fit are
#
# |         | $i_0$ | $R_e$ | $e$ | $n$ | $c$ | $W$ | $F$ |
# |:--------|:------|:------|:----|:----|:----|:----|:----|
# | Disk    | x     | x     | x   |     |     |     |     |
# | Bulge   | x     | x     | x   | x   |     |     |     |
# | Bar     | x     | x     | x   | x   | x   |     |     |
# | Spirals | x     |       |     |     |     | x   | x   |
#
# Where $W$ is the spiral spread and $F$ is spiral falloff (others are parameters of a boxy Sérsic profile).
#
# For the aggregate model we first fit without spiral arms, then introduce low-brightness arms and fit fully.
start_time = time.time()
loc = os.path.abspath(os.path.dirname(__file__))
lib = os.path.join(loc, 'lib')
sid_list = np.loadtxt(
    os.path.join(lib, 'subject-id-list.csv'),
    dtype='u8'
)
fitting_metadata = pd.read_pickle(
    os.path.join(lib, 'fitting_metadata.pkl')
)
AGGREGATE_LOCATION = os.path.join(lib, 'aggregation_results.pickle')
BEST_INDIVIDUAL_LOCATION = os.path.join(lib, 'best_individual.pickle')

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
                    default='tuned_models',
                    help='Whether to use a progress bar')
parser.add_argument('--progress', '-P', action='store_true',
                    help='Whether to use a progress bar')


args = parser.parse_args()
if args.index == -1 and args.subject == -1:
    print('Invalid arguments')
    parser.print_help()
    sys.exit(0)
elif args.subject >= 0:
    sid = args.subject
else:
    sid = sid_list[args.index]

print('Working on', sid)
AGGREGATE_LOCATION = os.path.join(lib, 'aggregation_results.pickle')
BEST_INDIVIDUAL_LOCATION = os.path.join(lib, 'best_individual.pickle')

aggregation_results = pd.read_pickle(AGGREGATE_LOCATION)
best_indiv = pd.read_pickle(BEST_INDIVIDUAL_LOCATION)


def make_model(subject_id, m):
    diff_data = fitting_metadata.loc[subject_id]
    psf = diff_data['psf']
    pixel_mask = np.array(diff_data['pixel_mask'])[::-1]
    sigma_image = np.array(diff_data['sigma_image'])[::-1]
    galaxy_data = np.array(diff_data['galaxy_data'])[::-1]
    return fitting.Model(m, galaxy_data, psf=psf, pixel_mask=pixel_mask, sigma_image=sigma_image)


def reset_spiral_intensity(s):
    points, params = s
    new_params = deepcopy(params)
    new_params['i0'] = 0.01
    return [points, new_params]


# ensure the output dir exists
agg_output_folder = os.path.join(args.output, 'agg')
bi_output_folder = os.path.join(args.output, 'bi')
if not os.path.isdir(args.output):
    os.makedirs(args.output)
if not os.path.isdir(agg_output_folder):
    os.makedirs(agg_output_folder)
if not os.path.isdir(bi_output_folder):
    os.makedirs(bi_output_folder)


# Optimization of Aggregate model
print('Optimizing aggregate model')
agg_model = make_model(sid, aggregation_results.loc[sid].Model)
original_agg_loss = fitting.loss(
    agg_model.render(),
    agg_model.data,
    pixel_mask=agg_model.pixel_mask,
    sigma_image=agg_model.sigma_image,
)
# reset spiral intensity to 0.01
agg_model = agg_model.copy_with_new_model({
    **deepcopy(agg_model._model),
    'spiral': np.array([
        reset_spiral_intensity(s) for s in agg_model['spiral']
    ]),
})
print('\tTuning slider values')
# fit only the slider values
tuned_agg_model_dict, res = fitting.fit(
    agg_model,
    progress=args.progress,
    template=SLIDER_FITTING_TEMPLATE
)

if res['success']:
    print('\tTuning all parameters')
    tuned_slider_agg_model = agg_model.copy_with_new_model(tuned_agg_model_dict)
    tuned_slider_agg_loss = fitting.loss(
        tuned_slider_agg_model.render(),
        tuned_slider_agg_model.data,
        pixel_mask=tuned_slider_agg_model.pixel_mask,
        sigma_image=tuned_slider_agg_model.sigma_image,
    )
    # fit all parameters
    tuned_agg_model_dict, res = fitting.fit(
        tuned_slider_agg_model,
        progress=args.progress
    )
    if res['success']:
        tuned_agg_model = agg_model.copy_with_new_model(tuned_agg_model_dict)
        final_agg_loss = fitting.loss(
            tuned_agg_model.render(),
            tuned_agg_model.data,
            pixel_mask=tuned_agg_model.pixel_mask,
            sigma_image=tuned_agg_model.sigma_image,
        )
        print('Aggregate Loss changed from {:.4e} to {:.4e} to {:.4e}'.format(
            original_agg_loss, tuned_slider_agg_loss, final_agg_loss
        ))
        outfile = os.path.join(
            agg_output_folder,
            '{}.json'.format(sid)
        )
        with open(outfile, 'w') as f:
            json_model = parsing.make_json(tuned_agg_model_dict)
            json.dump(json_model, f)
    else:
        print('Fitting failure on 2nd fit:', res['message'])
else:
    print('Fitting failure on 1st fit:', res['message'])


# Optimization of Best individual model
print('\nOptimizing best individual model')
bi_model = make_model(sid, best_indiv.loc[sid].Model)
original_bi_loss = fitting.loss(
    bi_model.render(),
    bi_model.data,
    pixel_mask=bi_model.pixel_mask,
    sigma_image=bi_model.sigma_image,
)
tuned_bi_model_dict, res = fitting.fit(bi_model, progress=args.progress)
if res['success']:
    tuned_bi_model = bi_model.copy_with_new_model(tuned_bi_model_dict)
    final_bi_loss = fitting.loss(
        tuned_bi_model.render(),
        tuned_bi_model.data,
        pixel_mask=tuned_bi_model.pixel_mask,
        sigma_image=tuned_bi_model.sigma_image,
    )
    print('BI Loss changed from {:.4e} to {:.4e}'.format(
        original_bi_loss, final_bi_loss
    ))
    outfile = os.path.join(
        bi_output_folder,
        '{}.json'.format(sid)
    )
    with open(outfile, 'w') as f:
        json_model = parsing.make_json(tuned_bi_model_dict)
        json.dump(json_model, f)
else:
    print('Fitting failure:', res['message'])

print('Total runtime: {:.2f}s'.format(time.time() - start_time))
