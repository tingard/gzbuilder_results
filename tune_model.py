import os
import sys
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


sid_list = np.loadtxt('lib/subject-id-list.csv', dtype='u8')
diff_data_df = pd.read_pickle('lib/diff-data.pkl')


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
AGGREGATE_LOCATION = 'lib/aggregation_results.pickle'
BEST_INDIVIDUAL_LOCATION = 'lib/best_individual.pickle'
FITTED_MODEL_LOCATION = 'lib/fitted_models.pickle'

aggregation_results = pd.read_pickle(AGGREGATE_LOCATION)
best_indiv = pd.read_pickle(BEST_INDIVIDUAL_LOCATION)


def make_model(subject_id, m):
    try:
        diff_data = diff_data_df.loc[subject_id]
        psf = diff_data['psf']
        pixel_mask = 1 - np.array(diff_data['mask'])[::-1]
        galaxy_data = np.array(diff_data['imageData'])[::-1]
        return fitting.Model(m, galaxy_data, psf=psf, pixel_mask=pixel_mask)
    except ZeroDivisionError:
        return 1E7


def reset_spiral_intensity(s):
    points, params = s
    new_params = deepcopy(params)
    new_params['i0'] = 0.001
    return [points, new_params]


# ensure the output dir exists
if not os.path.isdir(args.output):
    os.makedirs(args.output)

# Optimization of Aggregate model
# create a model object
print('Optimizing aggregate model')
model = make_model(sid, aggregation_results.loc[sid].Model)
original_loss = fitting.loss(model.render(), model.data, model.pixel_mask)
# reset spiral intensity to 0.01
agg_model = model.copy_with_new_model({
    **deepcopy(model._model),
    'spiral': np.array([
        reset_spiral_intensity(s) for s in deepcopy(model['spiral'])
    ]),
})
# fit only the slider values
new_model, res = fitting.fit(agg_model, progress=args.progress,
                             template=SLIDER_FITTING_TEMPLATE)

if res['success']:
    new_model_ = model.copy_with_new_model(new_model)
    final_loss = fitting.loss(new_model_.render(), new_model_.data,
                              new_model_.pixel_mask)
    print('Aggregate Loss changed from {:.4e} to {:.4e}'.format(
        original_loss, final_loss
    ))
    outfile = os.path.join(
        args.output,
        'fitted_agg_models',
        '{}.json'.format(sid)
    )
    with open(outfile, 'w') as f:
        json_model = parsing.make_json(new_model)
        json.dump(json_model, f)
else:
    print('Fitting failure:', res['message'])

# Optimization of Best individual model
print('Optimizing best individual model')
model = make_model(sid, best_indiv.loc[sid].Model)
original_loss = fitting.loss(model.render(), model.data, model.pixel_mask)
new_model, res = fitting.fit(model, progress=args.progress)
if res['success']:
    new_model_ = model.copy_with_new_model(new_model)
    final_loss = fitting.loss(new_model_.render(), new_model_.data,
                              new_model_.pixel_mask)
    print('BI Loss changed from {:.4e} to {:.4e}'.format(
        original_loss, final_loss
    ))
    outfile = os.path.join(
        args.output,
        'fitted_agg_models',
        '{}.json'.format(sid)
    )
    with open('fitted_bi_models/{}.json'.format(sid), 'w') as f:
        json_model = parsing.make_json(new_model)
        json.dump(json_model, f)
else:
    print('Fitting failure:', res['message'])
