"""This script accepts a subject id (or corresponding index in
lib/subject-id-list.csv), and performs optimization on a given model for that
subject.
We reparametrize the model such that bulge and bar radius, luminosity and
position are in relation to that of the disk
(i.e. Re -> Re / Re_disk and I -> L / (L + L_disk)).
We hope this allows better control over limits, and reduces convergence to an
unphysical result.
"""
import os
from os.path import join
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
from tqdm import tqdm
import gzbuilder_analysis.parsing as parsing
from gzbuilder_analysis.rendering.sersic import _b
import gzbuilder_analysis.fitting as fitting
import argparse


INITIAL_N = 50
MAX_N = 500


loc = os.path.abspath(os.path.dirname(__file__))
lib = join(loc, 'lib')
sid_list = np.loadtxt(
    os.path.join(lib, 'subject-id-list.csv'),
    dtype='u8'
)
models = pd.read_pickle(join(lib, 'models.pickle'))
diff_data_df = pd.read_pickle(join(lib, 'fitting_metadata.pkl'))

parser = argparse.ArgumentParser(
    description=(
        'Fit Aggregate model and best individual'
        ' model for a galaxy builder subject'
    )
)
parser.add_argument('--index', '-i', metavar='N', default=-1, type=int,
                    help='Subject id index (from liv/subject-id-list.csv)')
parser.add_argument('--type', metavar='model_type', default='best_individual',
                    help='Which model to fit (best_individual or aggregate)')
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

diff_data = diff_data_df.loc[subject_id]
psf = diff_data['psf']
data = diff_data['galaxy_data']
sigma_image = diff_data['sigma_image']

print(f'Working on subject {subject_id}, model type {args.type}')
model_dict = models.loc[subject_id][args.type]

model_obj = fitting.Model(
    model_dict,
    data,
    psf=psf,
    sigma_image=sigma_image,
)


def sersic_ltot(comp):
    return (
        2 * np.pi * comp['I'] * comp['Re']**2 * comp['n']
        * np.exp(_b(comp['n'])) / _b(comp['n'])**(2 * comp['n'])
        * gamma(2 * comp['n'])
    )


def sersic_I(comp):
    return comp['L'] / (
        2 * np.pi * comp['Re']**2 * comp['n']
        * np.exp(_b(comp['n'])) / _b(comp['n'])**(2 * comp['n'])
        * gamma(2 * comp['n'])
    )


def get_new_params(p):
    # go from original param specification to new
    p_new = p.copy()
    disk_l = sersic_ltot(p['disk'])
    p_new[('disk', 'L')] = disk_l
    p_new.drop(('disk', 'I'), inplace=True)
    for c in ('bulge', 'bar'):
        try:
            comp_l = sersic_ltot(p[c])
            p_new[(c, 'Re')] = p[(c, 'Re')] / p[('disk', 'Re')]
            p_new[(c, 'mux')] = p[(c, 'mux')] - p[('disk', 'mux')]
            p_new[(c, 'muy')] = p[(c, 'muy')] - p[('disk', 'muy')]
            p_new[(c, 'L')] = sersic_ltot(p[c]) / (disk_l + comp_l)
            p_new.drop((c, 'I'), inplace=True)
        except KeyError:
            pass
    return p_new


def get_original_params(p):
    # go from new param specification to original
    p_new = p.copy()
    disk_I = sersic_I(p['disk'])
    p_new[('disk', 'I')] = disk_I
    p_new.drop(('disk', 'L'), inplace=True)
    for c in ('bulge', 'bar'):
        try:
            p_new[(c, 'Re')] = p[(c, 'Re')] * p[('disk', 'Re')]
            p_new[(c, 'mux')] = p[(c, 'mux')] + p[('disk', 'mux')]
            p_new[(c, 'muy')] = p[(c, 'muy')] + p[('disk', 'muy')]
            comp_l = p[('disk', 'L')] * p[(c, 'L')] / (1 - p[(c, 'L')])
            p_new[(c, 'L')] = comp_l
            p_new[(c, 'I')] = sersic_I(p_new[c])
            p_new.drop((c, 'L'), inplace=True)
        except KeyError:
            pass
    return p_new


all_p = get_new_params(model_obj.params.dropna())

lims_df = pd.DataFrame([], index=all_p.index, columns=('lower', 'upper'))

lims_df['lower'] = -np.inf
lims_df['upper'] = np.inf

lims_df.loc[('disk', 'L')] = (0, np.inf)
lims_df.loc[('disk', 'Re')] = (0, np.inf)
lims_df.loc[('disk', 'q')] = (0.2, 1/0.2)
lims_df.drop(('disk', 'n'), inplace=True)  # do not fit disk n
lims_df.drop(('disk', 'c'), inplace=True)  # do not fit disk c
if model_obj['bulge'] is not None:
    lims_df.loc[('bulge', 'L')] = (0, 1-0.001)
    lims_df.loc[('bulge', 'Re')] = (0, 1)
    lims_df.loc[('bulge', 'q')] = (0.5, 1/0.5)
    lims_df.loc[('bulge', 'n')] = (0.3, 5)
    lims_df.drop(('bulge', 'c'), inplace=True)  # do not fit bulge c
if model_obj['bar'] is not None:
    lims_df.loc[('bar', 'L')] = (0, 1-0.001)
    lims_df.loc[('bar', 'Re')] = (0, 2)
    lims_df.loc[('bar', 'q')] = (0.05, 2/3)
    lims_df.loc[('bar', 'n')] = (0.3, 5)
    lims_df.loc[('bar', 'c')] = (0.5, 6)
if len(model_obj['spiral']) > 0:
    for i in range(len(model_obj['spiral'])):
        lims_df.loc[(f'spiral{i}', 'I')] = (0, np.inf)
        lims_df.loc[(f'spiral{i}', 'spread')] = (0, np.inf)
        lims_df.loc[(f'spiral{i}', 'falloff')] = (0.01, np.inf)

lims_df.sort_index(inplace=True)


# Initially only fit luminosity and size
lims_df_initial = lims_df.copy()

lims_df_initial.drop(('disk', 'q'), inplace=True)

if model_obj['bulge'] is not None:
    lims_df_initial.drop(('bulge', 'q'), inplace=True)
    lims_df_initial.drop(('bulge', 'n'), inplace=True)
if model_obj['bar'] is not None:
    lims_df_initial.drop(('bar', 'q'), inplace=True)
    lims_df_initial.drop(('bar', 'n'), inplace=True)
    lims_df_initial.drop(('bar', 'c'), inplace=True)
if len(model_obj['spiral']) > 0:
    for i in range(len(model_obj['spiral'])):
        lims_df_initial.loc[(f'spiral{i}', 'I')] = (0, np.inf)
        lims_df_initial.loc[(f'spiral{i}', 'spread')] = (0, np.inf)
        lims_df_initial.loc[(f'spiral{i}', 'falloff')] = (0.01, np.inf)

lims_df_initial.sort_index(inplace=True)


# define function to minimize
def _func(p, model_obj, lims, all_params):
    p_ = pd.Series(p, index=lims.index).combine_first(all_params).reindex_like(all_params)
    r = model_obj.render(params=get_original_params(p_))
    if np.any(np.isnan(r)):
        print('NaN in render')
        print(get_original_params(p_).unstack())
        raise(ValueError)
    cq = fitting.chisq(r, model_obj.data, model_obj.sigma_image)
    if np.isnan(cq):
        return 1E5
    return cq


# Perform initial tuning
if args.progress:
    pbar = tqdm(leave=True)

    def update_bar(*args):
        pbar.update(1)
        pbar.set_description(f'chisq={_func(args[0], model_obj, lims_df_initial, all_p):.4f}')
else:
    def update_bar(*args):
        pass
res = minimize(
    _func,
    all_p.reindex_like(lims_df_initial),
    args=(model_obj, lims_df_initial, all_p),
    bounds=lims_df_initial.values,
    callback=update_bar,
    options=dict(maxiter=INITIAL_N, disp=(not args.progress)),
)
if args.progress:
    pbar.close()


tuned_params = get_original_params(
    pd.Series(res['x'], index=lims_df_initial.index).combine_first(all_p)
).sort_index()

model_obj2 = fitting.Model(
    model_obj.to_dict(tuned_params),
    data,
    psf=psf,
    sigma_image=sigma_image,
)

# Perform full optimization
all_p2 = get_new_params(model_obj2.params.dropna())
if args.progress:
    pbar = tqdm(leave=True)

    def update_bar(*args):
        pbar.update(1)
        pbar.set_description(f'chisq={_func(args[0], model_obj2, lims_df, all_p2):.4f}')
else:
    def update_bar(*args):
        pass
res2 = minimize(
    _func,
    all_p2.reindex_like(lims_df),
    args=(model_obj2, lims_df, all_p2),
    bounds=lims_df.values,
    callback=update_bar,
    options=dict(maxiter=MAX_N, disp=(not args.progress)),
)
if args.progress:
    pbar.close()

final_params = get_original_params(
    pd.Series(res2['x'], index=lims_df.index).combine_first(all_p2)
).sort_index()

final_model_obj = fitting.Model(
    model_obj.to_dict(final_params),
    data,
    psf=psf,
    sigma_image=sigma_image,
)

final_model_dict = final_model_obj.to_dict()

output_path = join(args.output, f'{args.type}/{subject_id}.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f'Writing tuned aggregate model to {output_path}')
with open(output_path, 'w') as f:
    f.write(parsing.make_json(final_model_dict))
