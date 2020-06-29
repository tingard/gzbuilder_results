import pickle
import numpy as np
import pandas as pd
from gzbuilder_analysis.parsing import to_pandas
import lib.galaxy_utilities as gu


def get_errors(subject_id):
    with open(f'lib/clusters/{subject_id}.pickle', 'rb') as f:
        clusters = pickle.load(f)
    comps = pd.Series([], name=subject_id)
    for comp in ('disk', 'bulge', 'bar'):
        comp_df = clusters[comp].apply(pd.Series)
        comps[comp] = comp_df.std()
    return comps.apply(pd.Series) \
        .stack() \
        .rename(subject_id) \
        .rename_axis(('component', 'parameter'))


sid_list = pd.read_csv('lib/subject-id-list.csv').values[:, 0]
models = pd.read_pickle('lib/models.pickle')
model_df = models['tuned_aggregate'].apply(to_pandas)
# print(get_errors(20902040))
sid_list = pd.read_csv('lib/subject-id-list.csv').values[:, 0]
sid_list = pd.Series(sid_list, index=sid_list)
multipliers = sid_list.apply(lambda i: gu.get_diff_data(i)['multiplier'])
errors = sid_list.apply(get_errors)

sloan_asec_per_pixel = 0.396

# correct for scales - convert from sloan pixels to arcseconds
scaled_Re_errors = errors.T.xs('Re', level=1, drop_level=False).T * sloan_asec_per_pixel
scaled_mux_errors = errors.T.xs('mux', level=1, drop_level=False).T * sloan_asec_per_pixel
scaled_muy_errors = errors.T.xs('muy', level=1, drop_level=False).T * sloan_asec_per_pixel
errors.update(scaled_Re_errors)
errors.update(scaled_mux_errors)
errors.update(scaled_muy_errors)

# undo the brightness scaling in the subject creation process
scaled_I_errors = (errors.T.xs('I', level=1, drop_level=False) * multipliers).T
errors.update(scaled_I_errors)

errors.to_csv('lib/errors.csv')

name_changes = {
    'Re': r'$r_e$',
    'roll': r'$\psi$ (radians)',
    'mux': r'$\mu_x$ (arcseconds)',
    'muy': r'$\mu_y$ (arcseconds)',
    'q': r'$b/a$',
    'I': r'$\Sigma_e$ (nmgy)',
    'n': r'$n$',
    'c': r'$c$',
}
drops = [
    ('disk', 'n'), ('disk', 'c'),
    ('bulge', 'c'),
]
pivot = errors.drop(columns=drops).describe().T
pivot['count'] = pivot['count'].astype(int)
print(
    pivot.round(2)
         .rename(name_changes, level=1, axis=0)
         .to_latex(escape=False)
)
