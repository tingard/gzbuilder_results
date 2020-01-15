import os
import numpy as np
import pandas as pd
import json
import lib.galaxy_utilities as gu
from astropy.io import fits
from tqdm import tqdm


aggregated_models = pd.read_pickle('lib/models.pickle')['tuned_aggregate']


def get_n_arms(gal):
    keys = (
        't11_arms_number_a31_1_debiased',
        't11_arms_number_a32_2_debiased',
        't11_arms_number_a33_3_debiased',
        't11_arms_number_a34_4_debiased',
        't11_arms_number_a36_more_than_4_debiased',
    )
    return sum((i + 1) * gal[k] for i, k in enumerate(keys))


def get_winding_score(gal):
    keys = (
        't10_arms_winding_a28_tight_debiased',
        't10_arms_winding_a29_medium_debiased',
        't10_arms_winding_a30_loose_debiased',
    )
    return sum((i + 1) * gal[k] for i, k in enumerate(keys))


def get_pitch_angle(gal):
    m = get_n_arms(gal)
    w = get_winding_score(gal)
    return 6.37 * w + 1.3 * m + 4.34


def has_comp(annotation, comp=0):
    try:
        drawn_shapes = annotation[comp]['value'][0]['value']
        return len(drawn_shapes) > 0
    except (IndexError, KeyError):
        return False


if __name__ == '__main__':
    loc = os.path.abspath(os.path.dirname(__file__))
    # open the GZ2 catalogue
    NSA_GZ = fits.open(os.path.join(loc, '../source_files/NSA_GalaxyZoo.fits'))
    sid_list_loc = os.path.join(loc, 'lib/subject-id-list.csv')
    sid_list = pd.read_csv(sid_list_loc).values[:, 0]
    gz2_quants = pd.DataFrame([], columns=('hart_pa', 'winding', 'n_arms'))
    bad = []
    with tqdm(sid_list) as progress_bar:
        for subject_id in progress_bar:
            cls_for_s = gu.classifications.query(
                '(subject_ids == {}) & (workflow_version == 61.107)'.format(
                    subject_id
                )
            )
            ann_for_s = cls_for_s['annotations'].apply(json.loads)
            metadata = gu.metadata.loc[int(subject_id)]
            try:
                gz2_gal = NSA_GZ[1].data[
                    NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
                ][0]
                gz2_quants.loc[subject_id] = dict(
                    hart_pa=get_pitch_angle(gz2_gal),
                    winding=get_winding_score(gz2_gal),
                    n_arms=get_n_arms(gz2_gal),
                )
            except IndexError:
                bad.append(subject_id)
                gz2_quants.loc[subject_id] = (np.nan, np.nan, np.nan)

    gz2_quants.to_csv(os.path.join(loc, 'lib/gz2_spiral_data.csv'))
    print(f'Could not find objects for id {bad}')
