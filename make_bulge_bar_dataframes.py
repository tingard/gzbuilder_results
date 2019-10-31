import os
import numpy as np
import pandas as pd
import json
import lib.galaxy_utilities as gu
from astropy.io import fits
from tqdm import tqdm


aggregated_models = pd.read_pickle('lib/models.pickle')['tuned_aggregate']


def get_pnonebulge(gal):
    none = gal['t05_bulge_prominence_a10_no_bulge_debiased']
    # just = gal['t05_bulge_prominence_a11_just_noticeable_debiased'],
    # obvious = gal['t05_bulge_prominence_a12_obvious_debiased'],
    # dominant = gal['t05_bulge_prominence_a13_dominant_debiased'],
    return none


def get_pbulge(gal):
    none = gal['t05_bulge_prominence_a10_no_bulge_debiased']
    just = gal['t05_bulge_prominence_a11_just_noticeable_debiased']
    obvious = gal['t05_bulge_prominence_a12_obvious_debiased']
    dominant = gal['t05_bulge_prominence_a13_dominant_debiased']
    return none + just < obvious + dominant


def get_pbar(gal):
    n = gal['t03_bar_a06_bar_debiased'] + gal['t03_bar_a07_no_bar_debiased']
    return gal['t03_bar_a06_bar_debiased'] / n


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
    bulge_fractions = pd.DataFrame(
        [],
        columns=('GZB fraction', 'GZ2 bulge dominated', 'GZ2 no bulge')
    )
    bar_fractions = pd.DataFrame(
        [],
        columns=('GZB fraction', 'GZ2 bar fraction'),
    )
    bar_lengths = pd.Series([], name='GZB bar length')
    with tqdm(sid_list) as bar:
        for subject_id in bar:
            cls_for_s = gu.classifications.query(
                '(subject_ids == {}) & (workflow_version == 61.107)'.format(
                    subject_id
                )
            )
            ann_for_s = cls_for_s['annotations'].apply(json.loads)
            metadata = gu.metadata.loc[int(subject_id)]
            # gal, angle = gu.get_galaxy_and_angle(subject_id)
            try:
                gz2_gal = NSA_GZ[1].data[
                    NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
                ][0]
            except IndexError:
                print('Could not find object for id {}'.format(subject_id))
            # how frequently do people draw bulges?
            bulge_fractions.loc[subject_id] = {
                'GZB fraction': ann_for_s.apply(
                    lambda v: has_comp(v, comp=1)
                ).sum() / len(ann_for_s),
                'GZ2 bulge dominated': get_pbulge(gz2_gal),
                'GZ2 no bulge': get_pnonebulge(gz2_gal)
            }
            bar_fractions.loc[subject_id] = {
                'GZB fraction': ann_for_s.apply(
                    lambda v: has_comp(v, comp=2)
                ).sum() / len(ann_for_s),
                'GZ2 bar fraction': get_pbar(gz2_gal),
            }
            bar = aggregated_models.loc[subject_id]['bar']
            bar_length = bar['Re'] if bar is not None else np.nan
            bar_lengths.loc[subject_id] = bar_length


    bar_fractions['Strongly barred'] = bar_fractions['GZ2 bar fraction'] > 0.5
    bar_fractions['No bar'] = bar_fractions['GZ2 bar fraction'] < 0.2
    bar_fractions.to_pickle(os.path.join(loc, 'lib/bar_fractions.pkl'))
    bar_lengths = pd.concat((
        bar_lengths,
        bar_fractions[['GZ2 bar fraction', 'GZB fraction']],
    ), axis=1)
    bar_lengths.to_pickle(os.path.join(loc, 'lib/bar_lengths.pkl'))
