import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import lib.galaxy_utilities as gu
import gzbuilder_analysis.parsing as parsing
import gzbuilder_analysis.rendering as rendering
import gzbuilder_analysis.aggregation as aggregation
import gzbuilder_analysis.fitting as fitting
import argparse


def get_best_cls(subject_id, save=False):
    diff_data = fitting_metadata.loc[subject_id]
    psf = diff_data['psf']
    pixel_mask = np.array(diff_data['pixel_mask'])
    galaxy_data = np.array(diff_data['galaxy_data'])
    sigma_image = np.array(diff_data['sigma_image'])
    size_diff = diff_data['size_diff']

    classifications = gu.classifications.query(
        'subject_ids == {}'.format(subject_id)
    )
    annotations = classifications['annotations'].apply(json.loads)

    models = annotations.apply(
        parsing.parse_annotation,
        args=(galaxy_data.shape,),
        size_diff=size_diff,
        wcs_in=diff_data['montage_wcs'],
        wcs_out=diff_data['original_wcs'],
    )
    rendered = models.apply(
        rendering.calculate_model,
        image_size=galaxy_data.shape,
        psf=psf
    )
    losses = rendered.apply(
        fitting.loss,
        args=(galaxy_data,),
        pixel_mask=pixel_mask,
        sigma_image=sigma_image
    )
    if save:
        pd.concat(
            (
                models.rename('model'),
                rendered.rename('render'),
                losses.rename('chisq')
            ),
            axis=1
        ).to_pickle('lib/volunteer_models/{}.pickle'.format(subject_id))
    return losses.idxmin(), losses.min(), models.loc[losses.idxmin()]


def get_agg_model(subject_id):
    gal = gal_angle_df.loc[subject_id]
    angle = gal['angle']
    diff_data = fitting_metadata.loc[subject_id]
    size_diff = diff_data['size_diff']
    classifications = gu.classifications.query(
        'subject_ids == {}'.format(subject_id)
    )
    model, error, masks, arms = aggregation.make_model(
        classifications, gal, angle
    )
    scaled_model = parsing.scale_aggregate_model(
        model,
        size_diff=size_diff,
        image_size=np.array(diff_data['galaxy_data']).shape,
        wcs_in=diff_data['montage_wcs'],
        wcs_out=diff_data['original_wcs'],
    )
    scaled_errors = parsing.scale_model_errors(error, size_diff=size_diff)
    return scaled_model, scaled_errors, masks, arms


if __name__ == '__main__':
    loc = os.path.abspath(os.path.dirname(__file__))
    DEFAULT_AGG_LOCATION = os.path.join(
        loc, 'lib/aggregation_results.pickle'
    )
    DEFAULT_BI_LOCATION = os.path.join(
        loc, 'lib/best_individual.pickle'
    )
    sid_list = np.loadtxt(
        os.path.join(loc, 'lib/subject-id-list.csv'),
        dtype='u8'
    )
    gal_angle_df = pd.read_csv(
        os.path.join(loc, 'lib/gal-metadata.csv'),
        index_col=0
    )
    fitting_metadata = pd.read_pickle(
        os.path.join(loc, 'lib/fitting_metadata.pkl')
    )

    parser = argparse.ArgumentParser(
        description=(
            'Find Aggregate model and best individual'
            ' model for galaxy builder subjects'
        )
    )
    parser.add_argument(
        '--aggregate',
        metavar='/path/to/file.dat',
        default=DEFAULT_AGG_LOCATION,
        type=str,
        help='Where to output aggregate models'
    )
    parser.add_argument(
        '--best-individual',
        metavar='/path/to/file.dat',
        default=DEFAULT_BI_LOCATION,
        type=str,
        help='Where to output best individual models'
    )
    parser.add_argument('-S', '--save-each', action='store_true',
                        help='Save model information for each galaxy')
    args = parser.parse_args()

    # best_indiv = pd.Series([])
    # with tqdm(sid_list, desc='Finding best individual', leave=True) as bar:
    #     for subject_id in bar:
    #         try:
    #             best_indiv.loc[subject_id] = get_best_cls(
    #                 subject_id, save=args.save_each
    #             )
    #         except ValueError:
    #             print(subject_id)
    #             break
    # best_indiv = best_indiv.apply(pd.Series)
    # best_indiv.columns = ('cls_index', 'chisq', 'Model')
    # best_indiv.to_pickle(args.best_individual)

    tqdm.pandas(desc='Calculating aggregate', leave=True)
    aggregation_results = pd.Series(
        sid_list, index=sid_list
    ).progress_apply(get_agg_model)\
        .apply(pd.Series)
    aggregation_results.columns = ('Model', 'Errors', 'Masks', 'Arms')
    aggregation_results.to_pickle(args.aggregate)
