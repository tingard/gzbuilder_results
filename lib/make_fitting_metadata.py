import os
import numpy as np
try:
    import modin.pandas as pd
except ImportError:
    import pandas as pd
import galaxy_utilities as gu
from tqdm import tqdm
import make_cutouts as mkct
from astropy.wcs import WCS, FITSFixedWarning
import warnings

warnings.simplefilter('ignore', FITSFixedWarning)
# we'll be dividing by NaN a lot (regions of zero pixel coverage)
warnings.simplefilter('ignore', RuntimeWarning)

loc = os.path.abspath(os.path.dirname(__file__))
sid_list = np.loadtxt(os.path.join(loc, 'subject-id-list.csv'), dtype='u8')


def get_data(subject_id):
    diff_data = gu.get_diff_data(subject_id)

    # generate a cutout and sigma image
    frame_data = mkct.get_frame_data(subject_id)
    frame_data.to_pickle('frame_data/{}.pickle'.format(subject_id))
    try:
        stacked_image, sigma_image = mkct.generate_new_cutout(
            subject_id, frame_data=frame_data
        )

    except ValueError as e:
        print('Error on:', subject_id, e)
        print('Frame cutouts were not the same shape')
        return None
    # scale the image and std to match that used in modelling
    im_scaled = stacked_image / diff_data['multiplier']
    sd_scaled = sigma_image / diff_data['multiplier']

    # get the WCS objects so we can transform models back into the original
    # projection
    montage_wcs = mkct.get_montaged_cutout(subject_id).wcs
    original_wcs = frame_data.iloc[0].wcs

    return dict(
        psf=gu.get_psf(subject_id),
        pixel_mask=~im_scaled.mask,
        galaxy_data=im_scaled,
        montage_wcs=montage_wcs,
        original_wcs=original_wcs,
        sigma_image=sd_scaled,
        width=diff_data['width'],
        size_diff=diff_data['width'] / diff_data['imageWidth'],
    )


tqdm.pandas(desc='Making DataFrame')

df = pd.Series(sid_list, index=sid_list)
df = df.progress_apply(get_data)
# with tqdm(sid_list, desc='Making DataFrame') as bar:
#     for subject_id in bar:
#         df[subject_id] = get_data(subject_id)

df = df.apply(pd.Series)
df.to_pickle(os.path.join(loc, 'fitting_metadata2.pkl'))
