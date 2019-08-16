import os
import numpy as np
import pandas as pd
import galaxy_utilities as gu
from tqdm import tqdm

loc = os.path.abspath(os.path.dirname(__file__))
sid_list = np.loadtxt(os.path.join(loc, 'subject-id-list.csv'), dtype='u8')

df = pd.Series([])
with tqdm(sid_list, desc='Making DataFrame') as bar:
    for subject_id in bar:
        diff_data = gu.get_diff_data(subject_id)
        df[subject_id] = dict(
            psf=gu.get_psf(subject_id),
            pixel_mask=1 - np.array(diff_data['mask']),
            galaxy_data=np.array(diff_data['imageData']),
            width=diff_data['width'],
            size_diff=diff_data['width'] / diff_data['imageWidth'],
        )

df = df.apply(pd.Series)
df.to_pickle(os.path.join(loc, 'fitting_metadata.pkl'))
