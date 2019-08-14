import os
import shutil
import re
import requests
import json
from io import BytesIO
from PIL import Image
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u
import galaxy_utilities as gu


def download_files(subject_id, v):
    global bar
    diff_loc = v['0']
    image_loc = v['1']
    model_loc = v['2']
    res = [requests.get(i) for i in (diff_loc, model_loc)]
    if all(i.status_code == 200 for i in res):
        try:
            os.mkdir('subject_data/{}'.format(subject_id))
        except FileExistsError:
            pass
        for name, r in zip(('diff.json', 'model.json'), res):
            with open('subject_data/{}/{}'.format(subject_id, name), 'w') as f:
                d = r.json()
                json.dump(d, f)
    image = Image.open(BytesIO(requests.get(image_loc).content))
    image.save('subject_data/{}/image.png'.format(subject_id))


def recover_files(path_to_subject_sets):
    dr7objids = gu.metadata['SDSS dr7 id']
    coords = gu.metadata[['ra', 'dec']].dropna()
    match_count = 0
    # coords_catalog = SkyCoord(coords['ra'], coords['dec'], unit=u.degree)
    for subject_set in os.listdir(path_to_subject_sets):
        dir = os.path.join(path_to_subject_sets, subject_set)
        if not os.path.isdir(dir):
            continue
        metadata_files = [
            os.path.join(path_to_subject_sets, subject_set, f)
            for f in os.listdir(dir)
            if 'metadata' in f
        ]
        with tqdm(metadata_files, desc='Searching {}'.format(subject_set)) as bar:
            for f in bar:
                with open(f) as in_file:
                    metadata = json.load(in_file)
                    dr7objid = metadata.get('SDSS dr7 id', np.nan)
                    # ra = float(metadata.get('ra', False))
                    # dec = float(metadata.get('dec', False))
                    match_ids = gu.metadata[gu.metadata['SDSS dr7 id'] == dr7objid].index.values
                    for subject_id in match_ids:
                        match_count += 1
                        gu.metadata.loc[subject_id]['ra']
                        gu.metadata.loc[subject_id]['dec']
                        # sep = SkyCoord(ra=ra, dec=dec, unit='deg').sep(
                        #     SkyCoord(ra=ra2, dec=dec2, unit='deg')
                        # )
                        base_dir = os.path.join(path_to_subject_sets, subject_set)
                        ss_index = re.search('([0-9]+)\.json$', f).group(1)
                        output_dir = os.path.join('subject_data', str(subject_id))
                        try:
                            os.mkdir(output_dir)
                        except FileExistsError:
                            pass
                        files_to_copy = (
                            ('image', 'png'), ('metadata', 'json'),
                            ('difference', 'json'), ('model', 'json')
                        )
                        for name, type in files_to_copy:
                            shutil.copyfile(
                                os.path.join(
                                    base_dir,
                                    '{}_subject{}.{}'.format(name, ss_index, type),
                                ),
                                os.path.join(output_dir, '{}.{}'.format(name, type))
                            )
    print('Identified {} matches'.format(match_count))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            'Generate subsection of the NSA catalogue used by galaxy_utilities'
        )
    )
    parser.add_argument('S', metavar='/path/to/subject_sets',
                        type=str, help='Location of created subject sets')

    args = parser.parse_args()

    recover_files(args.S)

# if __name__ == '__main___':
#     subject_ids = np.unique(
#         gu.classifications.query('workflow_version == 61.107')['subject_ids']
#     )
#
#     locations = gu.subjects.set_index(
#         'subject_id'
#     ).locations.apply(
#         json.loads
#     )[subject_ids]
#     with Pool(4) as p:
#         async_res = p.starmap_async(
#             download_files,
#             locations.iteritems(),
#         )
#         async_res.wait()
