# Takes the NASA-Sloan Atlas in FITS format, extracts certain columns and saves
# as a pandas-friendly pickle file.
from astropy.table import Table
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(
    description=(
        ' Takes the NASA-Sloan Atlas in FITS format, extracts certain columns'
        'and saves as a pandas-friendly pickle file.'
    )
)
parser.add_argument('--catalog', metavar='/path/to/nsa_file.fits', required=True,
                    type=str, help='Location of NSA 1.0.1 FITS file')

args = parser.parse_args()

this_file_location = os.path.dirname(os.path.abspath(__file__))
nsa_catalog = Table.read(args.catalog, format='fits')

nsa_keys = (
    'NSAID', 'ISDSS', 'INED', 'IAUNAME',  # identifiers
    'RA', 'DEC', 'Z', 'ZDIST',  # position
    'SERSIC_BA', 'SERSIC_PHI',  # sersic photometry
    'PETRO_THETA',  # azimuthally averaged petrosean radius
    'PETRO_BA90', 'PETRO_PHI90',  # petrosean photometry at 90% light radius
    'PETRO_BA50', 'PETRO_PHI50',  # ... at 50% light radius
    'RUN', 'CAMCOL', 'FIELD', 'RERUN',
    'ELPETRO_MASS', 'SERSIC_MASS',
)

pd.DataFrame(
    {k: nsa_catalog[k] for k in nsa_keys}
).to_pickle(
    os.path.join(this_file_location, 'df_nsa.pkl')
)

# import galaxy_utilities as gu
# NSA_GZ = fits.open('lib/NSA_GalaxyZoo.fits')
# sid_list = pd.read_csv('subject-id-list.csv').values[:, 0]
# nsa_subsection = pd.Series([])
# with tqdm(sid_list) as bar:
#     for subject_id in bar:
#         metadata = gu.meta_map.get(int(subject_id), {})
#         try:
#             gz2_gal = NSA_GZ[1].data[
#                 NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
#             ][0]
#         except IndexError:
#             print('Could not find object for id {}'.format(subject_id))
#         nsa_subsection.loc[subject_id] = pd.Series({
#           k: v for k, v in zip(NSA_GZ[1].data.dtype.names, gz2_gal)
#         })
# nsa_subsection.to_pickle('df_nsa_gz.pkl')
