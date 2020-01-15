import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.affinity import scale, translate
from descartes import PolygonPatch
import lib.galaxy_utilities as gu
import gzbuilder_analysis.aggregation as ag
import gzbuilder_analysis.parsing as pg
from gzbuilder_analysis.config import COMPONENT_CLUSTERING_PARAMS
from asinh_cmap import asinh_cmap, asinh_cmap_r

st.title('Galaxy Builder classification clustering')
st.write('')

@st.cache
def load_models(subject_id):
    fitting_metadata = pd.read_pickle('lib/fitting_metadata.pkl')\
        .loc[subject_id]

    gal = gu.get_galaxy(subject_id)
    pic_array = np.array(gu.get_image(subject_id))
    cls_for_subject = gu.classifications.query(
        'subject_ids == {}'.format(subject_id)
    )

    zoo_models = cls_for_subject.apply(
        pg.parse_classification,
        axis=1,
        image_size=np.array(pic_array.shape),
        size_diff=fitting_metadata['size_diff'],
        ignore_scale=True  # ignore scale slider when aggregating
    )

    scaled_models = zoo_models.apply(
        pg.scale_model,
        args=(fitting_metadata['size_diff'],),
    )

    models = scaled_models.apply(
        pg.reproject_model,
        wcs_in=fitting_metadata['montage_wcs'],
        wcs_out=fitting_metadata['original_wcs']
    )

    sanitized_models = models.apply(pg.sanitize_model)

    geoms = pd.DataFrame(
        sanitized_models.apply(ag.get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )
    return fitting_metadata, gal, sanitized_models, geoms


subject_id = 20902040

fitting_metadata, gal, models, geoms = load_models(subject_id)
no_spiral_models = models.apply(lambda m: {**m, 'spiral': np.array([])})

st.write(no_spiral_models)

data = fitting_metadata.galaxy_data
psf = fitting_metadata.psf
sigma_image = fitting_metadata.sigma_image


ba = gal['PETRO_BA90']
phi = np.rad2deg(gal['original_angle'])

extent = (np.array([[-1, -1], [1, 1]]) * data.shape).T.ravel() / 2 * 0.396
imshow_kwargs = {
    'cmap': asinh_cmap_r, 'origin': 'lower',
    'extent': extent
}


def transform_patch(p):
    corrected_patch = scale(
        translate(p, xoff=-data.shape[1]/2, yoff=-data.shape[0]/2),
        xfact=0.396,
        yfact=0.396,
        origin=(0, 0),
    )
    # display patch at 3*Re
    return scale(corrected_patch, 3, 3)


def transform_arm(arm):
    return (arm - np.array(data.shape) / 2) * 0.396


n_classifications = st.slider(
    'Number of classifications to use',
    1, len(no_spiral_models), len(no_spiral_models)
)

fig = plt.figure(1, figsize=(18, 6), dpi=80)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.subplot(131)
plt.imshow(data, **imshow_kwargs)
for disk in geoms.iloc[:n_classifications]['disk'].dropna():
    p = PolygonPatch(transform_patch(disk), fc='C0', ec='k', alpha=0.05, zorder=2)
    plt.gca().add_patch(p)
plt.title('Drawn disks')
plt.xlabel('Arcseconds from galaxy centre')
plt.ylabel('Arcseconds from galaxy centre')

plt.subplot(132)
plt.imshow(data, **imshow_kwargs)
for bulge in geoms.iloc[:n_classifications]['bulge'].dropna():
    p = PolygonPatch(transform_patch(bulge), fc='C1', ec='k', alpha=0.1, zorder=2)
    plt.gca().add_patch(p)
# plt.yticks([])
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.title('Drawn bulges')
plt.xlabel('Arcseconds from galaxy centre')

plt.subplot(133)
plt.imshow(data, **imshow_kwargs)
for bar in geoms.iloc[:n_classifications]['bar'].dropna():
    p = PolygonPatch(transform_patch(bar), fc='C2', ec='k', alpha=0.1, zorder=2)
    plt.gca().add_patch(p)
plt.yticks([])
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.title('Drawn bars')
plt.xlabel('Arcseconds from galaxy centre')

plt.tight_layout()
st.pyplot(fig)


st.write('''Clustering is performed using DBSCAN and the Jaccard distance
''')

st.latex(r'D = 1 - \frac{|A \cap B|}{|A \cup B|}')


@st.cache
def do_cluster(n_cls, models=no_spiral_models):
    model_cluster = ag.cluster_components(
        models=models.iloc[:n_cls], image_size=data.shape,
    )
    aggregation_result = ag.aggregate_components(model_cluster)
    return model_cluster, aggregation_result


with st.spinner('Performing clustering and aggregation'):
    model_cluster, aggregation_result = do_cluster(n_classifications)

    disk_cluster_geoms = model_cluster['disk'].apply(ag.make_ellipse)
    bulge_cluster_geoms = model_cluster['bulge'].apply(ag.make_ellipse)
    bar_cluster_geoms = model_cluster['bar'].apply(ag.make_box)


agg_disk_geom = ag.make_ellipse(aggregation_result['disk'])
agg_bulge_geom = ag.make_ellipse(aggregation_result['bulge'])
agg_bar_geom = ag.make_box(aggregation_result['bar'])


def make_patches(c=('C0', 'C1', 'C2'), ls=('-.', ':', '--'), **kwargs):
    k = {'alpha': 0.3, 'zorder': 3, 'ec': 'k', **kwargs}
    patches = [
        PolygonPatch(
            transform_patch(geom),
            fc=c[i],
            linestyle=ls[i],
            **k,
        ) if geom is not None else None
        for i, geom in enumerate(
            (agg_disk_geom, agg_bulge_geom, agg_bar_geom)
        )
    ]
    return patches


disk_crop = 30
bulge_crop = bar_crop = 15

fig = plt.figure(2, figsize=(18, 6), dpi=80)
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
edge_patches = make_patches(
    c=('none', 'none', 'none'),
    alpha=1,
    lw=4,
)
face_patches = make_patches(
    c=('C0', 'C1', 'C2'),
    ec='none'
)
plt.subplot(131)
plt.imshow(data, **imshow_kwargs)
for disk in disk_cluster_geoms:
    p = PolygonPatch(transform_patch(disk), fc='none', ec='k', alpha=0.5, zorder=2)
    plt.gca().add_patch(p)
try:
    plt.gca().add_patch(edge_patches[0])
    plt.gca().add_patch(face_patches[0])
except AttributeError:
    pass
plt.title('Aggregate disk')
plt.xlabel('Arcseconds from galaxy centre')
plt.ylabel('Arcseconds from galaxy centre')
plt.xlim(-disk_crop, disk_crop)
plt.ylim(-disk_crop, disk_crop)
plt.subplot(132)
plt.imshow(data, **imshow_kwargs)
for bulge in bulge_cluster_geoms:
    p = PolygonPatch(transform_patch(bulge), fc='none', ec='k', alpha=0.5, zorder=2)
    plt.gca().add_patch(p)
try:
    plt.gca().add_patch(edge_patches[1])
    plt.gca().add_patch(face_patches[1])

except AttributeError:
    pass
plt.title('Aggregate bulge')
plt.xlabel('Arcseconds from galaxy centre')
plt.xlim(-bulge_crop, bulge_crop)
plt.ylim(-bulge_crop, bulge_crop)
plt.subplot(133)
plt.imshow(data, **imshow_kwargs)
for bar in bar_cluster_geoms:
    p = PolygonPatch(transform_patch(bar), fc='none', ec='k', alpha=0.5, zorder=2)
    plt.gca().add_patch(p)
try:
    plt.gca().add_patch(edge_patches[2])
    plt.gca().add_patch(face_patches[2])

except AttributeError:
    pass
plt.title('Aggregate bar')
plt.xlabel('Arcseconds from galaxy centre')
plt.xlim(-bar_crop, bar_crop)
plt.ylim(-bar_crop, bar_crop)
plt.tight_layout()
st.pyplot(fig)
