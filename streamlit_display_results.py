import os
import re
from PIL import Image
import requests
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from shapely.affinity import scale, translate
from astropy.visualization import AsinhStretch
from descartes import PolygonPatch
import gzbuilder_analysis.aggregation as ag
import seaborn as sns
import pandas as pd
import streamlit as st
from pdf_image_converter import convert


SKYSERVER_URL = (
    'http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg'
    '?TaskName=Skyserver.Chart.Navi&ra={}&dec={}&scale=0.396'
    '&width={}&height={}&opt='
)


def transform_patch(p, shape, s=2):
    corrected_patch = scale(
        translate(p, xoff=-shape[1]/2, yoff=-shape[0]/2),
        xfact=0.396,
        yfact=0.396,
        origin=(0, 0),
    )
#     display patch at s*Re
    return scale(corrected_patch, s, s)


def transform_arm(arm, shape):
    return (arm - np.array(shape) / 2) * 0.396


@st.cache
def load_agg_res(f, path='output_files/aggregation_results'):
    try:
        agg_res = pd.read_pickle(join(path, f'{f}.pkl.gz'))
        return pd.Series(dict(
            clusters=agg_res.clusters,
            params=agg_res.params,
            errors=agg_res.errors,
            spiral_arms=agg_res.spiral_arms,
            input_models=agg_res.input_models.apply(pd.Series),
        ))
    except IOError:
        return False

@st.cache
def load_fit_res(f, path='output_files/tuning_results'):
    try:
        return pd.read_pickle(join(path, f'{f}.pickle.gz'))
    except IOError:
        return False


@st.cache
def load_gal_df():
    return pd.read_csv('lib/gal-metadata.csv', index_col=0)


@st.cache
def get_sid_list(agg_res_path='output_files/aggregation_results'):
    return sorted(
        int(r.group(1))
        for r in (
            re.match(r'([0-9]+)\.pkl.gz', f)
            for f in os.listdir(agg_res_path)
        )
        if r is not None
    )


@st.cache
def get_fitting_metadata(subject_id):
    return pd.read_pickle('lib/fitting_metadata.pkl').loc[subject_id]



def get_imshow_kwargs(fm):
    extent = (np.array([[-1, -1], [1, 1]]) * fm.galaxy_data.shape).T.ravel() / 2 * 0.396
    return {
        'origin': 'lower',
        'extent': extent,
        'cmap': 'gray_r',
    }


def get_agg_geoms(agg_res):
    try:
        agg_disk_geom = ag.make_ellipse(agg_res.params.disk.to_dict())
    except AttributeError:
        agg_disk_geom = None
    try:
        agg_bulge_geom = ag.make_ellipse(agg_res.params.bulge.to_dict())
    except AttributeError:
        agg_bulge_geom = None
    try:
        agg_bar_geom = ag.make_box(agg_res.params.bar.to_dict())
    except AttributeError:
        agg_bar_geom = None
    return (agg_disk_geom, agg_bulge_geom, agg_bar_geom)


def make_patches(geoms, fm, c=('C0', 'C1', 'C2'), ls=('-.', ':', '--'), **kwargs):
    k = {'alpha': 0.3, 'zorder': 3, 'ec': 'k', **kwargs}
    agg_disk_geom, agg_bulge_geom, agg_bar_geom = geoms
    patches = [
        PolygonPatch(
            transform_patch(geom, fm.galaxy_data.shape),
            fc=c[i],
            linestyle=ls[i],
            **k,
        ) if geom is not None else None
        for i, geom in enumerate(
            (agg_disk_geom, agg_bulge_geom, agg_bar_geom)
        )
    ]
    return patches


def make_annotation_plot(agg_res, fm):
    drawn_arms = [j for i in (
        agg_res.input_models.spiral
        .apply(lambda a: [points for points, params in a])
    ) for j in i]
    geoms = agg_res.input_models.apply(ag.get_geoms, axis=1).apply(pd.Series)
    data = fm.galaxy_data
    imshow_kwargs = get_imshow_kwargs(fm)
    s = AsinhStretch()
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        ncols=2, nrows=2,
        figsize=(16, 16),
        sharex=True, sharey=True
    )
    ax0.imshow(s(data), **imshow_kwargs)
    for comp in geoms['disk'].values:
        if comp is not None:
            ax0.add_patch(
                PolygonPatch(transform_patch(comp, data.shape), fc='none', ec='k',
                             zorder=3)
            )
    ax1.imshow(s(data), **imshow_kwargs)
    for comp in geoms['bulge'].values:
        if comp is not None:
            ax1.add_patch(
                PolygonPatch(transform_patch(comp, data.shape), fc='none', ec='k',
                             zorder=3)
            )
    ax2.imshow(s(data), **imshow_kwargs)
    for comp in geoms['bar'].values:
        if comp is not None:
            ax2.add_patch(
                PolygonPatch(transform_patch(comp, data.shape), fc='none', ec='k',
                             zorder=3)
            )
    ax3.imshow(s(data), **imshow_kwargs)
    for arm in drawn_arms:
        ax3.plot(*transform_arm(arm, data.shape).T, 'k', alpha=0.5, linewidth=1)

    for i, ax in enumerate((ax0, ax1, ax2, ax3)):
        ax.set_xlim(imshow_kwargs['extent'][:2])
        ax.set_ylim(imshow_kwargs['extent'][2:])
        if i % 2 == 0:
            ax.set_ylabel('Arcseconds from center')
        if i > 1:
            ax.set_xlabel('Arcseconds from center')
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    st.pyplot()


def plot_clusters(agg_res, fm):
    disk_cluster_geoms = agg_res.clusters['disk'].apply(ag.make_ellipse)
    bulge_cluster_geoms = agg_res.clusters['bulge'].apply(ag.make_ellipse)
    bar_cluster_geoms = agg_res.clusters['bar'].apply(ag.make_box)
    drawn_arms = np.array([
        polyline
        for arm in agg_res.spiral_arms
        for polyline in arm.arms
    ])
    arms = agg_res.spiral_arms
    data = fm.galaxy_data
    s = AsinhStretch()
    imshow_kwargs = get_imshow_kwargs(fm)
    agg_geoms = get_agg_geoms(agg_res)
    disk_crop = min(np.abs(imshow_kwargs['extent']).min(), 1000)
    bulge_crop = bar_crop = min(np.abs(imshow_kwargs['extent']).min(), 15)
    # plot the component clusters
    plt.figure(figsize=(18/2, 18/2), dpi=200)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    edge_patches = make_patches(
        agg_geoms, fm,
        c=('none', 'none', 'none'),
        alpha=1,
        lw=4,
    )
    face_patches = make_patches(
        agg_geoms, fm,
        c=('C0', 'C1', 'C2'),
        ec='none'
    )

    plt.subplot(221)
    plt.imshow(s(data), **imshow_kwargs)
    for disk in disk_cluster_geoms:
        p = PolygonPatch(transform_patch(disk, data.shape), fc='none', ec='k', alpha=0.5, zorder=2)
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

    plt.subplot(222)
    plt.imshow(s(data), **imshow_kwargs)
    for bulge in bulge_cluster_geoms:
        p = PolygonPatch(transform_patch(bulge, data.shape), fc='none', ec='k', alpha=0.5, zorder=2)
        plt.gca().add_patch(p)
    try:
        plt.gca().add_patch(edge_patches[1])
        plt.gca().add_patch(face_patches[1])

    except AttributeError:
        pass
    plt.title('Aggregate bulge')
    plt.xlim(-bulge_crop, bulge_crop)
    plt.ylim(-bulge_crop, bulge_crop)

    plt.subplot(223)
    plt.imshow(s(data), **imshow_kwargs)
    for bar in bar_cluster_geoms:
        p = PolygonPatch(transform_patch(bar, data.shape), fc='none', ec='k', alpha=0.5, zorder=2)
        plt.gca().add_patch(p)
    try:
        plt.gca().add_patch(edge_patches[2])
        plt.gca().add_patch(face_patches[2])

    except AttributeError:
        pass
    plt.title('Aggregate bar')
    plt.xlabel('Arcseconds from galaxy centre')
    plt.ylabel('Arcseconds from galaxy centre')
    plt.xlim(-bar_crop, bar_crop)
    plt.ylim(-bar_crop, bar_crop)

    plt.subplot(224)
    plt.imshow(s(data), **imshow_kwargs)
    for i, arm in enumerate(arms):
        plt.plot(*transform_arm(arm.coords[arm.outlier_mask], data.shape).T, '.', alpha=0.4, ms=3, c=f'C{i}')
        plt.plot(*transform_arm(arm.coords[~arm.outlier_mask], data.shape).T,
                 'x', alpha=0.4, ms=6, c='r')
        plt.plot(*transform_arm(arm.reprojected_log_spiral, data.shape).T, 'k', lw=2.8)
        plt.plot(*transform_arm(arm.reprojected_log_spiral, data.shape).T, 'w', lw=2)
        plt.plot(*transform_arm(arm.reprojected_log_spiral, data.shape).T, c=f'C{i}', lw=2, alpha=0.6)
    plt.title('Aggregate spiral arms')
    plt.xlabel('Arcseconds from galaxy centre')
    plt.xlim(-disk_crop, disk_crop)
    plt.ylim(-disk_crop, disk_crop)

    plt.tight_layout()
    st.pyplot()


def plot_agg_model(agg_res, fm):
    arms = agg_res.spiral_arms
    data = fm.galaxy_data
    s = AsinhStretch()
    imshow_kwargs = get_imshow_kwargs(fm)

    agg_geoms = get_agg_geoms(agg_res)
    disk_crop = min(np.abs(imshow_kwargs['extent']).min(), 1000)
    bulge_crop = bar_crop = min(np.abs(imshow_kwargs['extent']).min(), 15)

    edge_patches = make_patches(
        agg_geoms, fm,
        c=('none', 'none', 'none'),
        alpha=1,
        lw=2,
    )
    face_patches = make_patches(
        agg_geoms, fm,
        ec='none',
    )

    # plot the aggregate model
    plt.figure(figsize=(8, 8))
    plt.imshow(s(data), **imshow_kwargs)
    try:
        plt.gca().add_patch(edge_patches[0])
        plt.gca().add_patch(face_patches[0])
    except AttributeError:
        pass
    try:
        plt.gca().add_patch(edge_patches[2])
        plt.gca().add_patch(face_patches[2])

    except AttributeError:
        pass
    try:
        plt.gca().add_patch(edge_patches[1])
        plt.gca().add_patch(face_patches[1])

    except AttributeError:
        pass

    for arm in arms:
        a = transform_arm(arm.reprojected_log_spiral, data.shape)
        plt.plot(*a.T, 'r', lw=2, zorder=4)


    plt.xlabel('Arcseconds from galaxy centre')
    plt.ylabel('Arcseconds from galaxy centre')
    plt.xlim(-disk_crop, disk_crop)
    plt.ylim(-disk_crop, disk_crop)

    plt.tight_layout()
    st.pyplot()


def main():
    st.title('Clustering Results Viewer')
    display_image = st.sidebar.checkbox('Display image', value=True)
    display_cls = st.sidebar.checkbox('Display classifications')
    display_cluster = st.sidebar.checkbox('Display Clusters', value=True)
    display_agg = st.sidebar.checkbox('Display Aggregation', value=True)
    display_fit = st.sidebar.checkbox('Display Tuning results', value=True)
    display_params = st.sidebar.checkbox('Display Model Parameters')
    gal_df = load_gal_df()
    sid_list = get_sid_list()
    subject_id = st.selectbox(options=sid_list, label='Subject ID')
    fm = get_fitting_metadata(subject_id)
    agg_res = load_agg_res(subject_id)
    fit_res = load_fit_res(subject_id)
    @st.cache
    def get_image(subject_id, shape):
        im = Image.open(
            requests.get(
                SKYSERVER_URL.format(
                    *gal_df.loc[subject_id][['RA', 'DEC']].values,
                    *np.array(shape)
                ),
                stream=True
            ).raw
        )
        return im.resize((256, 256), Image.HAMMING)
    if display_image:
        st.markdown('SDSS RGB image of Galaxy:')
        st.image(get_image(subject_id, (256, 256)), use_column_width=True)

    n_disk = agg_res.input_models.disk.notna().sum()
    n_bulge = agg_res.input_models.bulge.notna().sum()
    n_bar = agg_res.input_models.bar.notna().sum()
    n_spiral = agg_res.input_models.spiral.apply(len).sum()
    n_disk_c = len(agg_res.clusters['disk'])
    n_bulge_c = len(agg_res.clusters['bulge'])
    n_bar_c = len(agg_res.clusters['bar'])
    n_spiral_c = len(agg_res.spiral_arms)
    st.markdown(f'Recieved {n_disk} disks, {n_bulge} bulges, {n_bar} bars and {n_spiral} poly-lines')
    if display_cls:
        make_annotation_plot(agg_res, fm)

    st.markdown(f'Clustered {n_disk_c} disks, {n_bulge_c} bulges and {n_bar_c} bars')
    st.markdown(f'Identified {n_spiral_c} spiral arm(s)')
    if display_cluster:
        plot_clusters(agg_res, fm)
    if display_agg:
        plot_agg_model(agg_res, fm)

    if display_fit:
        png_path = f'fitting_plots/png/{subject_id}.png'
        pdf_path = f'fitting_plots/{subject_id}.pdf'
        if os.path.isfile(png_path):
            st.image(
                Image.open(png_path),
                use_column_width=True
            )
        elif os.path.isfile(pdf_path):
            with st.spinner('Converting fitting image from PDF to PNG'):
                convert(pdf_path, outfolder='fitting_plots/png')
            st.image(
                Image.open(png_path),
                use_column_width=True
            )
        else:
            st.markdown('Fitting image file not found')

    st.write(r'Final $\chi_\nu^2 = {:.4f}$'.format(fit_res['chisq']))

    if display_params:
        st.markdown('## Change in parameters')
        for k in ('disk', 'bulge', 'bar'):
            st.markdown('**{}**'.format(k.capitalize()))
            st.table(pd.concat((
                agg_res.params.get(k, pd.Series([], dtype=float)).rename('Aggregate'),
                agg_res.errors.get(k, pd.Series([], dtype=float))
                       .replace(np.inf, np.nan).rename('Sigma'),
                pd.Series(fit_res['final_model'].get(k, [])).rename('Tuned'),
            ), axis=1))

if __name__ == '__main__':
    main()
