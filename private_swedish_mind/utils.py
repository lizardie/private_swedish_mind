"""
Utilities
=========

Auxilliary routines for the main module
"""

import  pandas as pd
from shapely.ops import unary_union
import  numpy as np
import  matplotlib.pyplot  as plt



def make_hist_mpn_geoms(mpn_geoms, cell_rings):
    """
    taking list of `mpn_geoms` and corresponding list of cell rings `cell_rings`.
    Makes histogram for population of cell rings based in `mpn_geoms`.


    """

    hist = []
    for idx, ring in enumerate(cell_rings):
        for geom in mpn_geoms:
            if (geom in ring):
                hist.append(idx)
    return hist


def get_rings_around_cell(cell_idx, vcs, width):
    """
    given the Voronoi cells `vcs`, the index of the cell `cell_idx` and the number of
    layers `width`
    creates a list of lists, where the first list contains indexes  of `vcs` for the zero layer,
    the second -- for the first layer etc.
    """

    result = [vcs.iloc[cell_idx:cell_idx + 1]]
    t = [[cell_idx]]
    neighbours_dis = result[0].iloc[0].geometry
    for layer in range(width):
        neighbours = vcs[vcs.geometry.touches(neighbours_dis)]
        t.append(neighbours.index.to_list())
        result.append(neighbours)
        neighbours = pd.concat(result)
        neighbours_dis = unary_union(neighbours.geometry)

    return t


def get_vcs_used_area(vcs, data, area_max):
    """
    collecting Voronoi cells visited during all journeys and their area, if it below the given threshould
    """
    unique_vc_indexes_tracks = np.unique(data.explode('vc_index_mpn').dropna().values.flatten()
                                         )
    # print(unique_vc_indexes_tracks)
    vc_used = vcs.iloc[unique_vc_indexes_tracks]

    area = vc_used.geometry.area / 10**6
    area = area[area < area_max]

    return vc_used, area


def vc_area_splits(area, n_parts):
    """
    given the Pandas Series with Voronoi cell areas, sorts the areas by accending order
    and splits into given number of parts.

    Returns a list of tuples with splits borders, like `[(0.01, 2.63), (2.71, 24.67), (25.16, 184.09)]`
    """

    splits = np.array_split(area.sort_values().round(2), n_parts)
    size_borders = [(el.iloc[0], el.iloc[-1]) for el in splits]

    return size_borders



def make_group_col(row, vcs, size_):
    vc_idxs = [el for el in row]
    areas = vcs.iloc[vc_idxs].geometry.area / 10**6

    return [(el < size_[1]) & (el > size_[0]) for el in areas]



def make_diffs_ring_histogram_sample_size(hist_data, series_length):
    """
    we take samples of different size from `hist` column, make a histogram for each sample and
    observe how the difference between it and the  histogram for full `hist` column.
    We learn how the difference evolves with sample size.

    returns Pandas DF with the differences
    """

    series_diffs = []
    series_length = series_length
    sample_size = [10, 50, 100, 200, 300, 500, len(hist_data)]
    n_bins = 7

    for serie in range(series_length):
        hist_vals = []
        for sample in sample_size:
            data = hist_data.sample(sample)
            hst = [el for sublist in data.to_list() for el in sublist]
            his = np.histogram(hst, bins=range(n_bins), density=True)
            hist_vals.append(his[0])

        diffs = [sum(abs(element - hist_vals[-1])) for element in hist_vals]
        series_diffs.append(diffs)

    series_diffs_pd = pd.DataFrame(series_diffs, columns=sample_size)
    return series_diffs_pd




def _plot_ring_hist(data):
    """
    plotting histogram for rings
    """

    hst = [el for sublist in data.to_list() for el in sublist]
    his = np.histogram(hst,bins=range(max(hst)+2))


    plt.figure(figsize=(15,10))
    plt.xlabel('ring number')
    plt.ylabel('number of occurences')
    plt.title("""Voronoi cell ring histogram for a GPS position
    averaged over 5 minute intervals. Number of GPS positions is %i."""%(read_df.shape[0]))
    plt.bar(his[1][0:-1],his[0])


    plt.savefig('hist.png')


def _plot_visited_vcs_sizes(vc_used, area):
    """
    plotting Voronoi cells visited during all the collected tracks  along with their area distribution
    """
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # https://stackoverflow.com/questions/21535294/matplotlib-add-rectangle-to-figure-not-to-axes


    fig, ax1 = plt.subplots(figsize=(30, 20), )

    left, bottom, width, height = [0.45, 0.59, 0.25, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height], zorder=30)

    fig.patches.extend([plt.Rectangle((left - 0.02, bottom - 0.025), width + 0.04, height + 0.04,
                                      fill=True, color='w', alpha=0.9, zorder=20,
                                      transform=fig.transFigure, figure=fig)])

    vc_used.to_crs(WGS84_EPSG).plot(ax=ax1, alpha=0.5, linewidth=2, facecolor='none', hatch='////')
    ctx.add_basemap(ax=ax1)

    area.hist().plot(ax=ax2)
    ax2.set_title("Voronoi cells area distribution for areas<%s km^2" % area_max)
    ax2.set_xlabel("area, km^2")
    ax1.set_title("Voronoi cells visited during all trips", fontsize=20)

    plt.savefig('../docs/pics/visited_vcs_sizes.png', bbox_inches="tight")


def _plot_dist_hist(data, bins=None):
    """
    plotting histogram for rings
    """

    hst = [el for sublist in data.to_list() for el in sublist]
    #     print(hst)
    if bins:
        bns = np.arange(0, bins, .5)
        his = np.histogram(hst, bins=bns)
    else:
        his = np.histogram(hst, bins=range(max(hst) + 2))
    plt.figure(figsize=(15, 10))
    plt.xlabel('distance between GPS and MPN points, km')
    plt.ylabel('number of occurences')
    plt.title("""Voronoi cell distance histogram for a GPS position
    averaged over 5 minute intervals. Number of MPN positions is %i,  number of timestamps is %i""" %
              (len(hst), read_df.shape[0]))
    plt.xticks(bns, bns + .5, rotation=90)

    plt.bar(his[1][:-1], his[0], width=0.8 * (his[1][1] - his[1][0]))

    plt.savefig('../docs/pics/hist_dist.png', bbox_inches="tight")



def _plot_ring_histogram_by_group(data, size_borders):
    """
    plots ring histogram for different groups of Voronoi cells, split by their area
    """

    n_hist = data.shape[1]
    width_factor  = 0.8/n_hist
    shift_factor = (n_hist-1)/2.0

    plt.figure(figsize=(15,10))
    plt.xlabel('ring number')
    plt.ylabel('number of occurencies')


    for key in range(n_hist-1):
        hst = [el for sublist in data[data.columns[key+1]].to_list() for el in sublist]
    #     print(hst)

        his = np.histogram(hst,bins=range(max(hst)+2), density=True)


        plt.bar(his[1][:-1]+width_factor*(key-shift_factor), his[0], width=width_factor*(his[1][1]-his[1][0]),
                label="area is within "+ str(size_borders[key])+ ' km^2',
                linewidth=2, alpha=0.6,edgecolor='black'
    )

    hst = [el for sublist in data[data.columns[0]].to_list() for el in sublist]
    his = np.histogram(hst,bins=range(max(hst)+2), density=True)
    plt.bar(his[1][:-1]+width_factor*(key-shift_factor+1), his[0], width=width_factor*(his[1][1]-his[1][0]),
                label='without dividing  to classes',
                linewidth=2, alpha=0.6,edgecolor='black', facecolor='black',hatch=r"//"
    )
    plt.legend()
    plt.title("""Voronoi cell ring histogram for a GPS position. Number of MPN positions is %i,  number of timestamps is %i.
    Plots are for different size-based classes of VCs"""%
                  (len(hst), data.shape[0]))
    # https://stackoverflow.com/questions/46913184/how-to-make-a-striped-patch-in-matplotlib

    plt.savefig('../docs/pics/hist_ring_by_group.png', bbox_inches = "tight")



def _plot_ring_hist_diffs_sample_size(series_diffs_pd):
    """
    plots differences between the histograms for sample and full data as a function of sample size

    """

    means = series_diffs_pd.describe().loc['mean']
    std = series_diffs_pd.describe().loc['std']

    means.plot(yerr=std, marker='o', capsize=5, figsize=(12, 8), label='std deviation for errors')
    plt.xlabel('sample size')
    plt.ylabel('error')
    plt.title("""Ring histogram error based on the random sample size.
              The error bars for errors  are calculated for %i series. The full data size is %i.""" % (
    series_length, series_diffs_pd.columns[-1]))
    plt.legend()
    plt.grid()

    plt.savefig('../docs/pics/hist_ring_diffs_sample_size.png', bbox_inches="tight")



def make_plots():
    """
    producing all the plots

    :return:
    """

    _plot_dist_hist(read_df['distances'], bins=20)

    _plot_ring_hist(read_df['hist'], density=False)

    vc_used, area = get_vcs_used_area(read_df[['vc_index_gps', 'vc_index_mpn']], area_max=200)

    _plot_visited_vcs_sizes(vc_used, area)


    size_borders = vc_area_splits(area, 3)

    size_borders_lst = _vc_index_area_match()

    _plot_ring_histogram_by_group(size_borders_lst)

    series_length = 10
    series_diffs_pd = make_diffs_ring_histogram_sample_size(read_df['hist'], series_length)


    _plot_ring_hist_diffs_sample_size(series_diffs_pd)







