"""
Module for plotting
===================
"""


import  matplotlib.pyplot  as plt
import  numpy as np
import contextily as ctx
import  os

try:
    from consts import  *
except:
    from private_swedish_mind.consts import  *



def _plot_ring_hist(data, bins=None, density=True):
    """
    plotting histogram for rings
    """

    hst = [el for sublist in data.to_list() for el in sublist]
    #     print(hst)
    if bins:
        his = np.histogram(hst, bins=bins)
    else:
        his = np.histogram(hst, bins=range(max(hst) + 2), density=density)

    plt.figure(figsize=(15, 10))
    plt.xlabel('ring number')
    plt.ylabel('number of occurences')
    plt.title("""Voronoi cell ring histogram for a GPS position
    averaged over 5 minute intervals. Number of MPN positions is %i,  number of timestamps is %i.""" %
              (len(hst), data.shape[0]))
    plt.bar(his[1][:-1], his[0], width=0.8 * (his[1][1] - his[1][0]))

    plt.savefig('../docs/pic1/hist_ring.png', bbox_inches="tight")



def _plot_visited_vcs_sizes(vc_used, area, area_max):
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

    #     area.hist().plot( ax=ax2, edgecolor="black", color="red")
    ax2.hist(area, edgecolor="black", color="red", alpha=0.8)
    ax2.set_title("Voronoi cells area distribution for areas < %s $km^2$" % area_max)
    ax2.set_xlabel("area, $km^2$")
    ax2.set_ylabel('number of occurencies')
    ax1.set_title("Voronoi cells visited during all trips", fontsize=20)

    plt.savefig('../docs/pic1/visited_vcs_sizes.png', bbox_inches="tight")



def _plot_dist_hist(data, bins1=None, bins2=None):
    """
    plotting histogram for rings
    """

    hst = [el for sublist in data.to_list() for el in sublist]
    #     print(hst)
    bns1 = np.arange(0, bins1, .5)
    his1 = np.histogram(hst, bins=bns1)

    bns2 = np.arange(0, bins2, .1)
    his2 = np.histogram(hst, bins=bns2)

    fig, ax1 = plt.subplots(figsize=(15, 10), )

    left, bottom, width, height = [0.5, 0.5, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height], zorder=30)
    ax2.set_xlabel('distance between GPS and MPN points, km')
    ax2.set_ylabel('number of occurences')
    ax2.set_title("""Inset for ranges: 0 to %i km""" % bins2)
    ax2.set_xticks(bns2)
    ax2.set_xticklabels((bns2 + .1).round(1), rotation=90)

    ax1.set_xlabel('distance between GPS and MPN points, km')
    ax1.set_ylabel('number of occurences')
    ax1.set_title("""Voronoi cell distance histogram for a GPS position
    averaged over 5 minute intervals. Number of MPN positions is %i,  number of timestamps is %i""" %
                  (len(hst), data.shape[0]))
    ax1.set_xticks(bns1)
    ax1.set_xticklabels(bns1 + .5, rotation=90)

    ax1.bar(his1[1][:-1], his1[0], width=0.8 * (his1[1][1] - his1[1][0]))
    ax2.bar(his2[1][:-1], his2[0], width=0.8 * (his2[1][1] - his2[1][0]), color='red', edgecolor='black', alpha=0.5)

    plt.savefig('../docs/pic1/hist_dist.png', bbox_inches="tight")


def _plot_ring_histogram_by_group(data, size_borders, title, label, saveas):
    """
    plots ring histogram for different groups of Voronoi cells, split by their area
    """

    n_hist = data.shape[1]
    width_factor = 0.8 / n_hist
    shift_factor = (n_hist - 1) / 2.0

    plt.figure(figsize=(15, 10))
    plt.xlabel('ring number')
    plt.ylabel('number of occurences')

    for key in range(n_hist - 1):
        hst = [el for sublist in data[data.columns[key + 1]].to_list() for el in sublist]
        #     print(hst)

        his = np.histogram(hst, bins=range(max(hst) + 2), density=True)

        plt.bar(his[1][:-1] + width_factor * (key - shift_factor), his[0], width=width_factor * (his[1][1] - his[1][0]),
                #                 label="area is within "+ str(size_borders[key])+ ' $km^2$',
                label=label % str(size_borders[key]),
                linewidth=2, alpha=0.6, edgecolor='black'
                )

    hst = [el for sublist in data[data.columns[0]].to_list() for el in sublist]
    his = np.histogram(hst, bins=range(max(hst) + 2), density=True)
    plt.bar(his[1][:-1] + width_factor * (key - shift_factor + 1), his[0], width=width_factor * (his[1][1] - his[1][0]),
            label='without dividing  to classes',
            linewidth=2, alpha=0.6, edgecolor='black', facecolor='black', hatch=r"//"
            )
    plt.legend()
    #     plt.title("""Voronoi cell ring histogram for a GPS position. Number of MPN positions is %i,  number of timestamps is %i.
    #     Plots are for different size-based classes of VCs"""%
    #                   (len(hst), data.shape[0]))
    plt.title(title % (len(hst), data.shape[0]))
    # https://stackoverflow.com/questions/46913184/how-to-make-a-striped-patch-in-matplotlib

    plt.savefig(os.path.join(PICS_LOCATION, saveas), bbox_inches="tight")



def _plot_antennas_usage(data, nbins=20, ):
    x = [a[1] for a in data]
    plt.figure(figsize=(12, 8))
    # print(x)
    nbins = 50
    logbins = np.logspace(np.log10(min(x)), np.log10(max(x)), nbins + 2)
    # print(logbins)

    plt.hist(x, bins=logbins, edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Antennas usage distribution for %i distinct antennas, split into %i bins' % (len(x), nbins))
    plt.xlabel('antenna usage number, log scale')
    plt.ylabel("unique antenna number of occurrancies, log scale")
    plt.grid()
    plt.savefig('../docs/pic1/most_used_antennas.png', bbox_inches="tight")
    # https://stackoverflow.com/questions/47850202/plotting-a-histogram-on-a-log-scale-with-matplotlib




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






