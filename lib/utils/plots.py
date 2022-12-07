import imageio
import matplotlib.colors as col
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms
import numpy as np
from matplotlib import patches
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


### plotting ###
def text_box(ax, s, ori, W, H, **textkwargs):
    rectangle = patches.Rectangle(
        ori, W, H, edgecolor="k", facecolor="lightgray", linewidth=1
    )
    ax.add_patch(rectangle)
    rx, ry = rectangle.get_xy()
    cx = rx + rectangle.get_width() / 2.0
    cy = ry + rectangle.get_height() / 2.0
    ax.annotate(s, (cx, cy), ha="center", va="center", **textkwargs)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def make_cmap(colors, name):
    """
    Create a custom colormap

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :returns: colormap
    :rtype:
    """
    cc = []
    for c in colors:
        cc.append(c)
    new_map = col.LinearSegmentedColormap.from_list(name, cc, N=256, gamma=1)
    return new_map


def decorate_ax(
    ax,
    xlabel="",
    ylabel="",
    labelsize=12,
    xticks=[],
    yticks=[],
    xlim=None,
    ylim=None,
    spines=[True, True, True, True],
):
    """
    Decorate the axes

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :param list spines: list of visibility of axis spines (left, right, top, bottom)
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for k, name in enumerate(["left", "right", "top", "bottom"]):
        ax.spines[name].set_visible(spines[k])


def compute_mesh(grid_n, grid_size, func):
    """
    Create a uniformly spaced mesh over which to evaluate func.

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :returns: figure and axis
    :rtype: tuple
    """
    mesh = ()
    for n, gn in enumerate(grid_n):
        mesh += (np.linspace(grid_size[n][0], grid_size[n][1], gn),)

    cc = np.meshgrid(*mesh)
    z = np.stack(list(cc))
    field = func(z)
    return cc, field


def add_colorbar(
    figax,
    image,
    cbar_outline=False,
    cbar_ori="vertical",
    cbar_fontsize=12,
    cbar_pad=20,
    ticktitle=None,
    ticks=None,
    ticklabels=None,
    cbar_format=None,
):
    """
    Adds a colorbar object to the axis.

    :param string cbar_format: cbar label formatting, scientific mode '%.1e'
    """
    fig, ax = figax
    if cbar_ori == "vertical":
        rot = 270
    else:
        rot = 0

    cbar = fig.colorbar(image, cax=ax, orientation=cbar_ori, format=cbar_format)
    cbar.outline.set_visible(cbar_outline)

    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    if ticktitle is not None:
        cbar.ax.get_yaxis().labelpad = cbar_pad
        cbar.ax.set_ylabel(ticktitle, rotation=rot, fontsize=cbar_fontsize)

    return cbar


def mesh_plot(figax, mesh_tuple, cmap="gray", vmin=None, vmax=None):
    """
    Create a mesh plot from a mesh tuple given by compute_mesh.

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :returns: figure and axis
    :rtype: tuple
    """
    fig, ax = figax
    xx, yy, field = mesh_tuple
    im = ax.pcolormesh(xx, yy, field, vmin=vmin, vmax=vmax, cmap=cmap)
    return im


def draw_2d(
    figax,
    data,
    vmin=0,
    vmax=1,
    cmap="gray",
    origin="upper",
    aspect="auto",
    extent=None,
    interp_method=None,
):
    """
    Visualize a 2D array using imshow, the first axis of data is the y-axis.

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :returns: figure and axis
    :rtype: tuple
    """
    fig, ax = figax
    im = ax.imshow(
        data,
        cmap=cmap,
        origin=origin,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation=interp_method,
        extent=extent,
    )
    return im


def raster_plot(
    figax, spikes, time_bins, bin_time, units, colors=None, marker="|", markersize=2
):
    """
    Visualize a 2D array representing point events, spikes has shape (timstep, units).

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :returns: figure and axis
    :rtype: tuple
    """
    fig, ax = figax
    if colors is None:
        col = ["k"] * units
    else:
        col = colors
    for i in range(units):
        t = np.nonzero(spikes[i, :])[0] * bin_time
        ax.scatter(t, (i + 1) * np.ones_like(t), c=col[i], s=markersize, marker=marker)

    ax.set_xlim(0, time_bins * bin_time)
    ax.set_ylim(0.1, units + 0.9)
    ax.set_yticks(np.arange(1, units + 1))
    return ax


def grid_draw_2d(data, figsize, nrows, ncols, vmin=0, vmax=1, cmap="gray"):
    """
    Visualize batched image data on a grid.

    :param np.array data: input data of shape ()
    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :returns: figure and axis
    :rtype: tuple
    """
    fig = plt.figure(figsize=figsize)
    axes = [
        fig.add_subplot(nrows, ncols, r * ncols + c + 1)
        for r in range(0, nrows)
        for c in range(0, ncols)
    ]

    chans = data.shape[1]
    if chans == 1:
        data = data.squeeze(1)
    else:
        data = np.transpose(data, (0, 2, 3, 1))
    for k, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        if chans == 1:
            ax.imshow(data[k], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(data[k])

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig, axes


def render_image(fig):
    """
    Render an image from a figure buffer.

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    :returns: figure and axis
    :rtype: tuple
    """
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def generate_gif(images, out_file, fps=10):
    """
    Create a gif from an array of images.

    :param list colors: colors to be included in the colormap
    :param string name: name the colormap
    """
    kwargs_write = {"fps": fps, "quantizer": "nq"}
    imageio.mimsave(out_file, images, fps=fps)


def cmap_arrow(
    ax,
    start,
    end,
    cmap,
    n_segments=50,
    lw=3,
    head_size=10,
    head_width=0.8,
    head_length=0.5,
):
    """ """

    # Arrow shaft: LineCollection
    x = np.linspace(start[0], end[0], n_segments)
    y = np.linspace(start[1], end[1], n_segments)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0, 1, n_segments))
    ax.add_collection(lc)

    # Arrow head: Triangle
    tricoords = [
        (0, -head_width / 2.0),
        (head_length, 0),
        (0, head_width / 2.0),
        (0, -head_width / 2.0),
    ]
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(
        end[0], end[1], c=1, s=(2 * head_size) ** 2, marker=tri, cmap=cmap, vmin=0
    )


def visualize_field(
    figax,
    data,
    grid_extent,
    aspect="equal",
    spike_pos=None,
    cbar=True,
    vmin=0,
    vmax=None,
    cmap="viridis",
    ticktitle="firing rate (Hz)",
    ticks=None,
    ticklabels=None,
    cbar_format=None,
    cax=None,
    cbar_size="5%",
):
    """ """
    fig, ax = figax
    if vmax is None:
        vmax = data.max()

    im = draw_2d(
        figax,
        data,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(
            grid_extent[0][0],
            grid_extent[0][1],
            grid_extent[1][0],
            grid_extent[1][1],
        ),
        aspect=aspect,
    )
    decorate_ax(
        ax,
        xlim=[grid_extent[0][0], grid_extent[0][1]],
        ylim=[grid_extent[1][0], grid_extent[1][1]],
        spines=[False, False, False, False],
    )
    if cbar:
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=cbar_size, pad=0.2)

        add_colorbar(
            (fig, cax),
            im,
            ticktitle=ticktitle,
            ticks=ticks,
            ticklabels=ticklabels,
            cbar_format=cbar_format,
        )

    if spike_pos is not None:
        x_s, y_s = spike_pos
        ax.scatter(x_s, y_s, s=1, color="r", zorder=1, alpha=0.7)
    return im


def plot_dispersion(ax, q_cdf, s_KS, labelx=False, labely=False):
    r"""
    KS-plot for visualizing dispersion structure.

    :param string mode: either `count` or `ISI` version of quantiles
    """
    bins = np.linspace(0.0, 1.0, 11)
    q_order = np.append(np.array([0]), np.sort(q_cdf))

    samples = len(q_cdf)
    bb = np.arange(0, samples + 1) / samples
    (line,) = ax.plot(bb, bb, "r--")
    ax.fill_between(
        bins,
        bins - s_KS * np.ones_like(bins),
        bins + s_KS * np.ones_like(bins),
        color="grey",
        alpha=0.5,
    )
    (line,) = ax.plot(q_order, bb, "b")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    decorate_ax(ax)

    if labelx:
        ax.set_xticks([0, 1])
    if labely:
        ax.set_yticks([0, 1])

    return ax


def plot_circ_posterior(
    ax,
    times,
    wrap_y,
    y_std,
    col="k",
    linewidth=1.0,
    step=1,
    alpha=0.5,
    line_alpha=1.0,
    l=None,
    l_std=None,
):
    """
    Plot circular variables with (approximate) variational uncertainty.
    """
    if y_std is not None:
        upper = wrap_y + y_std
        lower = wrap_y - y_std

    T = len(wrap_y)
    for i in np.arange(T)[1::step]:
        lines = []
        delta = wrap_y[i] - wrap_y[i - 1]
        if delta > np.pi:
            lines.append(
                ax.plot(
                    [times[i - 1], times[i]],
                    [wrap_y[i - 1], wrap_y[i] - 2 * np.pi],
                    color=col,
                    linewidth=linewidth,
                    label=l,
                    alpha=line_alpha,
                )[0]
            )
            lines.append(
                ax.plot(
                    [times[i - 1], times[i]],
                    [wrap_y[i - 1] + 2 * np.pi, wrap_y[i]],
                    color=col,
                    linewidth=linewidth,
                    label=None,
                    alpha=line_alpha,
                )[0]
            )
        elif delta < -np.pi:
            lines.append(
                ax.plot(
                    [times[i - 1], times[i]],
                    [wrap_y[i - 1], wrap_y[i] + 2 * np.pi],
                    color=col,
                    linewidth=linewidth,
                    label=l,
                    alpha=line_alpha,
                )[0]
            )
            lines.append(
                ax.plot(
                    [times[i - 1], times[i]],
                    [wrap_y[i - 1] - 2 * np.pi, wrap_y[i]],
                    color=col,
                    linewidth=linewidth,
                    label=None,
                    alpha=line_alpha,
                )[0]
            )
        else:
            lines.append(
                ax.plot(
                    [times[i - 1], times[i]],
                    [wrap_y[i - 1], wrap_y[i]],
                    color=col,
                    linewidth=linewidth,
                    label=l,
                    alpha=line_alpha,
                )[0]
            )
            if y_std is not None:  # double locations for uncertainty
                if upper[i] > 2 * np.pi or upper[i - 1] > 2 * np.pi:
                    lines.append(
                        ax.plot(
                            [times[i - 1], times[i]],
                            [wrap_y[i - 1] - 2 * np.pi, wrap_y[i] - 2 * np.pi],
                            color=col,
                            linewidth=linewidth,
                            label=None,
                            alpha=line_alpha,
                        )[0]
                    )
                elif lower[i] < 0 or lower[i - 1] < 0:
                    lines.append(
                        ax.plot(
                            [times[i - 1], times[i]],
                            [wrap_y[i - 1] + 2 * np.pi, wrap_y[i] + 2 * np.pi],
                            color=col,
                            linewidth=linewidth,
                            label=None,
                            alpha=line_alpha,
                        )[0]
                    )

        if y_std is not None:
            for line in lines:
                ax.fill_between(
                    [times[i - 1], times[i]],
                    line.get_ydata() - y_std[i - 1 : i + 1],
                    line.get_ydata() + y_std[i - 1 : i + 1],
                    color=line.get_color(),
                    alpha=alpha,
                    label=l_std,
                )
                l_std = None

        l = None  # no label for each segment after first
