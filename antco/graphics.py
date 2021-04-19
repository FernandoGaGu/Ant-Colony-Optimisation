# Module containing the visualisation tools necessary to explore the convergence of the executed
# algorithms
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from .report import Report


def convergence(report: Report, **kwargs):
    """
    Function that displays the convergence using a antco.report.Report object.

    Parameters
    ----------
    report: antco.report.Report
        antco.report.Report instance returned by the antco.run() function.

    **kwargs
        figsize: tuple, default=(8, 5)
            Tuple indicating the size of the figure.

        title: str, default='Convergence'
            Plot title.

        alpha_grid: float, default=0.7
            Transparency of the grid lines of the plot.

        alpha_graph: float, default=0.2
            Transparency of the lines of the plot.

        save_plot: str, default=None
            File in which to save the generated graph, if no value is provided the graph will not
            be saved.

    Returns
    -------
    :matplotlib.pyplot.Fig
        Figure with convergence graph.
    """

    def _draw(ax_, params_: dict, alpha_: float, color_: str, label_: str, linestyle_: str, linewidth_: int):
        x = np.arange(len(params_))
        y = [np.mean(vals) for vals in params_.values()]
        ax_.plot(x, y, color=color_, label=label_, alpha=alpha_, linestyle=linestyle_, linewidth=linewidth_)

        return ax_

    # Check that the parameters necessary to represent convergence can be obtained.
    try:
        report.get('mean_cost')
    except Exception:
        raise Exception(
            'The Report instance does not have the "mean_cost" value, make sure you have saved the "mean_cost" value '
            'throughout the interactions of the algorithm using the method report.save("mean_cost").')
    try:
        report.get('max_cost')
    except Exception:
        raise Exception(
            'The Report instance does not have the "max_cost" value, make sure you have saved the "max_cost" value '
            'throughout the interactions of the algorithm using the method report.save("max_cost").')

    parameters = {
        'mean_cost': {'color': '#85C1E9', 'label': 'Average cost', 'linestyle': 'solid', 'linewidth': 3},
        'max_cost': {'color': '#AF7AC5', 'label': 'Max cost', 'linestyle': 'dashed', 'linewidth': 2}}

    # Get optional arguments                  
    figsize = kwargs.get('figsize', (8, 5))
    title = kwargs.get('title', 'Convergence')
    alpha_graph = kwargs.get('alpha_graph', 0.7)
    alpha_grid = kwargs.get('alpha_grid', 0.2)
    save_plot = kwargs.get('save_plot', None)

    fig, ax = plt.subplots(figsize=figsize)

    for param, values in parameters.items():
        ax = _draw(ax, report.get(param), alpha_graph,
                   values['color'], values['label'], values['linestyle'], values['linewidth'])

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=alpha_grid)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1),
              fancybox=True, shadow=True, ncol=len(parameters))
    ax.set_title(title)

    if save_plot is not None:
        plt.savefig(save_plot, dpi=150)

    return fig


def branchingFactor(report: Report, **kwargs):
    """
    Method representing the evolution of the lambda-branching factor.

    Parameters
    ----------
    report: antco.Report
        antco.report.Report instance returned by the antco.run() function.

    **kwargs
        figsize: tuple, default=(8, 5)
            Tuple indicating the size of the figure.

        title: str, default='Convergence'
            Plot title.

        alpha_grid: float, default=0.7
            Transparency of the grid lines of the plot.

        alpha_graph: float, default=0.2
            Transparency of the lines of the plot.

        color: str, default='viridis'
            Matplotlib colormap used to render the graph.

        show_legend: str, default=True
            Parameter indicating whether to display the legend or not.

        save_plot: str, default=None
            File in which to save the generated graph, if no value is provided the graph will not
            be saved.

    :return: matplotlib.pyplot.Fig
    """

    def _draw(ax_, params_: dict, alpha_: float, color_: str, label_: str):
        x = np.arange(len(params_))
        y = [np.mean(vals) for vals in params_.values()]
        ax_.plot(x, y, color=color_, label=label_, alpha=alpha_)

        return ax_

    assert len(report.values) > 0, 'The Report instance has not registered any value.'

    # Get optional arguments
    figsize = kwargs.get('figsize', (8, 5))
    title = kwargs.get('title', r'$\lambda$-branching factor')
    color = kwargs.get('color', 'viridis')
    alpha_graph = kwargs.get('alpha_graph', 0.7)
    alpha_grid = kwargs.get('alpha_grid', 0.2)
    show_legend = kwargs.get('show_legend', True)
    save_plot = kwargs.get('save_plot', None)

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique keys
    unique_keys = [
        key for key in list(report.values[1].keys()) if 'lambda' in key]

    cmap = plt.cm.get_cmap(color, len(unique_keys))

    for idx, key in enumerate(unique_keys):
        ax = _draw(ax, report.get(key), alpha_graph, cmap(idx), key)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of nodes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=alpha_grid)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if show_legend:
        ax.legend()
    ax.set_title(title)

    if save_plot is not None:
        plt.savefig(save_plot, dpi=150)

    return fig


def pheromoneConvergence(report: Report, output: str = 'PheromoneConvergence.mp4',
                         color: str = 'Oranges'):
    """
    Method that generates a video showing how the values of the pheromone matrix evolve throughout
    the iterations of the algorithm.

    Parameters
    ----------
    report: antco.Report
        antco.report.Report instance returned by the antco.run() function.

    output: str
        Name of the output file under which the video will be saved.

    color: str
        Matplotlib colormap used to represent the pheromone values.
    """
    # Check that the parameters necessary to represent convergence can be obtained.
    try:
        report.get('pheromones')
    except Exception:
        raise Exception(
            'The Report instance does not have the "pheromones" value, make sure you have saved '
            'the "pheromones" value throughout the interactions of the algorithm using the method '
            'report.save("pheromones").')

    pheromone_evolution = report.get('pheromones')

    # Get max and min value
    min_val, max_val = None, None
    for values in pheromone_evolution.values():
        p_min = np.min(values)
        p_max = np.max(values)
        if min_val is None or p_min < min_val:
            min_val = p_min
        if max_val is None or p_max > max_val:
            max_val = p_max

    cmap = plt.cm.get_cmap(color)

    fig, ax = plt.subplots()

    ims = []
    for it, values in pheromone_evolution.items():
        im = ax.imshow(values, vmin=min_val, vmax=max_val, cmap=cmap, animated=True)
        if it == 1:  # Initial frame
            ax.imshow(values, vmin=min_val, vmax=max_val, cmap=cmap)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    # Save animation
    ani.save(output)

    plt.show()
