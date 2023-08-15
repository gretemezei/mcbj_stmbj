import matplotlib.pyplot as plt
from matplotlib import cm, colormaps, cycler, rcParams, gridspec, ticker
import matplotlib.axes
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Union, Tuple, List, Optional
from tqdm import tqdm

from mcbj import TracePair, HoldTrace
import utils


def plot_measurement_scheme(trace_pair: TracePair, hold_trace: HoldTrace, home_folder: Path,
                            main_colors=('cornflowerblue', 'indianred'),
                            accent_colors=('royalblue', 'firebrick'),
                            smoothing: int = 1,
                            save_fig: bool = False,
                            xlim: Optional[Tuple[float, float]] = None):

    fig = plt.figure(figsize=utils.cm2inch(16, 5), dpi=600)  # figsize: (width, height) in inches
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=(0.35, 0.65), height_ratios=(1, 1),
                           figure=fig, left=0.2, right=0.95, top=0.95, bottom=0.15, wspace=0.2, hspace=0)

    ax_trace = fig.add_subplot(gs[:, 0])
    ax_pull = fig.add_subplot(gs[0, 1])
    ax_push = fig.add_subplot(gs[1, 1])

    ax_trace.xaxis.set_ticks_position('both')
    ax_trace.yaxis.set_ticks_position('both')

    ax_pull.xaxis.tick_top()
    ax_pull.xaxis.set_label_position('top')
    ax_pull.xaxis.set_ticks_position('both')
    ax_push.xaxis.set_ticks_position('both')

    ax_trace = trace_pair.plot_trace_pair(ax=ax_trace, xlim=xlim,
                                          main_colors=main_colors,
                                          accent_colors=accent_colors,
                                          smoothing=smoothing,
                                          plot_trigger=True)

    text_pos_pull = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1]-ax_trace.get_xlim()[0]) * 0.05,
                     1.5*trace_pair.hold_set_pull)
    text_pos_push = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1]-ax_trace.get_xlim()[0]) * 0.05,
                     0.8*trace_pair.hold_set_push)

    ax_trace.text(text_pos_pull[0], text_pos_pull[1], 'pull trigger', fontsize='xx-small', c=accent_colors[0])
    ax_trace.text(text_pos_push[0], text_pos_push[1], 'push trigger', fontsize='xx-small', c=accent_colors[1],
                  va='top')

    ax_pull, par_pull = hold_trace.plot_hold_traces(plot_step_ranges=False, plot_psd_intervals=True, conductance=True,
                                                    ax=ax_pull,
                                                    ax_colors=accent_colors,
                                                    smoothing=smoothing,
                                                    direction='pull')
    ax_push, par_push = hold_trace.plot_hold_traces(plot_step_ranges=False, plot_psd_intervals=True, conductance=True,
                                                    ax=ax_push,
                                                    ax_colors=accent_colors,
                                                    smoothing=smoothing,
                                                    direction='push')

    ax_pull.set_ylim(5e-5, 0.05)
    ax_pull.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_push.set_ylim(5e-5, 0.05)
    ax_push.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    return ax_trace, ax_pull, ax_push

    # if save_fig:
    #     plt.savefig(home_folder.joinpath(f'results/measurement_scheme_{trace_pair.trace_num}.png'), bbox_inches='tight')
    # else:
    #     plt.show()


def plot_measurement_scheme_combed(trace_pair: TracePair, hold_trace: HoldTrace, home_folder: Path, smoothing: int = 1,
                                   direction: str = 'pull',
                                   main_colors: Tuple[str, str] = ('cornflowerblue', 'indianred'),
                                   accent_colors: Tuple[str, str] = ('royalblue', 'firebrick'),
                                   ylim: Tuple[float, float] = (1e-6, 10),
                                   add_vlines_for_hold: bool = False,
                                   fig_size: Tuple[float, float] = utils.cm2inch(16, 5),
                                   save_fig: bool = False,
                                   to_axes: Optional[Tuple[matplotlib.axes.Axes,
                                                           matplotlib.axes.Axes]] = None):
    """
    Plot combed measurement scheme where the break junction trace and the corresponding hold measurement are plotted
    together as a function of time. Also add piezo movement!
    Parameters
    ----------
    trace_pair : TracePair
        trace pair to plot
    hold_trace : HoldTrace
        hold trace instance to plot
    home_folder : Path
        path to measured data
    smoothing : int, default: 1 (i.e. no smoothing)
        window size for moving average, determines the amount of smoothing applied to the trace
    direction : str, default: 'pull'
        which direction trace to plot
    main_colors : Tuple[str, str], default: ('cornflowerblue', 'indianred')
        main colors are used for the break junction parts of the trace.
        1st element is for pull traces, 2nd is for push traces
    accent_colors : Tuple[str, str], default: ('royalblue', 'firebrick')
    ylim : Tuple[float, float], default: (1e-6, 100)
    add_vlines_for_hold : bool, default: False
    fig_size: Tuple[float, float]

    save_fig : bool
        if True, the plotted figure is saved as 'measurement_scheme_combed_{trace_pair.trace_num}.png'
        in the results folder of the measurement
    to_axes : Tuple(matplotlib.axes.Axes, matplotlib.axes.Axes)
    Returns
    -------

    """
    # trace_pair = TracePair(trace_num, home_folder)
    # hold_trace = HoldTrace(trace_num, load_from=home_folder, bias_offset=0)

    if to_axes is None:
        fig = plt.figure(figsize=fig_size, dpi=600)  # figsize: (width, height) in inches
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(1, 2),
                               figure=fig, left=0.2, right=0.95, top=0.95, bottom=0.15, wspace=0.2, hspace=0)

        ax_piezo = fig.add_subplot(gs[0])
        ax_conductance = fig.add_subplot(gs[1])
    else:
        ax_piezo, ax_conductance = to_axes

    if direction == 'pull':
        cond_before = trace_pair.conductance_pull[np.nonzero(trace_pair.conductance_pull < 10)[0][0]:
                                                  trace_pair.hold_index_pull]
        cond_hold = hold_trace.hold_conductance_pull
        cond_after = trace_pair.conductance_pull[trace_pair.hold_index_pull:
                                                 np.nonzero(trace_pair.conductance_pull < 1e-5)[0][0] + 10_000]

        piezo_before = trace_pair.piezo_pull[np.nonzero(trace_pair.conductance_pull < 10)[0][0]:
                                             trace_pair.hold_index_pull]
        piezo_hold = np.ones_like(cond_hold) * piezo_before[-1]
        piezo_after = trace_pair.piezo_pull[trace_pair.hold_index_pull:
                                            np.nonzero(trace_pair.conductance_pull < 1e-5)[0][0] + 10_000]
        color_num = 0
    elif direction == 'push':
        cond_before = trace_pair.conductance_push[np.nonzero(trace_pair.conductance_push < 1e-5)[0][-1] - 10_000:
                                                  trace_pair.hold_index_push]
        cond_hold = hold_trace.hold_conductance_push
        cond_after = trace_pair.conductance_push[trace_pair.hold_index_push:
                                                 np.nonzero(trace_pair.conductance_push > 10)[0][0]]

        piezo_before = trace_pair.piezo_push[np.nonzero(trace_pair.conductance_push < 1e-5)[0][-1] - 10_000:
                                             trace_pair.hold_index_push]
        piezo_hold = np.ones_like(cond_hold) * piezo_before[-1]
        piezo_after = trace_pair.piezo_push[trace_pair.hold_index_push:
                                            np.nonzero(trace_pair.conductance_push > 10)[0][0]]
        color_num = 1
    else:
        raise ValueError(f'{direction} is an invalid parameter for direction. Use "pull", or "push"')

    time_before = np.arange(start=0, stop=len(cond_before), step=1)
    time_hold = np.arange(start=time_before[-1] + 1, stop=time_before[-1] + len(cond_hold) + 1, step=1)
    time_after = np.arange(start=time_hold[-1] + 1, stop=time_hold[-1] + len(cond_after) + 1, step=1)

    time_before = time_before / trace_pair.sample_rate
    time_hold = time_hold / trace_pair.sample_rate
    time_after = time_after / trace_pair.sample_rate

    # concatenate individual parts
    time_total = np.concatenate((time_before, time_hold, time_after))
    cond_total = np.concatenate((cond_before, cond_hold, cond_after))
    piezo_total = np.concatenate((piezo_before, piezo_hold, piezo_after))

    dividers = (len(time_before),                   # only need 2 dividers technically
                len(time_before) + len(time_hold))  # (the 3rd would be the last point)

    time_total_avg = utils.moving_average(time_total, smoothing)
    cond_total_avg = utils.moving_average(cond_total, smoothing)
    piezo_total_avg = utils.moving_average(piezo_total, smoothing)

    # plot the piezo
    ax_piezo.plot(time_total_avg[time_total_avg < time_total[dividers[0]]],
                  piezo_total_avg[time_total_avg < time_total[dividers[0]]], c=main_colors[color_num], lw=0.5)

    ax_piezo.plot(time_total_avg[
                   np.bitwise_and(time_total_avg > time_total[dividers[0]], time_total_avg < time_total[dividers[1]])],
                  piezo_total_avg[
                   np.bitwise_and(time_total_avg > time_total[dividers[0]], time_total_avg < time_total[dividers[1]])],
                  c=accent_colors[color_num], lw=0.5)

    ax_piezo.plot(time_total_avg[time_total_avg > time_total[dividers[1]]],
                  piezo_total_avg[time_total_avg > time_total[dividers[1]]], c=main_colors[color_num], lw=0.5)

    # plot the conductance
    ax_conductance.plot(time_total_avg[time_total_avg < time_total[dividers[0]]],
                        cond_total_avg[time_total_avg < time_total[dividers[0]]], c=main_colors[color_num], lw=0.5)

    ax_conductance.plot(time_total_avg[
                np.bitwise_and(time_total_avg > time_total[dividers[0]], time_total_avg < time_total[dividers[1]])],
            cond_total_avg[
                np.bitwise_and(time_total_avg > time_total[dividers[0]], time_total_avg < time_total[dividers[1]])],
            c=accent_colors[color_num], lw=0.5)

    ax_conductance.plot(time_total_avg[time_total_avg > time_total[dividers[1]]],
                        cond_total_avg[time_total_avg > time_total[dividers[1]]], c=main_colors[color_num], lw=0.5)
    ax_conductance.set_yscale('log')
    ax_conductance.set_ylim(ylim)

    ax_conductance.set_xlabel('Time [s]')
    ax_conductance.set_ylabel(r'Conductance [$G_{0}$]')
    ax_piezo.set_ylabel('Piezo [V]')
    ax_piezo.set_xlabel('Time [s]')

    ax_conductance.xaxis.set_ticks_position('both')
    ax_conductance.yaxis.set_ticks_position('both')

    ax_piezo.xaxis.tick_top()
    ax_piezo.xaxis.set_label_position('top')
    ax_piezo.xaxis.set_ticks_position('both')
    ax_piezo.yaxis.set_ticks_position('both')

    if add_vlines_for_hold:
        ax_conductance.axvline(time_total[dividers[0]], ls='--', lw=0.8, alpha=0.3, c='grey')
        ax_conductance.axvline(time_total[dividers[1]], ls='--', lw=0.8, alpha=0.3, c='grey')
        ax_piezo.axvline(time_total[dividers[0]], ls='--', lw=0.8, alpha=0.3, c='grey')
        ax_piezo.axvline(time_total[dividers[1]], ls='--', lw=0.8, alpha=0.3, c='grey')

    ax_conductance.set_xlim(time_total[0], time_total[-1])
    ax_piezo.set_xlim(ax_conductance.get_xlim())

    if save_fig:
        plt.savefig(home_folder.joinpath(f'results/measurement_scheme_combed_{trace_pair.trace_num}.png'),
                    bbox_inches='tight')

    return ax_piezo, ax_conductance


def plot_measurement_scheme_combed_both(trace_pair: TracePair, hold_trace: HoldTrace, home_folder: Path,
                                        smoothing: int = 1,
                                        main_colors: Tuple[str, str] = ('cornflowerblue', 'indianred'),
                                        accent_colors: Tuple[str, str] = ('royalblue', 'firebrick'),
                                        ylim: Tuple[float, float] = (1e-6, 10),
                                        add_vlines_for_hold: bool = False,
                                        add_points: bool = False,
                                        fig_size: Tuple[float, float] = utils.cm2inch(16, 5),
                                        save_fig: bool = False,
                                        time_text_pos: Optional[float] = None,
                                        to_axes: Optional[Tuple[matplotlib.axes.Axes,
                                                                matplotlib.axes.Axes,
                                                                matplotlib.axes.Axes,
                                                                matplotlib.axes.Axes]] = None):
    """
    Plot combed measurement scheme where the break junction trace and the corresponding hold measurement are plotted
    together as a function of time. Also add piezo movement!
    Parameters
    ----------
    trace_pair : TracePair
        trace pair to plot
    hold_trace : HoldTrace
        hold trace instance to plot
    home_folder : Path
        path to measured data
    smoothing : int, default: 1 (i.e. no smoothing)
        window size for moving average, determines the amount of smoothing applied to the trace
    main_colors : Tuple[str, str], default: ('cornflowerblue', 'indianred')
        main colors are used for the break junction parts of the trace.
        1st element is for pull traces, 2nd is for push traces
    accent_colors : Tuple[str, str], default: ('royalblue', 'firebrick')
    ylim : Tuple[float, float], default: (1e-6, 100)

    add_vlines_for_hold : bool, default: False
        if True plot verticel lines at beginning and end of hold interval
    add_points : bool, default: False
        if True plot the point where trigger level is reached and
        where G_avg is calculated at the end of the hold interval
    fig_size: Tuple[float, float]

    save_fig : bool
        if True, the plotted figure is saved as 'measurement_scheme_combed_{trace_pair.trace_num}.png'
        in the results folder of the measurement
    time_text_pos : Optional[float]
        vertical position of the text Time [s]. Use this to set it if it's way off
    to_axes : Tuple(matplotlib.axes.Axes, matplotlib.axes.Axes)
    Returns
    -------

    """

    if to_axes is None:
        fig = plt.figure(figsize=fig_size, dpi=600)  # figsize: (width, height) in inches
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=(1, 2),
                               figure=fig, left=0.2, right=0.95, top=0.95, bottom=0.15, wspace=0, hspace=0)

        ax_piezo_pull = fig.add_subplot(gs[0, 0])
        ax_piezo_push = fig.add_subplot(gs[0, 1], sharey=ax_piezo_pull)
        ax_cond_pull = fig.add_subplot(gs[1, 0], sharex=ax_piezo_pull)
        ax_cond_push = fig.add_subplot(gs[1, 1], sharey=ax_cond_pull, sharex=ax_piezo_push)
    else:
        ax_piezo_pull, ax_cond_pull, ax_piezo_push, ax_cond_push = to_axes

    # PULL
    cond_before_pull = trace_pair.conductance_pull[np.nonzero(trace_pair.conductance_pull > 10)[0][-1]:
                                                   trace_pair.hold_index_pull]
    cond_hold_pull = hold_trace.hold_conductance_pull
    cond_after_pull = trace_pair.conductance_pull[trace_pair.hold_index_pull:
                                                  np.nonzero(trace_pair.conductance_pull < 1e-5)[0][0] + 10_000]

    piezo_before_pull = trace_pair.piezo_pull[np.nonzero(trace_pair.conductance_pull > 10)[0][-1]:
                                              trace_pair.hold_index_pull]
    piezo_hold_pull = np.ones_like(cond_hold_pull) * piezo_before_pull[-1]
    piezo_after_pull = trace_pair.piezo_pull[trace_pair.hold_index_pull:
                                             np.nonzero(trace_pair.conductance_pull < 1e-5)[0][0] + 10_000]
    color_num = 0
    # PUSH
    cond_before_push = trace_pair.conductance_push[np.nonzero(trace_pair.piezo_push > piezo_after_pull[-1])[0][-1]:
                                                   trace_pair.hold_index_push]
    cond_hold_push = hold_trace.hold_conductance_push
    cond_after_push = trace_pair.conductance_push[trace_pair.hold_index_push:
                                                  np.nonzero(trace_pair.conductance_push < 10)[0][-1]]

    piezo_before_push = trace_pair.piezo_push[np.nonzero(trace_pair.piezo_push > piezo_after_pull[-1])[0][-1]:
                                              trace_pair.hold_index_push]
    piezo_hold_push = np.ones_like(cond_hold_push) * piezo_before_push[-1]
    piezo_after_push = trace_pair.piezo_push[trace_pair.hold_index_push:
                                             np.nonzero(trace_pair.conductance_push < 10)[0][-1]]
    color_num = 1

    time_before_pull = np.arange(start=0, stop=len(cond_before_pull), step=1)
    time_hold_pull = np.arange(start=time_before_pull[-1] + 1, stop=time_before_pull[-1] + len(cond_hold_pull) + 1,
                               step=1)
    time_after_pull = np.arange(start=time_hold_pull[-1] + 1, stop=time_hold_pull[-1] + len(cond_after_pull) + 1,
                                step=1)

    time_before_push = np.arange(start=time_after_pull[-1], stop=time_after_pull[-1] + len(cond_before_push), step=1)
    time_hold_push = np.arange(start=time_before_push[-1] + 1, stop=time_before_push[-1] + len(cond_hold_push) + 1,
                               step=1)
    time_after_push = np.arange(start=time_hold_push[-1] + 1, stop=time_hold_push[-1] + len(cond_after_push) + 1,
                                step=1)

    time_before_pull = time_before_pull / trace_pair.sample_rate
    time_hold_pull = time_hold_pull / trace_pair.sample_rate
    time_after_pull = time_after_pull / trace_pair.sample_rate

    time_before_push = time_before_push / trace_pair.sample_rate
    time_hold_push = time_hold_push / trace_pair.sample_rate
    time_after_push = time_after_push / trace_pair.sample_rate

    # concatenate individual parts
    time_total_pull = np.concatenate((time_before_pull, time_hold_pull, time_after_pull))
    cond_total_pull = np.concatenate((cond_before_pull, cond_hold_pull, cond_after_pull))
    piezo_total_pull = np.concatenate((piezo_before_pull, piezo_hold_pull, piezo_after_pull))

    time_total_push = np.concatenate((time_before_push, time_hold_push, time_after_push))
    cond_total_push = np.concatenate((cond_before_push, cond_hold_push, cond_after_push))
    piezo_total_push = np.concatenate((piezo_before_push, piezo_hold_push, piezo_after_push))

    dividers_pull = (len(time_before_pull),  # only need 2 dividers technically
                     len(time_before_pull) + len(time_hold_pull))  # (the 3rd would be the last point)

    dividers_push = (len(time_before_push),  # only need 2 dividers technically
                     len(time_before_push) + len(time_hold_push))  # (the 3rd would be the last point)

    time_total_avg_pull = utils.moving_average(time_total_pull, smoothing)
    cond_total_avg_pull = utils.moving_average(cond_total_pull, smoothing)
    piezo_total_avg_pull = utils.moving_average(piezo_total_pull, smoothing)

    time_total_avg_push = utils.moving_average(time_total_push, smoothing)
    cond_total_avg_push = utils.moving_average(cond_total_push, smoothing)
    piezo_total_avg_push = utils.moving_average(piezo_total_push, smoothing)

    # plot the piezo
    # PULL
    ax_piezo_pull.plot(time_total_avg_pull[time_total_avg_pull < time_total_pull[dividers_pull[0]]],
                       piezo_total_avg_pull[time_total_avg_pull < time_total_pull[dividers_pull[0]]], c=main_colors[0],
                       lw=0.5)

    ax_piezo_pull.plot(time_total_avg_pull[
                           np.bitwise_and(time_total_avg_pull > time_total_pull[dividers_pull[0]],
                                          time_total_avg_pull < time_total_pull[dividers_pull[1]])],
                       piezo_total_avg_pull[
                           np.bitwise_and(time_total_avg_pull > time_total_pull[dividers_pull[0]],
                                          time_total_avg_pull < time_total_pull[dividers_pull[1]])],
                       c=accent_colors[0], lw=0.5)

    ax_piezo_pull.plot(time_total_avg_pull[time_total_avg_pull > time_total_pull[dividers_pull[1]]],
                       piezo_total_avg_pull[time_total_avg_pull > time_total_pull[dividers_pull[1]]], c=main_colors[0],
                       lw=0.5)
    # PUSH
    ax_piezo_push.plot(time_total_avg_push[time_total_avg_push < time_total_push[dividers_push[0]]],
                       piezo_total_avg_push[time_total_avg_push < time_total_push[dividers_push[0]]], c=main_colors[1],
                       lw=0.5)

    ax_piezo_push.plot(time_total_avg_push[
                           np.bitwise_and(time_total_avg_push > time_total_push[dividers_push[0]],
                                          time_total_avg_push < time_total_push[dividers_push[1]])],
                       piezo_total_avg_push[
                           np.bitwise_and(time_total_avg_push > time_total_push[dividers_push[0]],
                                          time_total_avg_push < time_total_push[dividers_push[1]])],
                       c=accent_colors[1], lw=0.5)

    ax_piezo_push.plot(time_total_avg_push[time_total_avg_push > time_total_push[dividers_push[1]]],
                       piezo_total_avg_push[time_total_avg_push > time_total_push[dividers_push[1]]], c=main_colors[1],
                       lw=0.5)

    # plot the conductance
    # before hold
    ax_cond_pull.plot(time_total_avg_pull[time_total_avg_pull < time_total_pull[dividers_pull[0]]],
                      cond_total_avg_pull[time_total_avg_pull < time_total_pull[dividers_pull[0]]], c=main_colors[0],
                      lw=0.5)
    # hold
    ax_cond_pull.plot(time_total_avg_pull[
                          np.bitwise_and(time_total_avg_pull > time_total_pull[dividers_pull[0]],
                                         time_total_avg_pull < time_total_pull[dividers_pull[1]])],
                      cond_total_avg_pull[
                          np.bitwise_and(time_total_avg_pull > time_total_pull[dividers_pull[0]],
                                         time_total_avg_pull < time_total_pull[dividers_pull[1]])],
                      c=accent_colors[0], lw=0.5)
    # after hold
    ax_cond_pull.plot(time_total_avg_pull[time_total_avg_pull > time_total_pull[dividers_pull[1]]],
                      cond_total_avg_pull[time_total_avg_pull > time_total_pull[dividers_pull[1]]], c=main_colors[0],
                      lw=0.5)
    ax_cond_pull.set_yscale('log')
    ax_cond_pull.set_ylim(ylim)

    ax_cond_push.plot(time_total_avg_push[time_total_avg_push < time_total_push[dividers_push[0]]],
                      cond_total_avg_push[time_total_avg_push < time_total_push[dividers_push[0]]], c=main_colors[1],
                      lw=0.5)

    ax_cond_push.plot(time_total_avg_push[
                          np.bitwise_and(time_total_avg_push > time_total_push[dividers_push[0]],
                                         time_total_avg_push < time_total_push[dividers_push[1]])],
                      cond_total_avg_push[
                          np.bitwise_and(time_total_avg_push > time_total_push[dividers_push[0]],
                                         time_total_avg_push < time_total_push[dividers_push[1]])],
                      c=accent_colors[1], lw=0.5)

    ax_cond_push.plot(time_total_avg_push[time_total_avg_push > time_total_push[dividers_push[1]]],
                      cond_total_avg_push[time_total_avg_push > time_total_push[dividers_push[1]]], c=main_colors[1],
                      lw=0.5)
    ax_cond_push.set_yscale('log')
    ax_cond_push.set_ylim(ylim)

    ax_piezo_pull.yaxis.set_ticks_position('both')

    ax_piezo_push.xaxis.set_label_position('top')
    ax_piezo_push.xaxis.set_label_position('top')
    ax_piezo_pull.xaxis.tick_top()
    ax_piezo_push.xaxis.tick_top()
    ax_piezo_pull.xaxis.set_ticks_position('both')
    ax_piezo_push.xaxis.set_ticks_position('both')

    ax_cond_pull.yaxis.set_ticks_position('both')

    ax_piezo_push.yaxis.set_label_position('right')
    ax_piezo_push.yaxis.tick_right()
    ax_piezo_push.yaxis.set_ticks_position('both')

    ax_cond_push.yaxis.set_label_position('right')
    ax_cond_push.yaxis.tick_right()
    ax_cond_push.yaxis.set_ticks_position('both')

    ax_piezo_pull.set_xticks(np.arange(min(time_total_pull), max(time_total_pull), 1), minor=False)
    ax_piezo_push.set_xticks(np.arange(np.around(min(time_total_push), decimals=0), max(time_total_push), 1), minor=False)

    ax_piezo_pull.set_xticks(np.arange(min(time_total_pull), max(time_total_pull), 0.25), minor=True, alpha=0.5)
    ax_piezo_push.set_xticks(np.arange(np.around(min(time_total_push), decimals=0),
                                       max(time_total_push), 0.25), minor=True, alpha=0.5)

    ax_cond_pull.set_ylabel(r'G [$G_{0}$]')
    ax_cond_push.set_ylabel(r'G [$G_{0}$]')

    ax_piezo_pull.set_ylabel(r'Piezo [V]')
    ax_piezo_push.set_ylabel(r'Piezo [V]')

    ax_cond_pull.set_ylim(1e-6, 30)
    # ax_push_hist.set_ylim(0)

    ax_cond_pull.tick_params(axis='y', which='minor', left=False, right=False)
    ax_cond_push.tick_params(axis='y', which='minor', left=False, right=False)

    ax_piezo_pull.set_xlim(time_total_pull[0], time_total_pull[-1])
    ax_piezo_push.set_xlim(time_total_push[0], time_total_push[-1])

    # ax_cond_pull.text(time_after_pull[-1], 1e-7, 'Time [s]', va='top', ha='center', size=8)

    if time_text_pos is None:
        time_text_pos = ax_piezo_pull.get_ylim()[1] + 0.2 * (ax_piezo_pull.get_ylim()[1] - ax_piezo_pull.get_ylim()[0])

    ax_piezo_pull.text(x=time_after_pull[-1], y=time_text_pos,
                       s='Time [s]', va='bottom', ha='center',
                       size=rcParams['axes.labelsize'])

    if save_fig:
        plt.savefig(home_folder.joinpath(f'results/measurement_scheme_combed_{trace_pair.trace_num}_both.png'),
                    bbox_inches='tight')

    return ax_piezo_pull, ax_cond_pull, ax_piezo_push, ax_cond_push


def plot_ivs_scheme_one(trace_pair: TracePair, hold_trace: HoldTrace,
                        direction: str = 'push',
                        main_colors: Tuple[str, str] = ('cornflowerblue', 'indianred'),
                        accent_colors: Tuple[str, str] = ('royalblue', 'firebrick'),
                        vline_color: str = 'grey',
                        color_list: Optional[List] = None,
                        smoothing: int = 1,
                        iv_num_xticks: int = 5,
                        which_psds: Optional[List[int]] = None,
                        fig_size: Optional[Tuple[float, float]] = None):

    """
    Important: run hold_trace.analyse before plotting!
    Parameters
    ----------
    trace_pair : TracePair
        which trace to plot
    hold_trace : HoldTrace
        which hold trace to plot
    direction: str, default: 'push'
        which direction trace to plot
    main_colors : Tuple[str, str], default: ('cornflowerblue', 'indianred')
        main colors of the plots
    accent_colors : Tuple[str, str], default: ('royalblue', 'firebrick')
        accent colors used to emphasize or separate parts of plots
    vline_color: str
        color of the vertical lines
    color_list : Optional[List], default: None
    smoothing : int, default 1
        Amount of smoothing, the window size for moving average calculation, ie. 1 means no smoothing,
        and the greater this number the more smooth the plots become
    iv_num_xticks : int, default: 5
        number of ticks along the horizontal axis for the IV plot
    which_psds : Optional[List[int]], default: None
        use it to plot the PSDs only for the selected bias plateaus. If None, the PSDs of all bias plateaus are plotted.
    fig_size : Tuple[float, float]
        enter figsize manually, in units of inches. If not provided, the default figsize is applied: (15, 8.4) cm
    Returns
    -------

    """

    if fig_size is None:
        fig_size = utils.cm2inch(15, 8.4)
    fig = plt.figure(figsize=fig_size, dpi=600)  # figsize: (width, height) in inches

    gs_total = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(1, 1),
                                 figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.15)

    gs_top = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=(1, 2),
                                              subplot_spec=gs_total[0],
                                              wspace=0.25, hspace=0)

    gs_bottom = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=(1, 1),
                                                 subplot_spec=gs_total[1],
                                                 wspace=0.1, hspace=0)


    ax_trace = fig.add_subplot(gs_top[0])
    ax_hold = fig.add_subplot(gs_top[1])

    ax_iv = fig.add_subplot(gs_bottom[0])
    ax_psd = fig.add_subplot(gs_bottom[1])

    ax_trace.xaxis.set_label_position('top')
    ax_trace.xaxis.tick_top()
    ax_trace.xaxis.set_ticks_position('both')
    ax_trace.yaxis.set_ticks_position('both')

    ax_hold.xaxis.set_ticks_position('both')
    ax_hold.xaxis.set_label_position('top')
    ax_hold.xaxis.tick_top()
    # ax_pull.yaxis.set_ticks_position('both')
    # ax_pull.yaxis.set_label_position('right')
    # ax_pull.yaxis.tick_right()

    # ax_push.yaxis.set_ticks_position('both')
    # ax_push.yaxis.set_label_position('right')
    # ax_push.yaxis.tick_right()

    ax_iv.xaxis.set_ticks_position('both')
    ax_iv.yaxis.set_ticks_position('both')

    ax_psd.xaxis.set_ticks_position('both')
    ax_psd.yaxis.tick_right()
    ax_psd.yaxis.set_ticks_position('both')
    ax_psd.yaxis.set_label_position('right')

    ax_trace = trace_pair.plot_trace_pair(ax=ax_trace, xlim=None,
                                          main_colors=main_colors,
                                          accent_colors=accent_colors,
                                          smoothing=smoothing, plot_trigger=True)

    popt, perr = trace_pair.fit_tunnel()

    ax_trace.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
    ax_trace.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))

    # text_pos_pull = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
    #                  1.5 * trace_pair.hold_set_pull)
    # text_pos_push = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
    #                  0.8 * trace_pair.hold_set_push)
    #
    # ax_trace.text(text_pos_pull[0], text_pos_pull[1], 'pull trigger', fontsize='xx-small', c=accent_colors[0])
    # ax_trace.text(text_pos_push[0], text_pos_push[1], 'push trigger', fontsize='xx-small', c=accent_colors[1],
    #               va='top')

    # text_pos_pull_1 = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
    #                    1.5 * trace_pair.hold_set_pull)
    # text_pos_pull_2 = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
    #                    0.8 * trace_pair.hold_set_pull)
    # text_pos_push_1 = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
    #                    1.5 * trace_pair.hold_set_push)
    # text_pos_push_2 = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
    #                    0.8 * trace_pair.hold_set_push)
    #
    # ax_trace.text(text_pos_pull_1[0], text_pos_pull_1[1], 'pull', fontsize=5, c=accent_colors[0])
    # ax_trace.text(text_pos_pull_2[0], text_pos_pull_2[1], 'trigger', fontsize=5, c=accent_colors[0],
    #               va='top')
    # ax_trace.text(text_pos_push_1[0], text_pos_push_1[1], 'push', fontsize=5, c=accent_colors[1])
    # ax_trace.text(text_pos_push_2[0], text_pos_push_2[1], 'trigger', fontsize=5, c=accent_colors[1],
    #               va='top')

    # ax_trace.set_ylim(1e-6, 2)

    # I(V)

    if direction == 'pull':
        iv_bias = hold_trace.iv_bias_pull
        iv_current = hold_trace.iv_current_pull
        if color_list is None:
            cols = utils.blues
        else:
            cols = color_list
        pull = True
        if which_psds is None:
            psd_intervals = hold_trace.psd_interval_ranges_pull
        else:
            psd_intervals = hold_trace.psd_interval_ranges_pull[which_psds]
        col_index = 0
    elif direction == 'push':
        iv_bias = hold_trace.iv_bias_push
        iv_current = hold_trace.iv_current_push
        if color_list is None:
            cols = utils.reds
        else:
            cols = color_list
        pull = False
        if which_psds is None:
            psd_intervals = hold_trace.psd_interval_ranges_push
        else:
            psd_intervals = hold_trace.psd_interval_ranges_push[which_psds]
        col_index = 1
    else:
        raise ValueError

    ax_iv.plot(utils.moving_average(iv_bias, smoothing),
               utils.moving_average(iv_current, smoothing),
               c=accent_colors[col_index], lw=0.4)

    ax_iv.set_xlabel('Bias [V]')
    ax_iv.set_ylabel('Current [A]')

    max_curr = max(abs(iv_current))

    ax_iv.set_ylim(-1 * max_curr, max_curr)
    ax_iv.set_yticks(np.linspace(-1 * np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)),
                                 np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)), num=5)
                     * 10 ** utils.get_exponent(max_curr))
    ax_iv.set_yticklabels(np.linspace(-1 * np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)),
                                      np.ceil(max_curr / 10 ** utils.get_exponent(max_curr)), num=5))
    ax_iv.set_xticks(np.linspace(np.round(min(hold_trace.iv_bias_push)), np.round(max(hold_trace.iv_bias_push)),
                                 num=iv_num_xticks))

    ax_hold, par_hold = hold_trace.plot_hold_traces(direction=direction,
                                                    plot_step_ranges=False,
                                                    plot_psd_intervals=False,
                                                    conductance=False,
                                                    ax=ax_hold, log_scale_y=False,
                                                    ax_colors=accent_colors,
                                                    smoothing=smoothing)

    ax_hold.set_ylim(-1 * max_curr, max_curr)
    ax_hold.set_yticks(np.linspace(-1*np.ceil(max_curr/10**utils.get_exponent(max_curr)),
                                   np.ceil(max_curr/10**utils.get_exponent(max_curr)), num=5)
                       * 10**utils.get_exponent(max_curr))
    ax_hold.set_yticklabels(np.linspace(-1*np.ceil(max_curr/10**utils.get_exponent(max_curr)),
                                        np.ceil(max_curr/10**utils.get_exponent(max_curr)), num=5))

    if utils.get_exponent(max_curr) == -6:
        ax_hold.set_ylabel(r'Current [$10^{-6}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-6}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -7:
        ax_hold.set_ylabel(r'Current [$10^{-7}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-7}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -8:
        ax_hold.set_ylabel(r'Current [$10^{-8}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-8}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -9:
        ax_hold.set_ylabel(r'Current [$10^{-9}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-9}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -10:
        ax_hold.set_ylabel(r'Current [$10^{-10}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-10}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -5:
        ax_hold.set_ylabel(r'Current [$10^{-5}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-5}\;\mathrm{A}$]', labelpad=0.5)
    elif utils.get_exponent(max_curr) == -4:
        ax_hold.set_ylabel(r'Current [$10^{-4}\;\mathrm{A}$]', labelpad=0.5)
        ax_iv.set_ylabel(r'Current [$10^{-4}\;\mathrm{A}$]', labelpad=0.5)
    else:
        raise UserWarning(f'No axis label defined for this case. Refer to plots.py line 681, to add a label'
                          f'for the case when utils.get_exponent(max_curr) = {utils.get_exponent(max_curr)}')

    ax_hold.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_hold.set_xlabel('Time [s]')
    # ax_hold.set_xticklabels([''] * len(ax_hold.get_xticks()))

    bias_vals = np.concatenate((hold_trace.bias_steps, np.array([max(iv_bias)])))

    # par_hold.set_yticks(bias_vals)
    # par_hold.set_yticklabels(list(map(str, np.around(bias_vals, decimals=1))))

    par_hold.set_yticks(np.linspace(np.round(min(hold_trace.iv_bias_push)), np.round(max(hold_trace.iv_bias_push)),
                        num=iv_num_xticks))
    par_hold.set_yticklabels(np.linspace(np.round(min(hold_trace.iv_bias_push)), np.round(max(hold_trace.iv_bias_push)),
                                         num=iv_num_xticks))
    par_hold.set_ylabel('Bias [V]')

    for i, interval in enumerate(psd_intervals):
        ax_hold.axvspan(interval[0] / 50_000, interval[-1] / 50_000, color=cols[i], ec=None, alpha=0.5)

        for j in interval:
            ax_hold.axvline(j / 50_000, ls='--', c=vline_color, lw=0.6)

    ax_psd = hold_trace.plot_psds(ax=ax_psd, pull=pull, plot_legend=False, which_psds=which_psds, plot_guides=False,
                                  color_list=color_list)

    return fig, ax_trace, ax_hold, par_hold, ax_iv, ax_psd


def plot_ivs_scheme(trace_pair: TracePair, hold_trace: HoldTrace,
                    main_colors=('cornflowerblue', 'indianred'),
                    accent_colors=('royalblue', 'firebrick'),
                    smoothing: int = 1):

    """
    Important: run hold_trace.analyse before plotting!
    Parameters
    ----------
    self
    trace_pair
    hold_trace
    main_colors
    accent_colors
    smoothing

    Returns
    -------

    """

    fig = plt.figure(figsize=utils.cm2inch(15, 8.4), dpi=600)  # figsize: (width, height) in inches

    gs_total = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(3, 2),
                                 figure=fig, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.15)

    gs_top = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2, width_ratios=(1, 2),
                                              subplot_spec=gs_total[0],
                                              wspace=0.2, hspace=0)

    gs_bottom = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=(1, 2),
                                                 subplot_spec=gs_total[1],
                                                 wspace=0.2, hspace=0)

    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=(1, 1),
                                                       subplot_spec=gs_bottom[1],
                                                       wspace=0, hspace=0)

    ax_trace = fig.add_subplot(gs_top[:, 0])
    ax_pull = fig.add_subplot(gs_top[0, 1])
    ax_push = fig.add_subplot(gs_top[1, 1])

    ax_iv = fig.add_subplot(gs_bottom[0])
    ax_psd_pull = fig.add_subplot(gs_bottom_right[0])
    ax_psd_push = fig.add_subplot(gs_bottom_right[1], sharey=ax_psd_pull)

    ax_trace.xaxis.set_label_position('top')
    ax_trace.xaxis.tick_top()
    ax_trace.xaxis.set_ticks_position('both')
    ax_trace.yaxis.set_ticks_position('both')

    ax_pull.xaxis.set_ticks_position('both')
    ax_pull.xaxis.set_label_position('top')
    ax_pull.xaxis.tick_top()
    # ax_pull.yaxis.set_ticks_position('both')
    # ax_pull.yaxis.set_label_position('right')
    # ax_pull.yaxis.tick_right()

    # ax_push.yaxis.set_ticks_position('both')
    # ax_push.yaxis.set_label_position('right')
    # ax_push.yaxis.tick_right()

    ax_iv.xaxis.set_ticks_position('both')
    ax_iv.yaxis.set_ticks_position('both')

    ax_psd_pull.xaxis.set_ticks_position('both')
    ax_psd_push.xaxis.set_ticks_position('both')

    ax_psd_push.yaxis.tick_right()
    ax_psd_push.yaxis.set_ticks_position('both')
    ax_psd_push.yaxis.set_label_position('right')

    ax_trace = trace_pair.plot_trace_pair(ax=ax_trace, xlim=None,
                                          main_colors=main_colors,
                                          accent_colors=accent_colors,
                                          smoothing=smoothing, plot_trigger=True)

    text_pos_pull = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
                     1.5 * trace_pair.hold_set_pull)
    text_pos_push = (ax_trace.get_xlim()[0] + (ax_trace.get_xlim()[1] - ax_trace.get_xlim()[0]) * 0.05,
                     0.8 * trace_pair.hold_set_push)

    ax_trace.text(text_pos_pull[0], text_pos_pull[1], 'pull trigger', fontsize='xx-small', c=accent_colors[0])
    ax_trace.text(text_pos_push[0], text_pos_push[1], 'push trigger', fontsize='xx-small', c=accent_colors[1],
                  va='top')

    # I(V)

    ax_iv.plot(utils.moving_average(hold_trace.iv_bias_pull, smoothing),
               utils.moving_average(hold_trace.iv_current_pull, smoothing),
               c=accent_colors[0], lw=0.4)

    ax_iv.plot(utils.moving_average(hold_trace.iv_bias_push, smoothing),
               utils.moving_average(hold_trace.iv_current_push, smoothing),
               c=accent_colors[1], lw=0.4)

    ax_iv.set_xlabel('Bias [V]')
    ax_iv.set_ylabel('Current [A]')

    max_curr_pull = max(abs(hold_trace.iv_current_pull))
    max_curr_push = max(abs(hold_trace.iv_current_push))

    ax_iv.set_ylim(-1 * max(max_curr_pull, max_curr_push), max(max_curr_pull, max_curr_push))
    ax_iv.set_xticks([-0.7, -0.5, 0, 0.5, 0.7])
    ax_iv.set_xticklabels(['-0.7', '-0.5', '0', '0.5', '0.7'])

    ax_pull, par_pull = hold_trace.plot_hold_traces(plot_step_ranges=False, plot_psd_intervals=False,
                                              conductance=False,
                                              ax=ax_pull, log_scale_y=False,
                                              ax_colors=accent_colors,
                                              smoothing=smoothing)
    ax_push, par_push = hold_trace.plot_hold_traces(plot_step_ranges=False, plot_psd_intervals=False,
                                              conductance=False,
                                              ax=ax_push, log_scale_y=False,
                                              ax_colors=accent_colors,
                                              smoothing=smoothing,
                                              pull=False, push=True)

    ax_pull.set_ylim(-1 * max_curr_pull, max_curr_pull)
    ax_pull.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_push.set_ylim(-1 * max_curr_push, max_curr_push)
    ax_push.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax_push.set_xlabel('')
    ax_push.set_xticklabels([''] * len(ax_push.get_xticks()))

    for i, interval in enumerate(hold_trace.psd_interval_ranges_pull):
        ax_pull.axvspan(interval[0] / 50_000, interval[-1] / 50_000, color=utils.blues[i], ec=None, alpha=0.3)

        for j in interval:
            ax_pull.axvline(j / 50_000, ls='--', c='grey', lw=0.6)

    for i, interval in enumerate(hold_trace.psd_interval_ranges_push):
        ax_push.axvspan(interval[0] / 50_000, interval[-1] / 50_000, color=utils.reds[i], ec=None, alpha=0.3)

        for j in interval:
            ax_push.axvline(j / 50_000, ls='--', c='grey', lw=0.6)

    ax_psd_pull = hold_trace.plot_psds(ax=ax_psd_pull, plot_legend=False)
    ax_psd_push = hold_trace.plot_psds(ax=ax_psd_push, pull=False, plot_legend=False)

    return ax_psd_pull, ax_psd_push


def plot_correlation(n, correlation, axis=None, **kwargs):
    if axis is None:
        fig, ax = plt.subplots(1, figsize=utils.cm2inch(6, 4), dpi=600)
    else:
        ax = axis
    ax.plot(n, correlation, **kwargs)
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$C(\log(iPSD/G_{\mathrm{avg}}^{n}), \log(G_{\mathrm{avg}}))$')
    ax.axhline(y=0, xmin=0, xmax=1, ls='--', lw=0.5, c='k')
    ax.axvline(x=np.mean((n[correlation < 0][0], n[correlation > 0][-1])), ymin=0, ymax=1,
               ls='--', lw=0.5, c='k')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.text(x=0.15, y=0.15, s=f"n = {np.round(n[abs(correlation) == min(abs(correlation))][0], 2)}",
            transform=ax.transAxes, fontsize='xx-small', ha='left', va='bottom')

    return ax, n[abs(correlation) == min(abs(correlation))][0]


def calc_noise_mean_in_bins(xbins: np.ndarray, noise: np.ndarray, conductance: np.ndarray,
                            noise_type: str = 'noise_power'):
    noise_mean = np.zeros(len(xbins) - 1)
    noise_err = np.zeros(len(xbins) - 1)
    cond_mean = np.zeros(len(xbins) - 1)
    cond_err = np.zeros(len(xbins) - 1)
    # conductance_mask = []
    if noise_type == 'noise_power':
        for i in range(len(xbins) - 1):
            # conductance_mask.append(np.bitwise_and(conductance > xbins[i],
            #                                        conductance < xbins[i + 1]))

            conductance_mask = np.bitwise_and(conductance > xbins[i], conductance < xbins[i + 1])
            noise_mean[i] = utils.log_avg(noise[conductance_mask])
            cond_mean[i] = utils.log_avg(conductance[conductance_mask])
            noise_err[i] = utils.custom_error(noise[conductance_mask])
            cond_err[i] = utils.custom_error(conductance[conductance_mask])
    elif noise_type == 'dG/G':
        for i in range(len(xbins) - 1):
            conductance_mask = np.bitwise_and(conductance > xbins[i], conductance < xbins[i + 1])
            noise_mean[i] = utils.log_avg(np.sqrt(noise[conductance_mask])/conductance[conductance_mask])
            cond_mean[i] = utils.log_avg(conductance[conductance_mask])
            noise_err[i] = utils.custom_error(np.sqrt(noise[conductance_mask])/conductance[conductance_mask])
            cond_err[i] = utils.custom_error(conductance[conductance_mask])
    else:
        raise ValueError(f'Invalid value {noise_type} for parameter noise_type. Please enter: "noise_power" or "dG/G"')

    return noise_mean, noise_err, cond_mean, cond_err


def plot_noise_power_2dhist(conductance_avgs: np.ndarray, noise_power: np.ndarray,
                            xrange: Optional[Tuple[float, float]] = (1e-5, 0.1),
                            yrange: Optional[Tuple[float, float]] = (1e-12, 1e-5),
                            num_bins: Optional[Tuple[int, int]] = (10, 10),
                            n: Optional[Union[float, Tuple[float, ...]]] = None,
                            shift: Optional[Union[float, Tuple[float, ...]]] = 0,
                            line_colors: Optional[Union[List[str], Tuple[str, ...], np.ndarray, str]] = None,
                            normalize: bool = False,
                            axis=None, dpi: int = 600,
                            plot_noise_mean_in_bins: Optional[str] = None, **kwargs):

    line_color = 'k'
    if line_colors is None:
        line_colors = colormaps['tab10'](np.linspace(0, 1, 10))
    elif isinstance(line_colors, str):
        line_color = line_colors
    elif not isinstance(line_colors, (list, tuple, np.ndarray)):
        raise ValueError(f"Invalid value {line_colors} for parameter `line_colors`. See docs.")
    else:
        # line_colors is ok as it is
        pass

    num_of_decs_x = np.log10(xrange[1]) - np.log10(xrange[0])
    num_of_decs_y = np.log10(yrange[1]) - np.log10(yrange[0])

    xbins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), num=int(num_bins[0] * num_of_decs_x))
    ybins = np.logspace(np.log10(yrange[0]), np.log10(yrange[1]), num=int(num_bins[1] * num_of_decs_y))

    h, xedges, yedges = np.histogram2d(conductance_avgs.flatten(), noise_power.flatten(), bins=[xbins, ybins])
    x_mesh, y_mesh = np.meshgrid(xedges, yedges)

    h = h.T  # need to take the transpose of h

    if normalize:
        x, avg_cond_hist = utils.calc_hist_1d_single(data=conductance_avgs,
                                                     xrange=xrange,
                                                     xbins_num=num_bins[0],
                                                     log_scale=True)

        h = h / avg_cond_hist

    if axis is None:
        fig, ax = plt.subplots(1, figsize=utils.cm2inch(4, 4), dpi=dpi)
    else:
        ax = axis

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=9))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))

    ax.set_xlabel(r'$\langle G \rangle [G_0]$')

    if n is not None:
        if isinstance(n, float):
            ax.plot(xbins, 10 ** (n * np.log10(xbins) + shift), lw=0.5, ls='--', c=line_color)
        elif isinstance(n, tuple):
            for i, ni in enumerate(n):
                ax.plot(xbins, 10 ** (ni * np.log10(xbins) + shift), lw=0.5, ls='--',
                        c=line_colors[i])

    ax.text(x=0.85, y=0.15, s=f"n = {np.round(n, 2)}",
            transform=ax.transAxes, fontsize='xx-small', ha='right', va='bottom')

    im = ax.pcolormesh(x_mesh, y_mesh, h, **kwargs)

    if plot_noise_mean_in_bins is not None:
        noise_mean, noise_std, cond_mean, cond_std = calc_noise_mean_in_bins(xbins=xbins, noise=noise_power,
                                                                             conductance=conductance_avgs,
                                                                             noise_type=plot_noise_mean_in_bins)

        ax.plot(cond_mean[cond_mean > 0], noise_mean[cond_mean > 0], lw=1)

    return ax


def plot_noise_power(N: np.ndarray, corr: np.ndarray,
                     source: Optional[pd.DataFrame] = None,
                     avg_conductances: Optional[np.ndarray] = None,
                     noise_power: Optional[np.ndarray] = None,
                     conductance_noise: Optional[np.ndarray] = None,
                     step: int = 1, direction='push', **kwargs):

    if direction == 'pull':
        col = 'cornflowerblue'
    elif direction == 'push':
        col = 'indianred'
    else:
        raise ValueError('Invalid parameter for `direction`.')

    fig = plt.figure(figsize=utils.cm2inch(10, 7.5), dpi=600)  # figsize: (width, height) in inches
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=(1, 1), height_ratios=(1, 2),
                           figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)

    ax_corr = fig.add_subplot(gs[0, :])
    ax_corrhist_1 = fig.add_subplot(gs[1, 0])
    ax_corrhist_2 = fig.add_subplot(gs[1, 1])

    ax_corr.xaxis.tick_top()
    ax_corr.xaxis.set_label_position('top')
    ax_corr.xaxis.set_ticks_position('both')
    ax_corr.yaxis.set_ticks_position('both')

    ax_corrhist_1.xaxis.set_ticks_position('both')
    ax_corrhist_1.yaxis.set_ticks_position('both')

    ax_corrhist_2.yaxis.tick_right()
    ax_corrhist_2.yaxis.set_label_position('right')
    ax_corrhist_2.xaxis.set_ticks_position('both')
    ax_corrhist_2.yaxis.set_ticks_position('both')

    ax_corr, n = plot_correlation(n=N, correlation=corr, lw=0.5, c=col, axis=ax_corr)

    if source is not None:
        avg_conductances = np.array(source[f'avg_cond_on_step_{step}'])
        noise_power = np.array(source[f'noise_power_{step}'])
        conductance_noise = np.array(source[f'conductance_noise_{step}'])
    else:
        if avg_conductances is None or noise_power is None or conductance_noise is None:
            raise ValueError('If source is None, you have to provide '
                             '`avg_conductances`, `noise_power` and `conductance_noise`')

    plot_noise_power_2dhist(conductance_avgs=avg_conductances,
                            noise_power=noise_power,
                            n=n,
                            xrange=(1e-6, 0.1),
                            yrange=(1e-14, 1e-5),
                            num_bins=(10, 10),
                            shift=-3.5,
                            normalize=True,
                            cmap=utils.cmap_geo32, axis=ax_corrhist_1, **kwargs)

    plot_noise_power_2dhist(conductance_avgs=avg_conductances,
                            noise_power=conductance_noise,
                            n=(n - 2) / 2,
                            xrange=(1e-6, 0.1),
                            yrange=(1e-3, 1),
                            num_bins=(10, 10),
                            shift=-3.5,
                            normalize=True,
                            cmap=utils.cmap_geo32, axis=ax_corrhist_2, **kwargs)

    ax_corrhist_1.set_ylabel(r'Noise power [$G_{0}^{2}$]')
    ax_corrhist_2.set_ylabel(r'$\Delta G / G$')

    return ax_corr, ax_corrhist_1, ax_corrhist_2


def scatter_plots_from_source(source: pd.DataFrame, x: Union[str, Tuple[str, ...]], y = Union[str, Tuple[str, ...]]):
    ...


# def scatter_plots(x: np.ndarray, y: Union[np.ndarray, List[np.ndarray]], output: Path,
#                   nrows: Optional[int] = None, ncols: Optional[int] = None,
#                   xlim: Optional[Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]] = None,
#                   ylim: Optional[Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]] = None):
#
#     if isinstance(y, np.ndarray):
#         if len(x.shape) == 1 and x.shape == y.shape:
#             nfigs = 1
#         elif len(x.shape) == 1 and y.shape[1] == x.shape[0]:
#             nfigs = y.shape[0]
#         elif x.shape == y.shape:
#             nfigs = x.shape[0]*y.shape[0]
#             if nrows is None and ncols is None:
#                 ...
#             elif nrows is None:
#                 nrows = np.ceil(nfigs/ncols).astype(int)
#
#             elif ncols is None:
#                 ncols = np.ceil(nfigs/nrows).astype(int)
#             else:
#                 if ncols*nrows < nfigs:
#                     raise IndexError
#         else:
#             raise IndexError("x and y shapes don't match")
#
#     fig = plt.figure(figsize=utils.cm2inch(16, 10), dpi=600)  # figsize: (width, height) in inches
#     gs = gridspec.GridSpec(nrows=nrows, ncols=ncols,
#                            figure=fig, left=0.2, right=0.95, top=0.95, bottom=0.15, wspace=0, hspace=0)
#
#     which_cols = np.array(list(conductance_stat_push.columns[:-1])).reshape(4, 6)
#     ratios_push = []
#
#     for i, row in enumerate(which_cols):
#         for j, col in enumerate(row):
#             ax = fig.add_subplot(gs[i, j])
#             ax.xaxis.set_ticks_position('both')
#             ax.yaxis.set_ticks_position('both')
#             if j == (ncols-1):
#                 ax.yaxis.tick_right()
#                 ax.yaxis.set_label_position('right')
#                 ax.xaxis.set_ticks_position('both')
#             ax.yaxis.set_ticks_position('both')
#             if i == 0:
#                 ax.xaxis.tick_top()
#                 ax.xaxis.set_label_position('top')
#                 ax.xaxis.set_ticks_position('both')
#                 ax.yaxis.set_ticks_position('both')
#             indices = np.array(conductance_stat_push_above[:][
#                                    abs(np.log10(conductance_stat_push_above['G_avgs_25'] / conductance_stat_push_above[
#                                        col])) < np.log10(2)].index)
#             not_indices = conductance_stat_push_above.index.difference(indices)
#
#             perc = round(len(indices) / conductance_stat_push_above['trace_index'].count() * 100)
#             ratios_push.append(perc)
#
#             ax.text(5e-6, 1e-2, s=str(perc) + '%', size=6)
#             ax.scatter(conductance_stat_push_above[col][indices], conductance_stat_push_above['G_avgs_25'][indices],
#                        alpha=0.2, c='firebrick', edgecolor='None', s=2)
#             ax.scatter(conductance_stat_push_above[col][not_indices], conductance_stat_push_above['G_avgs_25'][not_indices],
#                        alpha=0.1, c='k', edgecolor='None', s=2)
#             #         ax.scatter(conductance_stat_push_above[col], conductance_stat_push_above['G_avgs_25'],
#             #                          alpha=0.2, c='firebrick', edgecolor='None', s=2)
#             ax.set_xscale('log')
#             ax.set_yscale('log')
#             ax.set_xlim(1e-6, 0.1)
#             ax.set_ylim(1e-6, 0.1)
#             ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=6))
#             #         ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=9))
#             ax.xaxis.set_minor_formatter(ticker.NullFormatter())
#             ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=6))
#             ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=1))
#             ax.yaxis.set_minor_formatter(ticker.NullFormatter())
#             #         ax.set_ylabel('G_avgs_25')
#             #         ax.set_xlabel(col)
#             if j > 0 and j < 5:
#                 ax.yaxis.set_ticklabels([])
#             if i > 0 and i < 3:
#                 ax.xaxis.set_ticklabels([])
#             #         ax.plot(np.logspace(-7, 0, num=10, base=10), np.logspace(-7, 0, num=10, base=10), c='k', ls='--', lw=1, alpha=0.7)
#             ax.plot(np.logspace(-7, 0, num=10, base=10), 2 * np.logspace(-7, 0, num=10, base=10), c='k', ls='--', lw=1,
#                     alpha=0.7)
#             ax.plot(2 * np.logspace(-7, 0, num=10, base=10), np.logspace(-7, 0, num=10, base=10), c='k', ls='--', lw=1,
#                     alpha=0.7)
#
#     plt.savefig(output.joinpath('results/push_relax.png'), bbox_inches='tight')