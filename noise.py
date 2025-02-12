from pathlib import Path
from typing import Union, Tuple, List, Optional

import matplotlib.pyplot as plt
from matplotlib import axes, colormaps, colors, cycler, gridspec
import numpy as np
import scipy
from tqdm.notebook import tqdm

from mcbj import TracePair
import utils


def temporal_noise_single_trace(trace_pair: TracePair, align_at, interpolate,
                                win_size=256, step_size=None, skip_points=50, end_point=3000,
                                tolerance=2, filter_method='start-end', freq_range=(2000, 5000),
                                return_window: bool = False):
    trace_pair.align_trace(align_at=align_at, interpolate=interpolate)

    if step_size is None:
        step_size = win_size//4

    conductance = trace_pair.conductance_pull[np.where(trace_pair.aligned_piezo_pull > 0)[0][0]:]
    piezo = trace_pair.aligned_piezo_pull[np.where(trace_pair.aligned_piezo_pull > 0)[0][0]:]

    if len(conductance) > end_point + skip_points + 1:
        psd_intervals = []
        avg_conductance_in_win = []
        areas = []
        conductance_windows = []

        if filter_method == 'start-end':
            def condition_check(arr):
                return (1/tolerance <
                        abs(utils.log_avg(arr[:len(arr) // 16]) / utils.log_avg(arr[-1 * len(arr) // 16:])) < tolerance
                        )
        elif filter_method == 'start-end_wrong_1':
            def condition_check(arr):
                return abs(utils.log_avg(arr[:len(arr) // 16]) / utils.log_avg(arr[-1 * len(arr) // 16:])) < tolerance
        elif filter_method == 'start-end_wrong_2':
            def condition_check(arr):
                return abs(utils.log_avg(arr[:len(arr) // 16]) / utils.log_avg(arr[-1 * len(arr) // 16:])) > 1/tolerance
        elif filter_method == 'min-max':
            def condition_check(arr):
                return abs(max(arr) / min(arr)) < tolerance
        elif filter_method == 'avg_dev':
            def condition_check(arr):
                return abs(np.log10(max(arr)) - np.mean(np.log10(arr))) < np.log10(tolerance)
        else:
            raise ValueError(f'Unknown value for parameter {filter_method} for parameter filter_method. '
                             f'Valid choices: "start-end", "min-max", "avg_dev", "start_end_wrong".'
                             f'See documentation for more info.')

        for i in range((len(conductance[skip_points:end_point]) - win_size) // step_size):
            conductance_in_win = conductance[skip_points + i * step_size: skip_points + i * step_size + win_size]
            conductance_windows.append((skip_points + i * step_size, skip_points + i * step_size + win_size))

            if condition_check(conductance_in_win):
                psd_intervals.append(conductance_in_win)
                avg_conductance_in_win.append(np.mean(conductance_in_win))
            else:
                psd_intervals.append(np.ones_like(conductance_in_win) * (-1) * 1e-10)
                avg_conductance_in_win.append((-1) * 1e-10)

        fft_freqs, psd_results = scipy.signal.welch(x=psd_intervals, fs=trace_pair.sample_rate, window='hann',
                                                    nperseg=win_size, noverlap=0, nfft=None,
                                                    detrend=False, scaling='density', average='mean', axis=1)

        freq_mask = np.bitwise_and(fft_freqs > freq_range[0], fft_freqs < freq_range[1])

        for i, psd_result in enumerate(psd_results):
            if avg_conductance_in_win[i] > 0:
                areas.append(scipy.integrate.trapz(psd_result[freq_mask], x=fft_freqs[freq_mask]))
            else:
                areas.append((-1) * 1e-10)
        if return_window:
            return conductance, piezo, psd_intervals, psd_results, fft_freqs, avg_conductance_in_win, areas, \
                   conductance_windows
        return conductance, piezo, psd_intervals, psd_results, fft_freqs, avg_conductance_in_win, areas
    else:
        raise utils.MyException(f'Cut conductance interval of trace not long enough. Cut length: {len(conductance)}, '
                                f'required length: {end_point + skip_points + 1}')


def plot_temporal_noise_single_trace(trace_pair: TracePair,
                                     psd_intervals: np.ndarray,
                                     fft_freqs: np.ndarray,
                                     psd_results: np.ndarray,
                                     avg_conductance_in_win: np.ndarray,
                                     areas: np.ndarray,
                                     noise_type: str = 'dG/G',
                                     noise_level_conductance: float = 0,
                                     fig_size: Tuple[float, float] = utils.cm2inch(15, 10),
                                     dpi: int = 300):
    fig = plt.figure(figsize=fig_size, dpi=dpi)  # figsize: (width, height) in inches
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=(2, 1), height_ratios=(1, 1),
                           figure=fig, left=0.2, right=0.95, top=0.95, bottom=0.15, wspace=0.1, hspace=0.1)

    ax_intervals = fig.add_subplot(gs[0, 0])
    ax_psd = fig.add_subplot(gs[1, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_noise = fig.add_subplot(gs[1, 1])

    ax_intervals.set_prop_cycle = cycler('color', colormaps['tab10'](np.linspace(0, 1, 10)))
    ax_psd.set_prop_cycle = cycler('color', colormaps['tab10'](np.linspace(0, 1, 10)))
    ax_noise.set_prop_cycle = cycler('color', colormaps['tab10'](np.linspace(0, 1, 10)))

    ax_intervals.xaxis.tick_top()
    ax_intervals.xaxis.set_label_position('top')
    ax_intervals.xaxis.set_ticks_position('both')
    ax_intervals.yaxis.set_ticks_position('both')

    ax_psd.xaxis.set_ticks_position('both')
    ax_psd.yaxis.set_ticks_position('both')

    ax_noise.yaxis.tick_right()
    ax_noise.yaxis.set_label_position('right')
    ax_noise.xaxis.set_ticks_position('both')
    ax_noise.yaxis.set_ticks_position('both')

    ax_trace.xaxis.tick_top()
    ax_trace.xaxis.set_label_position('top')
    ax_trace.yaxis.tick_right()
    ax_trace.yaxis.set_label_position('right')
    ax_trace.xaxis.set_ticks_position('both')
    ax_trace.yaxis.set_ticks_position('both')

    ax_psd.set_xscale('log')
    ax_psd.set_yscale('log')

    ax_noise.set_xscale('log')
    ax_noise.set_yscale('log')

    ax_trace.set_yscale('log')
    ax_intervals.set_yscale('log')

    ax_intervals.set_xlabel('Points')
    ax_intervals.set_ylabel(r'Conductance [$G_{0}$]')

    ax_psd.set_xlabel('Frequency [Hz]')
    ax_psd.set_ylabel(r'$S_{G}\;[\mathrm{G}_{0}^2/\mathrm{Hz}]$')

    ax_trace.set_xlabel('Aligned Piezo [V]')
    ax_trace.set_ylabel(r'Conductance [$G_{0}$]')

    ax_noise.set_xlabel(r'Conductance [$G_{0}$]')
    ax_noise.set_ylabel(r'$\Delta G/G$')

    ax_trace.plot(trace_pair.aligned_piezo_pull, trace_pair.conductance_pull, lw=1, c='b')

    for i in range(len(avg_conductance_in_win)):
        if avg_conductance_in_win[i] > noise_level_conductance:
            ax_intervals.plot(psd_intervals[i], alpha=0.6, lw=1.5)
            ax_psd.plot(fft_freqs, psd_results[i], alpha=0.6, lw=1, label=str(i))
            if noise_type == 'dG/G':
                ax_noise.scatter(avg_conductance_in_win[i], np.sqrt(areas[i])/avg_conductance_in_win[i],
                                 alpha=0.6, edgecolors='None', s=2)
            elif noise_type == 'noise_power':
                ax_noise.scatter(avg_conductance_in_win[i], areas[i],
                                 alpha=0.6, edgecolors='None', s=2)
            else:
                raise ValueError(f'Invalid value {noise_type} for parameter noise_type. '
                                 f'Please enter: "noise_power" or "dG/G"')

    return ax_intervals, ax_trace, ax_psd, ax_noise
