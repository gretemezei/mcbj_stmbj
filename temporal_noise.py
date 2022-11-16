from pathlib import Path
from typing import Union, Tuple, List, Optional

import matplotlib.pyplot as plt
from matplotlib import axes, cm, colors, cycler, gridspec
import numpy as np
import scipy.signal as scipy_signal
from tqdm.notebook import tqdm

from mcbj import TracePair
import utils


class TemporalNoise:
    def __init__(self, trace_pair: TracePair):
        self.trace_pair = trace_pair

    def calc_temporal_noise(self, align_at: Optional[float] = None, interpolate: bool = True,
                            mode='whole', win_size: int = 512, step_size: Optional[float] = None,
                            width: Optional[int] = None):
        if step_size is None:
            step_size = win_size // 2

        if width is None:
            width = min(len(self.trace_pair.conductance_pull), len(self.trace_pair.conductance_push))

        if align_at is not None:
            self.trace_pair.align_trace(align_at=align_at, interpolate=interpolate)

        if mode == 'whole':
            self.cond_avg_pull = utils.mov_avg(self.trace_pair.conductance_pull[:width],
                                               win_size=win_size, step_size=step_size,
                                               avg_type=utils.log_avg)

            self.f_pull, self.t_pull, self.Zxx_pull = scipy_signal.stft(self.trace_pair.conductance_pull[:width],
                                                                        fs=self.trace_pair.sample_rate,
                                                                        window='hann', nperseg=win_size,
                                                                        noverlap=win_size - step_size,
                                                                        nfft=None, detrend=False, return_onesided=True,
                                                                        boundary='even', padded=True, axis=- 1,
                                                                        scaling='psd')

            self.cond_avg_push = utils.mov_avg(self.trace_pair.conductance_push[:width],
                                               win_size=win_size, step_size=step_size,
                                               avg_type=utils.log_avg)

            self.f_push, self.t_push, self.Zxx_push = scipy_signal.stft(self.trace_pair.conductance_push[:width],
                                                                        fs=self.trace_pair.sample_rate,
                                                                        window='hann', nperseg=win_size,
                                                                        noverlap=win_size - step_size,
                                                                        nfft=None, detrend=False, return_onesided=True,
                                                                        boundary='even', padded=True, axis=- 1,
                                                                        scaling='psd')
        elif mode == 'positive':
            self.cond_avg_pull = utils.mov_avg(self.trace_pair.conductance_pull[self.trace_pair.aligned_piezo_pull >= 0][:width],
                                               win_size=win_size, step_size=step_size,
                                               avg_type=utils.log_avg)

            self.f_pull, self.t_pull, self.Zxx_pull = scipy_signal.stft(
                self.trace_pair.conductance_pull[self.trace_pair.aligned_piezo_pull >= 0][:width],
                fs=self.trace_pair.sample_rate,
                window='hann', nperseg=win_size, noverlap=win_size - step_size,
                nfft=None, detrend=False, return_onesided=True,
                boundary='even', padded=True, axis=- 1,
                scaling='psd')

            self.cond_avg_push = utils.mov_avg(self.trace_pair.conductance_push[self.trace_pair.aligned_piezo_push >= 0][:width],
                                               win_size=win_size, step_size=step_size,
                                               avg_type=utils.log_avg)

            self.f_push, self.t_push, self.Zxx_push = scipy_signal.stft(
                self.trace_pair.conductance_push[self.trace_pair.aligned_piezo_push >= 0][::-1][:width],
                fs=self.trace_pair.sample_rate,
                window='hann', nperseg=win_size, noverlap=win_size - step_size,
                nfft=None, detrend=False, return_onesided=True,
                boundary='even', padded=True, axis=- 1,
                scaling='psd')
        else:
            raise ValueError(f'Unknown parameter for mode {mode}. Valid choices: "whole", "positive".')

    def plot_temporal_noise(self, mode: str = 'whole', normalize: bool = False,
                            add_vlines: Tuple[float, ...] = tuple(),
                            add_hlines: Tuple[float, ...] = tuple(),
                            ax: Optional[Tuple[axes.Axes, axes.Axes]] = None,
                            piezo_range_pull: Optional[Tuple[float, float]] = None,
                            piezo_range_push: Optional[Tuple[float, float]] = None,
                            conductance_range: Tuple[float, float] = (1e-6, 100),
                            vmax_pull: float = 0.1, vmax_push: float = 0.1,
                            dpi: int = 300):
        """

        Parameters
        ----------
        mode
        normalize
        add_vlines
        add_hlines
        ax
        piezo_range_pull
        piezo_range_push
        conductance_range
        vmax_pull
        vmax_push
        dpi

        Returns
        -------
        To plot individual PSDs: ax.plot(self.f, np.abs(self.Zxx)[:,i]/self.cond_avg[i])
        """
        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(10, 10), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=2, ncols=1,
                                   figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0.3)

            ax_pull = fig.add_subplot(gs[0])
            ax_push = fig.add_subplot(gs[1])
        else:
            ax_pull = ax[0]
            ax_push = ax[1]  # width, height

        parx_pull = ax_pull.twinx()
        pary_pull = ax_pull.twiny()

        parx_push = ax_push.twinx()
        pary_push = ax_push.twiny()

        ax_pull.set_title('STFT Magnitude')
        ax_pull.set_ylabel('Frequency [Hz]')
        ax_pull.set_xlabel('Time [s]')
        pary_pull.set_xlabel('Piezo [V]')
        parx_pull.set_ylabel(r'Conductance [$G_{0}$]')
        ax_pull.set_yscale('log')
        ax_pull.set_ylim(100, 25000)
        # print(ax.get_xlim())
        parx_pull.set_yscale('log')

        # if mode == 'whole':
        shift_pull = min(self.trace_pair.aligned_time_pull)

        if piezo_range_pull is None:
            # set time limits
            ax_pull.set_xlim(min(self.trace_pair.aligned_time_pull),
                             max(self.trace_pair.aligned_time_pull))
            # set piezo limits
            pary_pull.set_xlim(min(self.trace_pair.aligned_piezo_pull),
                               max(self.trace_pair.aligned_piezo_pull))
        else:
            # set time limits
            ax_pull.set_xlim(min(piezo_range_pull) / self.trace_pair.rate,
                             max(piezo_range_pull) / self.trace_pair.rate)
            # set piezo limits
            pary_pull.set_xlim(min(piezo_range_pull),
                               max(piezo_range_pull))

        parx_pull.plot(self.trace_pair.aligned_time_pull,
                       self.trace_pair.conductance_pull, 'k', lw=0.5, zorder=100)

        shift_push = min(self.trace_pair.aligned_time_push)

        if piezo_range_push is None:
            # set time limits
            ax_push.set_xlim(max(self.trace_pair.aligned_time_push),
                             min(self.trace_pair.aligned_time_push))
            # set piezo limits
            pary_push.set_xlim(max(self.trace_pair.aligned_piezo_push),
                               min(self.trace_pair.aligned_piezo_push))
        else:
            # set time limits
            ax_push.set_xlim(max(piezo_range_push) / self.trace_pair.rate,
                             min(piezo_range_push) / self.trace_pair.rate)
            # set piezo limits
            pary_push.set_xlim(max(piezo_range_push),
                               min(piezo_range_push))

        parx_push.plot(self.trace_pair.aligned_time_push,
                       self.trace_pair.conductance_push, 'k', lw=0.5, zorder=100)

        if normalize:
            pcm_pull = ax_pull.pcolormesh(self.t_pull, self.f_pull,
                                          np.abs(self.Zxx_pull) / self.cond_avg_pull,
                                          vmin=0, vmax=vmax_pull, shading='gouraud',
                                          cmap='gist_rainbow', zorder=0)

            pcm_push = ax_push.pcolormesh(self.t_push, self.f_push,
                                          np.abs(self.Zxx_push) / self.cond_avg_push,
                                          vmin=0, vmax=vmax_push, shading='gouraud',
                                          cmap='gist_rainbow', zorder=0)
        else:
            pcm_pull = ax_pull.pcolormesh(self.t_pull, self.f_pull,
                                          np.abs(self.Zxx_pull),
                                          vmin=0, vmax=vmax_pull, shading='gouraud',
                                          cmap='gist_rainbow', zorder=0)

            pcm_push = ax_push.pcolormesh(self.t_push, self.f_push,
                                          np.abs(self.Zxx_push),
                                          vmin=0, vmax=vmax_push, shading='gouraud',
                                          cmap='gist_rainbow', zorder=0)

        # ax_push.set_title('STFT Magnitude')
        ax_push.set_ylabel('Frequency [Hz]')
        ax_push.set_xlabel('Time [s]')
        pary_push.set_xlabel('Piezo [V]')
        parx_push.set_ylabel(r'Conductance [$G_{0}$]')
        ax_push.set_yscale('log')
        ax_push.set_ylim(100, 25000)
        # print(ax.get_xlim())
        parx_push.set_yscale('log')

        parx_pull.set_ylim(conductance_range)
        parx_push.set_ylim(conductance_range)

        for i in add_hlines:
            parx_pull.axhline(i, lw=0.5, color='white', ls='--')
            parx_push.axhline(i, lw=0.5, color='white', ls='--')

        for i in add_vlines:
            parx_pull.axvline(i, lw=0.5, color='white', ls='--')
            parx_push.axvline(i, lw=0.5, color='white', ls='--')

        fig.colorbar(pcm_pull, ax=pary_pull, pad=0.1)
        fig.colorbar(pcm_push, ax=pary_push, pad=0.1)

        return ax_pull, ax_push