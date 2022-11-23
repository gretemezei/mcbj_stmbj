# IMPORTS
# python libraries
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps, cycler, rcParams, gridspec, ticker
import matplotlib.axes
from matplotlib.colors import ListedColormap
import numpy as np
from os import mkdir
from os.path import isdir, exists
import pandas as pd
from pathlib import Path
import re
import scipy.signal  # signal processing
import scipy.integrate
from scipy.optimize import curve_fit
from typing import Union, Tuple, List, Optional
from tqdm.notebook import tqdm
import warnings

# custom libraries
import utils
from utils import MeasurementNotComplete, MeasurementOverflow

# date = "21_12_01"
# utils.check_date(date)
# sample_rate = 50_000
# home_folder = Path(f"D:/BJ_Data/{date}")

# some matplotlib settings to have really nice figs
rcParams['figure.constrained_layout.use'] = False
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['mathtext.rm'] = 'serif'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['xtick.major.pad'] = 2
rcParams['xtick.minor.pad'] = 0.5
rcParams['ytick.major.pad'] = 1
rcParams['ytick.minor.pad'] = 0.5
rcParams['axes.labelsize'] = 6
rcParams['axes.titlepad'] = 6.0
rcParams['axes.labelpad'] = 1.0
rcParams['agg.path.chunksize'] = 10000


class TracePair:
    """
    Class to represent a break junction trace pair

    Parameters
    ----------
    trace : int or str
        trace number or trace name to work with
    load_from : `~pathlib.Path` or `~h5py.File`
        path to the file or an opened for read file instance to load data from
    low_bound_pull : float
        low bound value to find plateau and calculate plateau length
    high_bound_pull : float
        high bound value to find plateau and calculate plateau length
    low_bound_push : float
        low bound value to find plateau and calculate plateau length
    high_bound_push : float
        high bound value to find plateau and calculate plateau length

    Attributes
    ----------
    trace_num : int
        number of the trace
    trace_name : str
        name of the trace used as the key, format: 'trace_{trace_num}'
    conductance_pull : np.ndarray
        Conductance values for the pull direction in :math:`G_{0}` units
    conductance_push : np.ndarray
        Conductance values for the push direction in :math:`G_{0}` units
    piezo_pull : np.ndarray
        Piezo values for the pull direction in V
    piezo_push : np.ndarray
        Piezo values for the push direction
    dz_pull : float
        piezo change step size during breaking process
    dz_push : float
        piezo change step size during closing process
    z0_pull : float
        initial value of voltage applied to piezo when the breaking process started
    z0_push : float
        initial value of voltage applied to piezo when the closing process started
    R_parallel : float
        resistance of the resistor connected in parallel to the sample
    R_serial : float
        resistance of the resistor connected in series to the sample
    gain : float
        set gain for the current amplifier
    bias_mv : float
        set bias value for the measurement in mV
    rate : float
        set speed to change piezo value in V
    sample_rate : float
        set sample rate of the measurement in Hz

    G_limit_pull : float
        low conductance limit to stop the breaking process and turn around
    G_limit_push : float
        high conductance limit to stop closing process and turn around
    distance_pull : float
        total distance pulled
    distance_push : float
        total distance pushed
    excursion_pull : float
        extra distance to pull after reaching low conductance limit
    excursion_push : float
        extra distance to pull after reaching high conductance limit
    hold_excursion_pull : float
        extra distance to pull after reaching conductance limit for hold measurement
    hold_excursion_push : float
        extra distance to pull after reaching conductance limit for hold measurement
    speed_pull :
        speed to change piezo value, based on set rate
    speed_push :
        speed to change piezo value, based on set rate

    low_bound_pull : float
        low bound value to find plateau and calculate plateau length
    high_bound_pull : float
        high bound value to find plateau and calculate plateau length
    low_bound_push : float
        low bound value to find plateau and calculate plateau length
    high_bound_push : float
        high bound value to find plateau and calculate plateau length
    plateau_length_pull : int or float
        plateau length of the plateau defined by `low_bound_pull` and `high_bound_pull` values for the pull trace
    plateau_length_push : int or float
        plateau length of the plateau defined by `low_bound_push` and `high_bound_push` values for the push trace
    plateau_range_pull : Tuple[int, int]
        starting and ending point of the plateau defined by `low_bound_pull` and `high_bound_pull` for the pull trace
    plateau_range_push : Tuple[int, int]
        starting and ending point of the plateau defined by `low_bound_push` and `high_bound_push` for the push trace
    hold_set_pull : float
        set conductance value where breaking process is stopped to perform hold measurement
    hold_set_push : float
        set conductance value where closing process is stopped to perform hold measurement
    hold_index_pull : int
        index of the point where the hold measurement started in the pull direction,
        if there was no hold measurement for the trace, this value defaults to None or -1 CHECK!!!
    hold_index_push : int
        index of the point where the hold measurement started in the push direction,
        if there was no hold measurement for the trace, this value defaults to None
    hold_conductance_pull : float
        conductance value at index `hold_index_pull`
    hold_conductance_push : float
        conductance value at index `hold_index_push`

    f_pull
    t_pull
    Zxx_pull
    cond_avg_pull
    """

    def __init__(self, trace: Union[int, str], load_from: Union[Path, h5py._hl.files.File],
                 low_bound_pull: float = 0.5, high_bound_pull: float = 1.5,
                 low_bound_push: float = 0.5, high_bound_push: float = 1.5) -> None:

        if isinstance(trace, (int, np.int32)):
            self.trace_num = trace
            self.trace_name = f'trace_{self.trace_num}'
        elif isinstance(trace, str):
            self.trace_name = trace
            self.trace_num = utils.get_num_from_name(self.trace_name)
        else:
            raise ValueError('Invalid value for variable {trace}. It has to be an integer corresponding to the number'
                             'of the trace to be loaded or the name of the trace in the format: "trace_{trace_num}"')

        # initialize the data
        self.conductance_pull = None
        self.conductance_push = None
        self.piezo_pull = None
        self.piezo_push = None
        self.aligned_piezo_pull = None
        self.aligned_piezo_push = None
        self.time_pull = None
        self.time_push = None
        self.aligned_time_pull = None
        self.aligned_time_push = None
        self.dz_pull = None
        self.dz_push = None
        self.z0_pull = None
        self.z0_push = None
        self.align_at = None

        self.R_parallel = None
        self.R_serial = None
        self.gain = None
        self.bias_mv = None
        self.rate = None
        self.sample_rate = None

        self.G_limit_pull = None
        self.G_limit_push = None
        self.distance_pull = None
        self.distance_push = None
        self.excursion_pull = None
        self.excursion_push = None
        self.hold_excursion_pull = None
        self.hold_excursion_push = None
        self.speed_pull = None
        self.speed_push = None

        self.plateau_length_pull = None
        self.plateau_length_push = None
        self.plateau_range_pull = None
        self.plateau_range_push = None
        self.hold_set_pull = None
        self.hold_set_push = None
        self.hold_index_pull = None
        self.hold_index_push = None
        self.hold_conductance_pull = None
        self.hold_conductance_push = None

        self.low_bound_pull = low_bound_pull
        self.high_bound_pull = high_bound_pull
        self.low_bound_push = low_bound_push
        self.high_bound_push = high_bound_push

        self.f_pull = None
        self.t_pull = None
        self.Zxx_pull = None
        self.cond_avg_pull = None

        self.f_push = None
        self.t_push = None
        self.Zxx_push = None
        self.cond_avg_push = None

        # when a TracePair instance is created, immediately load given trace
        block_num, _ = utils.convert_to_block_and_trace_num(self.trace_num)

        if isinstance(load_from, Path):
            # if you entered the path to the file containing the data, 1st it needs to be opened
            self.file_path = load_from.joinpath(f'break_junction_{block_num}.h5')
            self.file = None
            with h5py.File(self.file_path, "r") as bj_file:
                self.load_trace_pair(bj_file)
        elif isinstance(load_from, h5py._hl.files.File):
            # file already open
            self.file_path = None
            self.file = load_from

            self.load_trace_pair(load_from)
        else:
            raise ValueError(f"Unknown datatype {type(load_from)} in parameter load_from."
                             f"It has to be either a Path object pointing to the folder containing the data file,"
                             f"or an opened h5py File instance.")

        # calculate the length of the 1 G_0 plateau
        self.plateau_length_pull, self.plateau_range_pull, self.plateau_length_push, self.plateau_range_push = \
            self.calc_plateau_length(low_bound_pull=self.low_bound_pull, high_bound_pull=self.high_bound_pull,
                                     low_bound_push=self.low_bound_push, high_bound_push=self.high_bound_push)

    def load_trace_pair(self, bj_file):
        """
        Loads the pull-push trace pair defined by `TracePair.trace_num` from `TracePair.file_path`.

        Parameters
        ----------
        bj_file : Path or h5py._hl.files.File
            path to the file or an opened for read file instance to load data from

        Returns
        -------
        None
        """

        trace_name = f'trace_{self.trace_num}'

        # same during the measurement, so no need to differentiate between pull and push
        self.R_parallel = bj_file[f'pull/{trace_name}/conductance'].attrs['R_parallel']
        self.R_serial = bj_file[f'pull/{trace_name}/conductance'].attrs['R_serial']
        self.bias_mv = bj_file[f'pull/{trace_name}/conductance'].attrs['bias_mv']
        self.gain = bj_file[f'pull/{trace_name}/conductance'].attrs['gain']
        self.rate = bj_file[f'pull/{trace_name}/conductance'].attrs['rate']
        self.sample_rate = bj_file[f'pull/{trace_name}/conductance'].attrs['sample_rate']

        # Pull
        z0 = bj_file[f'pull/{trace_name}/conductance'].attrs['z0']
        dz = bj_file[f'pull/{trace_name}/conductance'].attrs['dz']

        if dz == 0:
            raise ValueError(f"for pull trace {trace_name} dz=0")

        # Add values to the corresponding variables
        self.conductance_pull = bj_file[f'pull/{trace_name}/conductance'][:]
        self.hold_set_pull = bj_file[f'pull/{trace_name}/conductance'].attrs['holdG']
        if len(bj_file[f'pull/{trace_name}/hold_index'][:]) > 0:
            self.hold_index_pull = int(bj_file[f'pull/{trace_name}/hold_index'][0]) - 1
        self.piezo_pull = np.array([z0 - j * dz for j in range(self.conductance_pull.shape[0])])
        self.dz_pull = dz
        self.z0_pull = z0

        self.time_pull = self.piezo_pull / self.rate

        self.G_limit_pull = bj_file[f'pull/{trace_name}/conductance'].attrs['G_limit']
        self.distance_pull = bj_file[f'pull/{trace_name}/conductance'].attrs['distance']
        self.excursion_pull = bj_file[f'pull/{trace_name}/conductance'].attrs['excursion']
        self.hold_excursion_pull = bj_file[f'pull/{trace_name}/conductance'].attrs['hold_excursion']
        self.speed_pull = bj_file[f'pull/{trace_name}/conductance'].attrs['speed']

        if self.hold_index_pull is not None:
            self.hold_conductance_pull = self.conductance_pull[self.hold_index_pull]

        # Push
        z0 = bj_file[f'push/{trace_name}/conductance'].attrs['z0']
        dz = bj_file[f'push/{trace_name}/conductance'].attrs['dz']

        if dz == 0:
            raise ValueError(f"for push trace {trace_name} dz=0")

        # Add values to the corresponding variables
        self.conductance_push = bj_file[f'push/{trace_name}/conductance'][:]
        self.hold_set_push = bj_file[f'push/{trace_name}/conductance'].attrs['holdG']
        if len(bj_file[f'push/{trace_name}/hold_index'][:]) > 0:
            self.hold_index_push = int(bj_file[f'push/{trace_name}/hold_index'][0]) - 1
        self.piezo_push = np.array([z0 - j * dz for j in range(self.conductance_push.shape[0])])
        self.dz_push = dz
        self.z0_push = z0

        self.time_push = self.piezo_push / self.rate

        self.G_limit_push = bj_file[f'push/{trace_name}/conductance'].attrs['G_limit']
        self.distance_push = bj_file[f'push/{trace_name}/conductance'].attrs['distance']
        self.excursion_push = bj_file[f'push/{trace_name}/conductance'].attrs['excursion']
        self.hold_excursion_push = bj_file[f'push/{trace_name}/conductance'].attrs['hold_excursion']
        self.speed_push = bj_file[f'push/{trace_name}/conductance'].attrs['speed']

        if self.hold_index_push is not None:
            self.hold_conductance_push = self.conductance_push[self.hold_index_push]

    def align_trace(self, align_at: Optional[float] = None, interpolate: bool = True):

        if align_at is None:
            self.aligned_piezo_pull = self.piezo_pull
            self.aligned_piezo_push = self.piezo_push
        else:
            self.align_at = align_at
            if np.log10(self.align_at) > -1 * (1 - (-6)) * 0.5:
                # align value in the upper half of the conductance range
                which_point_pull = 0  # take the last cross point
                which_point_push = -1
            else:
                # align_at value in the lower half of the conductance range
                which_point_pull = 0  # take the 1st cross point
                which_point_push = -1  # take the 1st cross point

            # print(f'this: {which_point_pull}, {which_point_push}')

            x1_pull = self.piezo_pull[np.where(self.conductance_pull < self.align_at)[0][which_point_pull]]
            y1_pull = self.conductance_pull[np.where(self.conductance_pull < self.align_at)[0][which_point_pull]]

            x2_pull = self.piezo_pull[np.where(self.conductance_pull < self.align_at)[0][which_point_pull] + 1]
            y2_pull = self.conductance_pull[np.where(self.conductance_pull < self.align_at)[0][which_point_pull] + 1]

            # print(f'ind1 = ({x1_pull}, {y1_pull}), ind2 = ({x2_pull}, {y2_pull}')
            if interpolate:
                shift_pull = utils.interpolate(ind1=(x1_pull, y1_pull), ind2=(x2_pull, y2_pull), y=self.align_at)
            else:
                shift_pull = x2_pull
            self.aligned_piezo_pull = self.piezo_pull - shift_pull
            self.aligned_time_pull = self.aligned_piezo_pull / self.rate

            x1_push = self.piezo_push[np.where(self.conductance_push < self.align_at)[0][which_point_push] - 1]
            y1_push = self.conductance_push[np.where(self.conductance_push < self.align_at)[0][which_point_push] - 1]

            x2_push = self.piezo_push[np.where(self.conductance_push < self.align_at)[0][which_point_push]]
            y2_push = self.conductance_push[np.where(self.conductance_push < self.align_at)[0][which_point_push]]

            # print(f'ind1 = ({x1_push}, {y1_push}), ind2 = ({x2_push}, {y2_push}')
            if interpolate:
                shift_push = utils.interpolate(ind1=(x1_push, y1_push), ind2=(x2_push, y2_push), y=self.align_at)
            else:
                shift_push = x2_push
            self.aligned_piezo_push = self.piezo_push - shift_push
            self.aligned_time_push = self.aligned_piezo_push / self.rate

    def calc_plateau_length(self,
                            low_bound_pull: float = 0.5,
                            high_bound_pull: float = 1.5,
                            low_bound_push: float = 0.5,
                            high_bound_push: float = 1.5,
                            in_volts: bool = False) -> Tuple[Union[int, float],
                                                             Optional[Tuple[int, int]],
                                                             Union[int, float],
                                                             Optional[Tuple[int, int]]]:
        """
        Calculates plateau length in the given range. Default is the plateau length of the 1 G_0 plateau in the range of
        (0.5 G_0, 1.5 G_0). You can also set different range for pull and push traces

        Parameters
        ----------
        low_bound_pull : float
            the lowest value to be included in the plateau
        high_bound_pull : float
            the highest value to be included in the plateau
        low_bound_push : float
            the lowest value to be included in the plateau
        high_bound_push : float
            the highest value to be included in the plateau
        in_volts : bool
            if True, plateau length is returned in volts instead of points

        Returns
        -------
        plateau_length_pull : int or float
            length of the plateau of the pull trace
        plateau_range_pull : Tuple[int, int]
            starting and ending index of the plateau for the pull trace
            (plateau range is defined to be inclusive for the starting point and
            exclusive for the ending point (just as indexing in python))
        plateau_length_push : int or float:
            length of the plateau of the pull trace
        plateau_range_push : Tuple[int, int]
            starting and ending index of the plateau for the push trace

        """
        # pull
        condition = np.bitwise_and(self.conductance_pull < high_bound_pull, self.conductance_pull > low_bound_pull)
        plateau_length_pull = self.conductance_pull[condition].shape[0]
        if plateau_length_pull > 0:
            plateau_range_pull = (np.nonzero(condition)[0][0], np.nonzero(condition)[0][-1])
        else:
            plateau_range_pull = None
        # try:
        #     plateau_start = np.where(self.conductance_pull > high_bound_pull)[0][-1]
        #     plateau_end = np.where(self.conductance_pull < low_bound_pull)[0][0]
        #
        #     plateau_length_pull = plateau_end - plateau_start - 1
        #     plateau_range_pull = (plateau_start + 1, plateau_end)
        # except IndexError:
        #     plateau_range_pull = None
        #     plateau_length_pull = 0

        # push
        condition = np.bitwise_and(self.conductance_push < high_bound_push, self.conductance_push > low_bound_push)
        # not quite working additional condition for the case when the plateau that falls into the given range is
        # not continuous
        # condition[np.nonzero(condition)[0][0]:
        #           np.nonzero(condition)[0][0]+np.nonzero(np.diff(np.nonzero(condition)[0]) > 1)[0][-1] + 1] = False
        plateau_length_push = self.conductance_push[condition].shape[0]
        if plateau_length_push > 0:
            plateau_range_push = (np.nonzero(condition)[0][0], np.nonzero(condition)[0][-1])
        else:
            plateau_range_push = None

        # try:
        #     plateau_start = np.where(self.conductance_push < low_bound_push)[0][-1]
        #     plateau_end = np.where(self.conductance_push > high_bound_push)[0][0]
        #
        #     plateau_length_push = plateau_end - plateau_start - 1
        #     plateau_range_push = (plateau_start + 1, plateau_end)
        # except IndexError:
        #     plateau_range_push = None
        #     plateau_length_push = 0

        if in_volts:
            plateau_length_pull *= self.dz_pull
            plateau_length_push *= self.dz_push

        self.plateau_length_pull = plateau_length_pull
        self.plateau_length_push = plateau_length_push
        self.plateau_range_pull = plateau_range_pull
        self.plateau_range_push = plateau_range_push

        return plateau_length_pull, plateau_range_pull, plateau_length_push, plateau_range_push

    def fit_tunnel(self, conductance_range: Tuple[float, float] = (5e-5, 1e-2), direction: str = 'push',
                   fit_func: Optional[callable] = None):
        if direction == 'pull':
            conductance = self.conductance_pull
            piezo = self.piezo_pull
        elif direction == 'push':
            conductance = self.conductance_push
            piezo = self.piezo_push
        else:
            raise ValueError('Invalid value for direction')

        if fit_func is None:
            fit_func = utils.fit_func_lin

        popt, pcov = curve_fit(fit_func,
                               piezo[np.bitwise_and(conductance > conductance_range[0],
                                                    conductance < conductance_range[1])],
                               np.log10(conductance[np.bitwise_and(conductance > conductance_range[0],
                                                                   conductance < conductance_range[1])]))

        perr = np.sqrt(np.diag(pcov))

        return popt, perr

    def plot_trace_pair(self,
                        ax: Optional[matplotlib.axes.Axes] = None,
                        x_val: str = 'piezo',
                        aligned: bool = False,
                        ylim: Tuple[float, float] = (1e-6, 10),
                        xlim: Optional[Tuple[float, float]] = None,
                        main_colors: Tuple[str, str] = ('cornflowerblue', 'indianred'),
                        accent_colors: Tuple[str, str] = ('royalblue', 'firebrick'),
                        plot_1G0_range: bool = False, plot_trigger: bool = False,
                        smoothing: int = 1, dpi: int = 600,
                        **kwargs):
        """
        Plot the loaded pull-push trace pair

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            if there is a pre-existing axis you can plot it there.
            Otherwise, a new figure with a new axis will be created.
        x_val : str, default: 'piezo'
            x value to plot. Either the piezo voltage ('piezo') or time ('time')
        aligned : bool, default: False
            plot the trace after aligning it at a given conductance value. Raises an Error if any of the values
            self.aligned_piezo_pull/self.aligned_piezo_push are None.
        ylim : Tuple[float, float]
            conductance limits
        xlim : Tuple[float, float]
            piezo limits
        main_colors : Tuple[str, str]
            the two colors used to differentiate between pull and push traces, respectively
        accent_colors : Tuple[str, str]
            the two colors for the annotations for pull and push traces, respectively
        plot_1G0_range : bool, default: False

        plot_trigger : bool, default: False

        smoothing : int, default: 1

        dpi : resolution in dots per inch

        Returns
        -------
        ax : matplotlib.axes
            the axis with the resulting plot for further formatting if necessary
        """
        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(5, 5), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=1, ncols=1,
                                   figure=fig, left=0.2, right=0.95, top=0.95, bottom=0.15, wspace=0, hspace=0.1)
            ax = fig.add_subplot(gs[0])
        ax.set_yscale('log')
        ax.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.2), numticks=9))
        ax.grid(ls='--', lw=0.5, alpha=0.5)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        if aligned:
            if self.aligned_piezo_pull is not None and self.aligned_piezo_push is not None:
                piezo_pull = self.aligned_piezo_pull
                piezo_push = self.aligned_piezo_push
            else:
                warnings.warn('Plotting original. Explanation: Trace not aligned, not able to plot aligned trace. '
                              'Run align_trace() method first, to plot the aligned trace.')
                piezo_pull = self.piezo_pull
                piezo_push = self.piezo_push
        else:
            piezo_pull = self.piezo_pull
            piezo_push = self.piezo_push

        # PLot pull
        if x_val == 'piezo':
            ax.plot(utils.moving_average(piezo_pull, smoothing),
                    utils.moving_average(self.conductance_pull, smoothing),
                    color=main_colors[0], lw=0.5)
            ax.set_xlabel('Piezo [V]')
        elif x_val == 'time':
            ax.plot(
                utils.moving_average(np.arange(start=0, stop=self.conductance_pull.shape[0] / 50_000, step=1 / 50_000),
                                     smoothing),
                utils.moving_average(self.conductance_pull, smoothing),
                color=main_colors[0], lw=0.5)
            ax.set_xlabel('Time [s]')
        else:
            raise ValueError(f"Invalid parameter {x_val}, valid choices: 'piezo', 'time'.")

        # Plot push
        if x_val == 'piezo':
            ax.plot(utils.moving_average(piezo_push, smoothing),
                    utils.moving_average(self.conductance_push, smoothing),
                    color=main_colors[1], lw=0.5)
            ax.set_xlabel('Piezo [V]')
        elif x_val == 'time':
            ax.plot(
                utils.moving_average(np.arange(start=0, stop=self.conductance_push.shape[0] / 50_000, step=1 / 50_000),
                                     smoothing),
                utils.moving_average(self.conductance_push, smoothing),
                color=main_colors[1], lw=0.5)
            ax.set_xlabel('Time [s]')
        else:
            raise ValueError(f"Invalid parameter {x_val}, valid choices: 'piezo', 'time'.")

        # plot the bounds for the 1G0 plateau and denote the beginning and end of the plateau
        if plot_1G0_range:
            if self.plateau_range_pull is not None:
                ax.plot(piezo_pull[self.plateau_range_pull[0]], self.conductance_pull[self.plateau_range_pull[0]],
                        color=accent_colors[0], marker='+', markersize=3, markeredgewidth=0.5)
                ax.plot(piezo_pull[self.plateau_range_pull[1]], self.conductance_pull[self.plateau_range_pull[1]],
                        color=accent_colors[0], marker='+', markersize=3, markeredgewidth=0.5)

            if self.plateau_range_push is not None:
                ax.plot(piezo_push[self.plateau_range_push[0]], self.conductance_push[self.plateau_range_push[0]],
                        color=accent_colors[1], marker='+', markersize=3, markeredgewidth=0.5)
                ax.plot(piezo_push[self.plateau_range_push[1]], self.conductance_push[self.plateau_range_push[1]],
                        color=accent_colors[1], marker='+', markersize=3, markeredgewidth=0.5)

            ax.axhline(self.low_bound_pull, lw=0.5, ls='--', c='b')
            ax.axhline(self.high_bound_pull, lw=0.5, ls='--', c='b')
            ax.axhline(self.low_bound_push, lw=0.5, ls='--', c='r')
            ax.axhline(self.high_bound_push, lw=0.5, ls='--', c='r')

        if plot_trigger:
            # if there was hold measurement for this trace, plot the trigger value and the actual stopping point
            if self.hold_index_pull is not None:
                ax.plot(piezo_pull[self.hold_index_pull], self.conductance_pull[self.hold_index_pull],
                        color=accent_colors[0], marker='x', markersize=3, markeredgewidth=0.5)
                ax.axhline(self.hold_set_pull, lw=0.5, ls='-.', c=accent_colors[0])

            if self.hold_index_push is not None:
                ax.plot(piezo_push[self.hold_index_push], self.conductance_push[self.hold_index_push],
                        color=accent_colors[1], marker='x', markersize=3, markeredgewidth=0.5)
                ax.axhline(self.hold_set_push, lw=0.5, ls='-.', c=accent_colors[1])

        ax.set_ylabel(r'Conductance [$G_{0}$]')

        return ax

    # def cut_tunnel_part(self, leave_points: int = 3000):
    #
    #     cut_here = np.where(self.conductance_pull < 1e-5)[0][0]
    #     self.conductance_pull = self.conductance_pull[:cut_here + leave_points]
    #     self.piezo_pull = self.piezo_pull[:cut_here + leave_points]
    #
    #     cut_here = np.where(self.conductance_push < 1e-5)[0][-1]
    #     self.conductance_push = self.conductance_push[cut_here - leave_points:]
    #     self.piezo_push = self.piezo_push[cut_here - leave_points:]

    # def temporal_noise(self, align_at: Optional[float] = None, interpolate: bool = True,
    #                    mode='whole', win_size: int = 512, step_size: Optional[float] = None,
    #                    width: Optional[int] = None  # ,
    #                    # cut_tunnel_to_len: Optional[int] = None
    #                    ):
    #
    #     if step_size is None:
    #         step_size = win_size // 2
    #
    #     if width is None:
    #         width = min(len(self.conductance_pull), len(self.conductance_push))
    #
    #     # if cut_tunnel_to_len is not None:
    #     #     self.cut_tunnel_part(leave_points=cut_tunnel_to_len)
    #
    #     if align_at is not None:
    #         self.align_trace(align_at=align_at, interpolate=interpolate)
    #
    #     if mode == 'whole':
    #         self.cond_avg_pull = utils.mov_avg(self.conductance_pull[:width],
    #                                            win_size=win_size, step_size=step_size,
    #                                            avg_type=utils.log_avg)
    #
    #         self.f_pull, self.t_pull, self.Zxx_pull = scipy.signal.stft(self.conductance_pull[:width],
    #                                                                     fs=self.sample_rate,
    #                                                                     window='hann', nperseg=win_size,
    #                                                                     noverlap=win_size - step_size,
    #                                                                     nfft=None, detrend=False, return_onesided=True,
    #                                                                     boundary='even', padded=True, axis=- 1,
    #                                                                     scaling='psd')
    #
    #         self.cond_avg_push = utils.mov_avg(self.conductance_push[:width],
    #                                            win_size=win_size, step_size=step_size,
    #                                            avg_type=utils.log_avg)
    #
    #         self.f_push, self.t_push, self.Zxx_push = scipy.signal.stft(self.conductance_push[:width],
    #                                                                     fs=self.sample_rate,
    #                                                                     window='hann', nperseg=win_size,
    #                                                                     noverlap=win_size - step_size,
    #                                                                     nfft=None, detrend=False, return_onesided=True,
    #                                                                     boundary='even', padded=True, axis=- 1,
    #                                                                     scaling='psd')
    #     elif mode == 'positive':
    #         self.cond_avg_pull = utils.mov_avg(self.conductance_pull[self.aligned_piezo_pull >= 0][:width],
    #                                            win_size=win_size, step_size=step_size,
    #                                            avg_type=utils.log_avg)
    #
    #         self.f_pull, self.t_pull, self.Zxx_pull = scipy.signal.stft(
    #             self.conductance_pull[self.aligned_piezo_pull >= 0][:width],
    #             fs=self.sample_rate,
    #             window='hann', nperseg=win_size, noverlap=win_size - step_size,
    #             nfft=None, detrend=False, return_onesided=True,
    #             boundary='even', padded=True, axis=- 1,
    #             scaling='psd')
    #
    #         self.cond_avg_push = utils.mov_avg(self.conductance_push[self.aligned_piezo_push >= 0][:width],
    #                                            win_size=win_size, step_size=step_size,
    #                                            avg_type=utils.log_avg)
    #
    #         self.f_push, self.t_push, self.Zxx_push = scipy.signal.stft(
    #             self.conductance_push[self.aligned_piezo_push >= 0][::-1][:width],
    #             fs=self.sample_rate,
    #             window='hann', nperseg=win_size, noverlap=win_size - step_size,
    #             nfft=None, detrend=False, return_onesided=True,
    #             boundary='even', padded=True, axis=- 1,
    #             scaling='psd')
    #     else:
    #         raise ValueError(f'Unknown parameter for mode {mode}. Valid choices: "whole", "positive".')
    #
    # def plot_temporal_noise(self, mode: str = 'whole', normalize: bool = False,
    #                         add_vlines: Tuple[float, ...] = tuple(),
    #                         add_hlines: Tuple[float, ...] = tuple(),
    #                         ax: Optional[Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]] = None,
    #                         piezo_range_pull: Optional[Tuple[float, float]] = None,
    #                         piezo_range_push: Optional[Tuple[float, float]] = None,
    #                         conductance_range: Tuple[float, float] = (1e-6, 100),
    #                         vmax_pull: float = 0.1, vmax_push: float = 0.1,
    #                         dpi: int = 300):
    #     """
    #
    #     Parameters
    #     ----------
    #     mode
    #     normalize
    #     add_vlines
    #     add_hlines
    #     ax
    #     piezo_range_pull
    #     piezo_range_push
    #     conductance_range
    #     vmax_pull
    #     vmax_push
    #     dpi
    #
    #     Returns
    #     -------
    #     To plot individual PSDs: ax.plot(self.f, np.abs(self.Zxx)[:,i]/self.cond_avg[i])
    #     """
    #     if ax is None:
    #         fig = plt.figure(figsize=utils.cm2inch(10, 10), dpi=dpi)  # figsize: (width, height) in inches
    #         gs = gridspec.GridSpec(nrows=2, ncols=1,
    #                                figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0.3)
    #
    #         ax_pull = fig.add_subplot(gs[0])
    #         ax_push = fig.add_subplot(gs[1])
    #     else:
    #         ax_pull = ax[0]
    #         ax_push = ax[1]  # width, height
    #
    #     parx_pull = ax_pull.twinx()
    #     pary_pull = ax_pull.twiny()
    #
    #     parx_push = ax_push.twinx()
    #     pary_push = ax_push.twiny()
    #
    #     ax_pull.set_title('STFT Magnitude')
    #     ax_pull.set_ylabel('Frequency [Hz]')
    #     ax_pull.set_xlabel('Time [s]')
    #     pary_pull.set_xlabel('Piezo [V]')
    #     parx_pull.set_ylabel(r'Conductance [$G_{0}$]')
    #     ax_pull.set_yscale('log')
    #     ax_pull.set_ylim(100, 25000)
    #     # print(ax.get_xlim())
    #     parx_pull.set_yscale('log')
    #
    #     # if mode == 'whole':
    #     shift_pull = min(self.aligned_time_pull)
    #
    #     if piezo_range_pull is None:
    #         # set time limits
    #         ax_pull.set_xlim(min(self.aligned_time_pull),
    #                          max(self.aligned_time_pull))
    #         # set piezo limits
    #         pary_pull.set_xlim(min(self.aligned_piezo_pull),
    #                            max(self.aligned_piezo_pull))
    #     else:
    #         # set time limits
    #         ax_pull.set_xlim(min(piezo_range_pull) / self.rate,
    #                          max(piezo_range_pull) / self.rate)
    #         # set piezo limits
    #         pary_pull.set_xlim(min(piezo_range_pull),
    #                            max(piezo_range_pull))
    #
    #     parx_pull.plot(self.aligned_time_pull,
    #                    self.conductance_pull, 'k', lw=0.5, zorder=100)
    #
    #     shift_push = min(self.aligned_time_push)
    #
    #     if piezo_range_push is None:
    #         # set time limits
    #         ax_push.set_xlim(max(self.aligned_time_push),
    #                          min(self.aligned_time_push))
    #         # set piezo limits
    #         pary_push.set_xlim(max(self.aligned_piezo_push),
    #                            min(self.aligned_piezo_push))
    #     else:
    #         # set time limits
    #         ax_push.set_xlim(max(piezo_range_push) / self.rate,
    #                          min(piezo_range_push) / self.rate)
    #         # set piezo limits
    #         pary_push.set_xlim(max(piezo_range_push),
    #                            min(piezo_range_push))
    #
    #     parx_push.plot(self.aligned_time_push,
    #                    self.conductance_push, 'k', lw=0.5, zorder=100)
    #     # elif mode == 'positive':
    #     #     # set time limits
    #     #     ax_pull.set_xlim(min(self.aligned_time_pull[self.aligned_piezo_pull >= 0]),
    #     #                      max(self.aligned_time_pull[self.aligned_piezo_pull >= 0]))
    #     #     shift_pull = min(self.aligned_time_pull[self.aligned_piezo_pull >= 0])
    #     #     # set piezo limits
    #     #     pary_pull.set_xlim(min(self.aligned_piezo_pull[self.aligned_piezo_pull >= 0]),
    #     #                        max(self.aligned_piezo_pull[self.aligned_piezo_pull >= 0]))
    #     #     # plot conductance vs time
    #     #     parx_pull.plot(self.aligned_time_pull[self.aligned_piezo_pull >= 0],
    #     #                    self.conductance_pull[self.aligned_piezo_pull >= 0], 'k', lw=0.5, zorder=100)
    #     #
    #     #     # set time limits
    #     #     ax_push.set_xlim(min(self.aligned_time_push[self.aligned_piezo_push >= 0]),
    #     #                      max(self.aligned_time_push[self.aligned_piezo_push >= 0]))
    #     #     shift_push = min(self.aligned_time_push[self.aligned_piezo_push >= 0])
    #     #
    #     #     # set piezo limits
    #     #     pary_push.set_xlim(min(self.aligned_piezo_push[self.aligned_piezo_push >= 0]),
    #     #                        max(self.aligned_piezo_push[self.aligned_piezo_push >= 0]))
    #     #
    #     #     # plot conductance vs time
    #     #     parx_push.plot(self.aligned_time_push[self.aligned_piezo_push >= 0],
    #     #                    self.conductance_push[self.aligned_piezo_push >= 0][::-1], 'k', lw=0.5, zorder=100)
    #     # else:
    #     #     raise ValueError(f'Unknown parameter for mode {mode}. Valid choices: "whole", "positive".')
    #
    #     if normalize:
    #         pcm_pull = ax_pull.pcolormesh(self.t_pull, self.f_pull,
    #                                       np.abs(self.Zxx_pull) / self.cond_avg_pull,
    #                                       vmin=0, vmax=vmax_pull, shading='gouraud',
    #                                       cmap='gist_rainbow', zorder=0)
    #
    #         pcm_push = ax_push.pcolormesh(self.t_push, self.f_push,
    #                                       np.abs(self.Zxx_push) / self.cond_avg_push,
    #                                       vmin=0, vmax=vmax_push, shading='gouraud',
    #                                       cmap='gist_rainbow', zorder=0)
    #     else:
    #         pcm_pull = ax_pull.pcolormesh(self.t_pull, self.f_pull,
    #                                       np.abs(self.Zxx_pull),
    #                                       vmin=0, vmax=vmax_pull, shading='gouraud',
    #                                       cmap='gist_rainbow', zorder=0)
    #
    #         pcm_push = ax_push.pcolormesh(self.t_push, self.f_push,
    #                                       np.abs(self.Zxx_push),
    #                                       vmin=0, vmax=vmax_push, shading='gouraud',
    #                                       cmap='gist_rainbow', zorder=0)
    #
    #     # ax_push.set_title('STFT Magnitude')
    #     ax_push.set_ylabel('Frequency [Hz]')
    #     ax_push.set_xlabel('Time [s]')
    #     pary_push.set_xlabel('Piezo [V]')
    #     parx_push.set_ylabel(r'Conductance [$G_{0}$]')
    #     ax_push.set_yscale('log')
    #     ax_push.set_ylim(100, 25000)
    #     # print(ax.get_xlim())
    #     parx_push.set_yscale('log')
    #
    #     parx_pull.set_ylim(conductance_range)
    #     parx_push.set_ylim(conductance_range)
    #
    #     for i in add_hlines:
    #         parx_pull.axhline(i, lw=0.5, color='white', ls='--')
    #         parx_push.axhline(i, lw=0.5, color='white', ls='--')
    #
    #     for i in add_vlines:
    #         parx_pull.axvline(i, lw=0.5, color='white', ls='--')
    #         parx_push.axvline(i, lw=0.5, color='white', ls='--')
    #
    #     fig.colorbar(pcm_pull, ax=pary_pull, pad=0.1)
    #     fig.colorbar(pcm_push, ax=pary_push, pad=0.1)
    #
    #     return ax_pull, ax_push


class Histogram:
    """
    Class for statistical analysis of break junction traces

    Parameters
    ----------
    folder : Path
        path to reach files containing the data
    start_trace : int
        1st trace for statistical analysis
    end_trace : int
        last trace for statistical analysis
    conductance_range : Tuple[float, float]

    conductance_bins_num : int
        number of bins
    conductance_log_scale : bool
    conductance_bins_mode : str
    align_at : float


    Attributes
    ----------
    folder : Path
        path to reach files containing the data
    start_trace : int
        trace number of 1st trace taken in the statistics
    end_trace  : int
        trace number of last trace taken in the statistics
    traces : np.ndarray
        array of trace numbers that the histogram is calculated from
    conductance_range : Tuple[float, float]
        conductance range for the calculation of the statistics
    conductance_bins_num : np.ndarray
        bins constructed to calculate 1d histograms
    conductance_log_scale : bool
        if True, conductance axis is set to log scale and the histograms are calculated log binned
    bj_files : numpy.ndarray
        collected paths to break junction files that contain traces from `start_trace` and `end_trace`
    temporal_hist_pull : numpy.ndarray
        temporal histogram from individual 1d conductance histograms of pull traces
    temporal_hist_push : numpy.ndarray
        temporal histogram from individual 1d conductance histograms of push traces
    trace_length_pull : numpy.ndarray
        length of each pull trace
    trace_length_push : numpy.ndarary
        length of each push trace
    plateau_length_pull : numpy.ndarray
        length of the selected plateau in points or in V for each pull trace
    plateau_length_push  : numpy.ndarray
        length of the selected plateau in points or in V for each push trace
    hist_plateau_length_bins : numpy.ndarray
        location of bins of the plateau length histogram
    hist_plateau_length_pull : numpy.ndarray
        counts of the plateau length histogram of pull traces
    hist_plateau_length_push : numpy.ndarray
        counts of the plateau length histogram of push traces
    hist_1d_bins : numpy.ndarray
        location bins of the conductance histogram
    hist_1d_pull : numpy.ndarray
        counts of the conductance histogram of pull traces
    hist_1d_push : numpy.ndarray
        counts of the conductance histogram of push traces
    align_at : float
        conductance value where the traces are aligned for the calculation of 2d conductance histograms
    hist_2d_xmesh_pull : numpy.ndarray
        horizontal coordinates for the points of 2d histogram of the pull direction
    hist_2d_ymesh_pull : numpy.ndarray
        vertical coordinates for the points of 2d histogram of the pull direction
    hist_2d_pull : numpy.ndarray
        counts of the 2d histogram from pull traces
    hist_2d_xmesh_push : numpy.ndarray
        horizontal coordinates for the points of 2d histogram of the push direction
    hist_2d_ymesh_push : numpy.ndarray
        vertical coordinates for the points of 2d histogram of the push direction
    hist_2d_push : numpy.ndarray
        counts of the 2d histogram from push traces
    color_pull : str
        color for the pull histogram
    color_push : str
        color for the push histogram
    blues : List[str, ...]
        blue colors for plots
    reds : List[str, ...]
        red colors
    cmap_geo32 : colormap
        colormap for 2d histograms, same as in IgorPro

    """

    def __init__(self, folder: Path,
                 load_from: Optional[Union[str, Path]] = None,
                 traces: Optional[Union[np.array, List[int]]] = None,
                 start_trace: int = 1,
                 end_trace: Optional[int] = None,
                 conductance_range: Tuple[float, float] = (1e-6, 10),
                 conductance_bins_num: int = 100,
                 conductance_log_scale: bool = True,
                 conductance_bins_mode='total'):

        # Path to data
        self.folder = folder
        if traces is None:
            self.start_trace = start_trace
            if end_trace is None:
                self.end_trace = 0
            else:
                self.end_trace = end_trace
            self.traces = np.arange(start=self.start_trace, stop=self.end_trace+1, step=1)
        else:
            self.traces = traces
            self.start_trace = min(self.traces)
            self.end_trace = max(self.traces)
        self.conductance_range = conductance_range
        self.conductance_bins_num = conductance_bins_num
        self.conductance_log_scale = conductance_log_scale
        self.conductance_bins_mode = conductance_bins_mode

        # Temporal histogram
        self.temporal_hist_pull = None
        self.temporal_hist_push = None
        # Trace len histogram
        # self.trace_length_pull = None
        # self.trace_length_push = None
        # Plateau length histogram
        self.plateau_length_pull = []
        self.plateau_length_push = []
        self.hist_plateau_length_pull = None
        self.hist_plateau_length_push = None
        self.hist_plateau_length_bins = None
        #
        self.time_until_hold_pull = []
        self.time_until_hold_push = []
        self.times_until_hold_hist_pull = None
        self.times_until_hold_hist_bins = None
        # 1D histogram
        self.hist_1d_bins = None
        self.hist_1d_pull = None
        self.hist_1d_push = None
        # 2D histogram
        self.align_at = None

        self.hist_2d_xmesh_pull = None
        self.hist_2d_ymesh_pull = None
        self.hist_2d_pull = None
        self.hist_2d_xmesh_push = None
        self.hist_2d_ymesh_push = None
        self.hist_2d_push = None

        self.corr_2d_pull = None
        self.corr_2d_push = None
        self.cross_corr_2d = None

        # formatting
        self.color_pull = 'cornflowerblue'
        self.color_push = 'firebrick'

        self.all_traces = []
        self.filtered_traces = []

        self.collected_errors_alignment = []

        if load_from is not None:
            self.load_histogram(fname=load_from)
    
    def save_histogram(self, fname: Union[str, Path]):
        if isinstance(fname, str):
            if not self.folder.joinpath('results').exists():
                mkdir(self.folder.joinpath('results'))
            if not self.folder.joinpath('results/histograms').exists():
                mkdir(self.folder.joinpath('results/histograms'))
            fname = self.folder.joinpath(f'results/histograms/{fname}')
        with h5py.File(fname, 'w') as f:
            # f.create_dataset('all_traces', self.all_traces)
            # f.create_dataset('filtered_traces', self.filtered_traces)
            traces_dset = f.create_dataset('traces', data=self.traces)
            traces_dset.attrs['folder'] = str(self.folder)

            if self.hist_1d_bins is not None:
                hist_1d_dset = f.create_dataset('hist_1d_bins', data=self.hist_1d_bins)
                hist_1d_dset.attrs['conductance_bins_mode'] = self.conductance_bins_mode
                hist_1d_dset.attrs['conductance_bins_num'] = self.conductance_bins_num
                hist_1d_dset.attrs['conductance_log_scale'] = self.conductance_log_scale
                hist_1d_dset.attrs['conductance_range'] = self.conductance_range
                f.create_dataset('hist_1d_pull', data=self.hist_1d_pull)
                f.create_dataset('hist_1d_push', data=self.hist_1d_push)

            if self.hist_2d_pull is not None:
                f.create_dataset('hist_2d_pull', data=self.hist_2d_pull)
                f.create_dataset('hist_2d_xmesh_pull', data=self.hist_2d_xmesh_pull)
                f.create_dataset('hist_2d_ymesh_pull', data=self.hist_2d_ymesh_pull)
            if self.hist_2d_push is not None:
                f.create_dataset('hist_2d_push', data=self.hist_2d_push)
                f.create_dataset('hist_2d_xmesh_push', data=self.hist_2d_xmesh_push)
                f.create_dataset('hist_2d_ymesh_push', data=self.hist_2d_ymesh_push)

            if self.corr_2d_pull is not None:
                f.create_dataset('corr_2d_pull', data=self.corr_2d_pull)
                f.create_dataset('corr_2d_push', data=self.corr_2d_push)
                f.create_dataset('cross_corr_2d', data=self.cross_corr_2d)

            if self.temporal_hist_pull is not None:
                f.create_dataset('temporal_hist_pull', data=self.temporal_hist_pull)
                f.create_dataset('temporal_hist_push', data=self.temporal_hist_push)

            if self.hist_plateau_length_bins is not None:
                f.create_dataset('hist_plateau_length_bins', data=self.hist_plateau_length_bins)
                f.create_dataset('hist_plateau_length_pull', data=self.hist_plateau_length_pull)
                f.create_dataset('hist_plateau_length_push', data=self.hist_plateau_length_push)

            if self.plateau_length_pull is not None:
                f.create_dataset('plateau_length_pull', data=self.plateau_length_pull)
                f.create_dataset('plateau_length_push', data=self.plateau_length_push)

            if self.times_until_hold_hist_bins is not None:
                f.create_dataset('time_until_hold_pull', data=self.time_until_hold_pull)
                f.create_dataset('time_until_hold_push', data=self.time_until_hold_push)
                f.create_dataset('times_until_hold_hist_bins', data=self.times_until_hold_hist_bins)
                f.create_dataset('times_until_hold_hist_pull', data=self.times_until_hold_hist_pull)

        print(f'Histogram saved to {fname}.')

    def load_histogram(self, fname: Union[str, Path]):
        if isinstance(fname, str):
            fname = self.folder.joinpath(f'results/histograms/{fname}')
        with h5py.File(fname, 'r') as f:
            file_keys = f.keys()

            self.folder = Path(f['traces'].attrs['folder'])
            self.traces = f['traces'][:]

            if 'hist_1d_bins' in file_keys:
                self.hist_1d_bins = f['hist_1d_bins'][:]
                self.conductance_bins_mode = f['hist_1d_bins'].attrs['conductance_bins_mode']
                self.conductance_bins_num = f['hist_1d_bins'].attrs['conductance_bins_num']
                self.conductance_log_scale = f['hist_1d_bins'].attrs['conductance_log_scale']
                self.conductance_range = f['hist_1d_bins'].attrs['conductance_range']
                self.hist_1d_pull = f['hist_1d_pull'][:]
                self.hist_1d_push = f['hist_1d_push'][:]

            if 'hist_2d_pull' in file_keys:
                self.hist_2d_pull = f['hist_1d_pull'][:]
                self.hist_2d_xmesh_pull = f['hist_2d_xmesh_pull'][:]
                self.hist_2d_ymesh_pull = f['hist_2d_ymesh_pull'][:]
            if 'hist_2d_push' in file_keys:
                self.hist_2d_push = f['hist_1d_push'][:]
                self.hist_2d_xmesh_push = f['hist_2d_xmesh_push'][:]
                self.hist_2d_ymesh_push = f['hist_2d_ymesh_push'][:]

            if 'corr_2d_pull' in file_keys:
                self.corr_2d_pull = f['corr_2d_pull'][:]
                self.corr_2d_push = f['corr_2d_push'][:]
                self.cross_corr_2d = f['cross_corr_2d'][:]

            if 'temporal_hist_pull' in file_keys:
                self.temporal_hist_pull = f['temporal_hist_pull'][:]
                self.temporal_hist_push =f['temporal_hist_push'][:]

            if self.hist_plateau_length_bins is not None:
                self.hist_plateau_length_bins = f['hist_plateau_length_bins'][:]
                self.hist_plateau_length_pull = f['hist_plateau_length_pull'][:]
                self.hist_plateau_length_push = f['hist_plateau_length_push'][:]

            if self.plateau_length_pull is not None:
                self.plateau_length_pull = f['plateau_length_pull'][:]
                self.plateau_length_push = f['plateau_length_push'][:]

            if self.times_until_hold_hist_bins is not None:
                self.time_until_hold_pull = f['time_until_hold_pull'][:]
                self.time_until_hold_push = f['time_until_hold_push'][:]
                self.times_until_hold_hist_bins = f['times_until_hold_hist_bins'][:]
                self.times_until_hold_hist_pull = f['times_until_hold_hist_pull'][:]

    def calc_temporal_hist(self):
        """
        Calculate single conductance 1d histogram for each included trace, and collect them to create a
        temporal histogram of the break junction traces

        See Also
        --------
        utils.calc_hist_1d_single

        """
        single_hist_list_pull = []
        single_hist_list_push = []

        trace_len_pull = []
        trace_len_push = []

        for trace in self.traces:
            trace_pair = TracePair(trace=trace, load_from=self.folder)
            hist_bins, single_hist_1d = utils.calc_hist_1d_single(trace_pair.conductance_pull,
                                                                  xrange=self.conductance_range,
                                                                  xbins_num=self.conductance_bins_num,
                                                                  log_scale=self.conductance_log_scale,
                                                                  bin_mode=self.conductance_bins_mode)

            single_hist_list_pull.append(single_hist_1d)
            trace_len_pull.append(trace_pair.conductance_pull.shape[0])

            hist_bins, single_hist_1d = utils.calc_hist_1d_single(trace_pair.conductance_push,
                                                                  xrange=self.conductance_range,
                                                                  xbins_num=self.conductance_bins_num,
                                                                  log_scale=self.conductance_log_scale,
                                                                  bin_mode=self.conductance_bins_mode)

            single_hist_list_push.append(single_hist_1d)
            trace_len_push.append(trace_pair.conductance_push.shape[0])

        self.hist_1d_bins = hist_bins
        self.temporal_hist_pull = np.array(single_hist_list_pull)
        self.temporal_hist_push = np.array(single_hist_list_push)

    def plot_temporal_hist(self,
                           ax: Optional[Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]] = None,
                           add_hlines: Tuple[float, ...] = tuple(),
                           add_vlines: Tuple[float, ...] = tuple(),
                           dpi: int = 600,
                           vmin_pull: Optional[float] = None,
                           vmax_pull: Optional[float] = None,
                           vmin_push: Optional[float] = None,
                           vmax_push: Optional[float] = None,
                           **kwargs) -> Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
        """
        Plot temporal histograms

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            if there is a pre-existing axes for the 2 directions, you can plot it there.
            Otherwise, a new figure with a new axis will be created.
        add_hlines : Optional[Tuple[float, ...]]
            values to put horizontal lines at for annotation
        add_vlines : Optional[Tuple[float, ...]]
            values to put vertical lines at for annotation
        dpi : int, default: 600
            resolution in dots per inch
        vmin_pull : Optional[float], default: None,
        vmax_pull : Optional[float], default: None,
        vmin_push : Optional[float], default: None,
        vmax_push : Optional[float], default: None,
        kwargs :
            additional arguments for `pcolormesh` plots

        Returns
        -------
        ax_pull : `~matplotlib.axes.Axes`
            the axis with the resulting plot of the pull traces for further formatting if necessary
        ax_push : `~matplotlib.axes.Axes`
            the axis with the resulting plot of the push traces for further formatting if necessary

        """
        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(10, 10), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=2, ncols=1,
                                   figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0.1)

            ax_pull = fig.add_subplot(gs[0])
            ax_push = fig.add_subplot(gs[1])
        else:
            ax_pull = ax[0]
            ax_push = ax[1]

        ax_pull.set_ylabel(r'Conductance $[G_{0}]$')
        ax_push.set_ylabel(r'Conductance $[G_{0}]$')
        ax_pull.set_xlabel(r'Trace number')
        ax_push.set_xlabel(r'Trace number')

        ax_pull.xaxis.set_ticks_position('both')
        ax_push.xaxis.set_ticks_position('both')
        ax_pull.yaxis.set_ticks_position('both')
        ax_push.yaxis.set_ticks_position('both')

        ax_pull.xaxis.tick_top()
        ax_pull.xaxis.set_label_position('top')

        if self.conductance_log_scale:
            ax_pull.set_yscale('log')
            ax_push.set_yscale('log')

            ax_pull.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
            ax_push.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
            ax_pull.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=9))
            ax_push.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=9))
            ax_pull.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax_push.yaxis.set_minor_formatter(ticker.NullFormatter())

        x_edges = np.arange(0, self.temporal_hist_pull.shape[0], 1)
        y_edges = self.hist_1d_bins

        x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)

        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = utils.cmap_geo32

        im_norm_pull = ax_pull.pcolormesh(x_mesh, y_mesh, self.temporal_hist_pull.T,
                                          vmin=vmin_pull, vmax=vmax_pull, **kwargs)
        im_norm_push = ax_push.pcolormesh(x_mesh, y_mesh, self.temporal_hist_push.T,
                                          vmin=vmin_push, vmax=vmax_push, **kwargs)

        ax_pull.set_xlim(min(x_mesh.flatten()), max(x_mesh.flatten()))
        ax_push.set_xlim(min(x_mesh.flatten()), max(x_mesh.flatten()))

        for i in add_hlines:
            ax_pull.axhline(i, lw=0.5, color='k', ls='--')
            ax_push.axhline(i, lw=0.5, color='k', ls='--')

        for i in add_vlines:
            ax_pull.axvline(i, lw=0.5, color='k', ls='--')
            ax_push.axvline(i, lw=0.5, color='k', ls='--')

        return ax_pull, ax_push

    def calc_hist_1d(self):
        """
        Calculate 1d conductance histogram based on the parameters provided for the defined Histogram object

        See Also
        --------
        utils.calc_hist_1d_single

        """
        # reset values in case the 1d histogram was calculated previously
        self.hist_1d_pull = None
        self.hist_1d_push = None

        if self.temporal_hist_pull is not None and self.temporal_hist_push is not None:
            # if the temporal histograms are given, use them to sum the single histograms
            self.hist_1d_pull = np.mean(self.temporal_hist_pull, axis=0)
            self.hist_1d_push = np.mean(self.temporal_hist_push, axis=0)
        else:
            for trace in tqdm(self.traces):
                trace_pair = TracePair(trace=trace, load_from=self.folder)
                hist_bins, single_hist_1d = utils.calc_hist_1d_single(trace_pair.conductance_pull,
                                                                      xrange=self.conductance_range,
                                                                      xbins_num=self.conductance_bins_num,
                                                                      log_scale=self.conductance_log_scale,
                                                                      bin_mode=self.conductance_bins_mode)

                try:
                    self.hist_1d_pull = self.hist_1d_pull + single_hist_1d / len(self.traces)
                except (ValueError, TypeError):
                    self.hist_1d_pull = single_hist_1d / len(self.traces)

                hist_bins, single_hist_1d = utils.calc_hist_1d_single(trace_pair.conductance_push,
                                                                      xrange=self.conductance_range,
                                                                      xbins_num=self.conductance_bins_num,
                                                                      log_scale=self.conductance_log_scale,
                                                                      bin_mode=self.conductance_bins_mode)

                try:
                    self.hist_1d_push = self.hist_1d_push + single_hist_1d / len(self.traces)
                except (ValueError, TypeError):
                    self.hist_1d_push = single_hist_1d / len(self.traces)

            self.hist_1d_bins = hist_bins

    def plot_hist_1d(self,
                     ax=None,
                     ylims: Optional[Tuple[float, float]] = None,
                     add_hlines: Tuple[float, ...] = tuple(),
                     add_vlines: Tuple[float, ...] = tuple(),
                     dpi: int = 600):
        """
        Plot 1d conductance histogram, and add annotations

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            if there is a pre-existing axis, you can plot there.
            Otherwise, a new figure with a new axis will be created.
        ylims : Tuple[float, float]
            set the limits of the vertical axis
        add_hlines : Optional[Tuple[float, ...]]
            values to put horizontal lines at for annotation
        add_vlines : Optional[Tuple[float, ...]]
            values to put vertical lines at for annotation
        dpi : int, resolution in dots per inch

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            the axis with the resulting plot for further formatting if necessary

        """

        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(10, 5), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=1, ncols=1,
                                   figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.2, wspace=0, hspace=0.1)
            ax = fig.add_subplot(gs[0])

        ax.set_xlabel(r'Conductance $[G_{0}]$')
        ax.set_ylabel(r'Normalized counts [a.u.]')

        ax.set_xscale('log')
        ax.set_xlim(min(self.hist_1d_bins), max(self.hist_1d_bins))

        ax.plot(self.hist_1d_bins, self.hist_1d_pull, color=self.color_pull)
        ax.fill_between(self.hist_1d_bins, np.zeros_like(self.hist_1d_pull), self.hist_1d_pull, alpha=0.5,
                        color=self.color_pull)
        ax.plot(self.hist_1d_bins, self.hist_1d_push, color=self.color_push, alpha=0.5)

        if ylims is None:
            ax.set_ylim(bottom=0, top=None)
        else:
            ax.set_ylim(ylims)

        for i in add_hlines:
            ax.axhline(i, lw=0.5, color='k', ls='--')

        for i in add_vlines:
            ax.axvline(i, lw=0.5, color='k', ls='--')

        return ax

    def calc_hist_2d(self,
                     align_at: float,
                     interpolate: bool = True,
                     range_pull: Optional[Tuple[float, float]] = (-1.5, 1),
                     range_push: Optional[Tuple[float, float]] = (-3, 1),
                     xbins_pull: Optional[int] = 100,
                     xbins_push: Optional[int] = 100):
        """
        Calculate 2d conductance-displacement histogram for pull and push traces aligned at the conductance value
        *align_at* in the displacement range *range_pull* and *range_push*, respectively

        Parameters
        ----------
        align_at : float,
            conductance value where each array is scaled together
        interpolate : bool
            if True, the shift value for aligning the traces is determined via interpolation
        range_pull : Tuple[float, float]
            piezo range to calculate the pull 2d histogram
        range_push : Tuple[float, float]
            piezo range to calculate the push 2d histogram
        xbins_pull : int
            number of bins along the piezo axis for the pull histogram
        xbins_push : int
            number of bins along the piezo axis for the push histogram

        See Also
        --------
        utils.calc_hist_2d_single

        """

        self.align_at = align_at

        self.hist_2d_xmesh_pull = None
        self.hist_2d_ymesh_pull = None
        self.hist_2d_pull = None
        self.hist_2d_xmesh_push = None
        self.hist_2d_ymesh_push = None
        self.hist_2d_push = None

        x_mesh_pull = None
        y_mesh_pull = None
        x_mesh_push = None
        y_mesh_push = None

        count_pull = 0
        count_push = 0

        for trace in tqdm(self.traces):
            trace_pair = TracePair(trace=trace, load_from=self.folder)
            trace_pair.align_trace(align_at=self.align_at, interpolate=interpolate)
            try:
                x_mesh_pull, y_mesh_pull, single_hist_2d = utils.calc_hist_2d_single(
                    x=trace_pair.aligned_piezo_pull,
                    y=trace_pair.conductance_pull,
                    xrange=range_pull,
                    log_scale_x=False,
                    yrange=(1e-6, 10),
                    log_scale_y=True,
                    num_bins=(xbins_pull, self.conductance_bins_num),
                    bin_mode_y=self.conductance_bins_mode)

                try:
                    self.hist_2d_pull = self.hist_2d_pull + single_hist_2d
                except (ValueError, TypeError):
                    self.hist_2d_pull = single_hist_2d

                count_pull += 1

            except utils.MyException:
                pass

            try:
                x_mesh_push, y_mesh_push, single_hist_2d = utils.calc_hist_2d_single(
                    x=trace_pair.aligned_piezo_push[::-1],
                    y=trace_pair.conductance_push[::-1],
                    xrange=range_push,
                    log_scale_x=False,
                    yrange=(1e-6, 10),
                    log_scale_y=True,
                    num_bins=(xbins_push, self.conductance_bins_num),
                    bin_mode_y=self.conductance_bins_mode)

                try:
                    self.hist_2d_push = self.hist_2d_push + single_hist_2d
                except (ValueError, TypeError):
                    self.hist_2d_push = single_hist_2d

                count_push += 1

            except utils.MyException:
                pass

        print(f'2D histogram pull direction created from {count_pull} traces')
        print(f'2D histogram push direction created from {count_push} traces')

        self.hist_2d_xmesh_pull = x_mesh_pull
        self.hist_2d_ymesh_pull = y_mesh_pull
        self.hist_2d_xmesh_push = x_mesh_push
        self.hist_2d_ymesh_push = y_mesh_push

    def plot_hist_2d_one(self, direction: str = 'pull', ax: Optional[matplotlib.axes.Axes] = None,
                         dpi: int = 600, **kwargs):
        """
        Plot 2d conductance-displacement histogram for 'pull' or 'push' traces defined by the *direction* parameter

        Parameters
        ----------
        direction : str,
            the direction for which you want to plot the 2d histogram
        ax : `~matplotlib.axes.Axes`
            pre-defined axis for plot
        dpi : int, resolution in dots per inch
        kwargs :
            additional arguments for `~matplotlib.axes.Axes.pcolormesh`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            axis with plot

        """
        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(5, 5), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=1, ncols=1,
                                   figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.2, wspace=0, hspace=0.1)
            ax = fig.add_subplot(gs[0])

        if direction == 'pull':
            x_mesh = self.hist_2d_xmesh_pull
            y_mesh = self.hist_2d_ymesh_pull
            hist_2d = self.hist_2d_pull
            ax.set_xlim(min(x_mesh.flatten()), max(x_mesh.flatten()))
        elif direction == 'push':
            x_mesh = self.hist_2d_xmesh_push
            y_mesh = self.hist_2d_ymesh_push
            hist_2d = self.hist_2d_push
            ax.set_xlim(max(x_mesh.flatten()), min(x_mesh.flatten()))
        else:
            raise ValueError(f'Unknown parameter for direction: {direction}. Valid choices include: ["pull", "push"]')
        ax.set_yscale('log')

        ax.set_ylabel(r'Conductance $[G_{0}]$')
        ax.set_xlabel(r'Piezo $[V]$')

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=9))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        im_norm = ax.pcolormesh(x_mesh, y_mesh, hist_2d, cmap=utils.cmap_geo32, **kwargs)

        return ax

    def plot_hist_2d_both(self, dpi: int = 600, **kwargs):
        """
        Plot both pull and push 2d conductance-displacement histograms

        Parameters
        ----------
        dpi : int, default: 600
            resolution in dots per inch
        kwargs : additional parameters for `~matplotlib.axes.Axes.pcolormesh`

        Returns
        -------

        """
        fig = plt.figure(figsize=utils.cm2inch(15, 7), dpi=dpi)  # figsize: (width, height) in inches
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig,
                               left=0.15, right=0.9, top=0.9, bottom=0.15, wspace=0, hspace=0)

        ax_pull = fig.add_subplot(gs[0])
        ax_push = fig.add_subplot(gs[1])

        ax_push.yaxis.set_label_position("right")
        ax_push.yaxis.tick_right()

        ax_pull.xaxis.set_ticks_position('both')
        ax_pull.yaxis.set_ticks_position('both')
        ax_push.xaxis.set_ticks_position('both')
        ax_push.yaxis.set_ticks_position('both')

        ax_pull.set_yscale('log')
        ax_push.set_yscale('log')
        ax_pull.set_ylim(1e-6, 10)
        ax_push.set_ylim(1e-6, 10)

        ax_pull = self.plot_hist_2d_one(direction='pull', ax=ax_pull, **kwargs)
        ax_push = self.plot_hist_2d_one(direction='push', ax=ax_push, **kwargs)

        # push_xlims = ax_push.get_xlim()
        # ax_push.set_xlim(push_xlims[1], push_xlims[0])

        return ax_pull, ax_push

    def calc_corr_hist_2d(self):

        if self.temporal_hist_pull is None or self.temporal_hist_push is None or \
           self.hist_1d_pull is None or self.hist_1d_push is None:
            raise ValueError("Temporal histogram or 1D histogram empty. Run `hist.calc_stats` or "
                             "`hist.calc_temporal_hist` and `hist.calc_hist_1d` first.")

        hist_diff_pull = self.temporal_hist_pull - self.hist_1d_pull
        hist_diff_push = self.temporal_hist_push - self.hist_1d_push

        self.corr_2d_pull = np.zeros((hist_diff_pull.shape[1], hist_diff_pull.shape[1]))
        self.corr_2d_push = np.zeros((hist_diff_push.shape[1], hist_diff_push.shape[1]))
        self.cross_corr_2d = np.zeros((hist_diff_pull.shape[1], hist_diff_pull.shape[1]))

        for trace in tqdm(range(hist_diff_pull.shape[0])):
            self.corr_2d_pull += np.outer(hist_diff_pull[trace], hist_diff_pull[trace])
            self.corr_2d_push += np.outer(hist_diff_push[trace], hist_diff_push[trace])

            self.cross_corr_2d += np.outer(hist_diff_pull[trace], hist_diff_push[trace])

        self.cross_corr_2d /= np.sqrt(np.outer(np.diag(self.corr_2d_pull), np.diag(self.corr_2d_push)))
        self.corr_2d_pull /= np.sqrt(np.outer(np.diag(self.corr_2d_pull), np.diag(self.corr_2d_pull)))
        self.corr_2d_push /= np.sqrt(np.outer(np.diag(self.corr_2d_push), np.diag(self.corr_2d_push)))

    def plot_corr(self, mode: str = 'pull', dpi: int = 600, **kwargs):
        if mode == 'pull':
            corr_to_plot = self.corr_2d_pull
        elif mode == 'push':
            corr_to_plot = self.corr_2d_push
        elif mode == 'cross':
            corr_to_plot = self.cross_corr_2d
        else:
            raise ValueError("Unknown value {mode} for variable `mode`. Valid choices: 'pull', 'push', 'cross'.")

        fig = plt.figure(figsize=utils.cm2inch(5.25, 5), dpi=dpi)  # figsize: (width, height) in inches
        gs = gridspec.GridSpec(nrows=1, ncols=2,
                               figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.1,
                               wspace=0.1, width_ratios=(100, 5))

        ax_corr = fig.add_subplot(gs[0])
        ax_colorbar = fig.add_subplot(gs[1])

        ax_corr.set_xlabel(r'Conductance $[G_{0}]$')
        ax_corr.set_ylabel(r'Conductance $[G_{0}]$')

        ax_corr.xaxis.set_ticks_position('both')
        ax_corr.yaxis.set_ticks_position('both')

        if self.conductance_log_scale:
            ax_corr.set_xscale('log')
            ax_corr.set_yscale('log')
            ax_corr.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
            ax_corr.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=9))
            ax_corr.xaxis.set_minor_formatter(ticker.NullFormatter())

            ax_corr.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
            ax_corr.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.1), numticks=9))
            ax_corr.yaxis.set_minor_formatter(ticker.NullFormatter())

        x_mesh, y_mesh = np.meshgrid(self.hist_1d_bins, self.hist_1d_bins)

        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = 'bwr'

        im_norm = ax_corr.pcolormesh(x_mesh, y_mesh, corr_to_plot, **kwargs)

        # ax_corr.set_xlim(min(x_mesh.flatten()), max(x_mesh.flatten()))
        # ax_corr.set_ylim(min(y_mesh.flatten()), max(y_mesh.flatten()))

        plt.colorbar(im_norm, ax_colorbar)

        return ax_corr, ax_colorbar

    def calc_plateau_lengths(self,
                             low_bound_pull: float = 0.5, high_bound_pull: float = 1.5,
                             low_bound_push: float = 0.5, high_bound_push: float = 1.5):
        plateau_length_pull = []
        plateau_length_push = []

        for trace in self.traces:
            trace_pair = TracePair(trace=trace, load_from=self.folder,
                                   low_bound_pull=low_bound_pull, high_bound_pull=high_bound_pull,
                                   low_bound_push=low_bound_push, high_bound_push=high_bound_push)
            plateau_length_pull.append(trace_pair.plateau_length_pull)
            plateau_length_push.append(trace_pair.plateau_length_push)

        self.plateau_length_pull = np.array(plateau_length_pull)
        self.plateau_length_push = np.array(plateau_length_push)

    def calc_plateau_length_hist(self, xrange: Optional[Tuple[float, float]] = None,
                                 xbins: Optional[int] = 100):
        """
        Calculate plateau length histograms

        Parameters
        ----------
        xrange : Tuple[float, float]
            range for plateau length bins
        xbins : int
            number of bins

        Returns
        -------

        """
        plateau_length_pull = []
        plateau_length_push = []

        if self.plateau_length_pull is None or self.plateau_length_push is None:
            for trace in tqdm(self.traces):
                trace_pair = TracePair(trace=trace, load_from=self.folder)
                plateau_length_pull.append(trace_pair.plateau_length_pull)
                plateau_length_push.append(trace_pair.plateau_length_push)

            self.plateau_length_pull = np.array(plateau_length_pull)
            self.plateau_length_push = np.array(plateau_length_push)

        if xrange is None:
            xrange_max = (max(max(self.plateau_length_pull), max(self.plateau_length_push)) // 1000 + 1) * 1000
            xrange = (0, xrange_max)

        print(xrange)

        hist_bins, self.hist_plateau_length_pull = utils.calc_hist_1d_single(self.plateau_length_pull,
                                                                             xrange=xrange, xbins_num=xbins,
                                                                             log_scale=False)

        hist_bins, self.hist_plateau_length_push = utils.calc_hist_1d_single(self.plateau_length_push,
                                                                             xrange=xrange, xbins_num=xbins,
                                                                             log_scale=False)

        self.hist_plateau_length_bins = hist_bins

    def plot_plateau_length_hist(self,
                                 ax: matplotlib.axes.Axes = None,
                                 ylims: Tuple[float, float] = None,
                                 in_volts: Optional[bool] = False,
                                 add_hlines: Tuple[float, ...] = tuple(),
                                 add_vlines: Tuple[float, ...] = tuple(),
                                 dpi: int = 600):
        """
        Plot plateau length histograms

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            if there is a pre-existing axis, you can plot there.
            Otherwise, a new figure with a new axis will be created.
        ylims : Tuple[float, float]
            set the limits of the vertical axis
        in_volts : bool
            if True, plateau length is calculated in units of V, otherwise in points
        add_hlines : Optional[Tuple[float, ...]]
            values to put horizontal lines at for annotation
        add_vlines : Optional[Tuple[float, ...]]
            values to put vertical lines at for annotation
        dpi : int, default: 600
            resolution in dots per inch

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            the axis with the resulting plot for further formatting if necessary

        """

        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(10, 5), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=1, ncols=1,
                                   figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.2, wspace=0, hspace=0.1)
            ax = fig.add_subplot(gs[0])

        if in_volts:
            ax.set_xlabel(r'Plateau length [V]')
        else:
            ax.set_xlabel(r'Plateau length [points]')
        ax.set_ylabel(r'Counts [a.u.]')

        ax.set_xlim(min(self.hist_plateau_length_bins), max(self.hist_plateau_length_bins))

        ax.plot(self.hist_plateau_length_bins, self.hist_plateau_length_pull, color=self.color_pull)
        ax.fill_between(self.hist_plateau_length_bins, np.zeros_like(self.hist_plateau_length_pull),
                        self.hist_plateau_length_pull, alpha=0.5, color=self.color_pull)
        ax.plot(self.hist_plateau_length_bins, self.hist_plateau_length_push, color=self.color_push, alpha=0.5)

        if ylims is None:
            ax.set_ylim(bottom=0, top=None)
        else:
            ax.set_ylim(ylims)

        for i in add_hlines:
            ax.axhline(i, lw=0.5, color='k', ls='--')

        for i in add_vlines:
            ax.axvline(i, lw=0.5, color='k', ls='--')

        return ax

    def calc_time_until_hold_hist(self, xrange: Optional[Tuple[float, float]] = None, xbins: int = 100):
        # iterate over traces

        pull_times = []
        push_times = []

        for trace in tqdm(self.traces):
            trace_pair = TracePair(trace=trace, load_from=self.folder)

            try:
                pull_times.append((trace_pair.hold_index_pull - trace_pair.plateau_range_pull[1])
                                  / trace_pair.sample_rate)
                # push_times.append((trace_pair.plateau_range_push[0] - trace_pair.hold_index_push)
                #                   / trace_pair.sample_rate)
            except TypeError:
                continue

        self.time_until_hold_pull = np.array(pull_times)
        # self.time_until_hold_push = np.array(push_times)

        # now the histogram
        if xrange is None:
            # xrange_max = (max(max(self.time_until_hold_pull), max(self.time_until_hold_push)))
            xrange_max = max(self.time_until_hold_pull)
            xrange = (0, xrange_max)

        print(xrange)

        hist_bins, self.times_until_hold_hist_pull = utils.calc_hist_1d_single(self.time_until_hold_pull,
                                                                               xrange=xrange, xbins_num=xbins,
                                                                               log_scale=False)

        # hist_bins, self.times_until_hold_hist_push = utils.calc_hist_1d_single(self.time_until_hold_push,
        #                                                                        xrange=xrange, xbins_num=xbins,
        #                                                                        log_scale=False)

        self.times_until_hold_hist_bins = hist_bins

    def plot_time_until_hold_hist(self,
                                  ax: Optional[matplotlib.axes.Axes] = None,
                                  ylims: Tuple[float, float] = None,
                                  add_hlines: Tuple[float, ...] = tuple(),
                                  add_vlines: Tuple[float, ...] = tuple(),
                                  dpi: int = 600):
        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(10, 5), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=1, ncols=1,
                                   figure=fig, left=0.1, right=0.95, top=0.95, bottom=0.2, wspace=0, hspace=0.1)
            ax = fig.add_subplot(gs[0])

        ax.set_xlabel(r'Time [s]')
        ax.set_ylabel(r'Counts [a.u.]')

        ax.set_xlim(min(self.times_until_hold_hist_bins), max(self.times_until_hold_hist_bins))

        ax.plot(self.times_until_hold_hist_bins, self.times_until_hold_hist_pull, color=self.color_pull)
        ax.fill_between(self.times_until_hold_hist_bins, np.zeros_like(self.times_until_hold_hist_pull),
                        self.times_until_hold_hist_pull, alpha=0.5, color=self.color_pull)
        # ax.plot(self.times_until_hold_hist_bins, self.times_until_hold_hist_push, color=self.color_push, alpha=0.5)

        if ylims is None:
            ax.set_ylim(bottom=0, top=None)
        else:
            ax.set_ylim(ylims)

        for i in add_hlines:
            ax.axhline(i, lw=0.5, color='k', ls='--')

        for i in add_vlines:
            ax.axvline(i, lw=0.5, color='k', ls='--')

        return ax

    def calc_stats(self, align_at: float,
                   interpolate: bool = True,
                   range_pull: Tuple[float, float] = (-1.5, 1),
                   range_push: Tuple[float, float] = (-3, 1),
                   xbins_pull: int = 100,
                   xbins_push: int = 100,
                   plateau_length_bins: int = 100):
        """
        Perform all statistical analysis in one iteration

        Parameters
        ----------
        align_at : float,
            conductance value where each array is scaled together
        interpolate : bool
            if True, use interpolation for the determination of the shift of each trace
        range_pull : Tuple[float, float]
            piezo range to calculate the pull 2d histogram
        range_push : Tuple[float, float]
            piezo range to calculate the push 2d histogram
        xbins_pull : int
            number of bins along the piezo axis for the pull histogram
        xbins_push : int
            number of bins along the piezo axis for the push histogram
        plateau_length_bins : int

        Returns
        -------

        """

        self.align_at = align_at

        plateau_length_pull = []
        plateau_length_push = []

        single_hist_list_pull = []
        single_hist_list_push = []

        trace_len_pull = []
        trace_len_push = []

        count_pull = 0
        count_push = 0

        self.collected_errors_alignment = []

        for trace in tqdm(self.traces):
            # define TracePair instance
            trace_pair = TracePair(trace=trace, load_from=self.folder)
            try:
                trace_pair.align_trace(align_at=align_at, interpolate=interpolate)
            except IndexError:
                # warnings.warn(f'Trace {trace_pair.trace_num} did not cross align value {align_at}. '
                #               f'Skipping and deleting from included traces trace pair {trace_pair.trace_num}.')
                self.collected_errors_alignment.append(f'Trace {trace_pair.trace_num} did not cross align value '
                                                       f'{align_at}. skip and delete')
                self.traces = np.delete(self.traces, np.argwhere(self.traces == trace)[0][0])
                continue
            # Collect plateau length to create histogram
            plateau_length_pull.append(trace_pair.plateau_length_pull)
            plateau_length_push.append(trace_pair.plateau_length_push)
            # Calculate and collect single 1d histograms
            # pull
            hist_bins, single_hist_1d = utils.calc_hist_1d_single(trace_pair.conductance_pull,
                                                                  xrange=self.conductance_range,
                                                                  xbins_num=self.conductance_bins_num,
                                                                  log_scale=self.conductance_log_scale,
                                                                  bin_mode=self.conductance_bins_mode)

            single_hist_list_pull.append(single_hist_1d)
            trace_len_pull.append(trace_pair.conductance_pull.shape[0])
            # push
            hist_bins, single_hist_1d = utils.calc_hist_1d_single(trace_pair.conductance_push,
                                                                  xrange=self.conductance_range,
                                                                  xbins_num=self.conductance_bins_num,
                                                                  log_scale=self.conductance_log_scale,
                                                                  bin_mode=self.conductance_bins_mode)

            single_hist_list_push.append(single_hist_1d)
            trace_len_push.append(trace_pair.conductance_push.shape[0])
            # Calculate and sum 2d single histograms
            # pull
            try:
                x_mesh_pull, y_mesh_pull, single_hist_2d = utils.calc_hist_2d_single(
                    x=trace_pair.aligned_piezo_pull,
                    y=trace_pair.conductance_pull,
                    xrange=range_pull,
                    log_scale_x=False,
                    yrange=(1e-6, 10),
                    log_scale_y=True,
                    num_bins=(xbins_pull, self.conductance_bins_num),
                    bin_mode_y=self.conductance_bins_mode)

                try:
                    self.hist_2d_pull = self.hist_2d_pull + single_hist_2d
                except (ValueError, TypeError):
                    self.hist_2d_pull = single_hist_2d

                count_pull += 1

            except utils.MyException:
                pass
            # push
            try:
                x_mesh_push, y_mesh_push, single_hist_2d = utils.calc_hist_2d_single(
                    x=trace_pair.aligned_piezo_push[::-1],
                    y=trace_pair.conductance_push[::-1],
                    xrange=range_push,
                    log_scale_x=False,
                    yrange=(1e-6, 10),
                    log_scale_y=True,
                    num_bins=(xbins_push, self.conductance_bins_num),
                    bin_mode_y=self.conductance_bins_mode)

                try:
                    self.hist_2d_push = self.hist_2d_push + single_hist_2d
                except (ValueError, TypeError):
                    self.hist_2d_push = single_hist_2d

                count_push += 1

            except utils.MyException:
                pass

        self.hist_1d_bins = hist_bins
        self.hist_2d_xmesh_pull = x_mesh_pull
        self.hist_2d_ymesh_pull = y_mesh_pull
        self.hist_2d_xmesh_push = x_mesh_push
        self.hist_2d_ymesh_push = y_mesh_push

        # if self.end_trace is None and self.traces is None:
        #     self.end_trace = utils.get_num_from_name(trace)
        if self.end_trace is None:
            self.end_trace = max(self.traces)

        # Temporal histograms
        # use collected single 1d histograms
        # self.hist_1d_bins = hist_bins
        self.temporal_hist_pull = np.array(single_hist_list_pull)
        self.temporal_hist_push = np.array(single_hist_list_push)
        # 1D histograms
        # sum the temporal hist to create 1d histogram
        self.hist_1d_pull = np.sum(self.temporal_hist_pull, axis=0).flatten() / len(self.traces)
        self.hist_1d_push = np.sum(self.temporal_hist_push, axis=0).flatten() / len(self.traces)
        # Plateau length histograms
        self.plateau_length_pull = np.array(plateau_length_pull)
        self.plateau_length_push = np.array(plateau_length_push)
        hist_bins, self.hist_plateau_length_pull = utils.calc_hist_1d_single(self.plateau_length_pull,
                                                                             xrange=(0, 6000),
                                                                             xbins_num=plateau_length_bins,
                                                                             log_scale=False)

        hist_bins, self.hist_plateau_length_push = utils.calc_hist_1d_single(self.plateau_length_push,
                                                                             xrange=(0, 6000),
                                                                             xbins_num=plateau_length_bins,
                                                                             log_scale=False)
        self.hist_plateau_length_bins = hist_bins

        print(f"Pull 2D histogram created from {count_pull} traces")
        print(f"Push 2D histogram created from {count_push} traces")

    def plot_example_traces(self, traces: Union[int, Tuple[int, ...]], shift: float, ax=None,
                            ylims: Tuple[float, ...] = (1e-6, 10),
                            cut_conductances: Tuple[float, float] = (1e-5, 10), add_points_to_end: int = 2_000,
                            dpi=300, **kwargs):
        """

        Parameters
        ----------
        add_points_to_end
        cut_conductances
        traces : Union[int, Tuple[int, ...]
            if traces is an integer value, it is interpreted as the number of traces to plot
            if traces is a tuple of integer values, it is interpreted as the list of trace indices to plot
        shift : float
            the spacing between individual traces
        ylims : limits of the vertical axis
        ax :
        dpi : int
            dots per inch for the figure

        Returns
        -------

        """
        if isinstance(traces, int):
            traces = tuple(np.random.choice(self.traces, traces))

        if isinstance(traces, tuple):
            if ax is None:
                fig, ax = plt.subplots(1, dpi=dpi, figsize=utils.cm2inch((10, 5)))
            ax.set_prop_cycle = cycler('color', colormaps['tab10'](np.linspace(0, 1, 10)))
            ax.set_yscale('log')
            ax.set_ylim(ylims)
            for i, trace in enumerate(traces):
                trace_pair = TracePair(trace, load_from=self.folder)
                trace_pair.align_trace(self.align_at)
                first = np.where(trace_pair.conductance_pull > cut_conductances[1])[0][-1]
                last = np.where(trace_pair.conductance_pull < cut_conductances[0])[0][0] + add_points_to_end
                ax.plot(trace_pair.aligned_piezo_pull[first:last] + i*shift,
                        trace_pair.conductance_pull[first:last], label=trace_pair.trace_num, **kwargs)
            ax.legend(fontsize='xx-small')
        else:
            raise ValueError(f"Invalid type for parameter traces: {type(traces)}. It should be an int or "
                             f"a tuple of ints, see documentation for more info")

        return ax


class HoldTrace:
    """
    Class to represent a hold trace pair

    Parameters
    ----------
    trace : int or str
        trace number or trace name to work with
    load_from : Path or h5py._hl.files.File
        path to the file or an opened for read file instance to load data from
    bias_offset :
        bias offset in the measurement, see the spreadsheet, Default value: 0
    r_serial_ohm : int
        resistance of the resistor connected in series to the sample
    sample_rate : int
        sample rate of the measurement
    min_step_len : Optional[int]
        minimal required step length of a plateau to consider in the analysis. Default: None. If None, the minimal
        step length is the average step length of all bias steps found by the algorithm

    Attributes
    ----------
    bias_offset : float
        bias offset from measurement settings, usually you can find it in the spreadsheets
    R_ser : int
        resistance of resistor in series
    sample_rate : int
        sample rate in the measurement in Hertz
    hold_bias_pull = numpy.array
        measured bias during the pull hold measurement
    hold_current_pull = numpy.array
        measured current during the pull hold measurement
    hold_conductance_pull = numpy.array
        conductance during pull hold measurement calculated from hold bias and hold current
    hold_bias_push = numpy.array
        measured bias during the push hold measurement
    hold_current_push = numpy.array
        measured current during the push hold measurement
    hold_conductance_push = numpy.array
        conductance during push hold measurement calculated from hold bias and hold current
    time_axis_pull : numpy.array
        temporal array calculated using the sample rate, and has the same size as hold_bias_pull, hold_current_pull
        and hold_conductance_pull
    time_axis_push : numpy.array
        same as time_axis pull, except for the push direction
    fft_interval_length_pt : int
        length of the interval of fft calculation
    G_hold_pull = None
    G_hold_push = None
    G_avg_pull = None
    G_avg_push = Non
    bias_steps_ranges_pull : numpy.array
        range of identified bias steps for the pull hold measurement. Shape: (number of bias steps, 2),
        1st column is the starting point of each bias step, and the 2nd column is the end
    bias_steps_at_pull : numpy.array
    bias_steps_ranges_push : numpy.array
        range of identified bias steps for the push hold measurement. Shape: (number of bias steps, 2),
        1st column is the starting point of each bias step, and the 2nd column is the end
    bias_steps_at_push = numpy.array
    bias_steps : numpy.array
        bias values for each bias step in V, shape: (number of bias steps, )
    num_of_fft : int
        number of fft calculations to be performed for each bias step
    psd_interval_ranges_pull : numpy.array
        array containing starting and ending point of each interval on each bias step where psd calculation needs to be
        performed
    psd_intervals_pull = None
    psd_intervals_push = None
    psd_interval_ranges_push = None
    fft_freqs_pull = None
    psds_pull = None
    fft_freqs_push = None
    psds_push = None
    psd_fitparams_pull = None
    psd_fitparams_push = None
    areas_pull = None
    areas_push = None
    avg_cond_on_step_pull = None
    avg_cond_on_step_push = None
    avg_current_on_step_pull = None
    avg_current_on_step_push = None
    noise_power_pull = None
    noise_power_push = None
    conductance_noise_pull = None
    conductance_noise_push = None
    current_noise_pull = None
    current_noise_push = None
    test_intervals = None

    """

    def __init__(self, trace: Union[str, int], load_from: Union[Path, h5py._hl.files.File], bias_offset: float = 0,
                 r_serial_ohm: int = 100_000, sample_rate: int = 50_000,
                 min_step_len: Optional[Union[int, None]] = None, min_height: int = 100, iv: Optional[int] = None,
                 gain: float = 1e7):

        self.bias_offset = bias_offset
        self.R_ser = r_serial_ohm
        self.sample_rate = sample_rate
        self.gain = gain

        # constants:
        self.r0 = 12900  # resistance quantum in Ohm
        self.g0 = 1 / self.r0  # conductance quantum in 1/Ohm=Siemens

        self.time_axis_pull = None
        self.time_axis_push = None
        self.hold_current_pull = np.array([])
        self.hold_bias_pull = np.array([])
        self.hold_conductance_pull = np.array([])
        self.hold_current_push = np.array([])
        self.hold_bias_push = np.array([])
        self.hold_conductance_push = np.array([])
        self.fft_interval_length_pt = 0

        self.G_hold_pull = None
        self.G_hold_push = None
        self.G_avg_pull = None
        self.G_avg_push = None

        self.bias_steps_ranges_pull = None
        self.bias_steps_at_pull = None
        self.bias_steps_ranges_push = None
        self.bias_steps_at_push = None
        self.bias_steps = None
        self.bias_steps_total = None

        self.num_of_fft = 0
        self.psd_intervals_pull = None
        self.psd_interval_ranges_pull = None
        self.psd_intervals_push = None
        self.psd_interval_ranges_push = None
        self.freq_resolution = None
        self.fft_freqs_pull = None
        self.psds_pull = None
        self.fft_freqs_push = None
        self.psds_push = None
        self.psd_fitparams_pull = None
        self.psd_fitparams_push = None
        self.areas_pull = None
        self.areas_push = None
        self.avg_cond_on_step_pull = None
        self.avg_cond_on_step_push = None
        self.avg_current_on_step_pull = None
        self.avg_current_on_step_push = None
        self.noise_power_pull = None
        self.noise_power_push = None
        self.conductance_noise_pull = None
        self.conductance_noise_push = None
        self.current_noise_pull = None
        self.current_noise_push = None

        self.test_intervals = None

        if isinstance(trace, (int, np.int32, np.int64)):
            self.trace_num = trace
            self.trace_name = f'trace_{self.trace_num}'
        elif isinstance(trace, str):
            self.trace_name = trace
            self.trace_num = utils.get_num_from_name(self.trace_name)

        block_num, _ = utils.convert_to_block_and_trace_num(self.trace_num)

        if isinstance(load_from, Path):
            # if you entered the path to the file containing the data, 1st it needs to be opened
            self.file_path = load_from.joinpath(f'hold_data_{block_num}.h5')
            self.file = None
            with h5py.File(self.file_path, "r") as bj_file:
                self.load_hold_traces(bj_file)
        elif isinstance(load_from, h5py._hl.files.File):
            # file already open
            self.file_path = None
            self.file = load_from

            self.load_hold_traces(load_from)
        else:
            raise ValueError(f"Unknown datatype {type(load_from)} in parameter load_from."
                             f"It has to be either a Path object pointing to the folder containing the data file,"
                             f"or an opened h5py File instance.")

        if len(self.hold_bias_pull) > 0 and len(self.hold_bias_push) > 0:

            self.bias_steps_ranges_pull, self.bias_steps_at_pull = self.find_bias_steps(self.hold_bias_pull,
                                                                                        min_len=min_step_len,
                                                                                        height=min_height)
            self.bias_steps_ranges_push, self.bias_steps_at_push = self.find_bias_steps(self.hold_bias_push,
                                                                                        min_len=min_step_len,
                                                                                        height=min_height)
            self.bias_steps = self.hold_bias_pull[np.mean(self.bias_steps_ranges_pull, axis=1).astype(int)]
            self.bias_steps_total = self.bias_steps
            # Calculate avg conductances in the first and last 1 ms of every bias plateau to study stability throughout
            # the measurement

            if len(self.bias_steps) > 1:
                # calculate avg conductance for the beginning and end of each bias step
                self.conductance_avgs_pull = np.zeros((self.bias_steps.shape[0], 2))
                self.conductance_avgs_push = np.zeros((self.bias_steps.shape[0], 2))

                for i, step in enumerate(self.bias_steps):
                    self.conductance_avgs_pull[i, 0] = \
                        np.mean(self.hold_conductance_pull[self.bias_steps_ranges_pull[i, 0] + 100:
                                                           self.bias_steps_ranges_pull[i, 0] + 150])
                    self.conductance_avgs_push[i, 0] = \
                        np.mean(self.hold_conductance_push[self.bias_steps_ranges_push[i, 0] + 100:
                                                           self.bias_steps_ranges_push[i, 0] + 150])

                    self.conductance_avgs_pull[i, 1] = \
                        np.mean(self.hold_conductance_pull[self.bias_steps_ranges_pull[i, 1] - 150:
                                                           self.bias_steps_ranges_pull[i, 1] - 100])
                    self.conductance_avgs_push[i, 1] = \
                        np.mean(self.hold_conductance_push[self.bias_steps_ranges_push[i, 1] - 150:
                                                           self.bias_steps_ranges_push[i, 1] - 100])
            else:
                # calculate avg conductance inside the step around `num_points` equidistant points
                num_points = 25
                sub_points = 10

                test_points = np.linspace(start=self.bias_steps_ranges_pull[0, 0],
                                          stop=self.bias_steps_ranges_pull[0, 1],
                                          num=num_points, endpoint=True, retstep=False, dtype=int, axis=0)

                # test_points = np.unique(np.append(test_points,
                #                                   np.linspace(start=test_points[0], stop=test_points[3],
                #                                               num=sub_points, endpoint=True, retstep=False,
                #                                               dtype=int, axis=0)))

                # print('pull', test_points.shape)

                adjustment = np.ones_like(test_points) * 25
                adjustment[0] = -50
                adjustment[-1] = 100
                self.test_intervals_pull = np.hstack(((test_points - adjustment).reshape((-1, 1)),
                                                      (test_points + adjustment[::-1]).reshape((-1, 1))))

                test_points = np.linspace(start=self.bias_steps_ranges_push[0, 0],
                                          stop=self.bias_steps_ranges_push[0, 1],
                                          num=num_points, endpoint=True, retstep=False, dtype=int, axis=0)

                # test_points = np.unique(np.append(test_points,
                #                                   np.linspace(start=test_points[0], stop=test_points[3],
                #                                               num=sub_points, endpoint=True, retstep=False,
                #                                               dtype=int, axis=0)))

                # print('push', test_points.shape)

                adjustment = np.ones_like(test_points) * 25
                adjustment[0] = -50
                adjustment[-1] = 100
                self.test_intervals_push = np.hstack(((test_points - adjustment).reshape((-1, 1)),
                                                      (test_points + adjustment[::-1]).reshape((-1, 1))))

                self.conductance_avgs_pull = np.zeros(self.test_intervals_pull.shape[0])
                self.conductance_avgs_push = np.zeros(self.test_intervals_push.shape[0])

                for i in range(self.test_intervals_pull.shape[0]):
                    self.conductance_avgs_pull[i] = np.mean(self.hold_conductance_pull[self.test_intervals_pull[i, 0]:
                                                                                       self.test_intervals_pull[i, 1]])
                    self.conductance_avgs_push[i] = np.mean(self.hold_conductance_push[self.test_intervals_push[i, 0]:
                                                                                       self.test_intervals_push[i, 1]])

            # define G_hold as the avg conductance in the 1st 1 ms of the hold measurement
            self.G_hold_pull = np.mean(self.hold_conductance_pull[50:100])
            self.G_hold_push = np.mean(self.hold_conductance_push[50:100])

            self.G_avg_pull = np.mean(self.hold_conductance_pull[-100:-50])
            self.G_avg_push = np.mean(self.hold_conductance_push[-100:-50])

            if iv is not None:
                # I(V) measurement took place, iv refers to the last bias plateau before the I(V) signal

                if np.any(self.hold_current_pull > 10 ** (1 - np.log10(self.gain))):
                    raise MeasurementOverflow('Current overflow during I(V) cycle, in pull direction.')

                if np.any(self.hold_current_push > 10 ** (1 - np.log10(self.gain))):
                    raise MeasurementOverflow('Current overflow during I(V) cycle, in push direction.')

                ranges_pull = (self.bias_steps_ranges_pull[iv, 1] + 50, self.bias_steps_ranges_pull[iv + 1, 0])
                ranges_push = (self.bias_steps_ranges_push[iv, 1] + 50, self.bias_steps_ranges_push[iv + 1, 0])

                self.iv_current_pull = self.hold_current_pull[ranges_pull[0]:ranges_pull[1]]
                self.iv_bias_pull = self.hold_bias_pull[ranges_pull[0]:ranges_pull[1]]
                self.iv_current_push = self.hold_current_push[ranges_push[0]:ranges_push[1]]
                self.iv_bias_push = self.hold_bias_push[ranges_push[0]:ranges_push[1]]

        else:
            raise MeasurementNotComplete('Trace was not completed for some reason, skip this trace in the analysis')

    def analyse_hold_trace(self, num_of_fft: int = 1, subtract_bg: bool = False,
                           freq_range: Tuple[int, int] = (100, 1000)):
        """
        Analysis of steps of a single trace

        Parameters
        ----------
        num_of_fft : int
            number of fft calculations for each bias step
        subtract_bg : bool

        freq_range : Tuple[int, int], default: (100, 1000)
            frequency range of psd to fit

        """

        self.num_of_fft = num_of_fft

        self.psd_intervals_pull, self.psd_interval_ranges_pull = self.find_psd_intervals(self.bias_steps_ranges_pull,
                                                                                         self.bias_steps_at_pull,
                                                                                         self.hold_current_pull)

        self.psd_intervals_push, self.psd_interval_ranges_push = self.find_psd_intervals(self.bias_steps_ranges_push,
                                                                                         self.bias_steps_at_push,
                                                                                         self.hold_current_push)

        self.fft_freqs_pull, self.psds_pull = self.calculate_psds(psd_intervals=self.psd_intervals_pull)
        self.fft_freqs_push, self.psds_push = self.calculate_psds(psd_intervals=self.psd_intervals_push)

        self.psd_fitparams_pull, fit_errors = self.calc_exponents4psds(self.fft_freqs_pull, self.psds_pull,
                                                                       fit_range=freq_range)
        self.psd_fitparams_push, fit_errors = self.calc_exponents4psds(self.fft_freqs_push, self.psds_push,
                                                                       fit_range=freq_range)

        self.avg_current_on_step_pull = self.calculate_avg_value_on_step(value_array=self.hold_current_pull,
                                                                         in_ranges=self.psd_interval_ranges_pull)
        self.avg_current_on_step_push = self.calculate_avg_value_on_step(value_array=self.hold_current_push,
                                                                         in_ranges=self.psd_interval_ranges_push)

        # self.avg_cond_on_step_pull = self.calculate_avg_value_on_step(value_array=self.hold_conductance_pull,
        #                                                               in_ranges=self.psd_interval_ranges_pull)
        # self.avg_cond_on_step_push = self.calculate_avg_value_on_step(value_array=self.hold_conductance_push,
        #                                                               in_ranges=self.psd_interval_ranges_push)

        self.avg_cond_on_step_pull = utils.calculate_conductance_g0(self.bias_steps, self.avg_current_on_step_pull,
                                                                    r_serial_ohm=self.R_ser)

        self.avg_cond_on_step_push = utils.calculate_conductance_g0(self.bias_steps, self.avg_current_on_step_push,
                                                                    r_serial_ohm=self.R_ser)

        self.areas_pull = self.area_under_psds(self.fft_freqs_pull, self.psds_pull,
                                               freq_range=freq_range)
        self.areas_push = self.area_under_psds(self.fft_freqs_push, self.psds_push,
                                               freq_range=freq_range)

        if subtract_bg:
            non_zero_index = np.where(self.bias_steps > min(self.bias_steps))
            zero_index = np.where(self.bias_steps == min(self.bias_steps))

            self.bias_steps = self.bias_steps_total[non_zero_index]

            self.areas_pull = self.areas_pull[non_zero_index] - self.areas_pull[zero_index]

            self.areas_push = self.areas_push[non_zero_index] - self.areas_push[zero_index]

            self.avg_cond_on_step_pull = self.avg_cond_on_step_pull[non_zero_index]
            self.avg_cond_on_step_push = self.avg_cond_on_step_push[non_zero_index]

            self.avg_current_on_step_pull = self.avg_current_on_step_pull[non_zero_index]
            self.avg_current_on_step_push = self.avg_current_on_step_push[non_zero_index]

            self.noise_power_pull = self.calc_noise_value(self.areas_pull, self.bias_steps,
                                                          self.avg_cond_on_step_pull,
                                                          mode='noise_power')

            self.noise_power_push = self.calc_noise_value(self.areas_push, self.bias_steps,
                                                          self.avg_cond_on_step_push,
                                                          mode='noise_power')

            self.conductance_noise_pull = self.calc_noise_value(self.areas_pull, self.bias_steps,
                                                                self.avg_cond_on_step_pull,
                                                                mode='conductance_noise')

            self.conductance_noise_push = self.calc_noise_value(self.areas_push, self.bias_steps,
                                                                self.avg_cond_on_step_push,
                                                                mode='conductance_noise')

            self.current_noise_pull = self.calc_noise_value(self.areas_pull, self.bias_steps,
                                                            self.avg_current_on_step_pull,
                                                            mode='current_noise')

            self.current_noise_push = self.calc_noise_value(self.areas_push, self.bias_steps,
                                                            self.avg_current_on_step_push,
                                                            mode='current_noise')

        else:

            self.noise_power_pull = self.calc_noise_value(self.areas_pull, self.bias_steps, self.avg_cond_on_step_pull,
                                                          mode='noise_power')
            self.noise_power_push = self.calc_noise_value(self.areas_push, self.bias_steps, self.avg_cond_on_step_push,
                                                          mode='noise_power')

            self.conductance_noise_pull = self.calc_noise_value(self.areas_pull, self.bias_steps,
                                                                self.avg_cond_on_step_pull,
                                                                mode='conductance_noise')
            self.conductance_noise_push = self.calc_noise_value(self.areas_push, self.bias_steps,
                                                                self.avg_cond_on_step_push,
                                                                mode='conductance_noise')

            self.current_noise_pull = self.calc_noise_value(self.areas_pull, self.bias_steps,
                                                            self.avg_current_on_step_pull,
                                                            mode='current_noise')
            self.current_noise_push = self.calc_noise_value(self.areas_push, self.bias_steps,
                                                            self.avg_current_on_step_push,
                                                            mode='current_noise')

    def load_hold_traces(self, hold_file: Union[Path or h5py._hl.files.File]):
        """
        Load hold parameters to the corresponding attributes

        Parameters
        ----------
        hold_file : Path or h5py._hl.files.File
            path to the file or an opened for read file instance to load data from

        """

        self.hold_bias_pull = hold_file[f'pull/{self.trace_name}/hold_bias'][1:] + self.bias_offset
        self.hold_current_pull = hold_file[f'pull/{self.trace_name}/hold_current'][1:]
        # take the absolute value of conductance because at small conductances we can get small negative values due to
        # measurement error, and these can result in unexpected behaviour
        self.hold_conductance_pull = abs(utils.calculate_conductance_g0(self.hold_bias_pull, self.hold_current_pull,
                                                                        r_serial_ohm=self.R_ser))
        self.time_axis_pull = np.arange(start=0, stop=self.hold_current_pull.shape[0] / self.sample_rate,
                                        step=1 / self.sample_rate)
        self.hold_bias_push = hold_file[f'push/{self.trace_name}/hold_bias'][1:] + self.bias_offset
        self.hold_current_push = hold_file[f'push/{self.trace_name}/hold_current'][1:]
        self.hold_conductance_push = abs(utils.calculate_conductance_g0(self.hold_bias_push, self.hold_current_push,
                                                                        r_serial_ohm=self.R_ser))
        self.time_axis_push = np.arange(start=0, stop=self.hold_current_push.shape[0] / self.sample_rate,
                                        step=1 / self.sample_rate)

    def is_long_enough(self,
                       trace_array: np.ndarray,
                       min_length: Union[int, float],
                       in_sec: Optional[bool] = True) -> bool:
        """
        Checks whether the length of a single trace exceeds a certain value *min_length*

        Parameters
        ----------
        trace_array : np.ndarray
            single trace to check the length
        min_length : int or float
            minimal length required
        in_sec : bool
            whether the min_length is entered in seconds if False enter the minimum length in points

        Returns
        -------
        res : bool or list of bools
            True if a trace is long enough False if not
        """
        if min_length > 0:

            min_points = min_length * self.sample_rate * in_sec - (in_sec - 1) * min_length

            if isinstance(trace_array, np.ndarray):
                return trace_array.shape[0] > min_points

        return True

    def find_bias_steps(self, bias: np.ndarray, min_len: Optional[int] = None, height: int = 100) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the range of each bias step in the measurement

        Parameters
        ----------
        bias : np.ndarray
            measured bias during hold measurement
        min_len : Optional, int default: None
            minimum required length of a bias step in points
        height : minimum required height of the derivative

        Returns
        -------
        bias_steps_ranges: np.ndarray
            array containing the intervals of the steps (shape: (number_of_steps, 2))
        """

        # bias changes at these points:
        bias_steps_at = self.find_signal_steps(measured_signal=bias, height=height)
        # create 2d array where the (i)th row represents the beginning and ending point of the (i)th step
        # if no bias_steps are found, it takes the first and last index of the bias array as the range of the step
        bias_steps_ranges = np.pad(np.hstack((bias_steps_at.reshape(-1, 1), bias_steps_at.reshape(-1, 1))).flatten(),
                                   pad_width=1, mode='constant', constant_values=(0, len(bias) - 1)).reshape(-1, 2)

        bias_steps_lens = bias_steps_ranges[:, 1] - bias_steps_ranges[:, 0]

        if min_len is None:
            min_len = np.average(bias_steps_lens)  # for automatic analysis

        bias_steps_long = bias_steps_lens > min_len  # select those bias steps that are longer than min_len

        # the interesting steps are surrounded by these points:
        bias_steps_at = np.unique(bias_steps_ranges[bias_steps_long].flatten())

        return bias_steps_ranges[bias_steps_long], bias_steps_at

    def find_signal_steps(self, measured_signal: np.ndarray, use_sample_rate: bool = True, height: int = 100) \
            -> np.ndarray:
        """
        Finds the point of each signal step in the measurement

        Parameters
        ----------
        measured_signal : np.ndarray
            any measured array that changes step-like
        use_sample_rate: bool
            set to True, if you want to use the sample_rate in the calculation of the derivative
        height : int
            required height of the derivated signal, to be considered a step-change

        Returns
        -------
        steps_at: np.ndarray
            the points where `measured_signal` changes

        """
        # points of peaks in the derivative correspond to steps in the signal
        # calculate the derivative:
        if use_sample_rate:
            dsignal = np.gradient(measured_signal, 1 / self.sample_rate)
        else:
            dsignal = np.gradient(measured_signal)
        # find peaks in the derivative:
        steps_at = scipy.signal.find_peaks(abs(dsignal), height=height)[0]

        return steps_at

    def plot_hold_traces(self,
                         direction: str = 'pull',
                         ax: Optional[matplotlib.axes.Axes] = None,
                         log_scale_y: bool = True,
                         ax_colors: Tuple[str, str] = ('blue', 'red'),
                         conductance: bool = False,
                         plot_step_ranges: bool = False,
                         plot_psd_intervals: bool = False,
                         smoothing: int = 1,
                         add_hlines: Tuple[float, ...] = tuple(),
                         add_vlines: Tuple[float, ...] = tuple(),
                         dpi: int = 600):

        """
        Plot hold traces

        Parameters
        ----------
        direction : str

        ax : matplotlib.axes.Axes
            pre-defined axis for plot
        log_scale_y : bool, default: True
            scale vertical axis logarithmically
        ax_colors : Tuple[str, str]
        conductance : bool
            if True, the calculated conductance is plotted, otherwise the measured current is plotted
        plot_step_ranges : bool, default: False
            show starting and ending points of bias steps
        plot_psd_intervals : bool, default: False
            show starting and ending points of ranges for psd calculation
        smoothing : int, default: 1
            window size for moving average to smooth the plotted trace, 1 means no smoothing. The larger this number,
            the more smooth the trace
        add_hlines : Tuple[float, ...]
            add additional horizontal lines for annotations
        add_vlines : Tuple[float, ...]
            add additional vertical lines for annotations
        dpi : int, default: 600
            resolution in dots per inch

        Returns
        -------
        ax : matplotlib.axes.Axes
            axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=utils.cm2inch(10, 6), dpi=dpi)  # width, height

        par = ax.twinx()

        if direction == 'pull':
            pull = True
            push = False
        elif direction == 'push':
            pull = False
            push = True
        else:
            raise ValueError("Invalid value for direction. Use 'pull' or 'push'.")

        if pull:
            if conductance:
                data = utils.moving_average(self.hold_conductance_pull, smoothing)
                label_text = r"G $[G_{0}]$"
            else:
                data = utils.moving_average(self.hold_current_pull, smoothing)
                label_text = "Current [A]"
            self.time_axis_pull = utils.moving_average(np.arange(start=0,
                                                                 stop=self.hold_bias_pull.shape[0] / self.sample_rate,
                                                                 step=(1 / self.sample_rate)),
                                                       smoothing)

            par.plot(self.time_axis_pull, utils.moving_average(self.hold_bias_pull, smoothing),
                     c=ax_colors[1], lw=0.5, zorder=0)
            par.set_ylabel('Bias [mV]')

            ax.plot(self.time_axis_pull, data, c=ax_colors[0], lw=0.5, zorder=10)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(label_text)
            if log_scale_y:
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.2), numticks=9))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            if plot_step_ranges:
                ax.vlines(self.bias_steps_at_pull / self.sample_rate, 0, 1, transform=ax.get_xaxis_transform(),
                          color='k', alpha=1, lw=1)
            if plot_psd_intervals:
                ax.vlines(self.psd_interval_ranges_pull / self.sample_rate, 0, 1,
                          transform=ax.get_xaxis_transform(),
                          color='g', ls='--', lw=0.5)

            ax.spines['left'].set_color(ax_colors[0])
            par.spines['right'].set_color(ax_colors[1])
            par.spines['left'].set_color(ax_colors[0])
            ax.tick_params(axis='y', which='both', colors=ax_colors[0])
            par.tick_params(axis='y', colors=ax_colors[1])
            ax.yaxis.label.set_color(ax_colors[0])
            par.yaxis.label.set_color(ax_colors[1])
        elif push:
            if conductance:
                data = utils.moving_average(self.hold_conductance_push, smoothing)
                label_text = r"G $[G_{0}]$"
            else:
                data = utils.moving_average(self.hold_current_push, smoothing)
                label_text = f"Current [A]"
            self.time_axis_push = utils.moving_average(np.arange(start=0,
                                                                 stop=self.hold_bias_push.shape[0] / self.sample_rate,
                                                                 step=(1 / self.sample_rate)),
                                                       smoothing)
            #     time_total_avg = utils.moving_average(time_total, smoothing)

            par.plot(self.time_axis_push, utils.moving_average(self.hold_bias_push, smoothing),
                     c=ax_colors[0], lw=0.5, zorder=0)
            par.set_ylabel('Bias [mV]')

            ax.plot(self.time_axis_push, data, c=ax_colors[1], lw=0.5, zorder=10)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(label_text)
            if log_scale_y:
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=9))
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0, 1, 0.2), numticks=9))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            if plot_step_ranges:
                ax.vlines(self.bias_steps_at_push / self.sample_rate, 0, 1, transform=ax.get_xaxis_transform(),
                          color='k', alpha=1, lw=1)
            if plot_psd_intervals:
                ax.vlines(self.psd_interval_ranges_push / self.sample_rate, 0, 1,
                          transform=ax.get_xaxis_transform(),
                          color='g', ls='--', lw=0.5)

            ax.spines['left'].set_color(ax_colors[1])
            par.spines['right'].set_color(ax_colors[0])
            par.spines['left'].set_color(ax_colors[1])
            ax.tick_params(axis='y', which='both', colors=ax_colors[1])
            par.tick_params(axis='y', colors=ax_colors[0])
            ax.yaxis.label.set_color(ax_colors[1])
            par.yaxis.label.set_color(ax_colors[0])
        else:
            raise ValueError('One of the parameters pull or push has to be True.')

        ax.xaxis.set_ticks_position('both')

        # print('bias_steps', self.bias_steps)

        if len(self.bias_steps) == 1:
            par.set_ylim(self.bias_steps[0] - 0.005, self.bias_steps[0] + 0.005)
        par.set_yticks(self.bias_steps)
        par.set_yticklabels(list(map(str, np.around(self.bias_steps * 1000, decimals=1).astype(int))))

        for i in add_hlines:
            ax.axhline(i, lw=0.5, color='k', ls='--', alpha=0.5)
            # ax.axhline(i, lw=0.5, color='k', ls='--', alpha=0.5)

        for i in add_vlines:
            ax.axvline(i, lw=0.5, color='k', ls='--', alpha=0.5)
            # ax.axvline(i, lw=0.5, color='k', ls='--', alpha=0.5)

        return ax, par

    def plot_ivs(self, ax_colors=('cornflowerblue', 'firebrick'), smoothing: int = 1, dpi: int = 600):

        fig = plt.figure(figsize=utils.cm2inch(15, 5), dpi=dpi)  # figsize: (width, height) in inches
        gs = gridspec.GridSpec(nrows=1, ncols=2,
                               figure=fig, left=0.15, right=0.9, top=0.9, bottom=0.15, wspace=0, hspace=0)

        ax_pull = fig.add_subplot(gs[0])
        ax_push = fig.add_subplot(gs[1])

        ax_pull.plot(utils.moving_average(self.iv_bias_pull, smoothing),
                     utils.moving_average(self.iv_current_pull, smoothing),
                     c=ax_colors[0], lw=0.4)

        ax_push.plot(utils.moving_average(self.iv_bias_push, smoothing),
                     utils.moving_average(self.iv_current_push, smoothing),
                     c=ax_colors[1], lw=0.4)

        ax_pull.set_xlabel('Bias [V]')
        ax_pull.set_ylabel("Current [A]")
        ax_push.set_xlabel('Bias [V]')
        ax_push.set_ylabel("Current [A]")

        max_curr_pull = max(abs(self.iv_current_pull))
        max_curr_push = max(abs(self.iv_current_push))

        ax_pull.set_ylim(-1.05 * max_curr_pull, 1.05 * max_curr_pull)
        ax_push.set_ylim(-1.05 * max_curr_push, 1.05 * max_curr_push)

        ax_pull.yaxis.label.set_color(ax_colors[0])
        ax_push.yaxis.label.set_color(ax_colors[1])

        ax_pull.xaxis.set_ticks_position('both')
        # ax_pull.yaxis.set_ticks_position('both')

        ax_push.yaxis.tick_right()
        ax_push.yaxis.set_label_position('right')
        ax_push.xaxis.set_ticks_position('both')
        # ax_push.yaxis.set_ticks_position('both')

        return ax_pull, ax_push

    def save_iv_for_laci(self, home_folder, direction: str = 'pull'):

        # check if folder exists

        if home_folder.joinpath(f"results/IVs/selected/").is_dir():
            file = home_folder.joinpath(f"results/IVs/selected/IV_trace_{self.trace_num}_{direction}.txt")
        else:
            if home_folder.joinpath(f"results/IVs/").is_dir():
                mkdir(home_folder.joinpath(f"results/IVs/selected/"))
                file = home_folder.joinpath(f"results/IVs/selected/IV_trace_{self.trace_num}_{direction}.txt")
            else:
                if home_folder.joinpath(f"results/").is_dir():
                    mkdir(home_folder.joinpath(f"results/IVs/"))
                    mkdir(home_folder.joinpath(f"results/IVs/selected/"))
                    file = home_folder.joinpath(f"results/IVs/selected/IV_trace_{self.trace_num}_{direction}.txt")
                else:
                    mkdir(home_folder.joinpath(f"results/"))
                    mkdir(home_folder.joinpath(f"results/IVs/"))
                    mkdir(home_folder.joinpath(f"results/IVs/selected/"))
                    file = home_folder.joinpath(f"results/IVs/selected/IV_trace_{self.trace_num}_{direction}.txt")

        which_block, _ = utils.convert_to_block_and_trace_num(self.trace_num)

        if direction == 'pull':
            bias = self.iv_bias_pull
            current = self.iv_current_pull
        elif direction == 'push':
            bias = self.iv_bias_push
            current = self.iv_current_push
        else:
            raise ValueError('Invalid value for direction.')

        header = f"""I-V measurement: {file}
        {home_folder.name.replace('_', '.') + '.'}

        BiasInterval: {np.around(min(bias) * 1000).astype(int)} : {np.around(max(bias) * 1000).astype(int)}
        I/V gain: {self.gain}
        Gate Voltage: 0 V
        AcqFrequency: 50000
        NumberOfPoints : {len(bias)}
        NumberOfAveragedPoints : 1
        NumberOfCycles : 1
        BiasDivision 1
        Serial Resistance: {self.R_ser}
        Amplifier: Femto

        Program Bias (V)\tBias Voltage (V)\tCurrent (A)\n"""

        with open(file, "w") as f:
            f.write(header)

        with open(file, "a") as f:
            for i in range(len(bias)):
                f.write(str(bias[i].round(4)) + "\t" + str(bias[i]) + "\t" + str(current[i]) + "\n")

    def plot_psds(self,
                  pull: Optional[bool] = True,
                  colormap='Blues',
                  emph: Optional[int] = None,
                  plot_fit: bool = False,
                  fit_params: Optional[np.ndarray] = None,
                  ax: Optional[plt.axis] = None,
                  show: bool = False,
                  legend: Optional[List[str]] = None,
                  plot_legend: bool = True,
                  plot_guides: bool = True,
                  which_psds: Optional[Union[int, List[int]]] = None,
                  dpi: int = 600):
        """
        Plot calculated PSDs for each bias step with different color

        Parameters
        ----------
        pull : bool
            if True, the psds for the pull hold trace are plotted, otherwise for the push hold trace
        colormap : str
            which colormap to use
        emph : int
            the index of the bias plateau to be emphasized with opaque color, all others will appear more transparent
        plot_fit : bool
            whether to plot the lines fitted to PSDs. If True, you need to provide *fit_params*
        fit_params :
            fit parameters of lines fitted to PSDs
        ax : `~matplotlib.axes.Axes`
            pre-existing axis for plot
        show : bool
            if True, the plot is shown after creation
        legend : Optional[List[str]]
            legend to be added, if None, the legend is autogenerated from the voltage applied at each bias step
        plot_legend : bool, default: True
            whether to show legend or not
        plot_guides: bool, default: True
            whether to plot guidelines ~ 1/f, 1/f**1.4, 1/f**2
        which_psds

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            axis with plot

        """
        # blues = ['#000c33', '#002599', '#0032cc', '#003eff', '#406eff', '#809fff']  # blues
        # reds = ['#330000', '#660000', '#990000', '#cc0000', '#ff0000', '#ff4040']  # reds

        if ax is None:
            fig = plt.figure(figsize=utils.cm2inch(16, 6), dpi=dpi)  # figsize: (width, height) in inches
            gs = gridspec.GridSpec(nrows=1, ncols=1,
                                   figure=fig, left=0.15, right=0.9, top=0.9, bottom=0.15, wspace=0, hspace=0)

            ax_psd = fig.add_subplot(gs[0])
        else:
            ax_psd = ax

        if pull:
            bias_steps = self.bias_steps
            freq = self.fft_freqs_pull
            if which_psds is None:
                psds = self.psds_pull
            else:
                psds = self.psds_pull[which_psds]
            color_cyc = cycler(color=utils.blues)
        else:
            bias_steps = self.bias_steps
            freq = self.fft_freqs_push
            if which_psds is None:
                psds = self.psds_push
            else:
                psds = self.psds_push[which_psds]
            color_cyc = cycler(color=utils.reds)

        # color_cyc = cycler(color=cm.get_cmap(colormap)(np.linspace(1, 0.25, len(bias_steps), endpoint=False)))
        # color_cyc = cycler(color=cm.get_cmap(colormap)(np.linspace(0, 1, len(bias_steps), endpoint=False)))
        ax_psd.set_prop_cycle(color_cyc)

        ax_psd.set_xscale('log')
        ax_psd.set_yscale('log')

        ax_psd.set_ylim(1e-30, 1e-16)

        # ax_psd.xaxis.set_ticks_position('both')
        # ax_psd.yaxis.set_ticks_position('both')
        # ax_psd.yaxis.set_label_position('left')

        ax_psd.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=5))

        ax_psd.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=5))
        ax_psd.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=20))
        ax_psd.yaxis.set_minor_formatter(ticker.NullFormatter())

        alph = 1

        if legend is None:
            for psd, bias_step in zip(psds, bias_steps):
                #     print(freq_pull.shape)
                #     print(psd.shape)
                #     print(round(bias_step*1000).astype(int))

                bias = round(bias_step * 1000)

                if emph is not None:
                    if bias == round(bias_steps[emph] * 1000):
                        alph = 1
                    else:
                        alph = 0.2

                ax_psd.plot(freq, psd, lw=0.5, label=f'{bias} mV', alpha=alph)

        else:
            for psd, bias_step, leg in zip(psds, bias_steps, legend):
                #     print(freq_pull.shape)
                #     print(psd.shape)
                #     print(round(bias_step*1000).astype(int))

                bias = round(bias_step * 1000)

                if emph is not None:
                    if bias == round(bias_steps[emph] * 1000):
                        alph = 1
                    else:
                        alph = 0.2

                ax_psd.plot(freq, psd, lw=0.5, label=leg, alpha=alph)

        # if plot_fit:
        #   if fit_params is not None:
        #     for fit_param in fit_params:
        #         ax_psd.plot(freq[1:], 10 ** fit_func_lin(np.log10(freq[1:]), fit_param[0], fit_param[1]), alpha=alph)
        #   else:
        #       # do fit
        #       raise NotImplementedError

        if plot_guides:
            # add guide lines with exponents 1, 1.4, 2
            # scaling factors so they start from the same point = 1e-16
            x = 1 / freq[1] * 1 / max(psds[:, 1])  # 1e16
            y = 1 / freq[1] ** 1.4 * 1 / max(psds[:, 1])
            z = 1 / freq[1] ** 2 * 1 / max(psds[:, 1])

            # ax_psd.plot(freq[1:], (1 / freq[1:]) / x, lw=0.4, label=r'$1/f$', c='black', linestyle='dashed')
            ax_psd.plot(freq[1:], (1 / freq[1:]) / x, lw=0.4, c='black')
            ax_psd.text(freq[-300],
                        ((1 / freq[1:]) / x)[-600],
                        '1', size=4)
            # ax_psd.plot(freq[1:], (1 / freq[1:] ** 1.4) / y, lw=0.4, label=r'$1/f^{1.4}$', c='black')
            ax_psd.plot(freq[1:], (1 / freq[1:] ** 1.4) / y, lw=0.4, c='black', linestyle='dashed')
            ax_psd.text(freq[-300],
                        ((1 / freq[1:] ** 1.4) / y)[-450],
                        '1.4', size=4)

            # ax_psd.plot(freq[1:], (1 / freq[1:] ** 2) / z, lw=0.4, label=r'$1/f^{2}$', c='black', linestyle='dotted')
            ax_psd.plot(freq[1:], (1 / freq[1:] ** 2) / z, lw=0.4, c='black', linestyle='dotted')
            ax_psd.text(freq[-300],
                        ((1 / freq[1:] ** 2) / z)[-400],
                        '2', size=4)

        if plot_legend:
            ax_psd.legend(bbox_to_anchor=(0.02, 0.02, 0.4, .02), loc='lower left', ncol=2, mode="expand",
                          borderaxespad=0., handlelength=1.0,
                          fontsize='xx-small')

        ax_psd.axvspan(100, 1000, alpha=0.1, facecolor='grey', edgecolor=None)

        ax_psd.set_xlabel(r'Frequency [Hz]')

        ax_psd.set_ylabel(r'$S_{I}\;[\mathrm{A}^2/\mathrm{Hz}]$')

        # plt.savefig('psd.png', bbox_inches='tight')
        if show:
            plt.show()

        return ax_psd

    def calculate_avg_value_on_step(self, value_array: np.ndarray, in_ranges: np.ndarray) -> np.ndarray:
        """
        Calculate avg values for each step in the ranges: *in_ranges*

        Parameters
        ----------
        value_array : np.ndarray
            Array of the measured conductance or current.
        in_ranges : np.ndarray
            starting and ending point of each step, where avg value is calculated

        Returns
        -------
        avg_values : np.ndarray
            average values on each step

        """

        # average values for individual bias steps
        avg_values = np.array([np.average(value_array[single_range[0]:single_range[1]]) for single_range in in_ranges])

        return avg_values

    def find_max_fft_interval(self, min_step_length: int) -> int:
        """
        Calculates the maximal possible length (in points) of a sub-range in a step, that is a power of two
        and so it is fit to perform fft calculations on

        Parameters
        ----------
        min_step_length : int
            length of the shortest bias step during the measurement

        Returns
        -------
        max_fft_interval_pt: int
            maximal interval length for each fft calculation so that *self.num_of_fft* calculations
            fit inside each step

        """

        if self.num_of_fft > 0:
            i = 0
            max_fft_interval_pt = 0

            while 2 ** i < min_step_length / self.num_of_fft:
                max_fft_interval_pt = 2 ** i
                i += 1

            self.fft_interval_length_pt = max_fft_interval_pt

            return max_fft_interval_pt

        raise ValueError("Error: 'num_of_fft' should be greater than 0!")

    def calculate_freq_resolution(self) -> float:
        """
        Calculates the frequency resolution from the length of the fft intervals

        Returns
        -------
        freq_resolution : float
            frequency resolution

        """

        if self.fft_interval_length_pt > 0:
            freq_resolution = 1 / utils.convert_pt_to_sec(self.fft_interval_length_pt, self.sample_rate)
            self.freq_resolution = freq_resolution

            return freq_resolution

        raise ValueError("Attribute 'fft_interval_length_pt' has to be larger than 0!"
                         "Run method 'find_max_fft_interval' first!")

    def find_psd_intervals(self,
                           bias_steps_ranges: np.ndarray,
                           bias_steps_at: np.ndarray,
                           current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find intervals for individual PSD calculations

        Parameters
        ----------
        bias_steps_ranges : np.ndarray

        current : np.ndarray
            measured current during hold measurement

        Returns
        -------
        psd_intervals : np.ndarray
            parts cut from the *current* array to calculate the PSDs
        psd_interval_ranges : np.ndarray
            bounding points of the cut out intervals

        See Also
        --------
        find_max_fft_interval
        calculate_freq_resolution

        """

        bias_steps_lens = np.diff(self.bias_steps_ranges_pull).flatten()

        fft_interval_length_pt = self.find_max_fft_interval(min_step_length=min(bias_steps_lens))

        self.calculate_freq_resolution()

        psd_interval_start = np.empty_like(bias_steps_lens, dtype=np.int64)
        np.ceil(bias_steps_ranges[:, 0] + (bias_steps_lens - self.num_of_fft * fft_interval_length_pt) / 2,
                out=psd_interval_start, casting='unsafe')

        psd_interval_ranges = np.vstack(tuple([psd_interval_start] * (self.num_of_fft + 1))).T

        for i in range(1, self.num_of_fft + 1):
            psd_interval_ranges[:, i] = psd_interval_ranges[:, i - 1] + fft_interval_length_pt

        psd_intervals = np.array([current[i:i + self.num_of_fft * fft_interval_length_pt] for i in psd_interval_start])

        return psd_intervals, psd_interval_ranges

    def calculate_psds(self, psd_intervals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the power spectral densities on each bias step

        Parameters
        ----------
        psd_intervals: np.ndarray
            Intervals cut out from the measurement on *which_step* the FFT calculations are performed

        Returns
        -------
        psd_results: np.ndarray
            Power spectral density of *psd_intervals*
        fft_freqs: np.ndarray
            Array of sample frequencies

        """
        if self.num_of_fft > 0:
            num_steps, max_fft_interval = psd_intervals.shape
            max_fft_interval //= self.num_of_fft

            fft_freqs, psd_results = scipy.signal.welch(x=psd_intervals, fs=self.sample_rate, window='hann',
                                                        nperseg=max_fft_interval, noverlap=0, nfft=None,
                                                        detrend=False, scaling='density', average='mean', axis=1)

            return fft_freqs, psd_results

        raise ValueError("Error: 'num_of_fft' should be greater than 0!")

    def remove_thermal_background_1d(self,
                                     some_array: np.ndarray,
                                     bg_index: int = 0,
                                     using_fit: bool = False,
                                     criteria: Optional[int] = None) -> Tuple[np.ndarray, bool]:
        """
        Remove thermal background from *some_array*

        Parameters
        ----------
        some_array:
        bg_index : int
            index of the bias step of thermal background measurement, where the applied bias is 0 mV
        using_fit: bool, default: False
            whether to use the fit to this measurement, if True, raises Not Implemented Error
        criteria : Optional[int]
            filtering criteria. valid choices:
            None: checks if the area under the psd of a given bias plateau is above the background
            (0 V plateau psd)
            1: the area under the psd of a given bias plateau is at least 1 order of magnitude larger
            (10x as large)
            2: the area under the psd of a given bias plateau is at least 2 order of magnitude larger
            (100x as large)

        Returns
        -------
        some_array_wo_bg : np.ndarary
            the array after removing the thermal background
        above_bg : np.ndarray
            array of bools, showing whether a value was above of the thermal background or not
        """

        if using_fit:
            raise NotImplementedError("This option is not available yet.")
        else:
            ind_array = np.ones(some_array.shape[0]).astype(bool)
            ind_array[bg_index] = 0
            some_array_wo_bg = some_array[ind_array] - some_array[bg_index]

            if criteria is None:
                above_bg = np.all(some_array_wo_bg > 0)
            elif criteria == 1:
                above_bg = np.all(some_array[1:] > 10 * some_array[0])
            elif criteria == 2:
                above_bg = np.all(some_array[1:] > 100 * some_array[0])
            else:
                raise ValueError("Undefined criteria. Valid choices: None, 1 or 2 (see description for details)")
            return some_array_wo_bg, above_bg

    def remove_thermal_background_2d(self,
                                     some_array: np.ndarray,
                                     bg_index: Optional[int] = 0,
                                     using_fit=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove the thermal background by subtracting the first column from all other columns

        Parameters
        ----------
        bg_index:
        some_array:
        using_fit:

        Returns
        -------
        some_array_wo_bg: np.ndarray
            the array after the thermal background (measured at 0 V bias) is removed
        above_bg: np.ndarray
            array that tells you which traces are correct basically
            (we want PSDs measured at higher bias to be above than the thermal background)

        """

        if using_fit:
            raise NotImplementedError("This option is not available yet.")
        else:
            ind_array = np.ones(some_array.shape[0]).astype(bool)
            ind_array[bg_index] = 0
            some_array_wo_bg = some_array[:, ind_array] - some_array[:, 0]

            above_bg = np.array([np.all(some_array_wo_bg[i, :] > np.zeros_like(some_array_wo_bg[i, :]))
                                 for i in range(some_array_wo_bg.shape[0])])

            return some_array_wo_bg, above_bg

    def fit_line_to_psd(self, freq: np.ndarray, psd: np.ndarray, fit_range: Tuple[float, float]):
        """
        Fit a line to a single PSD and return the line parameters

        Parameters
        ----------
        freq : np.ndarray
            frequency
        psd : np.ndarray
            calculated PSD
        fit_range : Tuple[float, float]
            frequency range to fit line

        Returns
        -------
        popt : np.ndarray
            the slope and the intercept of the fitted line
        perr : np.ndarray
            standard deviation error of the slope and the intercept
        """
        popt, pcov = scipy.optimize.curve_fit(utils.fit_func_lin,
                                              np.log10(freq[np.bitwise_and(freq > fit_range[0], freq < fit_range[1])]),
                                              np.log10(psd[np.bitwise_and(freq > fit_range[0], freq < fit_range[1])]))
        perr = np.sqrt(np.diag(pcov))  # standard deviation error

        return popt, perr

    def calc_exponents4psds(self,
                            freq: np.ndarray, psds: np.ndarray,
                            fit_range: Optional[Tuple[float, float]] = (100, 1000)):
        """
        Calculate exponents for each PSD

        Parameters
        ----------
        freq : np.ndarray
            frequency
        psds : np.ndarray
            PSDs for each bias step
        fit_range : Tuple[float, float]
            frequency range for fitting line to PSDs

        Returns
        -------
        fit_params : np.ndarray
            line fit parameters. Each row of fit_params corresponds to a psd for a given bias step,
            the 0th column corresponds to the exponent, 1st column to the intercept
        fit_errors : np.ndarray
            line fit standard deviation errors (structure is the same as for fit_params)
        """

        if np.any(np.mean(psds[:, np.bitwise_and(freq > fit_range[0], freq < fit_range[1])], axis=1) < 1e-30):
            raise MeasurementOverflow('Current overflow in trace, skipping.')

        fit_params = np.zeros((psds.shape[0], 2))
        fit_errors = np.zeros((psds.shape[0], 2))
        for j, psd in enumerate(psds):
            popt, perr = self.fit_line_to_psd(freq, psd, fit_range=fit_range)

            fit_params[j] = popt  # popt[0] is the exponent, popt[1] is the interccept!
            fit_errors[j] = perr

        return fit_params, fit_errors

    def area_under_single_psd(self, freq: np.ndarray, psd_result: np.ndarray,
                              freq_range: Optional[Tuple[int, int]] = None) -> float:
        """
        Calculate the area under a PSD curve in the given frequency range (*start*, *end*)

        Parameters
        ----------
        freq: np.ndarray
            array of the frequency
        psd_result: np.ndarray
            PSD curve used to calculate the area under
        freq_range

        Returns
        -------
        area: float
            value of the area under the PSD curve

        """

        if freq_range is None:
            mask = np.ones_like(freq, dtype=bool)
        else:
            mask = np.bitwise_and(freq > freq_range[0], freq < freq_range[1])

        return scipy.integrate.trapz(psd_result[mask], x=freq[mask])

    def area_under_psds(self, freq: np.ndarray, psd_results: np.ndarray,
                        freq_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Calculate the areas under PSD curves in the given frequency range (*start*, *end*)

        Parameters
        ----------
        freq : np.ndarray
            frequency
        psd_results : np.ndarray
            array of PSDs for each individual bias step or the fit to the corresponding PSD curve
        freq_range : Tuple[int, int]

        Returns
        -------
        area : np.ndarray
            array of the calculated area under each PSD curve

        """

        area = np.zeros(shape=psd_results.shape[0])
        for i, psd in enumerate(psd_results):
            area[i] = self.area_under_single_psd(freq, psd, freq_range=freq_range)

        return area  # noise value in units of \Delta I^2

    def calc_noise_value(self, areas: np.ndarray, bias_steps: np.ndarray, avg_value: Optional[np.ndarray] = None,
                         mode='noise_power') -> np.ndarray:
        """
        Calculate the noise value from area under each PSD

        Parameters
        ----------
        areas : np.ndarray
            areas under psd curves corresponding to interesting bias plateaus
            (after removing the thermal background)
        bias_steps : np.ndarray
            applied bias at these bias plateaus
        avg_value : np.ndarray
            average values on individual bias steps of conductance
            (if mode is 'noise_power' or 'conductance_noise') or current (if mode is 'current_noise')
        mode : str
            method of the noise calculation: 'noise_power' for conductance noise power and
                                             'conductance_noise' for conductance noise
                                             'current noise' for current noise

        Returns
        -------
        noise_value : np.ndarray
            conductance noise power, conductance noise or current noise for each bias step,
            depending on the *mode* parameter
        """

        if len(avg_value) == len(bias_steps):
            multiplier = utils.Constants.r0 ** 2 / bias_steps ** 2
        else:
            raise ValueError(f"Shape of avg_value {avg_value.shape} does not match "
                             f"shape of bias_steps {bias_steps.shape}. Check `avg_conductances`!")

        if mode == 'noise_power':  # units of g0**2
            return multiplier * areas  # units of g0**2

        elif mode == 'conductance_noise':  # unitless
            return np.sqrt(multiplier * areas) / avg_value  # unitless

        elif mode == 'current_noise':
            return np.sqrt(np.ones_like(multiplier) * areas) / avg_value  # unitless \Delta I/ I

        raise ValueError("Undefined mode. Valid choices: 'noise_power', 'conductance_noise' or 'current_noise'.")

    # def short_time_fft(self):
    #     freq, t, Zxx = scipy.signal.stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False,
    #                                      return_onesided=True, boundary='zeros', padded=True, axis=- 1)


class NoiseStats:
    """
    Class for noise analysis

    Parameters
    ----------
    folder : `~pathlib.Path`
        path to reach files
    bias_offset : float
        bias offset in V
    start_trace : int
        first trace in the statistics
    end_trace : int
        last trace in the statistics
    num_of_fft : int
        number of FFT calculations to perform for the calculation of a single PSD curve
    min_step_len : Optional[int], default: None.
        minimum step length of a bias step. If None the algorithm uses the avg of all the found bias steps so that only
        intentional steps are considered in the analysis
    save_data : int, default: -1
        Determines how the data should be saved. If -1, the data is not written to disc, if > 0,
        the files containing the data will have this index

    Attributes
    ----------
    home_folder = folder
    start_trace = start_trace
    end_trace = end_trace
    hold_files : np.ndarray
        array of `~pathlib.Path` instances containing teh path to reach each file containing a block of hold traces
    num_of_fft : int
        number of FFT calculations for a single bias step
    sample_rate : int
        sample rate during measurement
    freq_range :
        frequency range
    fft_interval_length_pt : int
        interval length for single fft calculation in points, always a power of 2
    freq_resolution : float
        frequency resolution
    f0 : int
        constant ?
    bias_steps_ranges : np.ndarray
        bounding points of each bias step, determined from the derivative of the measured bias signal

    # for plots
    blues : List[str]
        continous list of hex colors of blues
        ['#000c33', '#002599', '#0032cc', '#003eff', '#406eff', '#809fff']
    reds : List[str]
        continous list of hey colors of reds
        ['#330000', '#660000', '#990000', '#cc0000', '#ff0000', '#ff4040']
    cmap_geo32 : list of rgb colors
        colormap of 32 individual colors based on the colormap in IgorPro

    G_set_pull : np.ndarray
        set trigger hold conductance(s) for pull traces
    G_set_push : np.ndarray
        set trigger hold conductance(s) for push traces
    G_stop_pull : np.ndarray
        conductance at stopping point of breaking the junction in the pull trace
    G_stop_push : np.ndarray
        conductance at stopping point of closing the junction in the push trace
    G_hold_pull : np.ndarray
        average conductance in a 1 ms wide interval in the beginning of the hold measurement (in range: [50, 100))
    G_hold_push : np.ndarray
        average conductance in a 1 ms wide interval in the beginning of the hold measurement (in range: [50, 100))
    G_avg_pull : np.ndarray
        average conductance in a 1 ms wide interval at the end of the hold measurement (in range: [-100,-50))
    G_avg_push : np.ndarray
        average conductance in a 1 ms wide interval at the end of the hold measurement (in range: [-100,-50))

    G_beg_pull = []  # only for those traces where we have more than 1 bias step
    G_beg_push = []
    G_end_pull = []
    G_end_push = []
    G_avgs_pull = []  # only for traces where we have exactly 1 bias step
    G_avgs_push = []

    fft_freqs_pull = []
    fft_freqs_push = []
    psds_pull = []
    psds_push = []
    psd_fitparams_pull = []
    psd_fitparams_push = []
    areas_pull = []
    areas_push = []

    avg_cond_on_step_pull = []
    avg_cond_on_step_push = []
    avg_current_on_step_pull = []
    avg_current_on_step_push = []
    noise_power_pull = []
    noise_power_push = []
    conductance_noise_pull = []
    conductance_noise_push = []
    current_noise_pull = []
    current_noise_push = []

    trace_index = []
    bias_steps = None
    """

    def __init__(self, folder: Path, bias_offset: float = 0, traces: Optional[Union[np.array, List[int]]] = None,
                 start_trace: Optional[int] = 1, end_trace: Optional[Union[None, int]] = None,
                 num_of_fft: Optional[int] = 1, subtract_bg: bool = False,
                 min_step_len: Optional[int] = None, min_height: int = 100,
                 iv: Optional[int] = None,
                 save_data=-1):

        self.home_folder = folder
        self.traces = traces

        if traces is None:
            self.start_trace = start_trace
            self.end_trace = end_trace
            start_block, _ = utils.convert_to_block_and_trace_num(self.start_trace)
            end_block, _ = utils.convert_to_block_and_trace_num(self.end_trace)
            self.hold_files = utils.choose_files(np.array(list(folder.glob(r'hold_data_*.h5'))),
                                                 start_block=start_block, end_block=end_block)
        else:
            self.start_trace = min(traces)
            self.end_trace = max(traces)
            self.block_nums, _ = utils.convert_to_block_and_trace_num(self.traces)
            self.hold_files = np.unique(np.array(list(map(lambda x: folder.joinpath(f'hold_data_{x}.h5'),
                                                          self.block_nums))))
        self.num_of_fft = num_of_fft
        # settings for noise calculations
        self.sample_rate = 0
        self.freq_range = 0
        self.fft_interval_length_pt = 0  # interval length for single fft calculation
        self.freq_resolution = 0  # frequency resolution
        self.f0 = 1  # constant
        self.bias_steps_ranges = np.array([])

        self.G_set_pull = []
        self.G_set_push = []
        self.G_stop_pull = []
        self.G_stop_push = []
        self.G_hold_pull = []
        self.G_hold_push = []
        self.G_avg_pull = []
        self.G_avg_push = []

        self.G_beg_pull = []  # only for those traces where we have more than 1 bias step
        self.G_beg_push = []
        self.G_end_pull = []
        self.G_end_push = []
        self.G_avgs_pull = []  # only for traces where we have exactly 1 bias step
        self.G_avgs_push = []

        self.fft_freqs_pull = []
        self.fft_freqs_push = []
        self.psds_pull = []
        self.psds_push = []
        self.psd_fitparams_pull = []
        self.psd_fitparams_push = []
        self.areas_pull = []
        self.areas_push = []

        self.avg_cond_on_step_pull = []
        self.avg_cond_on_step_push = []
        self.avg_current_on_step_pull = []
        self.avg_current_on_step_push = []
        self.noise_power_pull = []
        self.noise_power_push = []
        self.conductance_noise_pull = []
        self.conductance_noise_push = []
        self.current_noise_pull = []
        self.current_noise_push = []

        self.trace_index = []
        self.bias_steps = None
        self.bias_steps_total = None

        self.calc_stats(bias_offset=bias_offset, min_step_len=min_step_len, save_data=save_data,
                        subtract_bg=subtract_bg, iv=iv, min_height=min_height)

    def calc_stats(self, bias_offset: float, min_step_len: Optional[int] = None, min_height: int = 100,
                   save_data: int = -1, subtract_bg: bool = False, iv: Optional[int] = None):
        """
        Calculate all statistics for the chosen traces

        Parameters
        ----------
        bias_offset : float
        bias offset in V
        min_step_len : Optional[int], default: None.
        minimum step length of a bias step. If None the algorithm uses the avg of all the found bias steps so that only
        intentional steps are considered in the analysis
        min_height : int, default: 100

        save_data : int, default: -1
        Determines how the data should be saved. If -1, the data is not written to disc, if > 0,
        the files containing the data will have this index
        subtract_bg : bool

        iv : Optional[int], default: None


        Returns
        -------

        """
        collected_errors = []

        if self.traces is None:
            self.traces = np.arange(self.start_trace, self.end_trace)

        for trace in tqdm(self.traces):
            bj_trace = TracePair(trace=trace, load_from=self.home_folder)
            self.sample_rate = bj_trace.sample_rate
            self.freq_range = self.sample_rate / 2
            try:
                hold_trace = HoldTrace(trace=trace, load_from=self.home_folder,
                                       bias_offset=bias_offset, r_serial_ohm=bj_trace.R_serial,
                                       sample_rate=bj_trace.sample_rate,
                                       min_step_len=min_step_len, min_height=min_height, iv=iv)

                try:
                    hold_trace.analyse_hold_trace(num_of_fft=self.num_of_fft,
                                                  subtract_bg=subtract_bg)
                except ValueError:
                    # print(f'Error at {hold_trace.trace_name}')
                    collected_errors.append(f'Error at {hold_trace.trace_name}')
                    continue
                except MeasurementOverflow:
                    # print(f'Overflow at {hold_trace.trace_name}')
                    collected_errors.append(f'Overflow at {hold_trace.trace_name}')
                    continue

                self.G_set_pull.append(bj_trace.hold_set_pull)
                self.G_set_push.append(bj_trace.hold_set_push)
                self.G_stop_pull.append(bj_trace.hold_conductance_pull)
                self.G_stop_push.append(bj_trace.hold_conductance_push)
                self.G_hold_pull.append(hold_trace.G_hold_pull)
                self.G_hold_push.append(hold_trace.G_hold_push)
                self.G_avg_pull.append(hold_trace.G_avg_pull)
                self.G_avg_push.append(hold_trace.G_avg_push)

                if len(hold_trace.bias_steps) > 1:
                    self.G_beg_pull.append(hold_trace.conductance_avgs_pull[:, 0].flatten())
                    self.G_end_pull.append(hold_trace.conductance_avgs_pull[:, 1].flatten())

                    self.G_beg_push.append(hold_trace.conductance_avgs_push[:, 0].flatten())
                    self.G_end_push.append(hold_trace.conductance_avgs_push[:, 1].flatten())
                else:
                    self.G_avgs_pull.append(hold_trace.conductance_avgs_pull)
                    self.G_avgs_push.append(hold_trace.conductance_avgs_push)

                # self.psds_pull = []  # These are only needed if we want to calculate the 2D histograms
                # self.psds_push = []  # of the psds. Do we want that?
                self.psd_fitparams_pull.append(hold_trace.psd_fitparams_pull.flatten())
                self.psd_fitparams_push.append(hold_trace.psd_fitparams_push.flatten())
                self.areas_pull.append(hold_trace.areas_pull)
                self.areas_push.append(hold_trace.areas_push)

                self.avg_cond_on_step_pull.append(hold_trace.avg_cond_on_step_pull)
                self.avg_cond_on_step_push.append(hold_trace.avg_cond_on_step_push)
                self.avg_current_on_step_pull.append(hold_trace.avg_current_on_step_pull)
                self.avg_current_on_step_push.append(hold_trace.avg_current_on_step_push)
                self.noise_power_pull.append(hold_trace.noise_power_pull)
                self.noise_power_push.append(hold_trace.noise_power_push)
                self.conductance_noise_pull.append(hold_trace.conductance_noise_pull)
                self.conductance_noise_push.append(hold_trace.conductance_noise_push)
                self.current_noise_pull.append(hold_trace.current_noise_pull)
                self.current_noise_push.append(hold_trace.current_noise_push)
                self.trace_index.append(trace)

            except MeasurementNotComplete:
                collected_errors.append(f'Measurement incomplete at {trace}')
                continue

        self.fft_freqs_pull = hold_trace.fft_freqs_pull
        self.fft_freqs_push = hold_trace.fft_freqs_push

        # in the end convert lists to numpy arrays:
        self.G_set_pull = np.array(self.G_set_pull)
        self.G_set_push = np.array(self.G_set_push)
        self.G_stop_pull = np.array(self.G_stop_pull)
        self.G_stop_push = np.array(self.G_stop_push)
        self.G_hold_pull = np.array(self.G_hold_pull)
        self.G_hold_push = np.array(self.G_hold_push)
        self.G_avg_pull = np.array(self.G_avg_pull)
        self.G_avg_push = np.array(self.G_avg_push)

        self.G_beg_pull = np.array(self.G_beg_pull)
        self.G_beg_push = np.array(self.G_beg_push)
        self.G_end_pull = np.array(self.G_end_pull)
        self.G_end_push = np.array(self.G_end_push)
        self.G_avgs_pull = np.array(self.G_avgs_pull)
        self.G_avgs_push = np.array(self.G_avgs_push)

        self.psd_fitparams_pull = np.array(self.psd_fitparams_pull)
        self.psd_fitparams_push = np.array(self.psd_fitparams_push)
        self.areas_pull = np.array(self.areas_pull)
        self.areas_push = np.array(self.areas_push)

        self.avg_cond_on_step_pull = np.array(self.avg_cond_on_step_pull)
        self.avg_cond_on_step_push = np.array(self.avg_cond_on_step_push)
        self.avg_current_on_step_pull = np.array(self.avg_current_on_step_pull)
        self.avg_current_on_step_push = np.array(self.avg_current_on_step_push)
        self.noise_power_pull = np.array(self.noise_power_pull)
        self.noise_power_push = np.array(self.noise_power_push)
        self.conductance_noise_pull = np.array(self.conductance_noise_pull)
        self.conductance_noise_push = np.array(self.conductance_noise_push)
        self.current_noise_pull = np.array(self.current_noise_pull)
        self.current_noise_push = np.array(self.current_noise_push)

        self.trace_index = np.array(self.trace_index)

        self.bias_steps = hold_trace.bias_steps
        self.bias_steps_total = hold_trace.bias_steps_total

        if save_data > -1:
            # Saving data to files
            if len(self.bias_steps) > 1:
                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_pull,
                                        "G_stop": self.G_stop_pull,
                                        "G_hold": self.G_hold_pull,
                                        "G_avg": self.G_avg_pull},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_beg_{i + 1}" for i in range(self.bias_steps_total.shape[0])] + \
                       [f"G_end_{i + 1}" for i in range(self.bias_steps_total.shape[0])]
                my_df_2 = pd.DataFrame(np.hstack((np.vstack((self.bias_steps_total, self.G_beg_pull)),
                                                  np.vstack((self.bias_steps_total, self.G_end_pull)))),
                                       columns=cols)
                conductance_stats_pull = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_pull.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))

                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_push,
                                        "G_stop": self.G_stop_push,
                                        "G_hold": self.G_hold_push,
                                        "G_avg": self.G_avg_push},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_beg_{i + 1}" for i in range(self.bias_steps_total.shape[0])] + \
                       [f"G_end_{i + 1}" for i in range(self.bias_steps_total.shape[0])]
                my_df_2 = pd.DataFrame(np.hstack((np.vstack((self.bias_steps_total, self.G_beg_push)),
                                                  np.vstack((self.bias_steps_total, self.G_end_push)))),
                                       columns=cols)
                conductance_stats_push = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_push.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))
            else:
                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_pull,
                                        "G_stop": self.G_stop_pull,
                                        "G_hold": self.G_hold_pull,
                                        "G_avg": self.G_avg_pull},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_avgs_{i + 1}" for i in range(self.G_avgs_pull.shape[1])]
                my_df_2 = pd.DataFrame(np.vstack((np.ones(self.G_avgs_pull.shape[1]) * self.bias_steps_total[0],
                                                  self.G_avgs_pull)), columns=cols)
                conductance_stats_pull = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_pull.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))

                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_push,
                                        "G_stop": self.G_stop_push,
                                        "G_hold": self.G_hold_push,
                                        "G_avg": self.G_avg_push},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_avgs_{i + 1}" for i in range(self.G_avgs_push.shape[1])]
                my_df_2 = pd.DataFrame(np.vstack((np.ones(self.G_avgs_pull.shape[1]) * self.bias_steps_total[0],
                                                  self.G_avgs_push)), columns=cols)
                conductance_stats_push = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_push.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))

            noise_stats_pull = pd.DataFrame({'trace_index': np.concatenate((np.array([-1]), self.trace_index))})
            noise_stats_push = pd.DataFrame({'trace_index': np.concatenate((np.array([-1]), self.trace_index))})
            for i, step in enumerate(self.bias_steps):
                # pull
                noise_stats_pull[f'avg_cond_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                self.avg_cond_on_step_pull[:, i]))
                noise_stats_pull[f'avg_current_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                   self.avg_current_on_step_pull[:, i]))
                noise_stats_pull[f'noise_power_{i + 1}'] = np.concatenate((np.array([step]),
                                                                           self.noise_power_pull[:, i]))
                noise_stats_pull[f'conductance_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                 self.conductance_noise_pull[:, i]))
                noise_stats_pull[f'current_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                             self.current_noise_pull[:, i]))
                # push
                noise_stats_push[f'avg_cond_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                self.avg_cond_on_step_push[:, i]))
                noise_stats_push[f'avg_current_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                   self.avg_current_on_step_push[:, i]))
                noise_stats_push[f'noise_power_{i + 1}'] = np.concatenate((np.array([step]),
                                                                           self.noise_power_push[:, i]))
                noise_stats_push[f'conductance_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                 self.conductance_noise_push[:, i]))
                noise_stats_push[f'current_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                             self.current_noise_push[:, i]))
            try:
                conductance_stats_pull.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_pull_{save_data}.csv'), index=False)
                conductance_stats_push.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_push_{save_data}.csv'), index=False)
                noise_stats_pull.to_csv(self.home_folder.joinpath(f'results/noise_stats_pull_{save_data}.csv'),
                                        index=False)
                noise_stats_push.to_csv(self.home_folder.joinpath(f'results/noise_stats_push_{save_data}.csv'),
                                        index=False)
                print("Data saved.")
            except FileNotFoundError:
                mkdir(self.home_folder.joinpath('results'))
                conductance_stats_pull.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_pull_{save_data}.csv'), index=False)
                conductance_stats_push.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_push_{save_data}.csv'), index=False)
                noise_stats_pull.to_csv(self.home_folder.joinpath(f'results/noise_stats_pull_{save_data}.csv'),
                                        index=False)
                noise_stats_push.to_csv(self.home_folder.joinpath(f'results/noise_stats_push_{save_data}.csv'),
                                        index=False)
                print("Data saved.")

        if len(collected_errors) > 0:
            print("During the analysis the following errors occurred:")
            for i in collected_errors:
                print(i)

    def calc_stats_old(self, bias_offset: float, min_step_len: Optional[int] = None, min_height: int = 100,
                       save_data: int = -1, subtract_bg: bool = False, iv: Optional[int] = None):
        """
        Calculate all statistics for the chosen traces

        Parameters
        ----------
        bias_offset : float
        bias offset in V
        min_step_len : Optional[int], default: None.
        minimum step length of a bias step. If None the algorithm uses the avg of all the found bias steps so that only
        intentional steps are considered in the analysis
        min_height : int, default: 100

        save_data : int, default: -1
        Determines how the data should be saved. If -1, the data is not written to disc, if > 0,
        the files containing the data will have this index
        subtract_bg : bool

        iv : Optional[int], default: None


        Returns
        -------

        """
        collected_errors = []

        for hold_file in tqdm(self.hold_files, desc="Processing hold files"):
            bj_file = self.home_folder.joinpath(re.sub(r'hold_data', r'break_junction', hold_file.name))
            with h5py.File(bj_file, "r") as bj_in:
                with h5py.File(hold_file, "r") as hold_in:

                    if self.traces is None:
                        chosen_traces = utils.choose_traces(trace_array=np.array(list(hold_in['pull'].keys())),
                                                            first=self.start_trace, last=self.end_trace)

                        chosen_traces = np.array(list(map(utils.get_num_from_name, chosen_traces)))
                    else:
                        chosen_traces = self.traces[self.block_nums == utils.get_num_from_name(hold_file.name)]

                    for trace in chosen_traces:
                        trace_num = trace
                        bj_trace = TracePair(trace=trace, load_from=bj_in)
                        self.sample_rate = bj_trace.sample_rate
                        self.freq_range = self.sample_rate / 2
                        try:
                            hold_trace = HoldTrace(trace=trace, load_from=hold_in,
                                                   bias_offset=bias_offset, r_serial_ohm=bj_trace.R_serial,
                                                   sample_rate=bj_trace.sample_rate,
                                                   min_step_len=min_step_len, min_height=min_height, iv=iv)

                            try:
                                hold_trace.analyse_hold_trace(num_of_fft=self.num_of_fft,
                                                              subtract_bg=subtract_bg)
                            except ValueError:
                                # print(f'Error at {hold_trace.trace_name}')
                                collected_errors.append(f'Error at {hold_trace.trace_name}')
                                continue
                            except MeasurementOverflow:
                                # print(f'Overflow at {hold_trace.trace_name}')
                                collected_errors.append(f'Overflow at {hold_trace.trace_name}')
                                continue

                            self.G_set_pull.append(bj_trace.hold_set_pull)
                            self.G_set_push.append(bj_trace.hold_set_push)
                            self.G_stop_pull.append(bj_trace.hold_conductance_pull)
                            self.G_stop_push.append(bj_trace.hold_conductance_push)
                            self.G_hold_pull.append(hold_trace.G_hold_pull)
                            self.G_hold_push.append(hold_trace.G_hold_push)
                            self.G_avg_pull.append(hold_trace.G_avg_pull)
                            self.G_avg_push.append(hold_trace.G_avg_push)

                            if len(hold_trace.bias_steps) > 1:
                                self.G_beg_pull.append(hold_trace.conductance_avgs_pull[:, 0].flatten())
                                self.G_end_pull.append(hold_trace.conductance_avgs_pull[:, 1].flatten())

                                self.G_beg_push.append(hold_trace.conductance_avgs_push[:, 0].flatten())
                                self.G_end_push.append(hold_trace.conductance_avgs_push[:, 1].flatten())
                            else:
                                self.G_avgs_pull.append(hold_trace.conductance_avgs_pull)
                                self.G_avgs_push.append(hold_trace.conductance_avgs_push)

                            # self.psds_pull = []  # These are only needed if we want to calculate the 2D histograms
                            # self.psds_push = []  # of the psds. Do we want that?
                            self.psd_fitparams_pull.append(hold_trace.psd_fitparams_pull.flatten())
                            self.psd_fitparams_push.append(hold_trace.psd_fitparams_push.flatten())
                            self.areas_pull.append(hold_trace.areas_pull)
                            self.areas_push.append(hold_trace.areas_push)

                            self.avg_cond_on_step_pull.append(hold_trace.avg_cond_on_step_pull)
                            self.avg_cond_on_step_push.append(hold_trace.avg_cond_on_step_push)
                            self.avg_current_on_step_pull.append(hold_trace.avg_current_on_step_pull)
                            self.avg_current_on_step_push.append(hold_trace.avg_current_on_step_push)
                            self.noise_power_pull.append(hold_trace.noise_power_pull)
                            self.noise_power_push.append(hold_trace.noise_power_push)
                            self.conductance_noise_pull.append(hold_trace.conductance_noise_pull)
                            self.conductance_noise_push.append(hold_trace.conductance_noise_push)
                            self.current_noise_pull.append(hold_trace.current_noise_pull)
                            self.current_noise_push.append(hold_trace.current_noise_push)
                            self.trace_index.append(trace_num)

                        except MeasurementNotComplete:
                            collected_errors.append(f'Measurement incomplete at {trace}')
                            continue

        self.fft_freqs_pull = hold_trace.fft_freqs_pull
        self.fft_freqs_push = hold_trace.fft_freqs_push

        # in the end convert lists to numpy arrays:
        self.G_set_pull = np.array(self.G_set_pull)
        self.G_set_push = np.array(self.G_set_push)
        self.G_stop_pull = np.array(self.G_stop_pull)
        self.G_stop_push = np.array(self.G_stop_push)
        self.G_hold_pull = np.array(self.G_hold_pull)
        self.G_hold_push = np.array(self.G_hold_push)
        self.G_avg_pull = np.array(self.G_avg_pull)
        self.G_avg_push = np.array(self.G_avg_push)

        self.G_beg_pull = np.array(self.G_beg_pull)
        self.G_beg_push = np.array(self.G_beg_push)
        self.G_end_pull = np.array(self.G_end_pull)
        self.G_end_push = np.array(self.G_end_push)
        self.G_avgs_pull = np.array(self.G_avgs_pull)
        self.G_avgs_push = np.array(self.G_avgs_push)

        self.psd_fitparams_pull = np.array(self.psd_fitparams_pull)
        self.psd_fitparams_push = np.array(self.psd_fitparams_push)
        self.areas_pull = np.array(self.areas_pull)
        self.areas_push = np.array(self.areas_push)

        self.avg_cond_on_step_pull = np.array(self.avg_cond_on_step_pull)
        self.avg_cond_on_step_push = np.array(self.avg_cond_on_step_push)
        self.avg_current_on_step_pull = np.array(self.avg_current_on_step_pull)
        self.avg_current_on_step_push = np.array(self.avg_current_on_step_push)
        self.noise_power_pull = np.array(self.noise_power_pull)
        self.noise_power_push = np.array(self.noise_power_push)
        self.conductance_noise_pull = np.array(self.conductance_noise_pull)
        self.conductance_noise_push = np.array(self.conductance_noise_push)
        self.current_noise_pull = np.array(self.current_noise_pull)
        self.current_noise_push = np.array(self.current_noise_push)

        self.trace_index = np.array(self.trace_index)

        self.bias_steps = hold_trace.bias_steps
        self.bias_steps_total = hold_trace.bias_steps_total

        if save_data > -1:
            # Saving data to files
            if len(self.bias_steps) > 1:
                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_pull,
                                        "G_stop": self.G_stop_pull,
                                        "G_hold": self.G_hold_pull,
                                        "G_avg": self.G_avg_pull},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_beg_{i + 1}" for i in range(self.bias_steps_total.shape[0])] + \
                       [f"G_end_{i + 1}" for i in range(self.bias_steps_total.shape[0])]
                my_df_2 = pd.DataFrame(np.hstack((np.vstack((self.bias_steps_total, self.G_beg_pull)),
                                                  np.vstack((self.bias_steps_total, self.G_end_pull)))),
                                       columns=cols)
                conductance_stats_pull = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_pull.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))

                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_push,
                                        "G_stop": self.G_stop_push,
                                        "G_hold": self.G_hold_push,
                                        "G_avg": self.G_avg_push},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_beg_{i + 1}" for i in range(self.bias_steps_total.shape[0])] + \
                       [f"G_end_{i + 1}" for i in range(self.bias_steps_total.shape[0])]
                my_df_2 = pd.DataFrame(np.hstack((np.vstack((self.bias_steps_total, self.G_beg_push)),
                                                  np.vstack((self.bias_steps_total, self.G_end_push)))),
                                       columns=cols)
                conductance_stats_push = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_push.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))
            else:
                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_pull,
                                        "G_stop": self.G_stop_pull,
                                        "G_hold": self.G_hold_pull,
                                        "G_avg": self.G_avg_pull},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_avgs_{i + 1}" for i in range(self.G_avgs_pull.shape[1])]
                my_df_2 = pd.DataFrame(np.vstack((np.ones(self.G_avgs_pull.shape[1]) * self.bias_steps_total[0],
                                                  self.G_avgs_pull)), columns=cols)
                conductance_stats_pull = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_pull.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))

                my_df_1 = pd.DataFrame({"trace_index": self.trace_index,
                                        "G_set": self.G_set_push,
                                        "G_stop": self.G_stop_push,
                                        "G_hold": self.G_hold_push,
                                        "G_avg": self.G_avg_push},
                                       index=range(1, self.trace_index.shape[0] + 1))

                cols = [f"G_avgs_{i + 1}" for i in range(self.G_avgs_push.shape[1])]
                my_df_2 = pd.DataFrame(np.vstack((np.ones(self.G_avgs_pull.shape[1]) * self.bias_steps_total[0],
                                                  self.G_avgs_push)), columns=cols)
                conductance_stats_push = pd.concat((my_df_1, my_df_2), axis=1)
                conductance_stats_push.index = ['bias'] + list(range(1, self.trace_index.shape[0] + 1))

            noise_stats_pull = pd.DataFrame({'trace_index': np.concatenate((np.array([-1]), self.trace_index))})
            noise_stats_push = pd.DataFrame({'trace_index': np.concatenate((np.array([-1]), self.trace_index))})
            for i, step in enumerate(self.bias_steps):
                # pull
                noise_stats_pull[f'avg_cond_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                self.avg_cond_on_step_pull[:, i]))
                noise_stats_pull[f'avg_current_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                   self.avg_current_on_step_pull[:, i]))
                noise_stats_pull[f'noise_power_{i + 1}'] = np.concatenate((np.array([step]),
                                                                           self.noise_power_pull[:, i]))
                noise_stats_pull[f'conductance_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                 self.conductance_noise_pull[:, i]))
                noise_stats_pull[f'current_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                             self.current_noise_pull[:, i]))
                # push
                noise_stats_push[f'avg_cond_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                self.avg_cond_on_step_push[:, i]))
                noise_stats_push[f'avg_current_on_step_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                   self.avg_current_on_step_push[:, i]))
                noise_stats_push[f'noise_power_{i + 1}'] = np.concatenate((np.array([step]),
                                                                           self.noise_power_push[:, i]))
                noise_stats_push[f'conductance_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                                 self.conductance_noise_push[:, i]))
                noise_stats_push[f'current_noise_{i + 1}'] = np.concatenate((np.array([step]),
                                                                             self.current_noise_push[:, i]))
            try:
                conductance_stats_pull.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_pull_{save_data}.csv'), index=False)
                conductance_stats_push.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_push_{save_data}.csv'), index=False)
                noise_stats_pull.to_csv(self.home_folder.joinpath(f'results/noise_stats_pull_{save_data}.csv'),
                                        index=False)
                noise_stats_push.to_csv(self.home_folder.joinpath(f'results/noise_stats_push_{save_data}.csv'),
                                        index=False)
                print("Data saved.")
            except FileNotFoundError:
                mkdir(self.home_folder.joinpath('results'))
                conductance_stats_pull.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_pull_{save_data}.csv'), index=False)
                conductance_stats_push.to_csv(
                    self.home_folder.joinpath(f'results/conductance_stats_push_{save_data}.csv'), index=False)
                noise_stats_pull.to_csv(self.home_folder.joinpath(f'results/noise_stats_pull_{save_data}.csv'),
                                        index=False)
                noise_stats_push.to_csv(self.home_folder.joinpath(f'results/noise_stats_push_{save_data}.csv'),
                                        index=False)
                print("Data saved.")

        if len(collected_errors) > 0:
            print("During the analysis the following errors occurred:")
            for i in collected_errors:
                print(i)

    def scatterplot_pull_push(self, x_pull, x_push, y_pull, y_push, ax=None, dpi: int = 600, **kwargs):
        """
        Scatterpllot of values x and y, for both directions in one plot

        Parameters
        ----------
        x_pull : np.ndarray
            any value for the pull trace, horizontal axis
        x_push : np.ndarray
            any value for the push trace, horizontal axis
        y_pull : np.ndarray
            any value for the pull trace, vertical axis
        y_push : np.ndarray
            any value for the push trace, vertical axis
        ax : Optional, `~matplotlib.axes.Axes`, default: None. If None, a new axis is created
            pre-defined axis for plot
        kwargs : additional keyword arguments for `matplotlib.axes.Axes.scatter`

        Returns
        -------

        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=utils.cm2inch(7, 7), dpi=dpi)
        ax.scatter(x_pull, y_pull, edgecolors='none', **kwargs)
        ax.scatter(x_push, y_push, edgecolors='none', **kwargs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-5, 1e-1)
        ax.set_ylim(1e-5, 1e-1)
        ax.set_xlabel(r'$G_\mathrm{set} [G_{0}]$')
        ax.set_ylabel(r'$G_\mathrm{stop} [G_{0}]$')
        plt.show()

    def scatterplot(self, x: np.ndarray, y: np.ndarray,
                    ax_xlabel=r'$G_\mathrm{} [G_{0}]$',
                    ax_ylabel=r'$G_\mathrm{} [G_{0}]$',
                    ax: Optional[matplotlib.axes.Axes] = None, **kwargs):
        """

        Parameters
        ----------
        x : np.ndarray
            values for the horizontal axis
        y : np.ndarray
            values for the vertical axis
        ax_xlabel : str
            horizontal axis label
        ax_ylabel : str
            vertical axis label
        ax : `~matplotlib.axes.Axes`
            pre-defined axis
        kwargs : Optional
            additional keyword arguments for `~matplotlib.axes.Axes.scatter`

        Returns
        -------

        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=utils.cm2inch(7, 7), dpi=300)
        ax.scatter(x, y, **kwargs, edgecolors='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-5, 1e-1)
        ax.set_ylim(1e-5, 1e-1)
        ax.set_xlabel(ax_xlabel)
        ax.set_ylabel(ax_ylabel)

        return ax

    def scat_and_hist(self,
                      xlabel: str, ylabel: str,
                      x_pull: Optional[np.ndarray] = None,
                      y_pull: Optional[np.ndarray] = None,
                      x_push: Optional[np.ndarray] = None,
                      y_push: Optional[np.ndarray] = None,
                      set_arr_pull: Optional[np.ndarray] = None,
                      set_arr_push: Optional[np.ndarray] = None,
                      dpi: int = 600):
        """
        Plot scatterplots and histograms for each axis

        Parameters
        ----------
        xlabel : str
            label for horizontal axis
        ylabel : str
            label for vertical axis
        x_pull : Optional, np.ndarray
            values of horizontal axis, of the pull traces
        y_pull : Optional, np.ndarray
            values of vertical axis, of the pull traces
        x_push : Optional, np.ndarray
            values of horizontal axis, of the push traces
        y_push : Optional, np.ndarray
            values of vertical axis, of the push traces
        set_arr_pull : Optional, np.ndarray
            set values for the pull traces
        set_arr_push : Optional, np.ndarray
            set values for the push traces
        dpi : int, default: 600
            resolution in dots per inch

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            axis with scatterplot
        ax_histx : `~matplotlib.axes.Axes`
            axis with histogram of values at the horizontal axis
        ax_histy : `~matplotlib.axes.Axes`
            axis with histogram of values at the vertical axis

        """

        fig = plt.figure(figsize=utils.cm2inch(10, 10), dpi=dpi)  # figsize: (width, height) in inches
        gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=(7, 3), height_ratios=(3, 7),
                               figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax_histx.xaxis.set_label_position('top')
        ax_histx.xaxis.tick_top()
        ax_histx.xaxis.set_ticks_position('both')
        ax_histx.yaxis.set_ticks_position('both')
        ax_histx.yaxis.set_label_position('left')

        ax_histy.yaxis.set_label_position('right')
        ax_histy.yaxis.tick_right()
        ax_histy.yaxis.set_ticks_position('both')
        ax_histy.xaxis.set_label_position('bottom')
        ax_histy.xaxis.set_ticks_position('both')

        if set_arr_pull is not None:
            set_vals_pull = np.unique(set_arr_pull)
            histx_pull = []
            histy_pull = []

            for i in range(len(set_vals_pull)):
                ax = self.scatterplot(x_pull[set_arr_pull == set_vals_pull[i]],
                                      y_pull[set_arr_pull == set_vals_pull[i]],
                                      ax_xlabel=r'$G_\mathrm{set} [G_{0}]$', ax_ylabel=r'$G_\mathrm{stop} [G_{0}]$',
                                      ax=ax, c=utils.blues[i], alpha=0.5, s=1)

                binsx, hist = utils.calc_hist_1d_single(x_pull[set_arr_pull == set_vals_pull[i]],
                                                        xrange=(1e-5, 1e-1), xbins_num=100, log_scale=True)
                histx_pull.append(hist)

                binsy, hist = utils.calc_hist_1d_single(y_pull[set_arr_pull == set_vals_pull[i]],
                                                        xrange=(1e-5, 1e-1), xbins_num=100, log_scale=True)
                histy_pull.append(hist)

                ax_histx.plot(binsx, histx_pull[-1], c=utils.blues[i], lw=0.6, alpha=0.7)
                ax_histy.plot(histy_pull[-1], binsy, c=utils.blues[i], lw=0.6, alpha=0.7)

            histx_pull = np.array(histx_pull)
            histy_pull = np.array(histy_pull)

            ax_histx.set_ylim(0, max(histx_pull.flatten()))
            ax_histy.set_xlim(0, max(histy_pull.flatten()))

        if set_arr_push is not None:
            set_vals_push = np.unique(set_arr_push)
            histx_push = []
            histy_push = []

            for i in range(len(set_vals_push)):
                ax = self.scatterplot(x_push[set_arr_push == set_vals_push[i]],
                                      y_push[set_arr_push == set_vals_push[i]],
                                      ax_xlabel=r'$G_\mathrm{set} [G_{0}]$', ax_ylabel=r'$G_\mathrm{stop} [G_{0}]$',
                                      ax=ax, c=utils.reds[i], alpha=0.5, s=1)

                binsx, hist = utils.calc_hist_1d_single(x_push[set_arr_push == set_vals_push[i]],
                                                        xrange=(1e-5, 1e-1), xbins_num=100, log_scale=True)
                histx_push.append(hist)

                binsy, hist = utils.calc_hist_1d_single(y_push[set_arr_push == set_vals_push[i]],
                                                        xrange=(1e-5, 1e-1), xbins_num=100, log_scale=True)
                histy_push.append(hist)

                ax_histx.plot(binsx, histx_push[-1], c=utils.reds[i], lw=0.6, alpha=0.7)
                ax_histy.plot(histy_push[-1], binsy, c=utils.reds[i], lw=0.6, alpha=0.7)

            histx_push = np.array(histx_push)
            histy_push = np.array(histy_push)

            ax_histx.set_ylim(0, max(histx_push.flatten()))
            ax_histy.set_xlim(0, max(histy_push.flatten()))

        try:
            ax_histx.set_ylim(0, max(max(histx_pull.flatten()), max(histx_push.flatten())))
            ax_histy.set_xlim(0, max(max(histy_pull.flatten()), max(histy_push.flatten())))
        except UnboundLocalError:
            print("")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax_histx.set_ylabel('Counts')
        ax_histy.set_xlabel('Counts')

        return ax, ax_histx, ax_histy

    def plot_correlation(self, n, correlation, axis=None, dpi: int = 600, **kwargs):
        """

        Parameters
        ----------
        n : np.ndarray
            parameter
        correlation : np.ndarray
            calculated carrulation to plot
        axis : Optional
            pre-existing axis where plot should be placed
        kwargs : Any
            additional keyword arguments for plot formatting, see matplotlib.axes.plot

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            axis with resulting figure

        See Also
        --------
        utils.calc_correlation

        """

        if axis is None:
            fig, ax = plt.subplots(1, figsize=utils.cm2inch(6, 4), dpi=dpi)
        else:
            ax = axis
        ax.plot(n, correlation, **kwargs)
        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'$C(\log(iPSD/G_{\mathrm{avg}}^{n}), \log(G_{\mathrm{avg}}))$')
        ax.axhline(y=0, xmin=0, xmax=1, ls='--', lw=0.5, c='k')
        ax.axvline(x=n[abs(correlation) == min(abs(correlation))][0], ymin=0, ymax=1,
                   ls='--', lw=0.5, c='k')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.text(x=0.15, y=0.15, s=f"n = {np.round(n[abs(correlation) == min(abs(correlation))][0], 2)}",
                transform=ax.transAxes, fontsize='xx-small', ha='left', va='bottom')

        return ax

    def plot_noise_power_2dhist(self, conductance_avgs: np.ndarray, noise_power: np.ndarray, n: float,
                                xrange: Optional[Tuple[float, float]] = (1e-5, 0.1),
                                yrange: Optional[Tuple[float, float]] = (1e-12, 1e-5),
                                num_bins: Optional[Tuple[int, int]] = (10, 10),
                                shift: Optional[float] = 0, axis=None, dpi: int = 600):
        """

        Parameters
        ----------
        conductance_avgs
        noise_power
        n
        xrange
        yrange
        num_bins
        shift
        axis
        dpi

        Returns
        -------
        ax
        """

        num_of_decs_x = np.log10(xrange[1]) - np.log10(xrange[0])
        num_of_decs_y = np.log10(yrange[1]) - np.log10(yrange[0])

        xbins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), num=int(num_bins[0] * num_of_decs_x))
        ybins = np.logspace(np.log10(yrange[0]), np.log10(yrange[1]), num=int(num_bins[0] * num_of_decs_y))

        h, xedges, yedges = np.histogram2d(conductance_avgs.flatten(), noise_power.flatten(), bins=[xbins, ybins])
        x_mesh, y_mesh = np.meshgrid(xedges, yedges)

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

        ax.plot(xbins, 10 ** (n * np.log10(xbins) + shift), lw=0.5, ls='--', c='k')

        ax.text(x=0.85, y=0.15, s=f"n = {np.round(n, 2)}",
                transform=ax.transAxes, fontsize='xx-small', ha='right', va='bottom')

        im = ax.pcolormesh(x_mesh, y_mesh, h.T, cmap=utils.cmap_geo32)

        return ax
