import itertools
from time import time_ns
import numpy as np
from pathlib import Path
import re
import math
from typing import Union, Tuple, List, Optional
from matplotlib.colors import ListedColormap

cmap_geo32 = ListedColormap(np.array([[255 / 255, 255 / 255, 255 / 255, 1],
                                      [255 / 255, 235 / 255, 235 / 255, 1],
                                      [255 / 255, 215 / 255, 215 / 255, 1],
                                      [255 / 255, 196 / 255, 196 / 255, 1],
                                      [245 / 255, 179 / 255, 174 / 255, 1],
                                      [255 / 255, 158 / 255, 158 / 255, 1],
                                      [255 / 255, 124 / 255, 124 / 255, 1],
                                      [255 / 255, 90 / 255, 90 / 255, 1],
                                      [238 / 255, 80 / 255, 78 / 255, 1],
                                      [244 / 255, 117 / 255, 75 / 255, 1],
                                      [255 / 255, 160 / 255, 69 / 255, 1],
                                      [255 / 255, 189 / 255, 87 / 255, 1],
                                      [247 / 255, 215 / 255, 104 / 255, 1],
                                      [240 / 255, 236 / 255, 121 / 255, 1],
                                      [223 / 255, 245 / 255, 141 / 255, 1],
                                      [205 / 255, 255 / 255, 162 / 255, 1],
                                      [172 / 255, 245 / 255, 168 / 255, 1],
                                      [138 / 255, 236 / 255, 174 / 255, 1],
                                      [124 / 255, 235 / 255, 200 / 255, 1],
                                      [106 / 255, 235 / 255, 225 / 255, 1],
                                      [97 / 255, 225 / 255, 240 / 255, 1],
                                      [68 / 255, 202 / 255, 255 / 255, 1],
                                      [50 / 255, 190 / 255, 255 / 255, 1],
                                      [25 / 255, 175 / 255, 255 / 255, 1],
                                      [13 / 255, 129 / 255, 248 / 255, 1],
                                      [26 / 255, 102 / 255, 240 / 255, 1],
                                      [0 / 255, 40 / 255, 224 / 255, 1],
                                      [0 / 255, 25 / 255, 212 / 255, 1],
                                      [0 / 255, 10 / 255, 200 / 255, 1],
                                      [20 / 255, 5 / 255, 175 / 255, 1],
                                      [40 / 255, 0 / 255, 150 / 255, 1],
                                      [10 / 255, 0 / 255, 121 / 255, 1]]))

blues = ['#000c33', '#002599', '#0032cc', '#003eff', '#406eff', '#809fff']  # blues
reds = ['#330000', '#660000', '#990000', '#cc0000', '#ff0000', '#ff4040']  # reds


class MyException(Exception):
    def __init__(self, message):
        self.message = message


class MeasurementNotComplete(Exception):
    def __init__(self, message):
        self.message = message


class MeasurementOverflow(Exception):
    def __init__(self, message):
        self.message = message


class Constants:
    r0 = 12900  # Ohm
    g0 = 1 / r0  # Siemens = 1/Ohm


def check_date(date: str):
    """
    Check if a date is formatted right

    Parameters
    ----------
    date : str
        date string to check formatting

    Examples
    --------
    >>> check_date('21_12_08')

    >>> check_date('2021_12_08')
    ValueError: Invalid date format. Enter the date in the following format: 'yy_mm_dd'

    """
    if re.match(r'\d\d_\d\d_\d\d', date) is None:
        raise ValueError("Invalid date format. Enter the date in the following format: 'yy_mm_dd'")


def cm2inch(*tupl):
    """
    Converts cm to inch (mostly used for figsize)

    Parameters
    ----------
    tupl : float or tuple
        values to convert

    Returns
    -------
    res : tuple
        size in inch

    Examples
    --------
    >>> cm2inch(5)
    (1.968503937007874,)
    >>> cm2inch(5, 6, 7)
    (1.968503937007874, 2.3622047244094486, 2.7559055118110236)
    >>> cm2inch((5, 6))
    (1.968503937007874, 2.3622047244094486)

    """
    inch = 2.54  # 1 cm = 2.54 inch
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def point2inch(*tupl):
    """
    Converts cm to inch (mostly used for figsize)

    Parameters
    ----------
    tupl : float or tuple
        values to convert

    Returns
    -------
    res : tuple
        size in inch

    Examples
    --------
    >>> point2inch(72)
    (1.0,)
    >>> point2inch(240, 300, 504, 660)
    (3.3333333333333335, 4.166666666666667, 7.0, 9.166666666666666)

    Notes
    -----
    Second example contains the figure sizes in points recommended by ACS Publications:
    width of single column figure: 240 pt,
    min width of double column fig: 300 pt,
    max width of  double column fig: 504 pt,
    max height including the caption: 660 pt,
    minimum font size: 4.5 pt
    """
    inch = 72  # 1 point = 1/72 inch
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def calculate_conductance_g0(bias_v: Union[float, np.ndarray], current: Union[float, np.ndarray], r_serial_ohm=99900) \
        -> Union[float, np.ndarray]:
    """
    Calculates the conductance in units of the conductance quantum :math:`G_{0} = \\frac{1}{12900\\;\\mathrm{Ohm}}`

    Parameters
    ----------
    bias_v : float or numpy.ndarray
        measured bias in V
    current : float or numpy.ndarray
        measured current in A
    r_serial_ohm : int
        resistor in series in Ohm (default value: 99900)

    Returns
    -------
    res : float or numpy.array
        conductance in :math:`G_{0}` units
    """

    return 12900 * (bias_v / current - r_serial_ohm) ** (-1)


def calculate_resistance_r0(bias_v: Union[float, np.ndarray], current_a: Union[float, np.ndarray]) \
        -> Union[float, np.ndarray]:
    """
    Calculates the resistance in units of the resistance quantum :math:`R_{0} = 12900\\;\\mathrm{Ohm}`

    Parameters
    ----------
    bias_v : float or numpy.array
        measured bias in V
    current_a : float or numpy.array
        measured current in A

    Returns
    -------
    res : float or numpy.array
        resistance in ::math::`R_{0}` units

    """
    return bias_v / current_a / 12900


def calculate_resistance_ohm(bias_v: Union[float, np.ndarray], current_a: Union[float, np.ndarray]) \
        -> Union[float, np.ndarray]:
    """
    Calculates the resistance in units of Ohm

    Parameters
    ----------
    bias_v : float or numpy.array
        measured bias in V
    current_a : float or numpy.array
        measured current_step in A

    Returns
    -------
    res : float or numpy.array
        resistance in units of Ohm

    """

    return bias_v / current_a


def convert_g0_to_ohm(conductance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts conductance in units of :math:`G_{0}` to resistance in Ohm

    Parameters
    ----------
    conductance: float or numpy.array
        conductance in units of the conductance quantum ::math::`G_{0}`

    Returns
    -------
    res : float or numpy.array
        resistance in units of Ohm

    """

    return (conductance / 12900) ** -1


def convert_pt_to_sec(value_pt: Union[int, np.ndarray], sample_rate: int) -> Union[float, np.ndarray]:
    """
    Converts points to seconds using the sample rate of the measurement

    Parameters
    ----------
    value_pt : float or np.ndarray
        the value in points to be converted
    sample_rate : int
        sampling rate of the measurement

    Returns
    -------
    res : float or np.ndarray
        value in seconds
    """
    return value_pt / sample_rate


def convert_sec_to_pt(value_sec: Union[float, np.ndarray], sample_rate: int) -> Union[float, np.ndarray]:
    """
    Converts seconds to points using the sample rate of the measurement

    Parameters
    ----------
    value_sec : float or numpy.array
        the value in seconds to be converted
    sample_rate: int
        sampling rate of the measurement
    Returns
    -------
    res: float or numpy.array
        value in points
    """

    return value_sec * sample_rate


def execute_and_measure_time(function_name, *args, **kwargs):
    """
    Executes function `function_name` while measuring the time of execution.
    After execution is finished the time is printed

    Parameters
    ----------
    function_name : the name of the function you want to execute
    args : arguments of the executed function
    kwargs : keyword arguments of the executed function

    Returns
    -------
        res : the result(s) of the executed function

    """

    start = time_ns()
    result = function_name(*args, **kwargs)

    print(f"{str(function_name).split(' ')[1]} finished in {(time_ns() - start) / 10 ** 9} s")

    return result


def convert_current_noise_to_conductance(current_noise: Union[float, np.ndarray], bias: Union[float, np.ndarray]) \
        -> Union[float, np.ndarray]:
    """
    Converts current noise to conductance noise

    Parameters
    ----------
    current_noise : flot or np.ndarray
        calculated current noise
    bias : float or np.ndarray
        applied bias voltage

    Returns
    -------
    res : float or np.ndarray
        calculated conductance noise
    """

    return current_noise * 12900 ** 2 / bias ** 2


def collect_traces(array_of_all, array_of_single):
    """
    Add array to a pre-existing array collection by vertically stacking them
    (like when appending lists, because in numpy append function flattens the resulting array)

    Parameters
    ----------
    array_of_all : np.ndarray
        collection of arrays, shape: (n, m), or empty: (0, )
    array_of_single : np.ndarray
        single array to be added to collection, shape: (m,)
    Returns
    -------
    res : np.ndarray,
        collection of array, shape: (n+1, m), or if `array_of_all` was empty: (1, m)

    """
    if array_of_all.shape[0] > 0:
        array_of_all = np.vstack((array_of_all, [array_of_single]))
    else:
        array_of_all = np.array([array_of_single])

    return array_of_all


def load_block_to_list(path_to_block: Union[Path, str], data_type=np.float32, divider=-133) -> List[np.ndarray]:
    """
    Loads a single block into a list of traces
    (used for older measurements, stored in binary files using .dat extension)

    Parameters
    ----------
    path_to_block : `~pathlib.Path` or str
        the path to the file containing the block
    data_type : Optional, type of the data, default: np.float32
    divider : int
        number used to distinguish individual trace data

    Returns
    -------
    res : List[np.ndarray]

    """

    loaded_data = np.fromfile(path_to_block, dtype=data_type)
    indices = np.append(-1, np.nonzero(loaded_data == divider)[0])

    list_of_traces = []
    for i in range(len(indices) - 1):
        list_of_traces.append(loaded_data[indices[i] + 1:indices[i + 1]])

    return list_of_traces


def read_waveform_params(waveform_file: str, mode: Union[str, Tuple[str, ...]],
                         waveform_path: Path = Path("D:/DAQprogramming/SPM-Controller/UserFiles/WaveForms")):
    """
    Reads data from the waveform file used in the measurement

    Parameters
    ----------
    waveform_file : str
        file containing all the parameters about the used waveform
    mode : str or Tuple[str, ...]
        Depends on the parameter you are interested in. Valid choices are: 'all', 'bias', 'start', 'length'
    waveform_path : `~pathlib.Path`
        Path to the waveform. Check whether the default value is correct on your machine

    Returns
    -------
    res : depends on the 'mode' parameter:
             mode = 'all': dictionary containing key-value pairs of all the parameters
             mode = 'bias': numpy array containing the amplitudes of the bias bias_signal in V
             mode = 'start': numpy array containing the starting points of each part in msec
             mode = 'length': numpy array containing the length of each part in msec

    """

    with open(waveform_path.joinpath(waveform_file)) as f:
        loaded_waveform = f.read()

    data = np.array([i.strip().split("=") for i in loaded_waveform.split("\n") if len(i.strip().split("=")) == 2])
    data.T[0] = [i.strip() for i in data.T[0]]

    waveform_dict = dict(zip(data.T[0], data.T[1]))

    for keys in waveform_dict:
        waveform_dict[keys] = float(waveform_dict[keys])
    if isinstance(mode, str):
        if mode == 'all':
            return waveform_dict
        elif mode == 'bias':
            match_string = r"Amplitude.*_\d*"
            return_array = np.array([])
        elif mode == 'start':
            match_string = r"Start.*_\d*"
            return_array = np.array([])
        elif mode == 'length':
            match_string = r"Length.*_\d*"
            return_array = np.array([])
        else:
            raise ValueError(f"Error: mode={mode} is not valid. Valid choices are: 'all', 'start', 'length'."
                             f"See manual for further information.")

        for keys in [re.match(match_string, i)[0] for i in waveform_dict.keys()
                     if re.match(match_string, i) is not None]:
            return_array = np.append(return_array, waveform_dict[keys])

        return return_array
    else:
        params_tuple = ()
        for i in mode:
            params_tuple = params_tuple + (read_waveform_params(waveform_file, i),)

        return params_tuple


def convert_to_block_and_trace_num(trace: Union[None, int, np.array]) \
        -> Tuple[Union[None, int, np.array], Union[None, int, np.array]]:
    """
    Determines the block and trace numbers from the entered overall trace number

    Parameters
    ----------
    trace : int, np.array
        overall trace number

    Returns
    -------
    block_num : int, np.array
        the number of the block the trace belongs to
    trace_num : int, np.array
        the number of the trace in the corresponding block

    """

    if trace is None:
        return None, None

    block_num = trace // 100 + 1  # math.ceil(trace / 100)
    trace_num = (trace - 1) % 100 + 1

    return block_num, trace_num


def convert_to_trace(block_num: int, trace_num: int) -> int:
    """
    Calculates the trace number from all traces using the block number and the trace number in the given block

    Parameters
    ----------
    block_num : int
        number of the block, the trace is in
    trace_num : int
        trace number of the trace in the given block

    Returns
    -------
    trace : int
        trace number in the measurement
    """

    return (block_num - 1) * 100 + trace_num


def get_name_from_path(file_path):
    """
    Extracts filename from a given `~pathlib.Path`

    Parameters
    ----------
    file_path: `~pathlib.Path`
        path pointing to the file

    Returns
    -------
    filename : str
        name of the file with extension

    Examples
    --------
    >>> path_to_file = Path('D:/BJ_Data/21_04_20/break_junction_34.h5')
    >>> get_name_from_path(path_to_file)
    'break_junction_34.h5'
    """
    return file_path.name


def get_num_from_name(trace_name: str) -> int:
    """
    Extracts number from a string

    Parameters
    ----------
    trace_name : str
        string containing number

    Returns
    -------
    res : int
        number in the string

    Examples
    --------
    >>> get_num_from_name('break_junction_34.h5')
    34
    >>> get_num_from_name('trace_5')
    5
    >>> get_num_from_name('trace_12312')
    12312
    """

    try:
        return int(re.search(r"\d+", trace_name).group(0))
    except AttributeError:
        return -1


def choose_files(file_array: np.ndarray, start_block: int, end_block: Optional[int] = None):
    """
    Select files with block numbers between `start_block` and `end_block`

    Parameters
    ----------
    file_array: np.ndarray of `~pathlib.Path` elements
        path to all available files
    start_block: int
        first block to be included
    end_block: int or None,
        last block to be included

    Returns
    -------
    chosen_files : np.ndarray
        array containing the path to the selected files
    """

    block_nums = np.array(list(map(get_num_from_name, map(get_name_from_path, file_array))))

    if end_block is None:
        end_block = max(block_nums)
        # print(end_block)

    mask = np.bitwise_and(block_nums >= start_block, block_nums <= end_block)
    masked_file_arr = file_array[mask]

    chosen_files = masked_file_arr[np.argsort(np.array(list(map(get_num_from_name,
                                                                map(get_name_from_path, masked_file_arr)))))]

    return chosen_files


def choose_traces(trace_array: np.ndarray, first: int, last: Optional[int] = None):
    """
    Select traces with trace numbers between `first` and `last`

    Parameters
    ----------
    trace_array : np.ndarray
        array of trace names
    first : int
        first trace number to be included
    last : Optional, int,
        last trace number to be included (note: this trace will be included!)
        if None, the last included trace will be the very last trace

    Returns
    -------
    chosen_traces: np.ndarray,
        array of chosen traces sorted by the trace number in increasing order
    """

    trace_nums = np.array(list(map(get_num_from_name, trace_array)))

    if last is None:
        last = max(trace_nums)
        # print(last)

    mask = np.bitwise_and(trace_nums >= first, trace_nums <= last),

    chosen_traces = sort_by_trace_num(trace_array[mask])  # sort list so numbers are in increasing order

    return chosen_traces


def sort_by_trace_num(trace_names_array: np.ndarray):
    """
    Returns tha array of trace names sorted by trace numbers instead of alphabetical order

    Parameters
    ----------
    trace_names_array: np.ndarray
        array of trace names to be sorted

    Returns
    -------
    res : np.ndarray
        sorted array
    """

    return trace_names_array[np.argsort(np.array(list(map(get_num_from_name, trace_names_array))))]


def get_exponent(num: float) -> int:
    return int(np.floor(np.log10(num)))


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """
    Calculates the moving average of array `x` with a window size `w`

    Parameters
    ----------
    x : np.ndarray
        array for the calculation
    w : size of the used window

    Returns
    -------
    res : np.ndarray
        moving average
    """

    return np.convolve(x, np.ones(w), 'valid') / w


def round_bias_steps(bias_steps: np.ndarray) -> np.ndarray:
    """
    Transforms bias to mV and rounds the to integer values (use this for labeling in images)

    Parameters
    ----------
    bias_steps : np.ndarray
         bias value at each step

    Returns
    -------
    res : np.ndarray
        rounded bias values at steps
    """

    return np.round(bias_steps * 1000)


def fit_func_lin(x, a, b):
    """
    Linear function for line fitting

    Parameters
    ----------
    x : np.ndarray
        variable
    a : float
        (-1) * the slope of the line
    b : float
        intercept

    Returns
    -------
    res : np.ndarray
        line
    """
    # x = log(f)
    return -1 * a * x + b


def gaussian_fun(x, a, b, c):
    return a * np.exp(-1 * (x-b)**2 / (2*c**2))


def find_level(arr: np.ndarray, level: float, x: Optional[np.ndarray] = None) -> Union[float, None]:
    """
    Find the point where an array crosses a pre-defined level

    Parameters
    ----------
    arr : np.ndarray
        input array
    level : float
        find where arr crosses this value
    x : Optional np.ndarray, default: None
        scaling array

    Returns
    -------
    res : int or float or None
        if x is None, it returns the cross-point based on indexing
        if x is provided the cross-point is returned in terms of x
        None if the level is not crossed
    """

    above_or_below = np.sign(arr - level)  # 1: point above level
    # -1: point below level
    # 0: point on level
    level_on_point = np.argwhere(above_or_below == 0)

    if level_on_point.shape[0]:
        print('level on point')
        if x is None:
            return level_on_point[0][0]
        else:
            return x[level_on_point[0][0]]

    crosses = np.argwhere(np.diff(above_or_below))
    if crosses.shape[0] > 0:
        crossed_after = crosses[0][0]  # level is crossed where the diff is nonzero
    else:
        return None

    cross_ratio = ((arr[crossed_after] - level) / (arr[crossed_after] - arr[crossed_after + 1]))

    if x is None:
        return crossed_after + cross_ratio
    else:
        return x[crossed_after] + cross_ratio * (x[crossed_after + 1] - x[crossed_after])


def calc_hist_1d_single(data: np.ndarray, xrange: Tuple[float, float] = (1e-6, 10),
                        xbins_num: int = 100, log_scale: bool = True, bin_mode: Optional[str] = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a single 1d histogram of the *data* array in the range *xrange*

    Parameters
    ----------
    data : np.ndarray
        the array to calculate the 1d histogram for
    xrange : Tuple[float, float]
        range to calculate the histogram in
    xbins_num : int, default value: 100
        number of bins of the histogram.
    log_scale : bool, default value: True
        histogram calculated with logarithmically equidistant bins
    bin_mode : Optional, str, default value: None
        only valid when log_scale is True
        mode of definition of the bins: 'decade': xbins_num means the number of bins in a decade
                                        'total': xbins_num means the total number of bins
    Returns
    -------
    x : np.ndarray
        bin locations
    single_hist1d : np.ndarray
        counts at each bin

    """
    if log_scale:
        # if log scale is true, xbins_num means the number of bins in a decade
        # otherwise it means the number of bins overall

        if bin_mode is None:
            bin_mode = 'decade'

        if bin_mode == 'decade':
            num_of_decs = (np.log10(xrange[1]) - np.log10(xrange[0]))
            xbins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]),
                                num=int(np.ceil(num_of_decs * xbins_num + 1)), base=10)
        elif bin_mode == 'total':
            xbins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), num=xbins_num + 1, base=10)
        else:
            raise ValueError(f"Unknown value {bin_mode} for parameter mode. Valid choices: 'decade', 'total'.")
    else:
        xbins = np.linspace(xrange[0], xrange[1], num=xbins_num)

    single_hist1d = np.histogram(data, bins=xbins)[0]
    bar_width = np.diff(xbins)
    x = xbins[:-1] + bar_width / 2

    return x, single_hist1d


def calc_hist_2d_single(x: np.ndarray, y: np.ndarray,
                        xrange: Tuple[float, float], log_scale_x: bool = False,
                        yrange: Tuple[float, float] = (1e-6, 10), log_scale_y: bool = True,
                        num_bins: Tuple[int, int] = (100, 100),
                        bin_mode_x: Optional[str] = None,
                        bin_mode_y: Optional[str] = None):
    """
    Calculates a single 2d histogram from arrays *x* and *y* in the range *xrange*

    Parameters
    ----------
    x : np.ndarray
        values of the horizontal axis
    y : np.ndarray
        values of the vertical axis
    align_at : float
        y value where each array is scaled together
    xrange : Tuple[float, float]
        horizontal range of bins
    log_scale_x : bool, default: False
        if True, the scaling of the horizontal axis is logarithmic
    yrange : Tuple[float, float]
        vertical range of bins
    log_scale_y : bool, default: True
        if True, the scaling of the vertical axis is logarithmic
    num_bins : Tuple[int, int]
        number of bins along the horizontal, and vertical axis,
        if the scaling is set to logarithmic for the corresponding axis, then it is the number of bins per decade
    bin_mode_x : Optional, str, default value: None
        only valid when log_scale_x is True
        mode of definition of the bins: 'decade': num_bins[0] means the number of bins in a decade
                                        'total': num_bins[0] means the total number of bins
        if log_scale_x is True and bin_mode_x is None, the value defaults to 'decade'
    bin_mode_y : Optional, str, default value: None
        only valid when log_scale_y is True
        mode of definition of the bins: 'decade': num_bins[1] means the number of bins in a decade
                                        'total': num_bins[1] means the total number of bins
        if log_scale_y is True and bin_mode_y is None, the value defaults to 'decade'

    Returns
    -------
    x_mesh : np.ndarray
        coordinate matrix containing the x coordinates of each bin
    y_mesh : np.ndarray
        coordinate matrix containing the y coordinates of each bin
    h : np.ndarray
        counts at each bin (shape: (number of bins along x, number of bins along y))

    """
    # x: piezo
    # y: conductance
    if x.shape[0] != y.shape[0]:
        raise MyException(f"Shape mismatch: shape of piezo {x.shape} does not match shape of"
                          f"conductance {y.shape}")

    if log_scale_x:
        if bin_mode_x is None:
            bin_mode_x = 'decade'

        if bin_mode_x == 'decade':
            num_of_decs = np.log10(xrange[1]) - np.log10(xrange[0])
            xbins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), num=int(num_bins[0] * num_of_decs) + 1,
                                base=10)
        elif bin_mode_x == 'total':
            xbins = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), num=int(num_bins[0]) + 1, base=10)
        else:
            raise ValueError(f"Unknown value {bin_mode_x} for parameter mode. Valid choices: 'decade', 'total'.")
    else:
        xbins = np.linspace(xrange[0], xrange[1], num=num_bins[0]+1)

    if log_scale_y:
        if bin_mode_y is None:
            bin_mode_y = 'decade'

        if bin_mode_y == 'decade':
            num_of_decs = np.log10(yrange[1]) - np.log10(yrange[0])
            ybins = np.logspace(np.log10(yrange[0]), np.log10(yrange[1]), num=int(num_bins[1] * num_of_decs) + 1,
                                base=10)
        elif bin_mode_y == 'total':
            ybins = np.logspace(np.log10(yrange[0]), np.log10(yrange[1]), num=int(num_bins[1]) + 1, base=10)
        else:
            raise ValueError(f"Unknown value {bin_mode_y} for parameter mode. Valid choices: 'decade', 'total'.")
    else:
        ybins = np.linspace(yrange[0], yrange[1], num=num_bins[1])

    # x_aligned, y_aligned = align_trace(x=x, y=y, xrange=xrange, align_value=align_at)
    #
    # h, xedges, yedges = np.histogram2d(x_aligned, y_aligned, bins=[xbins, ybins])
    h, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    x_mesh, y_mesh = np.meshgrid(xedges, yedges)

    return x_mesh, y_mesh, h.T


def interpolate(ind1: Tuple[float, float],
                ind2: Tuple[float, float],
                x: Optional[float] = None,
                y: Optional[float] = None) -> float:
    """

    Parameters
    ----------
    ind1 : indices of first point
    ind2 : indices of second point
    x : x value where to determine the y value
    y : y value for which to get the x value

    Returns
    -------
    y (if x is given) or x (if y is given)
    """

    if x is not None and y is None:
        return ind1[1] + (x-ind1[0])*(ind2[1]-ind1[1])/(ind2[0]-ind1[0])
    elif y is not None and x is None:
        return ind1[0] + (y-ind1[1])*(ind2[0]-ind1[0])/(ind2[1]-ind1[1])
    else:
        raise ValueError('Please enter either `x` or `y`, but not both.')


def align_trace(x: np.ndarray, y: np.ndarray, xrange: Tuple[float, float], align_value: float):
    """
    Aligns traces based on the values of array `y` at the value `align_value`

    Parameters
    ----------
    x : np.ndarray
        scaling array
    y : np.ndarray
        main array, traces are aligned at the `x` value, where the values of this array cross `align_value`
    xrange : Tuple[float, float]
        displayed range along the horizontal axis
    align_value : float
        value in `y` at which traces are aligned
    Returns
    -------

    """

    crosses = np.argwhere(np.diff(np.sign(y - align_value)))
    # print("hello", crosses.size)
    if crosses.size > 0:
        # print(crosses[0])
        cross_at = x[crosses[0]]
    else:
        # print info and skip trace
        raise MyException(f'this trace does not cross align value {align_value}')

    scale_min = max(cross_at + xrange[0], min(x))
    scale_max = min(cross_at + xrange[1], max(x))

    y_aligned = y[np.bitwise_and(scale_min < x, x < scale_max)]
    x_aligned = np.linspace(start=scale_min - cross_at, stop=scale_max - cross_at, num=y_aligned.shape[0],
                            endpoint=True).flatten()

    return x_aligned, y_aligned


def calc_covariance(x, y):
    """
    Calculates the covariance of two selected quantities
    :math:`\\text{Cov}(x,y)=\\left\\langle(x-\\left\\langle x\\right\\rangle)\\cdot(y-\\left\\langle y\\right\\rangle)\\right\\rangle`

    Parameters
    ----------
    x : np.ndarray
        contains one quantity for each trace (x.shape = (num_traces,1) )
    y : np.ndarray
        contains another quantity for each trace (y.shape = (num_traces,1) )

    Returns
    -------
    covariance of quantities *x* and *y*

    """
    x_avg = np.mean(x)
    y_avg = np.mean(y)

    return np.mean((x - x_avg) * (y - y_avg))


def calc_correlation(x: np.ndarray, y: np.ndarray):
    """Calculates the correlation of two selected quantities.

    .. math::

        \\frac{\\text{Cov}(x,y)}{\\sqrt{\\left\\langle (x -\\left\\langle x \\right\\rangle ) ^2 \\right\\rangle \\cdot \\left\\langle (y -\\left\\langle y \\right\\rangle )^2\\right\\rangle}}

    Parameters
    ----------
    x : np.ndarray
        contains one quantity for each trace (x.shape = (num_traces,1) )
    y : np.ndarray
        contains another quantity for each trace (y.shape = (num_traces,1) )

    Returns
    -------
    correlation of quantities *x* and *y*

    """

    x_avg = np.mean(x)
    y_avg = np.mean(y)

    xy_cov = calc_covariance(x, y)

    return xy_cov / np.sqrt(np.mean((x - x_avg) ** 2) * np.mean((y - y_avg) ** 2))


def largest_divisor(num: int, limit: Optional[int] = None):
    if limit is None:
        limit = num

    divisor = 0

    for i in range(2, limit):
        if num % i == 0:
            divisor = i

    return divisor


def load_scopedata(file_name, home_dir):
    data_file = home_dir.joinpath(file_name)
    data = np.fromfile(data_file, dtype=np.float32).reshape((-1, 3))
    current = data[:, 0]
    bias = data[:, 1]
    piezo = data[:, 2]

    return current, bias, piezo


def count_bool_groups(bool_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    bool_arr

    Returns
    -------

    Examples
    >>> bool_arr = [True, True, True, False, False, True, True, False, True]
    >>> count_bool_groups(bool_arr)
    (array([True, False, True, False, True]), array([3, 2, 2, 1, 1]))
    """

    bool_vals = []
    bool_counts = []

    for key, group in itertools.groupby(bool_arr):
        bool_vals.append(key)
        bool_counts.append(sum(1 for _ in group))

    return np.array(bool_vals), np.array(bool_counts)

    # return np.array([sum(1 for _ in group) for key, group in itertools.groupby(bool_arr) if key])


def even_ext(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Extends array with n elements, by mirroring values adjacent to the edges
    Parameters
    ----------
    arr: np.ndarray
        array to extend
    n: int
        number of elements to add on each side

    Returns
    -------

    Examples
    >>> even_ext(np.array([0, 1, 2, 3, 4, 5, 6, 7]), 2)
    array([2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5])
    """
    left_ext = arr[n:0:-1]
    right_ext = arr[-n-1:-1][::-1]

    return np.concatenate((left_ext, arr, right_ext))


def mov_avg(arr, win_size, step_size, avg_type: callable = np.mean):
    arr_ext = even_ext(arr, n=step_size)
    arr_mov_avg = np.zeros(arr_ext.shape[0] // step_size)
    for i in range(len(arr_mov_avg)):
        arr_mov_avg[i] = avg_type(arr_ext[i * step_size: (i * step_size) + win_size])

    return arr_mov_avg


def log_avg(arr: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculates the avg of arr on a log scale
    Parameters
    ----------
    arr: np.ndarray
        array of the values you want to calculate the average
    kwargs: additional keyword arguments for np.mean

    Returns
    -------
    np.ndarray
    """
    return 10**np.mean(np.log10(arr), **kwargs)


def custom_error(arr: np.ndarray, **kwargs) -> np.ndarray:
    """

    Parameters
    ----------
    arr: np.ndarray
        array of the values you want to calculate the average
    kwargs: additional keyword arguments for np.mean

    Returns
    -------
    np.ndarray

    How to use
    ----------
    arr * custom_error(arr) and arr/custom_error(arr) gives the +-1 std interval for arr
    """

    # 10**(log(arr) +- sqrt(avg(abs(log(arr)-avg(log(arr)))**2)))
    # 10**(log( arr*10**sqrt(avg(abs(log(arr)-avg(log(arr)))**2)))
    # arr */ 10**sqrt(avg(abs(log(arr)-avg(log(arr)))**2)

    return 10**np.sqrt(np.mean(abs(np.log10(arr) - np.mean(np.log10(arr), **kwargs)) ** 2, **kwargs))
