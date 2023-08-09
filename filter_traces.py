from mcbj import *
import utils
from scipy.signal import find_peaks


def does_not_break(conductance, min_val=1e-5):
    #     return np.all(conductance > 1e-5)  # very strict
    #     return np.sum(conductance > 1e-5) > 0.998*len(conductance)  # still more strict than previously used ones
    return np.all(utils.moving_average(conductance, 50) > min_val)  # still more strict, but I choose this


def does_not_break_array(hold_trace: HoldTrace, min_val=1e-5):
    return (np.all(list(map(
                does_not_break, [hold_trace.hold_conductance_pull[
                                 hold_trace.bias_steps_ranges_pull[i][0]+50:hold_trace.bias_steps_ranges_pull[i][1]-50]
                                 for i in np.where(hold_trace.bias_steps > 0)[0]],
                [min_val, ]*np.where(hold_trace.bias_steps > 0)[0].shape[0]))),
            np.all(list(map(
                does_not_break, [hold_trace.hold_conductance_push[
                                 hold_trace.bias_steps_ranges_push[i][0]+50:hold_trace.bias_steps_ranges_push[i][1]-50]
                                 for i in np.where(hold_trace.bias_steps > 0)[0]],
                [min_val, ]*np.where(hold_trace.bias_steps > 0)[0].shape[0]))))


def check_plateau_length(trace_pair: TracePair, min_length: Union[int, Tuple[int, int]], **kwargs) -> Tuple[bool, bool]:
    """

    Parameters
    ----------
    trace_pair : TracePair
        object where you want to check that the plateau length is long enough. This function checks the attribute
        `plateau_length_pull` and `plateau_length_push`, so be careful, that these are either previously calculated
        for the plateau of interest or provide additional keyword arguments for the TracePair.calc_plateau_length
        method.
    min_length : Tuple
        minimum required plateau length in points
    kwargs : additional keyword arguments for the method trace_pair.calc_plateau_length

    Returns
    -------
    (pull, push) : Tuple[bool, bool]
        whether the plateau of the pull and push trace is long enough

    See Also
    TracePair.calc_plateau_length
    """

    pull = False
    push = False

    if len(kwargs) > 0:
        trace_pair.calc_plateau_length(**kwargs)

    if isinstance(min_length, tuple):
        if trace_pair.plateau_length_pull > min_length[0]:
            pull = True
        if trace_pair.plateau_length_push > min_length[1]:
            push = True
    else:
        if trace_pair.plateau_length_pull > min_length:
            pull = True
        if trace_pair.plateau_length_push > min_length:
            push = True

    return pull, push


def conductances_close(hold_trace: HoldTrace, max_ratio: float, plateaus: Optional[Tuple[int, ...]] = None):
    """

    Parameters
    ----------
    hold_trace
    max_ratio : float

    plateaus : Tuple[int, ...]
        index of the bias plateaus where we compare the avg conductances

    Returns
    -------

    """
    conductances_pull = []
    conductances_push = []

    if plateaus is None:

        conductance_pull = hold_trace.hold_conductance_pull[hold_trace.bias_steps_ranges_pull[0][0] + 50:
                                                            hold_trace.bias_steps_ranges_pull[0][1] - 50]
        pull = np.all(abs(np.diff(conductance_pull) / conductance_pull[-1]) < max_ratio)

        conductance_push = hold_trace.hold_conductance_push[hold_trace.bias_steps_ranges_push[0][0] + 50:
                                                            hold_trace.bias_steps_ranges_push[0][1] - 50]
        push = np.all(abs(np.diff(conductance_push) / conductance_push[-1]) < max_ratio)

    else:
        for i in plateaus:
            conductances_pull.append(np.mean(hold_trace.hold_conductance_pull[
                                             hold_trace.bias_steps_ranges_pull[i][0] + 50:
                                             hold_trace.bias_steps_ranges_pull[i][1] - 50]))

            conductances_push.append(np.mean(hold_trace.hold_conductance_push[
                                             hold_trace.bias_steps_ranges_push[i][0] + 50:
                                             hold_trace.bias_steps_ranges_push[i][1] - 50]))

        conductances_pull = np.array(conductances_pull)
        conductances_push = np.array(conductances_push)

        pull = np.all(abs(np.diff(conductances_pull) / conductances_pull[-1]) < max_ratio)
        push = np.all(abs(np.diff(conductances_push) / conductances_push[-1]) < max_ratio)

    return pull, push


def iv_difference(hold_trace: HoldTrace, direction: str = 'pull', smoothing: int = 1):
    if direction == 'pull':
        bias = hold_trace.iv_bias_pull
        current = hold_trace.iv_current_pull
    elif direction == 'push':
        bias = hold_trace.iv_bias_push
        current = hold_trace.iv_current_push
    else:
        raise ValueError(f'Unknown value {direction} for variable direction. Valid choices are: "pull", "push".')

    peaks, _ = find_peaks(abs(bias), height=0)
    segments = np.concatenate((np.array([-1]), peaks, np.array([-1])))

    # cut parts of the I(V)

    i1 = current[segments[0] + 1: segments[1]]
    u1 = bias[segments[0] + 1: segments[1]]
    i1 = i1[u1 > 0.05]
    u1 = u1[u1 > 0.05]
    i2 = current[segments[1] + 1: segments[2]]
    u2 = bias[segments[1] + 1: segments[2]]
    i2 = i2[u2 > 0.05]
    u2 = u2[u2 > 0.05]

    i3 = current[segments[1] + 1: segments[2]]
    u3 = bias[segments[1] + 1: segments[2]]
    i3 = i3[u3 < -0.05]
    u3 = u3[u3 < -0.05]
    i4 = current[segments[2] + 1: segments[3]]
    u4 = bias[segments[2] + 1: segments[3]]
    i4 = i4[u4 < -0.05]
    u4 = u4[u4 < -0.05]

    # make sure positive and negative parts have the same length

    if len(i1) != len(i2):
        if len(i1) < len(i2):
            i2 = i2[:len(i1)]
            # u2 = u2[:len(i1)]
        else:
            i1 = i1[-len(i2):]
            # u1 = u1[-len(i2):]
    if len(i3) != len(i4):
        if len(i3) < len(i4):
            i4 = i4[:len(i3)]
            # u4 = u4[:len(i3)]
        else:
            i3 = i3[-len(i4):]
            # u3 = u3[-len(i4):]

    # smooth

    i1 = utils.moving_average(i1, smoothing)
    i2 = utils.moving_average(i2, smoothing)
    i3 = utils.moving_average(i3, smoothing)
    i4 = utils.moving_average(i4, smoothing)

    return max(np.sum(np.sqrt(abs(i1 ** 2 - i2[::-1] ** 2)) / abs(i1) / len(i1)),
               np.sum(np.sqrt(abs(i3 ** 2 - i4[::-1] ** 2)) / abs(i3) / len(i3)))


def ivs_close(hold_trace: HoldTrace, max_diff: float, smoothing: int = 100):
    pull = iv_difference(hold_trace, direction='pull', smoothing=smoothing) < max_diff
    push = iv_difference(hold_trace, direction='push', smoothing=smoothing) < max_diff

    return pull, push


def measure_relaxation(conductance: np.array, conductance_limit: float = 2.0) -> (float, float, float):
    scaling_val = np.mean(conductance[-1*int(len(conductance) / 2):])
    scaled_conductance = conductance / scaling_val

    #     relax_ends_at = np.nonzero(scaled_conductance < 2)[0][0]  # first point that is smaller than 2
    try:
        relax_ends_at = np.nonzero(scaled_conductance > conductance_limit)[0][-1]  # last point that is larger than 2
    except IndexError:
        relax_ends_at = 0
    relax_time = relax_ends_at / 50_000
    relax_amount = conductance[0] - conductance[relax_ends_at]

    return relax_ends_at, relax_time, relax_amount


def is_stabil(conductance):
    relax_ends_at, _, _ = measure_relaxation(conductance)

    return relax_ends_at < len(conductance) / 3


def filter_bj(folder: Path,
              filter_condition: callable,
              traces: Optional[Union[np.array, List[int]]] = None,
              start_trace: int = 1,
              end_trace: Optional[int] = None,
              **kwargs):
    """

    Parameters
    ----------
    folder : Path
        path to reach break junction files
    traces : array_like, optional
        trace numbers of the traces you want to filter, useful if you have multiple filter conditions to consider
    start_trace : int
        number of the first trace to be filtered
    end_trace : int
        number of the last trace to be filtered
    filter_condition : callable
        function that defines the filter condition. It can be any function but has to return 2 bool values: one for the
        pull and one for the push trace, which tell us whether the pull/push traces meet the defined condition for
        selection
    kwargs : keyword arguments for the filter_condition function

    Returns
    -------

    See Also
    --------

    does_not_break
    check_plateau_length

    """

    start_block, _ = utils.convert_to_block_and_trace_num(start_trace)
    if end_trace is None:
        end_block = None
    else:
        end_block, _ = utils.convert_to_block_and_trace_num(end_trace)

    bj_files = utils.choose_files(np.array(list(folder.glob(r'break_junction_*.h5'))),
                                  start_block=start_block, end_block=end_block)

    met_condition_pull = []
    met_condition_push = []

    if traces is None:
        for file in tqdm(bj_files, desc="Collecting traces"):
            with h5py.File(file, "r") as input_file:
                for trace in utils.choose_traces(trace_array=np.array(list(input_file['pull'].keys())),
                                                 first=start_trace, last=end_trace):
                    trace_pair = TracePair(trace, load_from=folder)

                    bool_pull, bool_push = filter_condition(trace_pair, **kwargs)

                    if bool_pull:
                        met_condition_pull.append(trace_pair.trace_num)
                    if bool_push:
                        met_condition_push.append(trace_pair.trace_num)
    else:
        for trace in tqdm(traces):
            trace_pair = TracePair(trace, load_from=folder)

            bool_pull, bool_push = filter_condition(trace_pair, **kwargs)

            if bool_pull:
                met_condition_pull.append(trace_pair.trace_num)
            if bool_push:
                met_condition_push.append(trace_pair.trace_num)

    return np.array(met_condition_pull), np.array(met_condition_push)


def filter_hold(folder, filter_condition: callable,
                traces: Optional[Union[np.array, List[int]]] = None,
                start_trace: int = 0, end_trace: Optional[int] = None,
                bias_offset: float = 0, r_serial_ohm: int = 99_900,
                min_height=None, min_step_len=None, iv: Optional[int] = None, **kwargs):
    """

    Parameters
    ----------
    folder
    filter_condition : callable
        a function that returns True if the trace fulfills the condition criteria
        Important! This function has to return a bool for both the pull & push traces, in this order!
    traces : array-like, list of traces to be filtered, default: None
        if None, use start_trace and end_trace to generate the list of traces
    start_trace : int
        trace number of the first trace of the list you want to filter
    end_trace : Optional[int], default: None
        trace number of the first trace of the list you want to filter.
        If None, get the very last trace as trace number
    bias_offset : float
        bias offset set for the measurement
    r_serial_ohm : float
        resistance of the resistor in series with the sample
    min_step_len : int
        minimum step length of a single bias plateau in points
    iv  : Optional[int], default None
        If None, there was no I(V) measurement, and treat trace accordingly. If an integer, the I(V) measurement
        followed the bias plateau with the index iv
    kwargs :
        Additional keyword arguments for `filter_condition`

    Returns
    -------

    """
    # start_block, _ = utils.convert_to_block_and_trace_num(start_trace)
    # if end_trace is None:
    #     end_block = None
    # else:
    #     end_block, _ = utils.convert_to_block_and_trace_num(end_trace)
    #
    # hold_files = utils.choose_files(np.array(list(folder.glob(r'hold_data_*.h5'))),
    #                                 start_block=start_block, end_block=end_block)

    met_condition_pull = []
    met_condition_push = []

    # for file in tqdm(hold_files, desc="Filtering traces"):
    #     with h5py.File(file, "r") as input_file:

    if traces is None:
        traces = np.arange(start_trace, end_trace+1)

    # for trace in utils.choose_traces(trace_array=np.array(list(input_file['pull'].keys())),
    #                                  first=start_trace, last=end_trace):

    for trace in tqdm(traces):

        try:
            hold_trace = HoldTrace(trace, load_from=folder, bias_offset=bias_offset,
                                   r_serial_ohm=r_serial_ohm, min_step_len=min_step_len, min_height=min_height, iv=iv)

            bool_pull, bool_push = filter_condition(hold_trace, **kwargs)

            if bool_pull:
                met_condition_pull.append(hold_trace.trace_num)
            if bool_push:
                met_condition_push.append(hold_trace.trace_num)
        except (MeasurementOverflow, MeasurementNotComplete):
            continue

    return np.array(met_condition_pull), np.array(met_condition_push)
