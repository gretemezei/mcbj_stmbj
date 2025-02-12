from multiprocessing import Process, Manager, current_process

from pca_and_ml import *
from mcbj import *


def f(d, pc_key, pc):
    pc.calc_group_histograms()
    print(current_process().name, current_process().pid)
    d[pc_key] = test(pc_key)
    # i = 0
    # for pc_key in pc_keys:
    #     i += 1
    #     d[pc_key] = test(i)


def test(pc_key):
    if pc_key == 'PC1':
        return j+1
    if pc_key == 'PC2':
        return j+2
    if pc_key == 'PC3':
        return j+3
    if pc_key == 'PC4':
        return j+4
    if pc_key == 'PC5':
        return j+5


if __name__ == '__main__':

    date = "21_05_21"
    # home_folder = Path(f"D:/BJ_Data/{date}")
    home_folder = Path(f'//DATACENTER/BreakJunction_group/BJ_Data/{date}')

    manager = Manager()

    pc_dict = manager.dict()
    pc_keys = ('PC1', 'PC2', 'PC3', 'PC4', 'PC5')

    # p = Process(target=f, args=(pc_dict, pc_keys))
    # p.start()
    # p.join()

    # load histogram
    hist = Histogram(folder=home_folder, load_from='hist_BPY_hold_lim_40.h5')
    # if the correlation was not calculated, calculate the correlation
    hist.calc_corr_hist_2d()
    # instantiate pc and calculate the principal components
    pc = PCA(hist=hist, num_of_pcs=5)
    pc.calc_principal_components(direction='pull')
    # calculate projections and select 20% of traces for each group
    pc.project_to_pcs()
    pc.calc_pc_hist_all(num_of_bins=100, hist_min=None, hist_max=None)
    pc.select_percentage(percentage=20, calc_histograms=True)

    proc = [Process(target=f, args=(pc_dict, pc_key, pc)) for pc_key in pc_keys]
    for p in proc:
        p.start()

    for p in proc:
        p.join()

    print(pc_dict)

