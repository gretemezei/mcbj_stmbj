from pathlib import Path
from typing import Union, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps, cycler, rcParams, gridspec, ticker
from tqdm.notebook import tqdm

# custom libraries
from mcbj import TracePair, Histogram
import utils


class PCA:
    def __init__(self, hist: Histogram, num_of_pcs: int = 5, color_map=None):
        """

        Parameters
        ----------
        hist : Histogram
            Histogram instance where the correlation is calculated
        num_of_pcs : int, default: 5
            number of principal components to calculate and analyze
        """

        self.hist = hist
        self.num_of_pcs = num_of_pcs
        self.pc_keys = tuple(f'PC{i+1}' for i in range(self.num_of_pcs))

        if color_map is None:
            self.color_map = colormaps['tab10'](np.linspace(0, 1, 10))  # max 10 different colors
        else:
            self.color_map = color_map

        # PCs and eigenvalues
        self.principal_components = {}
        self.pc_vals = {}
        # Projection to PCs
        self.dot_prod = {}

        # PC histogram
        self.num_of_bins = None
        self.hist_min = None  # take the minimum value of the dot product
        self.hist_max = None  # take the maximum value of the dot product
        self.pc_hist_bins = {}
        self.pc_hist_1d = {}

        # Selected traces, their projections to the PCs, the histograms of the dot products
        self.traces_group1 = {}
        self.dot_prod_group1 = {}
        self.pc_hist_bins_group1 = {}
        self.pc_hist_1d_group1 = {}
        self.traces_group2 = {}
        self.dot_prod_group2 = {}
        self.pc_hist_bins_group2 = {}
        self.pc_hist_1d_group2 = {}

        # Histograms for traces selected via PCs
        self.hist_group1 = {}
        self.hist_group2 = {}

    def load_principal_components(self, source_file: Union[Path, str], source_type: str = 'igor'):
        if source_type == 'igor':
            loaded_pc = np.loadtxt(source_file, skiprows=1)
        else:
            raise NotImplementedError('Source type only accepts "igor" as input, other types are not implemented yet.')

        self.num_of_pcs = loaded_pc.shape[1]
        self.pc_keys = tuple(f'PC{i+1}' for i in range(self.num_of_pcs))

        for pc_key in self.pc_keys:
            self.principal_components[pc_key] = loaded_pc[:, self.pc_keys.index(pc_key)]

    def calc_principal_components(self, direction: str = 'pull'):
        """

        Parameters
        ----------
        direction : str, default: 'pull', valid: 'pull', 'push'
            for which direction you want the principal components

        Returns
        -------

        """
        if direction == 'pull':
            corr_2d = self.hist.corr_2d_pull
        elif direction == 'push':
            corr_2d = self.hist.corr_2d_push
        else:
            raise ValueError(f"Unknown value {direction} for parameter `direction`. Valid choices: 'pull' or 'push'.")

        eig_vals, eig_vecs = np.linalg.eigh(corr_2d)
        self.pc_vals = np.zeros(self.num_of_pcs)

        for i in range(self.num_of_pcs):
            self.principal_components[f'PC{i+1}'] = eig_vecs[:, ::-1][:, i]

        self.pc_vals = eig_vals[::-1][:self.num_of_pcs]/sum(eig_vals)

    def plot_pcs(self, dpi: int = 600, ax=None):
        """
        Plot the calculated principal components
        Parameters
        ----------
        dpi
        ax

        Returns
        -------

        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=utils.cm2inch(10, 5), dpi=dpi)

        ax.set_prop_cycle = cycler('color', self.color_map)  # set the colors for the PCs

        for pc_key in self.pc_keys:
            ax.plot(self.hist.hist_1d_bins, self.principal_components[pc_key], label=pc_key, lw=0.6)
            ax.set_xscale('log')
            ax.legend(loc='upper right', ncol=2, fontsize='xx-small')

        ax.plot(self.hist.hist_1d_bins, np.zeros_like(self.hist.hist_1d_bins),
                label=f'zero line', lw=0.6, ls='--', c='grey')

        return ax

    def project_to_pcs(self):
        """
        Projection to the selected principal component
        Returns
        -------

        """
        for pc_key in tqdm(self.pc_keys, desc=f'Calculating the projections to PCs.'):
            dot_prod = np.zeros(self.hist.temporal_hist_pull.shape[0])
            for i, single_hist in enumerate(self.hist.temporal_hist_pull):
                dot_prod[i] = np.dot(self.principal_components[pc_key], single_hist)
            self.dot_prod[pc_key] = dot_prod

    def calc_pc_hist_single(self, pc_key, num_of_bins: int = 100,
                            hist_min: Optional[float] = None,
                            hist_max: Optional[float] = None):

        if hist_min is None:
            hist_min = min(self.dot_prod[pc_key])
        if hist_max is None:
            hist_max = max(self.dot_prod[pc_key])

        self.pc_hist_bins[pc_key], self.pc_hist_1d[pc_key] = \
            utils.calc_hist_1d_single(data=self.dot_prod[pc_key], xrange=(hist_min, hist_max),
                                      xbins_num=num_of_bins, log_scale=False, bin_mode='total')

    def calc_pc_hist_all(self, num_of_bins: Union[int, Tuple[int, ...]] = 100,
                         hist_min: Optional[Union[float, Tuple[float, ...]]] = None,
                         hist_max: Optional[Union[float, Tuple[float, ...]]] = None):

        if isinstance(num_of_bins, int):
            self.num_of_bins = (num_of_bins,)*self.num_of_pcs
        elif isinstance(num_of_bins, tuple):
            if len(num_of_bins) == self.num_of_pcs:
                self.num_of_bins = num_of_bins
            else:
                raise ValueError(f'Number of parameters wrong! The number of parameters must be equal to the '
                                 f'number of principal components. You entered {len(num_of_bins)} numbers, '
                                 f'but {self.num_of_pcs} are necessary.')
        else:
            raise ValueError(f'Invalid value {num_of_bins} for parameter `hist_min`. See documentation...')

        if hist_min is None:
            self.hist_min = tuple(map(lambda x: min(x), self.dot_prod.values()))
        elif isinstance(hist_min, int):
            self.hist_min = (hist_min,)*self.num_of_pcs
        elif isinstance(hist_min, tuple):
            if len(hist_min) == self.num_of_pcs:
                self.hist_min = hist_min
            else:
                raise ValueError(f'Number of parameters wrong! The number of parameters must be equal to the '
                                 f'number of principal components. You entered {len(hist_min)} numbers, '
                                 f'but {self.num_of_pcs} are necessary.')
        else:
            raise ValueError(f'Invalid value {hist_min} for parameter `hist_min`. See documentation...')

        if hist_max is None:
            self.hist_max = tuple(map(lambda x: max(x), self.dot_prod.values()))
        elif isinstance(hist_max, int):
            self.hist_max = (hist_max,)*self.num_of_pcs
        elif isinstance(hist_max, tuple):
            if len(hist_max) == self.num_of_pcs:
                self.hist_max = hist_max
            else:
                raise ValueError(f'Number of parameters wrong! The number of parameters must be equal to the '
                                 f'number of principal components. You entered {len(hist_max)} numbers, '
                                 f'but {self.num_of_pcs} are necessary.')
        else:
            raise ValueError(f'Invalid value {hist_max} for parameter `hist_max`. See documentation...')

        for pc_key in tqdm(self.pc_keys):
            self.calc_pc_hist_single(pc_key=pc_key,
                                     num_of_bins=self.num_of_bins[self.pc_keys.index(pc_key)],
                                     hist_min=self.hist_min[self.pc_keys.index(pc_key)],
                                     hist_max=self.hist_max[self.pc_keys.index(pc_key)])

    def select_percentage(self, percentage: int = 20, calc_histograms: bool = True):
        """
        Select `percentage` percent of traces from the left-most and right-most of the PCA histogram
        Parameters
        ----------
        percentage : the amount of traces to select from each side, stored in group1 and group2, respectively

        calc_histograms : bool, default: True

        Returns
        -------

        """
        cut_off_index = int(percentage / 100 * self.hist.traces.shape[0])
        for pc_key in self.pc_keys:
            group1_indxs = np.argsort(self.dot_prod[pc_key])[:cut_off_index]
            self.traces_group1[pc_key] = self.hist.traces[group1_indxs]
            self.dot_prod_group1[pc_key] = self.dot_prod[pc_key][group1_indxs]
            group2_indxs = np.argsort(self.dot_prod[pc_key])[-1*cut_off_index:]
            self.traces_group2[pc_key] = self.hist.traces[group2_indxs]
            self.dot_prod_group2[pc_key] = self.dot_prod[pc_key][group2_indxs]

            if calc_histograms:
                self.pc_hist_bins_group1[pc_key], self.pc_hist_1d_group1[pc_key] = \
                    utils.calc_hist_1d_single(data=self.dot_prod_group1[pc_key],
                                              xrange=(self.hist_min[self.pc_keys.index(pc_key)],
                                                      self.hist_max[self.pc_keys.index(pc_key)]),
                                              xbins_num=self.num_of_bins[self.pc_keys.index(pc_key)],
                                              log_scale=False, bin_mode='total')

                self.pc_hist_bins_group2[pc_key], self.pc_hist_1d_group2[pc_key] = \
                    utils.calc_hist_1d_single(data=self.dot_prod_group2[pc_key],
                                              xrange=(self.hist_min[self.pc_keys.index(pc_key)],
                                                      self.hist_max[self.pc_keys.index(pc_key)]),
                                              xbins_num=self.num_of_bins[self.pc_keys.index(pc_key)],
                                              log_scale=False, bin_mode='total')

    def plot_pc_hist(self, pc_key: str, plot_groups: bool = False, dpi: int = 600, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=utils.cm2inch(10, 5), dpi=dpi)

        ax.bar(x=self.pc_hist_bins[pc_key],
               height=self.pc_hist_1d[pc_key],
               width=np.mean(np.diff(self.pc_hist_bins[pc_key])),
               align='edge', color='white', edgecolor='k')
        if plot_groups:
            ax.bar(x=self.pc_hist_bins_group1[pc_key],
                   height=self.pc_hist_1d_group1[pc_key],
                   width=np.mean(np.diff(self.pc_hist_bins_group1[pc_key])),
                   align='edge', color='b', edgecolor='b', lw=0)
            ax.bar(x=self.pc_hist_bins_group2[pc_key],
                   height=self.pc_hist_1d_group2[pc_key],
                   width=np.mean(np.diff(self.pc_hist_bins_group1[pc_key])),
                   align='edge', color='r', edgecolor='r', lw=0)

        return ax

    def calc_group_histograms(self):
        for pc_key in self.pc_keys:
            hist_group1 = Histogram(folder=self.hist.folder, traces=self.traces_group1[pc_key],
                                    conductance_range=self.hist.conductance_range,
                                    conductance_bins_num=self.hist.conductance_bins_num,
                                    conductance_log_scale=self.hist.conductance_log_scale,
                                    conductance_bins_mode=self.hist.conductance_bins_mode)
            print(f'Calculating 1D and 2D histograms of {pc_key} group1.')
            hist_group1.calc_hist_1d()
            hist_group1.calc_hist_2d(align_at=self.hist.align_at,
                                     range_pull=(min(self.hist.hist_2d_xmesh_pull.flatten()),
                                                 max(self.hist.hist_2d_xmesh_pull.flatten())),
                                     range_push=(min(self.hist.hist_2d_xmesh_push.flatten()),
                                                 max(self.hist.hist_2d_xmesh_push.flatten())))

            hist_group2 = Histogram(folder=self.hist.folder, traces=self.traces_group2[pc_key],
                                    conductance_range=self.hist.conductance_range,
                                    conductance_bins_num=self.hist.conductance_bins_num,
                                    conductance_log_scale=self.hist.conductance_log_scale,
                                    conductance_bins_mode=self.hist.conductance_bins_mode)
            print(f'Calculating 1D and 2D histograms of {pc_key} group2.')
            hist_group2.calc_hist_1d()
            hist_group2.calc_hist_2d(align_at=self.hist.align_at,
                                     range_pull=(min(self.hist.hist_2d_xmesh_pull.flatten()),
                                                 max(self.hist.hist_2d_xmesh_pull.flatten())),
                                     range_push=(min(self.hist.hist_2d_xmesh_push.flatten()),
                                                 max(self.hist.hist_2d_xmesh_push.flatten())))

            self.hist_group1[pc_key] = hist_group1
            self.hist_group2[pc_key] = hist_group2

    def plot_group_histograms_1d(self, ylims: Tuple[float, float] = (0, 50), group1_color='b', group2_color='r'):
        fig, ax = plt.subplots(nrows=self.num_of_pcs, dpi=300, figsize=utils.cm2inch(10, 4 * self.num_of_pcs))

        for i, pc_key in enumerate(self.pc_keys):
            ax[i].plot(self.hist.hist_1d_bins, self.hist.hist_1d_pull, label='total', c='cornflowerblue', lw=1)
            ax[i].fill_between(self.hist.hist_1d_bins, np.zeros_like(self.hist.hist_1d_pull), self.hist.hist_1d_pull,
                               alpha=0.5, color='cornflowerblue')
            ax[i].plot(self.hist_group1[pc_key].hist_1d_bins,
                       self.hist_group1[pc_key].hist_1d_pull, label='group1', c=group1_color,
                       lw=1)
            ax[i].plot(self.hist_group1[pc_key].hist_1d_bins,
                       self.hist_group2[pc_key].hist_1d_pull, label='group2', c=group2_color,
                       lw=1)
            ax[i].set_xscale('log')
            ax[i].set_xlim(self.hist_group1[pc_key].conductance_range)
            ax[i].set_ylim(ylims)
            ax[i].legend(fontsize='xx-small')
            ax[i].set_title(pc_key, fontsize='xx-small', pad=1)

        return ax

    def plot_group_histograms_2d(self, dpi=300, vmax: Optional[Union[float, Tuple[float, float]]] = None):
        fig, ax = plt.subplots(nrows=self.num_of_pcs, ncols=2, dpi=300, figsize=utils.cm2inch(10, 5 * self.num_of_pcs))
        if vmax is None or isinstance(vmax, (int, float)):
            vmax = (vmax, vmax)
        elif isinstance(vmax, tuple):
            vmax = vmax
        else:
            raise ValueError(f'Invalid value {vmax} for parameter `vmax`.')

        for i, pc_key in enumerate(self.pc_keys):
            ax[i, 0] = self.hist_group1[pc_key].plot_hist_2d_one(direction='pull', ax=ax[i, 0], dpi=dpi, vmax=vmax[0])
            ax[i, 1] = self.hist_group2[pc_key].plot_hist_2d_one(direction='pull', ax=ax[i, 1], dpi=dpi, vmax=vmax[1])
            ax[i, 0].set_title(f'{pc_key} group 1', fontsize='xx-small', pad=1)
            ax[i, 1].set_title(f'{pc_key} group 2', fontsize='xx-small', pad=1)

        return ax


# def compare_principal_components(pc1: PCA, pc2: PCA, labels=('pc1', 'pc2'), fig_size=None, dpi=300):
#     use_cmap = colormaps['tab20'](np.linspace(0, 1, 20))  # 20 different color, to compare 2 sets of 10 PCs
#     if pc1.num_of_pcs == pc2.num_of_pcs:
#         if fig_size is None:
#             fig_size = utils.cm2inch(10, pc1.num_of_pcs * 4)
#         fig, ax = plt.subplots(nrows=pc1.num_of_pcs, figsize=fig_size, dpi=dpi)
#
#         for i, pc_key in enumerate(pc1.pc_keys):
#             ax[i].set_prop_cycle, cycler('color', use_cmap)
#             ax[i].plot(pc1.hist.hist_1d_bins, pc1.principal_components[pc_key], label=labels[0])
#             ax[i].plot(pc2.hist.hist_1d_bins, pc2.principal_components[pc_key], label=labels[1])
#             ax[i].set_xscale('log')
#             ax[i].plot(pc1.hist.hist_1d_bins, np.zeros_like(pc1.hist.hist_1d_bins),
#                        label=f'zero line', lw=0.6, ls='--', c='grey')
#             ax[i].set_title(pc_key, fontsize='xx-small', pad=1)
#             ax[i].legend(fontsize='xx-small')
#
#         return ax
#
#     else:
#         raise IndexError(f"The PCA instances entered don't have the same number of principal components calculated: "
#                          f"pc1 has {pc1.num_of_pcs} PCs and pc2 has {pc2.num_of_pcs}. Please check your inputs.")


def compare_principal_components(pcs: Tuple[PCA, ...], labels=None, fig_size=None, dpi=300, **kwargs):
    use_cmap = colormaps['tab20'](np.linspace(0, 1, 20))  # 20 different color, to compare 2 sets of 10 PCs
    number_of_pcs_each = list(map(lambda x: x.num_of_pcs, pcs))
    if np.unique(number_of_pcs_each).shape[0] == 1:
        if labels is None:
            labels = tuple(f'pc{i+1}' for i in range(len(pcs)))
        if fig_size is None:
            fig_size = utils.cm2inch(10, pcs[0].num_of_pcs * 4)
        fig, ax = plt.subplots(nrows=pcs[0].num_of_pcs, figsize=fig_size, dpi=dpi)

        for i, pc_key in enumerate(pcs[0].pc_keys):
            ax[i].set_prop_cycle, cycler('color', use_cmap)
            for pc in pcs:
                ax[i].plot(pc.hist.hist_1d_bins, pc.principal_components[pc_key], label=labels[pcs.index(pc)], **kwargs)
                ax[i].set_ylim((-1.1 * max(abs(pc.principal_components[pc_key])),
                                1.1 * max(abs(pc.principal_components[pc_key]))))
            # ax[i].plot(pcs[0].hist.hist_1d_bins, np.zeros_like(pcs[0].hist.hist_1d_bins),
            #            label=f'zero line', lw=0.6, ls='--', c='grey')
            ax[i].axhline(0, ls='--', lw=0.6, c='grey')
            ax[i].set_xscale('log')
            ax[i].set_title(pc_key, fontsize='xx-small', pad=1)
            ax[i].legend(fontsize='xx-small')

        return ax

    else:
        raise IndexError(
            f"The PCA instances entered don't have the same number of principal components calculated: "
            f"the entered PCA instances have {list(map(lambda x: x.num_of_pcs, pcs))} PCs, respectively. "
            f"Please check your inputs.")
