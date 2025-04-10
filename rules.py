import numpy as np
from utils import IndexedBidirectionalSet, TransitionMatrix, ProbabilityTransitionMatrix
from utils import norm, entropy, kmp_search
from collections import OrderedDict


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mask(grid, bool_mask, start_axs=(0,0), exception_value=0, copy=True, filter_mode=True):
    """ if copy == True then mask grid's copy rather than grid itself.
        if filter_mode=True then bool_mask's True location of grid will be filtered to exception_value. 
        actually, if filter_mode == False, means the reserve_mode
    """
    if copy:
        grid = grid.copy()
    if not filter_mode:
        bool_mask = ~bool_mask
    mask_rows, mask_cols = bool_mask.shape
    start_row, start_col =  start_axs
    target_region = grid[start_row:start_row + mask_rows, start_col:start_col + mask_cols]
    target_region[bool_mask] = exception_value
    return grid


class MetaRules:
    def __init__(self, null_value=0):
        self.null_value = null_value

class PatchKernel:
    def __init__(self, name='point', kernel_map=[[1]], center=(0,0), exception_value=-1):
        self.patch_name = name
        self.kernel = np.array(kernel_map).astype(bool)
        self.row_center = center[0]
        self.col_center = center[1]
        self.exception_value = exception_value
        self.patch_dict = IndexedBidirectionalSet()

    def patching(self, grid):
        k_rows, k_cols = self.kernel.shape
        pad_width = ((self.row_center, k_rows-self.row_center-1), (self.col_center, k_cols-self.col_center-1))
        pad_grid = np.pad(grid, pad_width=pad_width, mode='constant', constant_values=self.exception_value)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                yield mask(pad_grid[i:i+k_rows,j:j+k_cols], self.kernel, exception_value=self.exception_value, filter_mode=False)
    
    def patch_flatten(self, patch):
        flatten_str = ''
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                flatten_str += str(patch[i][j]) if patch[i][j] != self.exception_value else 'x'
        patch_id = self.patch_dict.add_item(flatten_str)
        return patch_id

class PatchSequence:
    def __init__(self, grid, kernel):
        self.kernel = kernel
        self.grid = grid
        self.patch_list = []
        self.state_list = []
        self.states = IndexedBidirectionalSet()
        self.state_count = {}
        for patch in self.kernel.patching(grid):
            self.patch_list.append(patch)
            state = self.kernel.patch_flatten(patch)
            self.state_list.append(state)    
            if state not in self.states:
                self.states.add_item(state)
                self.state_count[state] = 1
            else:
                self.state_count[state] += 1

    def analyse(self, analyzer):
        analyzer.fit(self)
        return analyzer

    def get_random_entropy(self):
        return np.log2(len(self.state_count))

    def get_info_entropy(self):
        return entropy(norm(np.array([v for _, v in self.state_count.items()])))

    def get_markov_entropy(self):
        bigram_count = {}
        for i in range(len(self.state_list)-1):
            bigram = (self.state_list[i], self.state_list[i+1])
            if bigram not in bigram_count:
                bigram_count[bigram] = 1
            else:
                bigram_count[bigram] += 1
        return np.log2(len(bigram_count)),entropy(norm(np.array([v for _, v in bigram_count.items()])))

    def get_entropy_rate(self, n=None):
        """Also known as Real-Entropy."""
        if n is None:
            n = len(self.state_list)
        gamma_sum = 0
        occured_list = []
        for i in range(0, n):
            hit = False
            for j in range(i+1, n+1):
                s = self.state_list[i:j]
                if kmp_search(occured_list, s) == []:
                    gamma_sum += len(s)
                    hit = True
                    break
            if not hit:
                gamma_sum += n - (i + 1) + 2
            occured_list.append(self.state_list[i])
        return n*np.log2(n)/gamma_sum  # Lempel-Ziv data compression as entropy_rate's estimation

class StaticFeaturesOnRule(MetaRules):
    def __init__(self, grid):
        super().__init__()
        self.origin_grid = grid
        self.grid = np.array(grid)
        self.patch_sequences = {}
    
    def add_patching_sequence(self, patch_kernel=None):
        if patch_kernel is None:
            patch_kernel = PatchKernel()
            self.patch_sequences[patch_kernel.patch_name] = PatchSequence(self.grid, patch_kernel)
        else:
            self.patch_sequences[patch_kernel.patch_name] = PatchSequence(self.grid, patch_kernel)
        return self.patch_sequences[patch_kernel.patch_name]
    
    def get_sequence(self, patch_name):
        return self.patch_sequences[patch_name]

class TransFeaturesOnRule(MetaRules):
    def __init__(self, g1, g2):
        super().__init__()
        self.grids = [g1, g2]
        self.np_grids = [np.array(g1, dtype=np.uint8), np.array(g2, dtype=np.uint8)]
        self.sizes = [self.np_grids[0].shape, self.np_grids[1].shape]
        self.scale_shape = [gcd(self.sizes[0][0], self.sizes[1][0]), gcd(self.sizes[0][1], self.sizes[1][1])]
        self.null_size = [np.count_nonzero(self.null_value == self.np_grids[0]), np.count_nonzero(self.null_value == self.np_grids[1])]
        self.densitys = [float(self.null_size[0])/self.np_grids[0].size, float(self.null_size[1])/self.np_grids[1].size]
        self.overlap_grids, self.overlap_axs, self.overlap_count, self.overlap_shape = self.cal_overlap(self.np_grids[0], self.np_grids[1], self.null_value)
        self.cal_appear()
        self.cal_disappear()

    def cal_appear(self):
        bool_mask = self.overlap_grids[0].astype(bool)
        o_axs = self.overlap_axs[0]
        grid_shape = self.np_grids[1].shape
        start_row = min(o_axs[0], grid_shape[0]) - self.overlap_shape[0]
        start_col = min(o_axs[1], grid_shape[1]) - self.overlap_shape[1]
        self.appear = mask(self.np_grids[1], ~bool_mask, (start_row, start_col), exception_value=0)

    def cal_disappear(self):
        bool_mask = self.overlap_grids[0].astype(bool)
        o_axs = self.overlap_axs[0]
        grid_shape = self.np_grids[0].shape
        start_row = min(o_axs[0], grid_shape[0]) - self.overlap_shape[0]
        start_col = min(o_axs[1], grid_shape[1]) - self.overlap_shape[1]
        self.disappear = mask(self.np_grids[0], ~bool_mask, (start_row, start_col), exception_value=0)

    @staticmethod
    def cal_overlap(g0, g1, null_value):
        row_min = min(g0.shape[0], g1.shape[0])
        row_max = max(g0.shape[0], g1.shape[0])
        column_min = min(g0.shape[1], g1.shape[1])
        column_max = max(g0.shape[1], g1.shape[1])
    
        overlap_sum = -1
        overlap_grid = []
        overlap_axis = []
        overlap_count = 0

        for i in range(row_min, row_max+1):
            for j in range(column_min, column_max+1):
                r_index_0 = min(g0.shape[0], i)
                c_index_0 = min(g0.shape[1], j)
                r_index_1 = min(g1.shape[0], i)
                c_index_1 = min(g1.shape[1], j)
                o0 = g0[r_index_0-row_min:r_index_0,c_index_0-column_min:c_index_0]
                o1 = g1[r_index_1-row_min:r_index_1,c_index_1-column_min:c_index_1]
                overlap = np.where((o0 == o1) & (o0 != null_value), 1, 0)
                o_sum = overlap.sum()
                if o_sum > overlap_sum:
                    overlap_sum = o_sum
                    overlap_grid = [overlap]
                    overlap_axis = [(i,j)]
                    overlap_count = 1
                elif o_sum == overlap_sum:
                    overlap_grid.append(overlap)
                    overlap_axis.append((i,j))
                    overlap_count += 1

        return overlap_grid, overlap_axis, overlap_count, [row_min, column_min]
    
class MarkovChainAnalyzer:
    def __init__(self):
        self.transition_counts = TransitionMatrix()

    def fit(self, sequence:PatchSequence):
        for i in range(len(sequence.state_list) - 1):
            self.transition_counts.add((sequence.state_list[i], sequence.state_list[i + 1]), 1)
        
        self.probability_matrix = ProbabilityTransitionMatrix(self.transition_counts)



    