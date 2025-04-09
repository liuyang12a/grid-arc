import numpy as np
from utils import IndexedBidirectionalSet, TransitionMatrix, norm, entropy, kmp_search
from collections import OrderedDict


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mask(grid, bool_mask, start_axs=(0,0), exception_value=0, copy=True):
    if copy:
        grid = grid.copy()
    mask_rows, mask_cols = bool_mask.shape
    start_row, start_col =  start_axs
    target_region = grid[start_row:start_row + mask_rows, start_col:start_col + mask_cols]
    target_region[bool_mask] = exception_value
    return grid


class MetaRules:
    def __init__(self, null_value=0):
        self.null_value = null_value

class PatchKernel:
    def __init__(self, name, kernel, center, exception_value=-1):
        self.patch_name = name
        self.kernel = kernel
        self.row_center = center[0]
        self.col_center = center[1]
        self.exception_value = exception_value
        self.patch_dict = IndexedBidirectionalSet()

    def patching(self, grid):
        k_rows, k_cols = self.kernel.shape
        pad_width = ((self.row_center, k_rows-self.row_center), (self.col_center, k_cols-self.col_center))
        pad_grid = np.pad(grid, pad_width=pad_width, mode='constant', constant_values=self.exception_value)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                yield mask(pad_grid, self.kernel, start_axs=(i,j), exception_value=self.exception_value)
    
    def patch_flatten(self, patch):
        flatten_str = ''
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                flatten_str = str(patch[i][j]) if patch[i][j] != self.exception_value else 'x'
        patch_id = self.patch_dict.add_item(flatten_str)
        return patch_id

class PatchSequence:
    def __init__(self, kernel, grid):
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

    def axs2idx(self, axs):
        return axs[0]*self.grid.shape[0] + axs[0]
    
    def idx2axs(self, idx):
        return (idx//self.grid.shape[0], idx%self.grid.shape[0])

    def sequence_analyse(self, analyzer):
        pass

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
        return entropy(norm(np.array([v for _, v in bigram_count.items()])))

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
        self.grid = grid
        self.patch_sequences = {}
    
    def add_patching_sequence(self, patch_kernel):
        self.patch_sequences[patch_kernel.patch_name] = PatchSequence(patch_kernel)
        self.patch_sequences[patch_kernel.patch_name].patching(self.grid)

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
        for i in range(len(state_sequence) - 1):
            current_state = state_sequence[i]
            next_state = state_sequence[i + 1]

            self.transition_counts.add((current_state, next_state), 1)

            # 更新状态总出现次数
            if current_state not in self.state_counts:
                self.state_counts[current_state] = 0
            self.state_counts[current_state] += 1

        # 处理序列中的最后一个状态
        last_state = state_sequence[-1]
        if last_state not in self.state_counts:
            self.state_counts[last_state] = 0
        self.state_counts[last_state] += 1

    def get_transition_matrix(self):
        """
        根据状态转移计数计算状态转移概率矩阵
        :return: 状态转移概率矩阵，以字典形式表示
        """
        transition_matrix = {}
        for current_state, next_states in self.transition_counts.items():
            transition_matrix[current_state] = {}
            total_count = self.state_counts[current_state]
            for next_state, count in next_states.items():
                # 计算转移概率
                transition_matrix[current_state][next_state] = count / total_count
        return transition_matrix



    