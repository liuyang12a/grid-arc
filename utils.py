import os
import pickle
import numpy as np
from collections import OrderedDict

def compute_lps(pattern: list) -> list:
    """
    计算模式串的最长前缀后缀数组
    :param pattern: 模式串
    :return: 最长前缀后缀数组
    """
    length = 0
    lps = [0] * len(pattern)
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(text: list, pattern: list) -> list:
    """
    KMP（Knuth-Morris-Pratt）
    使用 KMP 算法在文本中查找模式串的所有出现位置
    :param text: 主文本
    :param pattern: 模式串
    :return: 模式串在文本中出现的所有起始位置
    """
    indices = []
    m = len(pattern)
    n = len(text)
    lps = compute_lps(pattern)
    i = 0  # 文本的索引
    j = 0  # 模式串的索引
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            indices.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return indices


def norm(mtx):
    row_sums = mtx.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mtx / row_sums

def entropy(probability_distribution):
    return -np.sum([p * np.log2(p) for p in probability_distribution if p > 0])

class IndexedBidirectionalSet:
    def __init__(self):
        self.item_list = []
        self.item_to_index = {}

    def add_item(self, item):
        if item in self.item_to_index:
            return self.item_to_index[item]
        index = len(self.item_list)
        self.item_list.append(item)
        self.item_to_index[item] = index
        return index

    def get_item(self, index):
        return self.item_list[index]

    def get_index(self, item):
        return self.item_to_index.get(item)

    def __contains__(self, item):
        return item in self.item_to_index.keys()

    def __len__(self):
        return len(self.item_list)
    
    def __str__(self):
        return str(self.item_list)

    def items(self):
        return enumerate(self.item_list)

class PickleObjectFileCache: 
    @staticmethod
    def save(cache_file, obj):
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            print(f"保存对象到缓存文件时出错: {e}")

    @staticmethod
    def load(cache_file):
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"从缓存文件加载对象时出错: {e}")
        return None
    
class TransitionMatrix:

    def __init__(self):
        self.matrix = {}
        self.states = OrderedDict()

    def get(self, indices):
        i, j = indices
        if i in self.matrix.keys():
            if j in self.matrix[i].keys():
                return self.matrix[i][j]
        return None
    
    def _check_null(self, indices):
        for x in indices:
            if x not in self.states.keys():
                self.states[x] = len(self.states)
                self.matrix[x] = {}

    def set(self, indices, value):
        self._check_null(indices)
        i, j = indices
        self.matrix[i][j] = {j:value}

    def add(self, indices, value):
        self._check_null(indices)
        i, j = indices
        if j not in self.matrix[i].keys():
            self.matrix[i][j] = value
        else:
            self.matrix[i][j] += value
    
class ProbabilityTransitionMatrix:
    def __init__(self, trans_mtx: TransitionMatrix, epsilon=0):
        self.states = trans_mtx.states.copy()
        state_num = len(self.states)
        self.matrix = np.zeros(state_num, state_num)
        for cur, nexts in trans_mtx.matrix.items():
            for nxt, count in nexts.items():
                self.matrix[cur][nxt] = count
        self.matrix = norm(self.matrix)
        for i, row in enumerate(self.matrix):
            if sum(row) == 0:
                self.matrix[i][i]=1.0
        if epsilon != 0:
            self.matrix[self.matrix == 0] = epsilon
        self.matrix = norm(self.matrix)
