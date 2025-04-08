import os
import pickle

class IndexStringBidirectionalDict:
    def __init__(self):
        self.index_to_string = {}
        self.string_to_index = {}
        self.current_index = 0

    def add_string(self, string):
        if string in self.string_to_index:
            return self.string_to_index[string]
        index = self.current_index
        self.index_to_string[index] = string
        self.string_to_index[string] = index
        self.current_index += 1
        return index

    def get_string(self, index):
        return self.index_to_string.get(index)

    def get_index(self, string):
        return self.string_to_index.get(string)

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.index_to_string
        elif isinstance(item, str):
            return item in self.string_to_index
        return False

    def __len__(self):
        return len(self.index_to_string)

    def items(self):
        return self.index_to_string.items()

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