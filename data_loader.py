from collections import OrderedDict


def load_json(file_path):
    import json
    with open(file_path) as f:
        l = f.readline()
        return json.loads(l)

class GridDataSet:
    def __init__(self, examples=None):
        self.demonstration = {}
        self.test = {}
        self.solution = {}
        self.ids = OrderedDict()

    def load_challenges(self, raw_data):
        if isinstance(raw_data, str):
            self.load_challenges(load_json(raw_data))
        else:
            for k, v in raw_data.items():
                if k not in self.ids.keys():
                    self.ids[k] = len(self.ids)
                self.demonstration[k] = v["train"]
                self.test[k] = v["test"]

    def load_solution(self, raw_data):
        if isinstance(raw_data, str):
            self.load_solution(load_json(raw_data))
        else:
            self.solution = raw_data

    def get_task_example_by_id(self, id, eva_mode=False):
        example = {
            'id': id,
            'train': None,
            'test': None,
            'solution': None
        }
        if id in self.demonstration.keys():
            example['train'] = self.demonstration[id]
        if id in self.test.keys():
            example['test'] = self.test[id]
        if eva_mode is True and id in self.solution.keys():
            example['solution'] = self.solution[id]
        return example

    def iter_task_examples(self, eva_mode=False):
        for index, id in enumerate(list(self.ids.keys())):
            ex = self.get_task_example_by_id(id, eva_mode=eva_mode)
            ex['index'] = index
            yield ex

    def iter_demonstration_set(self):
        for k,exs in self.demonstration.items():
            index = 0
            for ex in exs:
                index+=1
                yield {
                    'id': k,
                    'index': index,
                    'input': ex['input'],
                    'output': ex['output']
                }