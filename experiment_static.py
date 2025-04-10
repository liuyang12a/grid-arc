from visualization import show_grids, show_grid, show_task_example
from data_loader import GridDataSet
from rules import TransFeaturesOnRule, StaticFeaturesOnRule, PatchKernel

kernel_params = [
    ['3x3', [[1,1,1],[1,1,1],[1,1,1]], (1,1)]
]

def init_patch_kernel(kernel_param):
    return PatchKernel(kernel_param[0], kernel_param[1], kernel_param[2])

train_loader = GridDataSet()
train_loader.load_challenges('arc-prize-2024/arc-agi_training_challenges.json')
train_loader.load_solution('arc-prize-2024/arc-agi_training_solutions.json')

ex = train_loader.get_task_example_by_id('06df4c85')
grid = ex['train'][0]['input']

static_rule = StaticFeaturesOnRule(grid)
kernel = init_patch_kernel(kernel_params[0])
seq = static_rule.add_patching_sequence(kernel)

# print(kernel.patch_dict)
print(seq.state_list)
print(seq.get_random_entropy())
print(seq.get_info_entropy())
print(seq.get_markov_entropy())
print(seq.get_entropy_rate())