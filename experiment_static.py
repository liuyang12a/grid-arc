from visualization import show_grids, show_grid, show_task_example, attention_to_image, save_image, plot_dataframe, grid_to_image
from data_loader import GridDataSet
from rules import TransFeaturesOnRule, StaticFeaturesOnRule, PatchKernel, MarkovChainAnalyzer


import pandas as pd
import matplotlib.pyplot as plt



kernel_params = [
    '1',
    '11', '1|1', '10|01', '01|10',
    '*11', '1*1', '11*', '*|1|1', '1|*|1', '1|1|*', '*1|10', '*1|01', '*0|11',
    '*1|11', '1*|11', '11|*1', '11|1*', '1*1|010', '010|1*1', '10|*1|10', '01|1*|01',
    '010|1*1|010', '101|0*0|101',
    '1*1|111', '11|*1|11',
    '111|1*1|111'
]
kernels = [PatchKernel(kn) for kn in kernel_params]


train_loader = GridDataSet()
train_loader.load_challenges('arc-prize-2024/arc-agi_training_challenges.json')
train_loader.load_solution('arc-prize-2024/arc-agi_training_solutions.json')

ex = train_loader.get_task_example_by_id('06df4c85')
grid = ex['train'][0]['input']

count = 0
for ex in train_loader.iter_task_examples():
    count+=1
    if count == 100:
        break
    iid = ex['id']
    grid = ex['train'][0]['input']
    save_image(grid_to_image(grid), 'images/%s/0.png'%(iid))

    entropys = []
    statics = StaticFeaturesOnRule(grid)
    for i, kernel in enumerate(kernels):
        seq = statics.add_patching_sequence(kernel)
        entropys.append(seq.get_entropys())
        attention = seq.attention_analysis(MarkovChainAnalyzer())
        save_image(attention_to_image(attention), 'images/%s/%02d.(%d)%s.png'%(iid, i, kernel.kernel_size, kernel.name.replace('|', '_').replace('*', 'c')))
    entropys = pd.DataFrame(entropys)
    # img = plot_dataframe(entropys, 'kernel_name')
    save_image(plot_dataframe(entropys, 'kernel_name'), 'images/%s/entorpys.png'%(iid))