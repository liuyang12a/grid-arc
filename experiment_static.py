from visualization import show_grids, show_grid, show_task_example
from data_loader import GridDataSet
from rules import TransFeaturesOnRule, StaticFeaturesOnRule, PatchKernel


import pandas as pd
import matplotlib.pyplot as plt


def plot_dataframe(df, x_label, title=None, y_label=None):

    x = df[x_label]
    ys= df.columns.drop(x_label)
    for column in ys:
        plt.plot(x, df[column], label=column)

    # 添加图标题、x 轴标签、y 轴标签和图例
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    # 显示网格线
    plt.grid(True)
    plt.xticks(rotation=90)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # 从内存中的数据创建 PIL 的 Image 对象
    img = Image.open(buf)

    # 关闭当前的 matplotlib 图形
    plt.close()

    # 返回 Image 对象
    return img


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

entropys = []

statics = StaticFeaturesOnRule(grid)
for kernel in kernels:
    seq = statics.add_patching_sequence(kernel)
    entropys.append(seq.get_entropys())

entropys = pd.DataFrame(entropys)
img = plot_dataframe(entropys, 'kernel_name')
plt.imshow(img)
plt.axis('off')

# 显示图形
plt.show()