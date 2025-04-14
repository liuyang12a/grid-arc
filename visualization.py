import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
from PIL import Image

# 定义颜色映射，ARC 数据集中通常有 10 种颜色（0 - 9）
colors = [
    [0, 0, 0],      # 黑色
    [0, 116, 217],  # 蓝色
    [255, 65, 54],  # 红色
    [46, 204, 64],  # 绿色
    [255, 220, 0],  # 黄色
    [170, 170, 170],# 灰色
    [240, 18, 190], # 粉色
    [255, 133, 27], # 橙色
    [127, 219, 255],# 浅蓝色
    [135, 12, 37]   # 深紫色
]
line_color = [85,85,85] #深灰色

def attention_to_image(grid, grid_pixels=15, line_pixels=0):
    if isinstance(grid, list):
        grid = np.array(grid)
    height, width = grid.shape
    image = np.zeros((line_pixels+height*(grid_pixels+line_pixels), line_pixels+width*(grid_pixels+line_pixels), 3), dtype=np.uint8)
    image[:,:] = 0
    for i in range(height):
        for j in range(width):
            attention = grid[i, j]
            strength = int(255*attention)
            image[i*(line_pixels+grid_pixels)+line_pixels:(i+1)*(line_pixels+grid_pixels), j*(line_pixels+grid_pixels)+line_pixels:(j+1)*(line_pixels+grid_pixels)] = [strength,strength,strength]     
    return image

def grid_to_image(grid, grid_pixels=15, line_pixels=1):
    """
    将二维网格转换为可视化图像
    :param grid: 二维数组，表示网格
    :param grid_pixels: 边线宽度，像素数
    :line_pixels_pixels: 网格尺寸，像素数 
    :return: 可视化的图像
    """
    if isinstance(grid, list):
        grid = np.array(grid)
    height, width = grid.shape
    image = np.zeros((line_pixels+height*(grid_pixels+line_pixels), line_pixels+width*(grid_pixels+line_pixels), 3), dtype=np.uint8)
    image[:,:] = line_color
    for i in range(height):
        for j in range(width):
            color_index = grid[i, j]
            image[i*(line_pixels+grid_pixels)+line_pixels:(i+1)*(line_pixels+grid_pixels), j*(line_pixels+grid_pixels)+line_pixels:(j+1)*(line_pixels+grid_pixels)] = colors[color_index]      
    return image

def show_grid(grid, title=None):
    """
    绘制单个网格图
    :param grid: 二维数组，表示网格
    """
    image = grid_to_image(grid)
    plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def show_grids(grids, columns=None, title=None, sub_titles=None):
    """
    绘制一组网格图
    :param grid: 以网格为元素的二维数组
    :param rows: 行数
    :param columns: 列数
    """
    if columns is None or len(grids)<=columns:
        _, axes = plt.subplots(1, len(grids))
        for j in range(len(grids)):
            axes[j].imshow(grid_to_image(grids[j]))
            if sub_titles is not None:
                axes[j].set_title(sub_titles[j])
            axes[j].axis('off')
    else:
        rows = len(grids)//columns + 1
        _, axes = plt.subplots(rows, columns)
        for i in range(rows):
            for j in range(columns):
                index = i*columns+j
                if index < len(grids):
                    axes[i,j].imshow(grid_to_image(grids[index]))
                    if sub_titles is not None:
                        axes[i,j].set_title(sub_titles[index])
                axes[i,j].axis('off')
    plt.figtext(0.5, 0.02, title, ha='center')
    plt.tight_layout()
    plt.show()

def show_task_example(ex, columns=6):
    title = "%d-%s"%(ex['index'], ex['id'])
    grids = []
    sub_titles = []
    for n, pair in enumerate(ex['train']):
        grids.append(pair['input'])
        sub_titles.append('ex%d.input'%(n+1))
        grids.append(pair['output'])
        sub_titles.append('ex%d.output'%(n+1))
    grids.append(ex['test'][0]['input'])
    sub_titles.append('test')
    if ex['solution'] is not None:
        for n, s in ex['solution']:
            grids.append(s)
            sub_titles.append('solution%d')%(n+1)


    show_grids(grids, columns, title, sub_titles)

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

def save_image(image, file_path):
    directory = os.path.dirname(file_path)
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.cla()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(file_path)
    plt.close('all')