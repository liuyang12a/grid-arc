import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
from PIL import Image
import torch

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

def convert_color(grid):  # grid dims must end in c
    return np.clip(np.matmul(grid, colors), 0, 255).astype(np.uint8)

def plot_problem(logger):
    """
    Draw a plot of an ARC-AGI problem, and save it in plots/
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
    """

    # Put all the grids beside one another on one grid
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y
    pixels = 255+np.zeros([n_train+n_test, 2*n_x+2, 2, 2*n_y+8, 3], dtype=np.uint8)
    for example_num in range(n_examples):
        if example_num < n_train:
            subsplit = 'train'
            subsplit_example_num = example_num
        else:
            subsplit = 'test'
            subsplit_example_num = example_num - n_train
        for mode_num, mode in enumerate(('input', 'output')):
            if subsplit == 'test' and mode == 'output':
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
            grid = convert_color(grid)  # x, y, c
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],mode_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid
    pixels = pixels.reshape([(n_train+n_test)*(2*n_x+2), 2*(2*n_y+8), 3])
    
    os.makedirs("plots/", exist_ok=True)

    # Plot the combined grid and make gray dividers between the grid cells, arrows, and a question mark for unsolved examples.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for example_num in range(n_examples):
        for mode_num, mode in enumerate(('input', 'output')):
            if example_num < n_train:
                subsplit = 'train'
                subsplit_example_num = example_num
            else:
                subsplit = 'test'
                subsplit_example_num = example_num - n_train
            ax.arrow((2*n_y+8)-3-0.5, (2*n_x+2)*example_num+1+n_x-0.5, 6, 0, width=0.5, fc='k', ec='k', length_includes_head=True)
            if subsplit == 'test' and mode == 'output':
                ax.text((2*n_y+8)+4+n_y-0.5, (2*n_x+2)*example_num+1+n_x-0.5, '?', size='xx-large', ha='center', va='center')
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            for xline in range(grid.shape[0]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]-0.5, (2*n_y+8)*mode_num+4+n_y+grid.shape[1]-0.5),
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(grid.shape[1]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]-0.5, (2*n_x+2)*example_num+1+n_x+grid.shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    plt.axis('off')
    plt.savefig('plots/' + logger.task.task_name + '_problem.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_solution(logger, fname=None):
    """
    Draw a plot of a model's solution to an ARC-AGI problem, and save it in plots/
    Draws four plots: A model output sample, the mean of samples, and the top two most common samples.
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
    """
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y

    # Four plotted solutions
    solutions_list = [
            torch.softmax(logger.current_logits, dim=1).cpu().numpy(),
            torch.softmax(logger.ema_logits, dim=1).cpu().numpy(),
            logger.solution_most_frequent,
            logger.solution_second_most_frequent,
            ]
    masks_list = [
            (logger.current_x_mask, logger.current_y_mask),
            (logger.ema_x_mask, logger.ema_y_mask),
            None,
            None,
            ]
    solutions_labels = [
            'sample',
            'sample average',
            'guess 1',
            'guess 2',
            ]
    n_plotted_solutions = len(solutions_list)

    # Put all the grids beside one another on one grid
    pixels = 255+np.zeros([n_test, 2*n_x+2, n_plotted_solutions, 2*n_y+8, 3], dtype=np.uint8)
    shapes = []
    for subsplit_example_num in range(n_test):
        subsplit = 'test'
        example_num = subsplit_example_num + n_train
        shapes.append([])

        for solution_num, (solution, masks, label) in enumerate(zip(solutions_list, masks_list, solutions_labels)):
            grid = np.array(solution[subsplit_example_num])  # c, x, y if 'sample' in label else x, y, c
            if 'sample' in label:
                grid = np.einsum('dxy,dc->xyc', grid, colors[logger.task.colors])  # x, y, c
                if logger.task.in_out_same_size or logger.task.all_out_same_size:
                    x_length = logger.task.shapes[example_num][1][0]
                    y_length = logger.task.shapes[example_num][1][1]
                else:
                    x_length = None
                    y_length = None
                x_start, x_end = logger._best_slice_point(masks[0][subsplit_example_num,:], x_length)
                y_start, y_end = logger._best_slice_point(masks[1][subsplit_example_num,:], y_length)
                grid = grid[x_start:x_end,y_start:y_end,:]  # x, y, c
                grid = np.clip(grid, 0, 255).astype(np.uint8)
            else:
                grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
                grid = convert_color(grid)  # x, y, c

            shapes[subsplit_example_num].append((grid.shape[0], grid.shape[1]))
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[subsplit_example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],solution_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid

    pixels = pixels.reshape([n_test*(2*n_x+2), n_plotted_solutions*(2*n_y+8), 3])
    
    # Plot the combined grid and make gray dividers between the grid cells, and labels.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for subsplit_example_num in range(n_test):
        for solution_num in range(n_plotted_solutions):
            subsplit = 'test'
            grid = np.array(solutions_list[solution_num][subsplit_example_num])  # x, y
            shape = shapes[subsplit_example_num][solution_num]
            for xline in range(shape[0]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]-0.5, (2*n_y+8)*solution_num+4+n_y+shape[1]-0.5),
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(shape[1]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]-0.5, (2*n_x+2)*subsplit_example_num+1+n_x+shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    for solution_num, solution_label in enumerate(solutions_labels):
        ax.text((2*n_y+8)*solution_num+4+n_y-0.5, -3, solution_label, size='xx-small', ha='center', va='center')
    plt.axis('off')
    if fname is None:
        fname = 'plots/' + logger.task.task_name + '_solutions.pdf'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

