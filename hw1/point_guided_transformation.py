import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None


# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img


# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标

    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点

    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点

    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射

    return marked_image


# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8, lam=0):
    """
    Return
    ------
        A deformed image.
    """
    # d = 3
    sz = source_pts.shape[0]
    si_0 = image.shape[0]
    si_1 = image.shape[1]
    warped_image = 255 * np.ones_like(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    sx = source_pts[:, 0]
    sy = source_pts[:, 1]
    source_pts_x = sx.reshape((-1, 1))
    source_pts_y = sy.reshape((-1, 1))
    m = np.sqrt((source_pts_x - sx) ** 2 + (source_pts_y - sy) ** 2)
    m = np.where(m != 0, m ** 2 * np.log(m), 0)
    #m = 1 / (d + m ** 2)
    m = m + lam * np.eye(sz)
    m1 = np.hstack((np.ones((sz, 1)), source_pts))
    m2 = np.hstack((m1.T, np.zeros((3, 3))))
    m = np.hstack((m, m1))
    m = np.vstack((m, m2))
    b = np.vstack((target_pts, np.zeros((3, 2))))
    parm = np.linalg.solve(m, b)
    si_x0 = np.arange(0, si_0, 1)
    si_y0 = np.arange(0, si_1, 1)
    si_x = np.kron(np.ones_like(si_x0), si_y0)
    si_y = np.kron(si_x0, np.ones_like(si_y0))
    m_i = np.sqrt((source_pts_x - si_x) ** 2 + (source_pts_y - si_y) ** 2)
    m_i = np.where(m_i != 0, m_i ** 2 * np.log(m_i), 0)
    # m_i = 1 / (d + m_i ** 2)
    m_i = np.vstack((m_i, np.ones_like(si_x), si_x, si_y))
    d_uv = np.matmul(parm.T, m_i)
    d_u = d_uv[0, :]
    d_u = d_u.reshape((si_0, si_1))
    d_v = d_uv[1, :]
    d_v = d_v.reshape((si_0, si_1))
    for i in range(0, si_0):
        for j in range(0, si_1):
            u = int(d_u[i, j])
            if u >= si_1:
                u = si_1 - 1
            elif u < 0:
                u = 0
            v = int(d_v[i, j])
            if v >= si_0:
                v = si_0 - 1
            elif v < 0:
                v = 0
            warped_image[v, u] = image[i, j]
            if v < si_0 - 1:
                warped_image[v + 1, u] = image[i, j]
            if u < si_1 - 1:
                warped_image[v, u + 1] = image[i, j]
            if v > 1:
                warped_image[v - 1, u] = image[i, j]
            if u > 1:
                warped_image[v, u - 1] = image[i, j]
    return warped_image


def run_warping():
    global points_src, points_dst, image  ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image


# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图


# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", width=800, height=800)

        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)

    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮

    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)

# 启动 Gradio 应用
demo.launch()
