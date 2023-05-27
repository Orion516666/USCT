import math
import time

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def cartesian_to_polar(x, y, grid_size):
    center = (grid_size - 1) / 2
    # 计算相对于中心点的偏移量
    dx = x - center
    dy = y - center

    # 计算极坐标系中的距离和角度
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # 弧度
    # 将角度转换为度数
    theta_deg = np.degrees(theta)
    theta_deg = np.mod(theta_deg, 360.0)

    # 如果那条线过原点，那要它的法线是90度，如果不判断，会变成0度
    if dx == 0 and dy == 0:
        theta_deg = 90

    return float(r), float(theta_deg)


def calculate_sinogram(travelTimes, grid_size):
    times = travelTimes['times']
    x_idx = travelTimes['x_idx']
    y_idx = travelTimes['z_idx']

    num_sensors = x_idx.shape[0]

    sensor_coordinates = list(zip(x_idx - 1, y_idx - 1))

    sensors = [(x, y) for x, y in sensor_coordinates]  # 将传感器的x和y坐标组成列表

    # 对每对传感器之间的直线进行转换
    line_segments = []
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            x1, y1 = sensors[i]
            x2, y2 = sensors[j]
            tof = times[i][j]
            # # 将直线的两个端点从笛卡尔坐标系转换为极坐标系
            # r1, theta1 = cartesian_to_polar(x1, y1, grid_size)
            # r2, theta2 = cartesian_to_polar(x2, y2, grid_size)

            # 计算垂线段的rho和theta值
            x_midpoint = (x1 + x2) / 2
            y_midpoint = (y1 + y2) / 2
            rho_normal, theta_normal = cartesian_to_polar(x_midpoint, y_midpoint, grid_size)
            if theta_normal >= 180:
                rho_normal = -rho_normal
            rho_normal += 458
            theta_normal = float(np.mod(theta_normal, 180))
            # theta_segment = (theta1 + theta2) * 0.5
            # cz = theta_segment - min(theta1, theta2)
            # cz = math.radians(cz)
            # rho_segment = 458 * math.cos(cz) + 458
            # theta_segment = float(np.mod(theta_segment, 180))
            # 构建极坐标系中的直线表示（以端点的距离和角度表示）

            line_segments.append((rho_normal, theta_normal, tof))


    print(len(line_segments))
    # raw_sinogram: The range of rho is from 0 to 916, and the range of theta is from 0 to 180 degrees.
    np.save('raw_sinogram.npy', line_segments)

    # # 输出转换后的直线
    # for i, line in enumerate(line_segments):
    #     rho_segment, theta_segment, tof = line
    #     print(f"Rho: {rho_segment}, Theta: {theta_segment}, tof: {tof}")
    #
    #     if i == 126:  # 输出前10个直线后退出循环
    #         break


def USCT_interpolation(raw_sinogram):
    # 将前两列的元素四舍五入为整数
    rounded_sinogram = np.round(raw_sinogram[:, :2])
    raw_sinogram[:, :2] = rounded_sinogram

    # 创建一个458x360的网格，用于存储tof值
    grid = np.zeros((181, 917))

    for rho, theta, tof in raw_sinogram:
        x = int(theta)
        y = int(rho)
        grid[x, y] = tof


    # # Plotting the grid as a heatmap
    # plt.imshow(grid, cmap='viridis', origin='lower')
    # plt.colorbar()
    # plt.xlabel('Rho')
    # plt.ylabel('Theta')
    # plt.title('Grid Heatmap')
    # plt.show()


    # 生成非零值的坐标和对应的值
    nonzero_indices = np.nonzero(grid)
    nonzero_points = np.column_stack(nonzero_indices)
    nonzero_values = grid[nonzero_indices]

    x_coords, y_coords = np.mgrid[0:181, 0:917]

    # 使用插值函数进行插值
    interpolated_grid = griddata(nonzero_points, nonzero_values, (x_coords, y_coords), method='linear')

    # print(interpolated_grid.shape)

    # # Plotting the grid as a heatmap
    # plt.imshow(interpolated_grid, cmap='viridis', origin='lower')
    # plt.colorbar()
    # plt.xlabel('Rho')
    # plt.ylabel('Theta')
    # plt.title('Sinogram Heatmap')
    # plt.show()

    return grid[0:180, :]

def USCT_FBP(proj):
    proj[np.isnan(proj)] = 0
    n = proj.shape[1]
    theta = np.arange(0, 180, 1)
    x_range = np.arange(np.ceil(-n / 2), np.ceil(n / 2))
    y_range = np.arange(np.ceil(-n / 2), np.ceil(n / 2))
    x, y = np.meshgrid(x_range, y_range)

    proj_fft = np.fft.fftshift(np.fft.fft(proj, axis=1), axes=1)
    freqs = np.arange(-n/2, n/2)

    # 构造斜坡滤波器
    ramp_filter = np.abs(freqs)

    # 应用滤波器到投影数据
    proj_filtered = np.zeros_like(proj_fft)
    for i in range(proj_fft.shape[0]):
        proj_filtered[i] = proj_fft[i] * ramp_filter
    # 对滤波后的投影数据进行IFFT和频移逆操作
    filtered_sino = np.fft.ifft(np.fft.ifftshift(proj_filtered, axes=1), axis=1).real
    # 创建空白的重建图像
    reconstructed_image = np.zeros((n, n))

    # 循环遍历所有投影角度
    for angle in range(len(theta)):
        # 计算索引
        index = np.round((n + 1) / 2 + x * np.sin(np.deg2rad(theta[angle])) - y * np.cos(np.deg2rad(theta[angle])))

        # 创建临时的反投影图像
        back_projection_temp = np.zeros((n, n))

        # 筛选有效的索引
        flag = np.where((index >= 0) & (index < n))
        valid_index = index[flag].astype(int)

        # 将滤波后的投影值赋给反投影图像
        back_projection_temp[flag] = filtered_sino[angle][valid_index]

        # 将临时的反投影图像添加到重建图像中
        reconstructed_image += back_projection_temp

    # Plotting the grid as a heatmap
    plt.imshow(reconstructed_image, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title('Reconstructed Image')
    plt.show()


if __name__ == "__main__":
    #
    # travelTimes = scio.loadmat('travelTimes.mat')
    # grid_size = 1001
    # calculate_sinogram(travelTimes, grid_size)
    #
    # time.sleep(2)


    # sorted_sinogram = raw_sinogram[raw_sinogram[:, 0].argsort()]
    #
    # # # 将前两列的元素四舍五入为整数
    # rounded_sinogram = np.round(raw_sinogram[:, :2])
    # #
    # raw_sinogram[:, :2] = rounded_sinogram
    # #
    # print(raw_sinogram[:6])
    # print(rounded_sinogram[:6])

    # 打印四舍五入后的结果
    # print(sorted_sinogram[:128])


    raw_sinogram = np.load('raw_sinogram.npy', allow_pickle=True)
    proj = USCT_interpolation(raw_sinogram)
    USCT_FBP(proj)
