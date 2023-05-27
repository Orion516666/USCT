import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import time

from numpy import mod


def getSinoGram(times):
    # 定义投影角度和传感器对的索引
    num_angles = times.shape[0]
    num_sensor_pairs = times.shape[1]

    # 初始化正弦图谱
    sinogram = np.zeros((num_sensor_pairs, num_angles))

    # 从行进时间矩阵中提取对应的行进时间
    for angle in range(num_angles):
        for sensorPair in range(num_sensor_pairs):
            time = times[angle, sensorPair]
            sinogram[sensorPair, angle] = time

    # 现在你可以使用生成的正弦图谱进行反投影重建或其他处理

    print(sinogram.shape)

def reconstruction_from_times(times, numAngles, numSensorPairs, reconX, reconY, fov):
    # 定义重建参数
    nx = reconX
    ny = reconY
    dx = fov / nx
    dy = fov / ny

    # 初始化重建图像
    reconstruction = np.zeros((ny, nx))

    # 根据行进时间进行反投影
    for angle in range(numAngles):
        for sensorPair in range(numSensorPairs):
            time = times[angle, sensorPair]
            x = sensorPair * dx - fov / 2
            y = angle * dy - fov / 2

            # 反投影到重建图像
            x_index = int((x + fov / 2) / dx)
            y_index = int((y + fov / 2) / dy)
            if x_index >= 0 and x_index < nx and y_index >= 0 and y_index < ny:
                reconstruction[y_index, x_index] += time

    # 绘制重建图像
    plt.imshow(reconstruction, cmap='gray', extent=[-fov / 2, fov / 2, -fov / 2, fov / 2])
    plt.colorbar()
    plt.xlabel('X Coordinate [m]')
    plt.ylabel('Y Coordinate [m]')
    plt.title('Reconstructed Image')
    plt.show()


def fan_beam_reconstruction(projections, angles, detector_size, image_size, pixel_size, source_to_detector, source_to_axis):
    num_projections = len(projections)
    num_detectors = len(projections[0])
    image = np.zeros(image_size)
    center_x = image_size[0] // 2
    center_y = image_size[1] // 2

    for projection_index in range(num_projections):
        angle = angles[projection_index]
        for detector_index in range(num_detectors):
            detector_position = detector_index - num_detectors // 2
            theta = np.deg2rad(angle)
            x = detector_position * pixel_size
            y = source_to_axis
            t = x * np.cos(theta) + y * np.sin(theta)
            s = -x * np.sin(theta) + y * np.cos(theta)
            beta = np.arctan2(s, source_to_detector + t)
            r = np.sqrt(t**2 + (source_to_detector + t)**2)
            x_image = int(center_x + r * np.cos(beta))
            y_image = int(center_y - r * np.sin(beta))

            if 0 <= x_image < image_size[0] and 0 <= y_image < image_size[1]:
                image[x_image, y_image] += projections[projection_index, detector_index]

    return image

def ultrasound_backprojection(projections, angles, slowness_map, image_size, pixel_size, source_to_detector, source_to_axis):
    num_projections = len(projections)
    num_detectors = len(projections[0])
    image = np.zeros(image_size)
    center_x = image_size[0] // 2
    center_y = image_size[1] // 2

    for projection_index in range(num_projections):
        angle = angles[projection_index]
        for detector_index in range(num_detectors):
            detector_position = detector_index - num_detectors // 2
            theta = np.deg2rad(angle)
            x = detector_position * pixel_size
            y = source_to_axis
            t = x * np.cos(theta) + y * np.sin(theta)
            s = -x * np.sin(theta) + y * np.cos(theta)
            beta = np.arctan2(s, source_to_detector + t)
            r = np.sqrt(t**2 + (source_to_detector + t)**2)
            x_image = int(center_x + r * np.cos(beta))
            y_image = int(center_y - r * np.sin(beta))

            if 0 <= x_image < image_size[0] and 0 <= y_image < image_size[1]:
                image[x_image, y_image] += projections[projection_index, detector_index] / slowness_map[x_image, y_image]

    return image

def t():
    proj = scio.loadmat('projection_data.mat')['proj'].T
    n = 185
    theta = np.arange(0, 180, 1)
    x_range = np.arange(np.ceil(-n / 2), np.ceil(n / 2))
    y_range = np.arange(np.ceil(-n / 2), np.ceil(n / 2))
    x, y = np.meshgrid(x_range, y_range)

    proj_fft = np.fft.fftshift(np.fft.fft(proj, axis=1), axes=1)
    freqs = np.arange(-n / 2, n / 2)

    # 构造斜坡滤波器
    ramp_filter = np.abs(freqs)

    # 应用滤波器到投影数据
    proj_filtered = np.zeros_like(proj_fft)
    for i in range(proj_fft.shape[0]):
        proj_filtered[i] = proj_fft[i] * ramp_filter
    # 对滤波后的投影数据进行IFFT和频移逆操作
    filtered_sino = np.fft.ifft(np.fft.ifftshift(proj_filtered, axes=1), axis=1).real

    # filtered_sino = np.fft.ifftshift(np.fft.ifft(proj_filtered, axis=1), axes=1).real
    # 创建空白的重建图像
    reconstructed_image = np.zeros((n, n))

    # 循环遍历所有投影角度
    for angle in range(len(theta)):
        # 计算索引
        index = np.round((n + 1) / 2 + x * np.sin(np.deg2rad(theta[angle])) - y * np.cos(np.deg2rad(theta[angle])) - 1)

        # 创建临时的反投影图像
        back_projection_temp = np.zeros((n, n))

        # 筛选有效的索引
        # flag = np.flatnonzero((index > 0) & (index <= n))
        flag = np.where((index >= 0) & (index < n))

        valid_index = index[flag].astype(int)
        # print(type(valid_index))
        # print(valid_index.shape)
        # 将滤波后的投影值赋给反投影图像
        # temp_filtered_sino = filtered_sino[angle]

        back_projection_temp[flag] = filtered_sino[angle][valid_index]
        # for i in range(flag.shape[0]):
        #     for j in range(flag.shape[1]):
        #         back_projection_temp[i][j] = filtered_sino[angle, valid_index]

        # 将临时的反投影图像添加到重建图像中
        reconstructed_image += back_projection_temp

    # Plotting the grid as a heatmap
    plt.imshow(reconstructed_image, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title('Reconstructed Image')
    plt.show()


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

    return r, float(theta_deg)


if __name__ == '__main__':
    # array = np.array([1, 2, 3])
    # index = np.array([1, 2, 1, 0])
    # print(type(index))
    # print(index.shape)
    # new_array = array[index]
    #
    # print(new_array)

    # t()

    # rows = 3
    # cols = 4
    # index = np.array([[1, 2], [3, 4], [5, 6]])
    # print(index)
    # flag = np.where((index > 0) & (index <= 3))
    #
    # index[flag] = [6, 66, 666]
    # print(index)

    # # 定义两个点的坐标
    # x1, y1 = 0, 0
    x2, y2 = 0, 0
    #
    # # 计算斜率
    # m = (y2 - y1) / (x2 - x1)
    #
    # # 计算角度
    theta = np.arctan2(y2, x2)
    #
    # r, a = cartesian_to_polar(x2, y2, grid_size=3)
    # print(r)
    # print(a)
    #
    # # 将角度转换为度数
    theta_deg = np.degrees(theta)
    theta_deg = np.mod(theta_deg, 360.0)
    print("角度（弧度）:", theta)
    print("角度（度数）:", theta_deg)


    # a = np.arange(0, 2, 1)
    # print(a)