import os
import numpy as np
import cv2


class Calib:
    def __init__(self, dict_calib):
        super(Calib, self).__init__()
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.P1 = dict_calib['P1'].reshape(3, 4)
        self.P2 = dict_calib['P2'].reshape(3, 4)
        self.P3 = dict_calib['P3'].reshape(3, 4)
        self.R0_rect = dict_calib['R0_rect'].reshape(3, 3)
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape(3, 4)
        self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape(3, 4)


class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        # content 就是一个字符串，根据空格分隔开来
        lines = content.split()

        # 去掉空字符
        lines = list(filter(lambda x: len(x), lines))

        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(lines[3])

        self.bbox = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox = np.array([float(x) for x in self.bbox])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])
        #这一行是模型训练后的label通常最后一行是阈值，可以同个这个过滤掉概率低的object
        #如果只要显示kitti本身则不需要这一行
        #self.ioc = float(lines[15])


class Kitti_Dataset:
    def __init__(self, dir_path, split="training"):
        super(Kitti_Dataset, self).__init__()
        self.dir_path = os.path.join(dir_path, split)
        # calib矫正参数文件夹地址
        self.calib = os.path.join(self.dir_path, "calib")
        # RGB图像的文件夹地址
        self.images = os.path.join(self.dir_path, "image_2")
        # 点云图像文件夹地址
        self.pcs = os.path.join(self.dir_path, "velodyne")
        # 标签文件夹的地址
        self.labels = os.path.join(self.dir_path, "label_2")

    # 得到当前数据集的大小
    def __len__(self):
        file = []
        for _, _, file in os.walk(self.images):
            pass

        # 返回rgb图片的数量
        return len(file)

    # 得到矫正参数的信息
    def get_calib(self, index):
        # 得到矫正参数文件
        calib_path = os.path.join(self.calib, "{:06d}.txt".format(index))
        with open(calib_path) as f:
            lines = f.readlines()

        lines = list(filter(lambda x: len(x) and x != '\n', lines))
        dict_calib = {}
        for line in lines:
            key, value = line.split(":")
            dict_calib[key] = np.array([float(x) for x in value.split()])
        return Calib(dict_calib)

    def get_rgb(self, index):
        # 首先得到图片的地址
        img_path = os.path.join(self.images, "{:06d}.png".format(index))
        return cv2.imread(img_path)

    def get_pcs(self, index):
        pcs_path = os.path.join(self.pcs, "{:06d}.bin".format(index))
        # 点云的四个数据（x, y, z, r)
        aaa = np.fromfile(pcs_path, dtype=np.float32, count=-1).reshape([-1, 4])
        return aaa[:, :3]

    def get_labels(self, index):
        labels_path = os.path.join(self.labels, "{:06d}.txt".format(index))
        with open(labels_path) as f:
            lines = f.readlines()
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))

        return [Object3d(x) for x in lines]
