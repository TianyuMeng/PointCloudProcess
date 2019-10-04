from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def rotate_point_cloud_by_angle(data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    :param batch_data: Nx3 array, original batch of point clouds
    :param rotation_angle: range of rotation
    :return:  Nx3 array, rotated batch of point clouds
    """
    angle = rotation_angle * np.pi / 180
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    # rotation_matrix = np.array([[cosval, 0, sinval],
    # [0, 1, 0],
    # [-sinval, 0, cosval]])
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data


def show_pointcloud(point_cloud_np, gif_name=''):

    #if point_cloud_np.shape[1] != 3:
        #return

    fig_ori = plt.figure()
    ps = point_cloud_np
    imgs = []
    for angle in range(0, 360, 10):
        ps = rotate_point_cloud_by_angle(ps, 10)
        plt.clf()
        a1 = fig_ori.add_subplot(111, projection='3d')
        a1.set_xlabel('x', size=15)
        a1.set_xlim(-0.5, 0.5)
        a1.set_ylabel('y', size=15)
        a1.set_ylim(-0.5, 0.5)
        a1.set_zlabel('z', size=15)
        a1.set_zlim(-0.5, 0.5)
        a1.scatter(ps[:, 0], ps[:, 1], ps[:, 2])

        pic_name = gif_name + '_' + str(angle) + '_points_show.png'
        plt.savefig(pic_name)
        temp = Image.open(pic_name)
        imgs.append(temp)

    save_name = gif_name + '_points_show.gif'
    imgs[0].save(save_name, save_all=True, loop=0, append_images=imgs, duration=0.1)
    print(save_name + ' gif has been saved')
    for angle in range(0, 360, 10):
        pic_name = gif_name + '_' + str(angle) + '_points_show.png'
        os.remove(pic_name)

