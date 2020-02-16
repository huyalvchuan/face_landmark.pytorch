import math
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
plt.ion()


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=3, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()


def rotate(angle, x, y, pointx, pointy):
    angle = -angle * math.pi / 180.0
    srx = (x-pointx)*math.cos(angle) - (y-pointy)*math.sin(angle)+pointx
    sry = (x-pointx)*math.sin(angle) + (y-pointy)*math.cos(angle)+pointy


    return srx, sry
    