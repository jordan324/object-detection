import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from setuptools import setup
from Cython.Build import cythonize

def convert_dim(image_data, width, height):
    return [image_data[i * width:(i + 1) * width] for i in range(height)]


def gaussian_kernel(kernel_size, sigma):
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))

    for x in range(kernel_size):
        for y in range(kernel_size):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-((x - center)**2.0 + (y - center)**2.0) / (2.0 * sigma**2.0))
            
            kernel[x, y] = normal * exp_term

    # Normalize the kernel
    kernel /= np.sum(kernel)

    return kernel


def convolution(image_array, width, height, kernel, kernel_size):
    conv_img = np.zeros((width, height, 3), dtype=np.float32)
    kernel_center = kernel_size // 2

    for img_x in range(width):
        for img_y in range(height):
            conv_red, conv_green, conv_blue = 0.0, 0.0, 0.0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    # Calculate the indices after applying the kernel offset
                    x_index = max(0, min(width - 1, img_x + i - kernel_center))
                    y_index = max(0, min(height - 1, img_y + j - kernel_center))
                    red, green, blue = image_array[x_index][y_index]  

                    conv_red += red * kernel[i, j]
                    conv_green += green * kernel[i, j]
                    conv_blue += blue * kernel[i, j]

            conv_img[img_x, img_y] = tuple([conv_red, conv_green, conv_blue])
    return conv_img




def apply_gaussian(img_path, kernel_size, sigma):
    image = Image.open(img_path)
    width, height = image.size
    
    image_array = list(image.getdata())
    image_array = convert_dim(image_array, width, height)
    kernel = gaussian_kernel(kernel_size, sigma)

    return convolution(image_array, width, height, kernel, kernel_size)


