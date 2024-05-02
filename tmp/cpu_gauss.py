# Gaussia blur
"""
We will start by implementing a CPU version with the fixed kernel of the previous section. Here are some advices

Process each RGB value separately, i.e. compute the new value R using the R values only, then G...
Do not modify the original image, write the result to a new (empty) image
Enforce typing on your array to uint8, otherwise you will end up with float when doing the computations
Don't forget that numpy swap axes when extracting the array from an image.

"""

import sys
import numpy as np
from PIL import Image
from timeit import default_timer as timer

def pixel_cov(matrix, gaussian):
    height, width = matrix.shape
    sum = 0
    for i in range(height):
        for j in range(width):
            sum = sum + matrix[i, j] * gaussian[i, j]
    return sum // 256

def to_gauss_blur(input_img):
    height, width, _ = input_img.shape
    dst = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the Gaussian matrix
    gaussian = np.array([[1, 4, 7, 4, 1],
                         [4, 16, 26, 16, 4],
                         [7, 26, 41, 26, 7],
                         [4, 16, 26, 16, 4],
                         [1, 4, 7, 4, 1]])

    for i in range(height):
        for j in range(width):
            # array of 5*5 of the pixels around if the pixel is on the edge set to 0
            matrix = np.zeros((5, 5, 3), dtype=np.uint8)
            for k in range(-2, 3):
                for l in range(-2, 3):
                    if i + k < 0 or i + k >= height or j + l < 0 or j + l >= width:
                        matrix[k + 2, l + 2] = [0, 0, 0]
                    else:
                        matrix[k + 2, l + 2] = input_img[i + k, j + l]
            dst[i, j] = [pixel_cov(matrix[:, :, c], gaussian) for c in range(3)]
    return dst


inputFile = sys.argv[1]
outputFile=sys.argv[2]

# Ouvrir l'image
img = Image.open(inputFile)

# Convertir l'image en tableau numpy
src = np.array(img)

# Appliquer le flou gaussien
dst = to_gauss_blur(src)

Image.fromarray(dst, mode="RGB").save(outputFile)