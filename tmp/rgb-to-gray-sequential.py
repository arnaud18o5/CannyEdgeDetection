# met une image en niveau de gris
import sys
if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} SRC_IMAGE DST_IMAGE")
src_img = sys.argv[1]
dst_img = sys.argv[2]

from numba import cuda
import numpy as np
from PIL import Image

@cuda.jit(device=True)
def to_gray(rgb):
    R, G, B = rgb
    return 0.3 * R + 0.59 * G + 0.11 * B

img = Image.open(src_img)
src = np.array(img)
height, width, _ = src.shape

# Allocate the array for the B&W image directly on the GPU
dst = cuda.device_array((height, width), dtype=np.int8)

threadsPerBlock = 16
blocksPerGrid = (height + threadsPerBlock - 1) // threadsPerBlock

# Define the kernel function
@cuda.jit
def convert_to_gray(src, dst):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        dst[x, y] = to_gray(src[x, y])

# Call the kernel function
convert_to_gray[blocksPerGrid, threadsPerBlock](src, dst)

# Copy the result back to the host and save the image
result = dst.copy_to_host()
Image.fromarray(result, mode="L").save(dst_img)