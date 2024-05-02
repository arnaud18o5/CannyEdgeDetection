from numba import cuda
import numba as nb
import sys

@cuda.jit
def coordinates1D():
    local_id = cuda.threadIdx.x
    # global_id = cuda.grid(1)
    # print("global_id", global_id, "and local_id", local_id)
    block_size = cuda.blockDim.x
    block_id = cuda.blockIdx.x
    # print("block_id", block_id, "and block_size", block_size)
    man_global_id = block_id * block_size + local_id
    print("man_global_id", man_global_id)

@cuda.jit
def coordiantes2D():
    # Thread 2D local ID
    local_id_x = cuda.threadIdx.x
    local_id_y = cuda.threadIdx.y
    local_id_z = cuda.threadIdx.z
    (global_id_x, global_id_y) = cuda.grid(2)
    # print("local ID (", local_id_x, ",", local_id_y, ",", local_id_z, ") ")
    print("global ID (", global_id_x, ",", global_id_y, ") ")


def run1D():
    threadsPerBlock = 8
    blocksPerGrid = 2
    print("Starting",sys._getframe(  ).f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    coordinates1D[blocksPerGrid,threadsPerBlock]()
    cuda.synchronize()

def run2D():
    threadsPerBlock = (4,2,1)
    blocksPerGrid = (2,2,1)
    print("Starting",sys._getframe(  ).f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    coordiantes2D[blocksPerGrid,threadsPerBlock]()
    cuda.synchronize()

if __name__ == '__main__':
    # run1D() 
    run2D()(numba)