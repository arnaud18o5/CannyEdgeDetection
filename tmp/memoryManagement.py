from numba import cuda
import numba as nb
import numpy as np
import sys
import math

# In this exercise you will instantiate an array on the host, send it to the device. A kernel will write in the array and finally the host will get the data back.

# 1. Instantiate an array of size 32 on the host and fill it with 0
# 2. Write the code to send the array to the device
# 3. Write a kernel where each thread write its local ID in the corresponding array cell. For example, thread with local ID 4 will do array[4]=4
# 4. Write the code to copy back the array after the execution of the kernel and print its content
# 5. Call your kernel with a grid size of 1 and and 32 threads. Is it working?
    # Yes it is

@cuda.jit
def writeGlobalID(array) :
    local_id = cuda.threadIdx.x
    # global_id = cuda.grid(1)
    array[local_id]= local_id

def runGlobalID():
    print("Starting",sys._getframe(  ).f_code.co_name)
    #Generate array with size matching total number of threads
    A = np.zeros(32, dtype=np.uint16)
    #Send array to device
    d_A = cuda.to_device(A)
    #Execute kernel
    threadsPerBlock=32
    blocksPerGrid=1
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    writeGlobalID[ blocksPerGrid,threadsPerBlock](d_A)
    cuda.synchronize()
    #Copy back the modified array
    A = d_A.copy_to_host()
    #Print a subsequence to check
    # print(A[-10:])
    print(A)

if __name__ == '__main__':
    runGlobalID()(numba)