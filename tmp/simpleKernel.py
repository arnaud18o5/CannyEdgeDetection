from numba import cuda
import numba as nb
import sys

@cuda.jit
def kernel1D() :
    print("hello")

def run1D():
    #4 blocks of 16 threads
    threadsPerBlock = 4
    blocksPerGrid = 1
    print("Starting",sys._getframe(  ).f_code.co_name)
    print("threadsPerBlock ", threadsPerBlock)
    print("blocksPerGrid", blocksPerGrid)
    kernel1D[blocksPerGrid,threadsPerBlock]()
    cuda.synchronize()

run1D()


#lorsque nous faisons le test avec une grid size de 1 et 4 threads, le programme affiche hello 4 fois
# le nombre de print est égale au nombre de threads * le nombre de blocks
