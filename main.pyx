cimport cython

from augmentation import apply_gaussian

cimport cython
cpdef gaussian():
    return apply_gaussian(r"C:\Users\test0\Downloads\car.jpg", 5, 2)