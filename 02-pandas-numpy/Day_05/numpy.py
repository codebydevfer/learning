#numpy arrays

import numpy as np

#1
arr = np.array([1, 2, 3, 4, 5])

print(arr)

print(type(arr))

#2 - Checking number of dimensions (0-D; 1-D; 2-D; 3-D...)
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)


#slicing & indexing

# Slicing in python means taking elements from one given index to another given index.
# We pass slice instead of index like this: [start:end].
# We can also define the step, like this: [start:end:step].
# If we don't pass start its considered 0
# If we don't pass end its considered length of array in that dimension
# If we don't pass step its considered 1

#1
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])

#2
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])

#3
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])

#4 (every other element)
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[::2])

#5 (slicing from the second element)
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4])

#6 (slicing from both elements)
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 2])

#random

#https://www.w3schools.com/python/numpy/numpy_random.asp

#mean / std

#mean (average)
# numpy.mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True)

#std (standard deviation)
data = np.array([10, 20, 30, 40, 50])
std_dev = np.std(data)
print(f"Standard Deviation: {std_dev}")

#matrix operations and broadcasting

#broadcasting
import numpy as np

# create 1-D array
array1 = np.array([1, 2, 3])

# create 2-D array
array2 = np.array([[1], [2], [3]])

# add arrays of different dimension
# size of array1 expands to match with array2
sum = array1 + array2

print(sum)

#broadcastable examples

# Broadcastable Shapes

# (6, 7) and (6, 7)
# (6, 7) and (6, 1)
# (6, 7) and (7, )

#non-broadcastable examples

# Non-Broadcastable Shapes

# (6, 7) and (7, 6)
# (6, 7) and (6, )

#Broadcasting with Scalars

import numpy as np

# 1-D array
array1 = np.array([1, 2, 3])

# scalar
number = 5

# add scalar and 1-D array
sum = array1 + number

print(sum)

#matrix operations

# Functions	Descriptions
# array()	creates a matrix
# dot()	performs matrix multiplication
# transpose()	transposes a matrix
# linalg.inv()	calculates the inverse of a matrix
# linalg.det()	calculates the determinant of a matrix
# flatten()	transforms a matrix into 1D array

#1
import numpy as np

# create a 2x2 matrix
matrix1 = np.array([[1, 3], 
                   [5, 7]])

print("2x2 Matrix:\n",matrix1)

# create a 3x3  matrix
matrix2 = np.array([[2, 3, 5],
             	    [7, 14, 21],
                    [1, 3, 5]])
                    
print("\n3x3 Matrix:\n",matrix2)


#2 - matrix multiplication
import numpy as np

# create two matrices
matrix1 = np.array([[1, 3], 
             		[5, 7]])
             
matrix2 = np.array([[2, 6], 
                    [4, 8]])

# calculate the dot product of the two matrices
result = np.dot(matrix1, matrix2)

print("matrix1 x matrix2: \n",result)

#3 - transpose matrix
import numpy as np

# create a matrix
matrix1 = np.array([[1, 3], 
             		[5, 7]])

# get transpose of matrix1
result = np.transpose(matrix1)

print(result)

#4 - calculate the inverse of a matrix
import numpy as np

# create a 3x3 square matrix
matrix1 = np.array([[1, 3, 5], 
             		[7, 9, 2],
                    [4, 6, 8]])

# find inverse of matrix1
result = np.linalg.inv(matrix1)

print(result)

#5 - find a determinant of a matrix
import numpy as np

# create a matrix
matrix1 = np.array([[1, 2, 3], 
             		[4, 5, 1],
                    [2, 3, 4]])

# find determinant of matrix1
result = np.linalg.det(matrix1)

print(result)

#6 - flatten matrix
import numpy as np

# create a 2x3 matrix
matrix1 = np.array([[1, 2, 3], 
             		[4, 5, 7]])

result = matrix1.flatten()
print("Flattened 2x3 matrix:", result)








#Reference - W3Schools
#Reference - Programiz