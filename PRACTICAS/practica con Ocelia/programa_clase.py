import numpy as np

a = np.zeros((2,3), int)
b = np.array(([1,2,0],[1.2,0,3]),np.float32)
c = b*b
print(c)

#producto vectorial/matricial
d = b.dot(b.transpose())
print(d)

e = np.random.choice([0,1,2,3,4,5,6,7,8,9,1,5,6,45,78,98,32,65,156,78,654,15,5], (6), False)
print(e)

array_cargado = np.loadtxt('/tmp/arrays.txt')
################################################
#Hackerrun para hacer ejercicios de programación
################################################

import matplotlib.pyplot as mat
"""
y = [4,1,2,5,8.7]
x = range(1, len(y)+1)
mat.plot(x, y)
mat.xlabel('I am x axis')
mat.ylabel('I am y axis')
mat.title('Example 1')
mat.show()
"""
###########################

max_val = 5.
t = np.arange(0., max_val+0.5, 0.5)
mat.plot(t, t, 'r-', label='linear')
mat.plot(t, t**2, 'b--', label='quadratic')
mat.plot(t, t**3, 'g-.', label='cubic')
mat.plot(t, 2**t, 'y:', label='exponential')
mat.xlabel('I am x axis')
mat.ylabel('I am y axis')
mat.title('Example 2')
mat.legend()
mat.axis([0,max_val,0,2**max_val])
mat.show()
