# -*- coding: utf-8 -*-
#https://www.youtube.com/watch?v=-_A_AAxqzCg
"""
Algoritmo de gradiente descendente aplicado a la función:
f(x,y) = sin(1/2*x² - 1/4*y² + 3)*cos(2x + 1 - e^y)
Sabemos que la derivada de f es:
f'(x) = 2x
El punto mínimo (x,y) de la funcion f es aquel que verifica:
f'(x) = 0
Este punto es:
f'(x) = 2x = 0; x = 0;
y = f(0) = 0² + 1 = 1;
Punto mínimo = (0,1)
"""

import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

"""
th es nuestro vector de parametros (x,y)
th[0] == x
th[1] == y
"""
#Declaración de nuestra funcion f(x,y)
func = lambda th: np.sin(1/2 * th[0]**2 - 1/4 * th[1]**2 + 3) * np.cos(2*th[0] + 1 - np.e**th[1])

#Declaramos una variable con el número de parámetros de entrada llamada resolucion (res)
res = 100
#Declaramos otra variable con el rango mínimo y otra con el rango máximo
r_min = -2
r_max = 2


#Creamos dos vectores (_X e _Y) donde guardaremos, respectivamente, los valores de cada parámetro
_X = np.linspace(r_min,r_max,res)
_Y = np.linspace(r_min,r_max,res)

#Creamos una matriz nula(_Z) donde guardaremos los valores de _X e _Y
_Z = np.zeros((res,res))

#Guardamos los valores resultantes de la funcion func en la matriz _Z
for ix, x in enumerate(_X):
	for iy, y in enumerate(_Y):
		"""
		Como los valores de X son los que nos indican las columnas y
		los valores de Y los valores de las filas pondremos los índices
		[iy,ix] para indexar correctamente los valaores en _Z
		"""
		_Z[iy, ix] = func([x,y])


"""
Mostramos en pantalla la planta (vista superior) del gráfico
formado por los puntos (x,y,z)
"""
plt.contourf(_X,_Y,_Z,100)
#Añadimos una barra como leyenda que nos explique los colores del grafo
plt.colorbar()

#Creamos un punto inicial en el grafo el cual debemos dirigir al punto mínimo
Theta = np.random.rand(2) * r_max*2 -r_max
#Theta = [-1.3, 0.5]

_T = np.copy(Theta)

plt.plot(Theta[0], Theta[1], "o", c="white")


h = 0.001
lr = 0.0001

grad = np.zeros(2)

for _ in range(50000):

	for it, th in enumerate(Theta):

		_T  =np.copy(Theta)

		_T[it] = _T[it] + h

		deriv = (func(_T) - func(Theta)) / h

		grad[it] = deriv

	Theta = Theta - lr * grad

	#print(func(Theta))

	if(_ % 100 == 0):
		plt.plot(Theta[0], Theta[1], ".", c="red")

plt.plot(Theta[0], Theta[1], "o", c="green")
plt.show()
#print(_Z)












