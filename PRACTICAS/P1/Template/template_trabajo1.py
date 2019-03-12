# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: JJavier Alonso Ramos
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

#Función
def E(u,v):
    return (np.e**v*u**2-2*v**2*np.e**(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(np.e**v*u**2-2*v**2*np.e**(-u))*(2*v**2*np.e**(-u)+2*np.e**v*u)

#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**2*np.e**v-4*np.e**(-u)*v)*(u**2*np.e**v-2*np.e**(-u)*v**2)

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(u,v):
    #
    # gradiente descendente
    # 
	# Declaramos un error mínimo (epsilon)
	epsilon = 1e-14
	# Declaramos un máximo de iteraciones
	maxIter = 10000000000
	# Declaramos learning-rate
	lr = 0.01
	#Creamos un contador de iteraciones
	it = 0
	#Creamos una lista donde guardar todas las aproximaciones que realiza el algoritmo
	points2min = []

	"""
	Realizamos el cálculo de un nuevo punto
	hasta superar nuestra precisión epsilon
	o hasta alcanzar el máximo de iteraciones
	"""
	while E(u,v) > epsilon and it < maxIter:
    	#Calculamos las pendientes respecto a u e v
		_pend = gradE(u,v)

		#Calculamos el nuevo punto más próximo al mínimo local
		u = u - lr*_pend[0]
		v = v - lr*_pend[1]

		#Guardamos las coordenadas del punto calculado en una dupla
		w = [u,v]
		points2min.append([u,v,E(u, v)])

		#Aumentamos el número de iteraciones realizadas
		it = it+1

	return w, it, points2min

####################### MAIN #################################
#Declaramos el punto inicial
initial_point = np.array([1.0,1.0])

w, it, points2min = gradient_descent(initial_point[0], initial_point[1])


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', c='red')
#for i in range(len(points2min)):
	#ax.plot(points2min[i],points2min[i],points2min[i], '.', c='black')
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
#Añadimos una barra como leyenda que nos explique los colores del grafo
#plt.colorbar()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...




"""

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return 

# Gradiente Descendente Estocastico
def sgd(?):
    #
    return w

# Pseudoinversa	
def pseudoinverse(?):
    #
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...



"""