# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Estudiante: JJavier Alonso Ramos

"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

#Función 1.2 - (np.e**v*u**2-2*v**2*np.e**(-u))**2
def E(u,v):
    return (np.e**v*u**2-2*v**2*np.e**(-u))**2

#Derivada parcial de E con respecto a u - 2*(np.e**v*u**2-2*v**2*np.e**(-u))*(2*v**2*np.e**(-u)+2*np.e**v*u)
def dEu(u,v):
    return 2*(np.e**v*u**2-2*v**2*np.e**(-u))*(2*v**2*np.e**(-u)+2*np.e**v*u)

#Derivada parcial de E con respecto a v - 2*(u**2*np.e**v-4*np.e**(-u)*v)*(u**2*np.e**v-2*np.e**(-u)*v**2)
def dEv(u,v):
    return 2*(u**2*np.e**v-4*np.e**(-u)*v)*(u**2*np.e**v-2*np.e**(-u)*v**2)

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#Función 1.3 - u**2+2*v**2+2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v)
def F(u,v):
    return u**2+2*v**2+2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v)

#Derivada parcial de F con respecto a u - 4*np.pi*np.sin(2*np.pi*v)*np.cos(2*np.pi*u)+2*u
def dFu(u,v):
    return 4*np.pi*np.sin(2*np.pi*v)*np.cos(2*np.pi*u)+2*u

#Derivada parcial de F con respecto a v - 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v)+4*v
def dFv(u,v):
    return 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v)+4*v

#Gradiente de F
def gradF(u,v):
    return np.array([dFu(u,v), dFv(u,v)])


################################################################################################
######################################## 1.1 ###################################################
################################################################################################

def gradient_descent(func,grad,u,v,maxIter,epsilon=np.NINF,learning_rate=0.01):
	"""
	Gradiente Descendente
	Aceptamos como parámetro un punto mínimo (epsilon)
	Aceptamos como parámetro el número máximo de iteraciones a realizar
	Aceptamos como parámetro un learning-rate que por defecto será 0.01
	"""

	#Creamos un contador de iteraciones
	it = 0
	#Creamos una lista donde guardar todas las aproximaciones que realiza el algoritmo
	points2min = []
	"""
	Creamos una variable donde guardaremos el último valor de Z obtenido
		con el fin de acabar el algoritmo si es necesario
	Creamos también un booleano para indicar la salida del bucle
	"""
	continuar = True
	last_z=np.Inf

	"""
	Realizamos el cálculo de un nuevo punto
		hasta alcanzar nuestro mínimo objetivo(epsilon)
		o hasta superar el máximo de iteraciones
	"""
	while func(u,v) > epsilon and it < maxIter and continuar:
    	# Calculamos las pendientes respecto a u e v
		_pend = grad(u,v)

		# Calculamos el nuevo punto más próximo al mínimo local
		u = u - learning_rate*_pend[0]
		v = v - learning_rate*_pend[1]

		"""
		La dupla w guardará y devolverá las coordenas (u,v) del último valor calculado,
			es decir, el valor mínimo alcanzado
		points2min almacena todas las coordenadas (u,v) de los puntos que se han ido calculando
		"""
		w = [u,v]
		points2min.append([u,v])
		# Almacenamos la "altura" de todos los puntos (u,v) calculados
		new_z = func(u,v)

		if new_z < last_z:
			last_z = new_z
		else:
			continuar = False

		# Aumentamos el número de iteraciones realizadas
		it = it+1

	# Devolvemos las coordenadas (x,y) del punto mínimo alcanzado
	# junto con el nº de iteraciones y todos los valores que se han ido recorriendo
	return w, it, points2min

################################################################################################
######################################## 1.2 ###################################################
################################################################################################

#Declaramos el punto inicial
initial_point_E = np.array([1.0,1.0])

"""
Realizamos el algoritmo del Gradiente Descendiente para la función E partiendo del punto (1,1)
Como tope de iteraciones indicamos 10000000000
Como altura mínima a encontrar marcamos 1e-14

En w guardamos las coordenadas (x,y) del punto con z mínimo alcanzado
En it almacenamos el número de iteraciones que han sido necesarias para calcular w
En points2min guardamos la secuencia de (x,y) que se ha ido generando hasta llegar a w
"""
w, it, points2min = gradient_descent(E,gradE,initial_point_E[0], initial_point_E[1],10000000000,1e-14)


# Mostramos por pantalla los datos más relevantes de aplicar el algoritmo a la función E
print ('Función E')
print ('Punto inicial: (', initial_point_E[0], ', ', initial_point_E[1], ')' )
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Valor mínimo en estas coordenadas: ', E(w[0], w[1]), '\n')

# Creamos una gráfica con los valores de Z para cada una de las iteraciones
valores_z = []
for punto in points2min:
	valores_z.append(E(punto[0], punto[1]))
figura = 'Ejercicio 1.2. Valor de Z en las distintas iteraciones del algoritmo'
titulo = 'Punto inicial: ('+ str(initial_point_E[0])+ ', '+ str(initial_point_E[1])+ ')'
subtitulo = 'Función E'
plt.figure(figura)
plt.title(titulo)
plt.suptitle(subtitulo)
plt.xlabel('iteraciones')
plt.ylabel('z')
plt.plot(valores_z)
plt.show()

"""
Creamos una figura 3D donde pintaremos la función para un conjunto de valores y marcaremos el mínimo encontrado
	con una estrella roja.
"""
# Importamos el módulo para hacer el gráfico 3D
from mpl_toolkits.mplot3d import Axes3D
# Tomamos 50 valores entre [-30,30] para la representación del gráfico
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
# Calculamos los valores de z para los (x,y) obtenidos antes
Z = E(X, Y) #E_w([X, Y])
# Creamos la figura 3D y la dibujamos
figura = 'Ejercicio 1.2. Representacion 3D de la función E'
fig = plt.figure(figura)
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')

"""
Dibujamos el punto mínimo encontrado como una estrella roja,
	los puntos intermedios como puntos verdes
	y el punto inicial como una estrella blanca
"""
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ini_point = np.array([initial_point_E[0], initial_point_E[1]])
ini_point_ = ini_point[:, np.newaxis]
ax.plot(ini_point_[0], ini_point_[1], E(ini_point_[0], ini_point_[1]), 'r*', c='black')
for punto in points2min:
	point = np.array([punto[0], punto[1]])
	point_ = point[:, np.newaxis]
	ax.plot(point_[0], point_[1], E(point_[0], point_[1]), '.', c='green')
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', c='red')

# Ponemos título y nombre a los ejes de la gráfica
ax.set(title='Punto inicial: (' + str(initial_point_E[0]) + ', ' + str(initial_point_E[1]) + ')')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
# Imprimimos por pantalla el resultado
plt.show()

################################################################################################
######################################## 1.3 ###################################################
################################################################################################

# Creamos una tabla donde almacenaremos los distintos resultados del algoritmo dependiendo de nuestro punto de partida
tabla = []
# Como primera fila de la tabla ponemos un índice para indicar qué será cada columna que incorporemos después
tabla.append(['punto inicial','u','v','F(u,v)'])

# Realizamos el algoritmo para una lista de puntos iniciales
for initial_point_F in ([0.1,0.1],[2.1,-2.1],[-0.5,-0.5],[-1,-1],[22.0,22.0]):

	"""
	Realizamos el algoritmo del Gradiente Descendiente para la función F
		partiendo desde los puntos ([0.1,0.1],[1,1],[-0.5,-0.5],[-1,-1])
	Como tope de iteraciones indicamos 50

	En w guardamos las coordenadas (x,y) del punto con z mínimo alcanzado
	En it almacenamos el número de iteraciones que han sido necesarias para calcular w
	En points2min guardamos la secuencia de (x,y) que se ha ido generando hasta llegar a w
	"""
	w, it, points2min = gradient_descent(F,gradF,initial_point_F[0], initial_point_F[1],50)

	# Incluimos en la tabla los resultados obtenidos
	tabla.append([tuple(initial_point_F), w[0],w[1],F(w[0], w[1])])

	"""
	Mostramos por pantalla los datos más relevantes de aplicar el algoritmo a la función F
		con punto inicial initial_point_F
	"""
	print ('Función F')
	print ('Punto inicial: (', initial_point_F[0], ', ', initial_point_F[1], ')' )
	print ('Numero de iteraciones: ', it)
	print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
	print ('Valor mínimo en estas coordenadas: ', F(w[0], w[1]), '\n\n')

	# Creamos una gráfica con los valores de Z para cada una de las iteraciones
	valores_z = []
	for punto in points2min:
		valores_z.append(F(punto[0], punto[1]))
	figura = 'Ejercicio 1.3. Valor de Z en las distintas iteraciones del algoritmo'
	titulo = 'Punto inicial: ('+ str(initial_point_F[0])+ ', '+ str(initial_point_F[1])+ ')'
	subtitulo = 'Función F'
	plt.figure(figura)
	plt.title(titulo)
	plt.suptitle(subtitulo)
	plt.xlabel('iteraciones')
	plt.ylabel('z')
	plt.plot(valores_z)
	plt.show()

	"""
	Creamos una figura 3D donde pintaremos la función para un conjunto de valores y marcaremos el mínimo encontrado
		con una estrella roja.
	"""
	# Importamos el módulo para hacer el gráfico 3D
	from mpl_toolkits.mplot3d import Axes3D
	# Tomamos 50 valores entre [-30,30] para la representación del gráfico
	x = np.linspace(-30, 30, 50)
	y = np.linspace(-30, 30, 50)
	X, Y = np.meshgrid(x, y)
	# Calculamos los valores de z para los (x,y) obtenidos antes
	Z = F(X, Y) #F_w([X, Y])
	# Creamos la figura 3D y la dibujamos
	figura = 'Ejercicio 1.3. Representacion 3D de la función F'
	fig = plt.figure(figura)
	ax = Axes3D(fig)
	surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
	                        cstride=1, cmap='jet')
	"""
	Dibujamos el punto mínimo encontrado como una estrella roja,
		los puntos intermedios como puntos verdes
		y el punto inicial como una estrella blanca
	"""
	min_point = np.array([w[0],w[1]])
	min_point_ = min_point[:, np.newaxis]
	ini_point = np.array([initial_point_F[0], initial_point_F[1]])
	ini_point_ = ini_point[:, np.newaxis]
	ax.plot(ini_point_[0], ini_point_[1], F(ini_point_[0], ini_point_[1]), 'r*', c='black')
	for punto in points2min:
		point = np.array([punto[0], punto[1]])
		point_ = point[:, np.newaxis]
		ax.plot(point_[0], point_[1], F(point_[0], point_[1]), '.', c='green')
	ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', c='red')
	# Ponemos título y nombre a los ejes de la gráfica
	ax.set(title='Punto inicial: (' + str(initial_point_F[0]) + ', ' + str(initial_point_F[1]) + ')')
	ax.set_xlabel('u')
	ax.set_ylabel('v')
	ax.set_zlabel('F(u,v)')
	# Imprimimos por pantalla el resultado
	plt.show()


print('   Tabla de datos con función F\n')
for i in range(len(tabla)):
	print(tabla[i])

input("\n--- Pulsar intro para continuar ---\n")

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