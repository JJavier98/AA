# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Estudiante: JJavier Alonso Ramos

"""

# Importamos módulo para trabajar con datos matemáticos
import numpy as np
# Importamos módulo para gráficos 2D
import matplotlib.pyplot as plt
# Importamos el módulo para formater tablas
import pandas as pd
# Importamos el módulo para hacer el gráfico 3D
from mpl_toolkits.mplot3d import Axes3D
# Importamos el módulo para generar números aleatorios
import random as rnd

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

# Función 1.2 - (np.e**v*u**2-2*v**2*np.e**(-u))**2
def E(u, v):
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

def gradient_descent(func,grad,u,v,maxIter,epsilon=1e-14,learning_rate=0.01, ejer1_2=False, ejer1_3a=False):
	"""
	Gradiente Descendente
	Aceptamos como parámetros:
	La fución sobre la que calcularemos el gradiente
	Las coordenadas con las que evaluaremos la función (u,v)
	El número máximo de iteraciones a realizar
	Un valor de Z mínimo (epsilon)
	Un learning-rate que por defecto será 0.01
	Los parámetros ejer* ayudan a la impresión o ejecución con factores determinados
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
	while it < maxIter and continuar:
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

		"""
		Si de una iteración a otra no se produce una mejora considerable
			salimos del bucle, en caso contrario, actualizamos
		"""
		if last_z - new_z > epsilon:
			last_z = new_z
		else:
			continuar = False

		# Si realizamos el ejercicio 1.2 y el valor calculado es menor que epsilon, salimos.
		if ejer1_2 and new_z < epsilon:
			continuar = False

		# En el ejercicio 1.3 iteramos hasta final de iteraciones.
		if(ejer1_3a):
			continuar=True

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
w, it, points2min = gradient_descent(E,gradE,initial_point_E[0], initial_point_E[1],10000000000,1e-14,ejer1_2=True)


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
Creamos una figura 3D donde pintaremos la función para un conjunto de valores
"""
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
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet', alpha=0.5)

"""
Dibujamos el punto mínimo encontrado como una estrella roja,
	los puntos intermedios como puntos verdes
	y el punto inicial como una estrella negra
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

input("\n--- Pulsar intro para continuar con el ejercicio 1.3 a) ---\n")

################################################################################################
###################################### 1.3 a) ##################################################
################################################################################################
"""
Vamos a comparar los resultados obtenidos aplicando el gradiente descendente al punto inicial (0.1,0.1)
con un learning-rate de 0.1 y 0.01
"""
#Inicializamos las columnas que indicarán el punto de inicio y el learning-rate utilizado en cada caso
columna1 = [[0.1,0.1],[0.1,0.1]]
columna2 = []
columna3 = []
columna4 = [0.01,0.1]
columna5 = []
columna6 = []

# Aplicamos el algoritmo para un lr=0.01 y almacenamos los resultados obtenidos en la tabla
w, it, points2min = gradient_descent(F,gradF,0.1,0.1,50)
columna2.append(w[0])
columna3.append(w[1])
columna5.append(F(w[0],w[1]))
columna6.append(it)

# Creamos una gráfica con los valores de Z para cada una de las iteraciones
valores_z = []
for punto in points2min:
	valores_z.append(F(punto[0], punto[1]))
figura = 'Ejercicio 1.3 a). Valor de Z para lr = 0.01'
titulo = 'Punto inicial: (0.1, 0.1)'
subtitulo = 'Función F'
plt.figure(figura)
plt.title(titulo)
plt.suptitle(subtitulo)
plt.xlabel('iteraciones')
plt.ylabel('z')
plt.plot(valores_z)
plt.show()

# Realizamos lo mismo pero para un lr=0.1
w, it, points2min = gradient_descent(F,gradF,0.1,0.1,50,learning_rate=0.1, ejer1_3a=True)
columna2.append(w[0])
columna3.append(w[1])
columna5.append(F(w[0],w[1]))
columna6.append(it)

# Creamos una gráfica con los valores de Z para cada una de las iteraciones
valores_z = []
for punto in points2min:
	valores_z.append(F(punto[0], punto[1]))
figura = 'Ejercicio 1.3 a). Valor de Z para lr = 0.1'
titulo = 'Punto inicial: (0.1, 0.1)'
subtitulo = 'Función F'
plt.figure(figura)
plt.title(titulo)
plt.suptitle(subtitulo)
plt.xlabel('iteraciones')
plt.ylabel('z')
plt.plot(valores_z)
plt.show()

# Creamos la tabla con los rersultados almacenados anteriormente y la imprimimos
dict_tabla = {'Initial Point':columna1, 'u':columna2, 'v':columna3, 'lr': columna4,
			'F(u,v)':columna5, 'iteraciones':columna6}
dataframe = pd.DataFrame(dict_tabla)

print('   Tabla de datos con función F\n')
print(dataframe)
print('\n\n')

input("\n--- Pulsar intro para continuar con el ejercicio 1.3 b) ---\n")

################################################################################################
###################################### 1.3 b) ##################################################
################################################################################################

# Creamos una tabla donde almacenaremos los distintos resultados del algoritmo dependiendo de nuestro punto de partida
# La crearemos como un objeto 'pandas' al que le pasaremos las columnas en el siguiente orden:
# punto incial - u - v - learning-rate - f(u,v) - iteraciones

columna1 = [[0.1,0.1],[1.0,1.0],[-0.5,-0.5],[-1,-1],[22.0,22.0]]
columna2 = []
columna3 = []
columna4 = [0.01, 0.01, 0.01, 0.01, 0.01]
columna5 = []
columna6 = []

# Realizamos el algoritmo para una lista de puntos iniciales
for initial_point_F in ([0.1,0.1],[1.0,1.0],[-0.5,-0.5],[-1,-1],[22.0,22.0]):

	"""
	Realizamos el algoritmo del Gradiente Descendiente para la función F
		partiendo desde los puntos ([0.1,0.1],[1,1],[-0.5,-0.5],[-1,-1], [22.0,22.0])
		He añadido el punto (22.0, 22.0) para obtener una gráfica en la que se vea más
		claramente el dibujo de los distintos puntos calculados hasta llegar al mínimo
	Como tope de iteraciones indicamos 50

	En w guardamos las coordenadas (x,y) del punto con z mínimo alcanzado
	En it almacenamos el número de iteraciones que han sido necesarias para calcular w
	En points2min guardamos la secuencia de (x,y) que se ha ido generando hasta llegar a w
	"""
	w, it, points2min = gradient_descent(F,gradF,initial_point_F[0], initial_point_F[1],50)

	# Incluimos en la tabla los resultados obtenidos
	####tabla.append([tuple(initial_point_F), w[0],w[1],F(w[0], w[1])])
	columna2.append(w[0])
	columna3.append(w[1])
	columna5.append(F(w[0],w[1]))
	columna6.append(it)

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
	Creamos una figura 3D donde pintaremos la función para un conjunto de valores
	"""
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
	surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet', alpha=0.5)
	"""
	Dibujamos el punto mínimo encontrado como una estrella roja,
		los puntos intermedios como puntos verdes
		y el punto inicial como una estrella negra
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


	input("\n--- Pulsar intro para continuar ---\n")

dict_tabla = {'Initial Point':columna1, 'u':columna2, 'v':columna3, 'lr': columna4,
			'F(u,v)':columna5, 'iteraciones':columna6}
dataframe = pd.DataFrame(dict_tabla)

print('   Tabla de datos con función F\n')
print(dataframe)

input("\n--- Pulsar intro para continuar con el ejercicio 2 ---\n")


###############################################################################
############################### EJERCICIO 2.1 #################################
###############################################################################

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2.1\n')

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
	"""
	Vamos a calcular el error cuadrático medio. Para ello:
		Dividiremos por el número de elementos que en este caso es igual al numero de filas de X
		Calculamos la diferencia entre nuestra aproximación y el valor real y la elevamos al cuadrado
		Realizamos la sumatoria de todas estas diferencias
		Hacemos la media y devolvemos el resultado
	"""
	denominador = len(x) # Número de elementos
	numerador = (np.dot(x,w)-y)**2 # Diferencia cuadrática
	numerador = np.sum(numerador) # Sumatoria de las diferencias
	res = numerador/denominador # Media del error cuadrático
	return res

# Gradiente Descendente Estocastico
def sgd(X,Y,epsilon = 1e-14, lr = 0.001):
	"""
	Gradiente Descendente Estocástico
	Calculamos el vector de pesos W
	Aceptamos como parámetros:
		Un conjunto de datos (muestra) a partir de los cuales debemos obtener los valores pasados como segundo argumento
		Un valor de error mínimo (epsilon) que marcará el final de la ejecución del algoritmo (por defecto será 1e-14)
		Un learning-rate que por defecto será 0.001
	"""
	size_of_x = len(X) # calculamos el número de filas que tiene X (el número de muestras)
	minibatch_size = 64 # establecemos un tamaño de minibatch
	minibatch_num = size_of_x // minibatch_size # calculamos el número de minibatchs en que podemos dividir X
	cols_of_x = len(X[0]) # calculamos el número de columnas de X (su número de características)
	w = np.zeros(cols_of_x) # inicializamos un vector de pesos a 0 con longitud = cols_of_x
	error_antiguo = 999.0 # inicializamos a un valor suficientemente alto para asegurarnos que entra en la condición de actualización de su valor
	continuar = True # inicializamos a true para que entre en el bucle

	# mientras la diferencia entre el anterior error calculado y el recién calculado sea mayor que 1e-14 continuamos realizando el algoritmo
	while(continuar):
		# recorremos todos los minibatchs en los que hemos dividido X
		for i in range(minibatch_num):
			# recorremos las características de X (sus columnas)
			for j in range(cols_of_x):
				# multiplicamos vectorialmente toda la submatriz de X que conforma un minibatch por su peso asociado
				h_x = np.dot(X[i*minibatch_size : (i+1)*minibatch_size, :],w)
				# restamos al valor obtenido su verdadero valor para ver cuanta precisión tenemos por ahora
				diff = h_x - Y[i*minibatch_size : (i+1)*minibatch_size]
				# multiplicamos individualmente la característica correspondiente a la columna j, fila a fila del minibatch, por la diferencia anterior
				mul = np.dot(X[i*minibatch_size : (i+1)*minibatch_size , j], diff)
				# realizamos la sumatoria de los valores obtenidos en el vector anterior (mul)
				sumatoria = np.sum(mul)
				# actualizamos w[j] (el peso de esa característica) restándole el valor anterior multiplicado por el learning rate
				w[j] = w[j] - lr*sumatoria

		"""
		si el número de filas de x no es múltiplo del tamaño del minibach sobrarán elementos en x que no se recorran con los bucles anteriores
			con esta condición nos aseguramos de recorrerlos
		"""
		if size_of_x % minibatch_size != 0:
			n = minibatch_num*minibatch_size
			for j in range(cols_of_x):
				h_x = np.dot(X[n : size_of_x, :],w)
				diff = h_x - Y[n : size_of_x]
				mul = np.dot(X[n : size_of_x , j], diff)
				sumatoria = np.sum(mul)
				w[j] = w[j] - lr*sumatoria

		# calculamos el error que obtenemos con una primera vuelta a los minibatchs
		error = Err(X,Y,w)
		# si todavía no llegamos a la precisión requerida repetimos el algoritmo
		if(error_antiguo - error > epsilon):
			error_antiguo = error
		# si hemos alcanzado la precisión requerida salimos
		else:
			continuar = False

	return w

# Pseudoinversa	
def pseudoinverse(X,Y):
	"""
	Calculamos el vector de pesos W
	Aceptamos como parámetros:
		La muestra que tenemos que acercar a los valores de Y por medio de W
		Los valores de Y
	"""
	px = np.linalg.pinv(X) # Calculamos la pseudoinversa de X por medio de una función del módulo numpy
	w = np.dot(px, Y) # Calculamos W multiplicando vectorialmente la pseudoinversa de X por los valores Y
	return w # Devolvemos el vector de pesos

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

#Imprimimos la matrix de muestra
print(x)
#Imprimimos el vector de valores reales a alcanzar
print(y)

# Calculamos el vector de pesos W por medio del Gradiente Descendente Estocástico
w_sgd = sgd(x,y)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w_sgd))
print ("Eout: ", Err(x_test, y_test, w_sgd))


# Calculamos el vector de pesos W por medio de la pseudoinversa de X
w_pinv = pseudoinverse(x,y)
print ('\n\n\nBondad del resultado para pseudoinversa de X:\n')
print ("Ein: ", Err(x,y,w_pinv))
print ("Eout: ", Err(x_test, y_test, w_pinv))



#Ponemos un título a la figura
figura = 'Ejercicio 2.1. Representacion 3D de las soluciones obtenidas con los datos usados en el ajuste'
# cremaos la figura
fig = plt.figure(figura)
ax = Axes3D(fig)

#Preparamos los datos para poder representarlos
x_11_ = np.array(x[np.where(y==1),1].T) # Valores de la columna 1 de x cuyo target es 1
x_1_1 = np.array(x[np.where(y==-1),1].T) # Valores de la columna 1 de x cuyo target es -1
x_21_ = np.array(x[np.where(y==1),2].T) # Valores de la columna 2 de x cuyo target es 1
x_2_1 = np.array(x[np.where(y==-1),2].T) # Valores de la columna 3 de x cuyo target es -1
#y_r1 = np.array(y[np.where(y==1)]) # Lista de targets == 1
#y_r_1 = np.array(y[np.where(y==-1)]) # Lista de targets == -1
y_ = x.dot(w_pinv) # Calculos de los targets de X a traves del vector de pesos W
y1_ = np.array(y_[np.where(y==1)])
y_1 = np.array(y_[np.where(y==-1)])


# Pintamos los puntos con target == 1 de rojo y los de target == -1 de cian
ax.plot(x_11_, x_21_, y1_, '.', c='r')
ax.plot(x_1_1, x_2_1, y_1, '.', c='c')
#ax.plot(x_11_, x_21_, y_r1, '.', c='C1', alpha=0.3)
#ax.plot(x_1_1, x_2_1, y_r_1, '.', c='C1', alpha=0.3)
# Ponemos título y nombre a los ejes de la gráfica
ax.set(title='Representacion 3D de las soluciones obtenidas con los datos usados en el ajuste')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')
# Imprimimos por pantalla el resultado
plt.show()


"""
Dibujamos en un diagrama de puntos la muestra y la separamos por medio de la recta:
	y = w[0] + w[1]*x1 + w[2]*x2
Los puntos que coincidan con una etiqueta igual a 1 los pintaremos de rojo mientras que los
	que tengan una etiqueta = -1 serán azul cian
Aplicaremos transparencia a estos puntos para ver más claramente la densidad de puntos
Dibujaremos las rectas asociadas a la regresión:
	De color azul la correspondiente al SGD
	De color magenta la correspondiente a la Pseudoinversa
"""
# Lo hacemos con W calculado con el gradiente descendente estocástico
plt.scatter(x[np.where(y==1),1], x[np.where(y==1),2], c='r', alpha=0.5)
plt.scatter(x[np.where(y==-1),1], x[np.where(y==-1),2], c='c', alpha=0.5)
plt.plot([0.0,1.0],[-w_sgd[0]/w_sgd[2], (-w_sgd[0]-w_sgd[1])/w_sgd[2]], c='dodgerblue')
plt.plot([0.0,1.0],[-w_pinv[0]/w_pinv[2], (-w_pinv[0]-w_pinv[1])/w_pinv[2]], c='magenta')
# Esrablecemos un título a la gráfica
plt.title(u'Gráfica de regresión lineal. Pesos calculados con SGD')
# La imprimimos
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2 a) ---\n")



###############################################################################
############################### EJERCICIO 2.2 #################################
###############################################################################


print('Ejercicio 2.2 a)\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

###############################################################################
############################## EJERCICIO 2.2 a) ###############################
###############################################################################


num_muestras = 1000
dimension = 2
size = 1
# Generamos una muestra de 100 puntos 2D en el cuadrado X = [−1, 1] × [−1, 1]
points = simula_unif(num_muestras, dimension, size)
# Dibujamos los puntos obtenidos en un plano
plt.plot(points[:,0], points[:,1], '.', c='c')
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2 b) ---\n")
###############################################################################
############################## EJERCICIO 2.2 b) ###############################
###############################################################################

# Declaramos la función f que usaremos para asignar una atiqueta a cada punto anterior
def f_sign(x1, x2):
    return np.sign((x1-0.2)**2 + x2**2 - 0.6)

# Calculamos el 10% de la muestra para saber cuántas etiquetas deben ser alteradas
porcentaje = 10
proporcion = porcentaje/100
num_muestras_a_alterar = int(proporcion*num_muestras)

# Generamos las etiquetas para cada coordenada
etiquetas = f_sign(points[:,0], points[:,1])
# Seleccionamos aleatoriamente los puntos que serán alterados
indices_alterar = rnd.sample(range(num_muestras),num_muestras_a_alterar)
# Agregamos las etiquetas a sus correspondientes coordenadas creando una matriz de tres atributos
matriz = np.c_[points,etiquetas]
# Agrupamos los puntos segun sus etiquetas para distinguirlos por colores
matriz_positiva = matriz[matriz[:,2]==1]
matriz_negativa = matriz[matriz[:,2]==-1]
# Dibujamos el mapa de puntos según etiquetas
plt.plot(matriz_positiva[:,0], matriz_positiva[:,1], '.', c='c')
plt.plot(matriz_negativa[:,0], matriz_negativa[:,1], '.', c='r')
plt.show()

# Alteramos la etiqueta del 10% de la muestra
matriz_alterada = matriz
for i in indices_alterar:
    matriz_alterada[i,2] = -matriz_alterada[i,2]

# Volvemos a calcular las matrices según su etiqueta
matriz_positiva = matriz[matriz[:,2]==1]
matriz_negativa = matriz[matriz[:,2]==-1]
# Dibujamos el mapa de puntos según etiquetas
plt.plot(matriz_positiva[:,0], matriz_positiva[:,1], '.', c='c')
plt.plot(matriz_negativa[:,0], matriz_negativa[:,1], '.', c='r')
plt.show()


input("\n--- Pulsar tecla para continuar al ejercicio 2.2 c) ---\n")
###############################################################################
############################## EJERCICIO 2.2 c) ###############################
###############################################################################

# Creamos el conjunto de muestra a partir de los puntos 2D aleatorios
muestra = points
# Como etiqueta usamos el signo obtenido al aplicar la función de signo del apartado b)
etiqueta_real  = matriz_alterada[:,2]
# Creamos una columna de unos para sacar el término independiente de W
independiente = np.ones_like(points[:,0])
# Terminamos de conformar la muestra añadiento esta columna de unos a la izquierda de los puntos 2D
muestra = np.c_[independiente, muestra]
# Calculamos los pesos W por medio del Gradiente Descendiente Estocástico
W = sgd(muestra, etiqueta_real)
# Nos sale un Ein muy alto devido a que, por la distribución de los puntos según su etiqueta,
# no podemos realizar un buen ajuste con un modelo lineal
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(muestra, etiqueta_real, W))


input("\n--- Pulsar tecla para continuar al ejercicio 2.2 d) ---\n")
###############################################################################
############################## EJERCICIO 2.2 d) ###############################
###############################################################################

# DUPLICAMOS CÓDIGO PARA TENER PERFECTAMENTE SEPARADOS LOS APARTADOS DEL EJERCICIO 

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Declaramos la función f que usaremos para asignar una atiqueta a cada punto anterior
def f_sign(x1, x2):
    return np.sign((x1-0.2)**2 + x2**2 - 0.6)

Error_in_med = 0.0
Error_out_med = 0.0

iteraciones = 1000

for i in range(iteraciones):
    num_muestras = 1000
    dimension = 2
    size = 1
    # Generamos una muestra de 100 puntos 2D en el cuadrado X = [−1, 1] × [−1, 1]
    points = simula_unif(num_muestras, dimension, size)
    # Creamos un conjusto de test de 1000 valores para calcular Eout
    test = simula_unif(num_muestras, dimension, size)
    
    # No imprimimos las gráficas porque son muchas iteraciones y son inecesarias
    
    # Calculamos el 10% de la muestra para saber cuántas etiquetas deben ser alteradas
    porcentaje = 10
    proporcion = porcentaje/100
    num_muestras_a_alterar = int(proporcion*num_muestras)
    
    # Generamos las etiquetas para cada valor de points
    etiquetas = f_sign(points[:,0], points[:,1])
    # Generamos las etiquetas para cada valor de test
    etiquetas_test = f_sign(test[:,0], test[:,1])
    
    # Seleccionamos aleatoriamente los datos de points que serán alterados
    indices_alterar = rnd.sample(range(num_muestras),num_muestras_a_alterar)
    # Seleccionamos aleatoriamente los datos de test que serán alterados
    indices_alterar_test = rnd.sample(range(num_muestras),num_muestras_a_alterar)
    
    # Agregamos las etiquetas a sus correspondientes coordenadas de pointscreando una matriz de tres atributos
    matriz = np.c_[points,etiquetas]
    # Agregamos las etiquetas a sus correspondientes coordenadas de test creando una matriz de tres atributos
    matriz_test = np.c_[test,etiquetas_test]
    
    # No imprimimos las gráficas porque son muchas iteraciones y son inecesarias
    
    # Alteramos la etiqueta del 10% de la muestra y del test
    matriz_alterada = matriz
    matriz_alterada_test = matriz_test
    for i in indices_alterar:
        matriz_alterada[i,2] = -matriz_alterada[i,2]
        matriz_alterada_test[i,2] = -matriz_alterada_test[i,2]
    
    # No imprimimos las gráficas porque son muchas iteraciones y son inecesarias
    
    # Creamos el conjunto de muestra a partir de los puntos 2D aleatorios
    muestra = points
    # Creamos el conjunto de test a partir de los puntos 2D aleatorios
    Test = test
    
    # Como etiqueta usamos el signo obtenido al aplicar la función de signo del apartado b)
    etiqueta_real  = matriz_alterada[:,2]
    etiqueta_real_test  = matriz_alterada_test[:,2]
    
    # Creamos una columna de unos para sacar el término independiente de W
    independiente = np.ones_like(points[:,0])
    independiente_test = np.ones_like(test[:,0])
    
    # Terminamos de conformar la muestra añadiento esta columna de unos a la izquierda de los puntos 2D
    muestra = np.c_[independiente, muestra]
    Test = np.c_[independiente_test, Test]
    
    # Calculamos los pesos W por medio del Gradiente Descendiente Estocástico
    W = sgd(muestra, etiqueta_real)
    # Nos sale un Ein muy alto devido a que, por la distribución de los puntos según su etiqueta,
    # no podemos realizar un buen ajuste con un modelo lineal
    ei = Err(muestra, etiqueta_real, W)
    eo = Err(Test, etiqueta_real_test, W)
    # Sumamos para calcular posteriormente el error medio
    Error_in_med = Error_in_med + ei
    Error_out_med = Error_out_med + eo
    # No mostramos el error de manera individual para ahorrar tiempo en la ejecución
#    print ('Bondad del resultado para grad. descendente estocastico:\n')
#    print ("Ein: ", ei)
#    print ("Eout: ", eo)

# Calculamos el error medio
Error_in_med = Error_in_med/iteraciones
Error_out_med = Error_out_med/iteraciones
print ('\n\nError medio tras ' + str(iteraciones) + ' iteraciones:\n')
print ("Ein medio: ", Error_in_med)
print ("Eout medio: ", Error_out_med, '\n\n')


input("\n--- Pulsar tecla para continuar al ejercicio 3 ---\n")
###############################################################################
########################## BONUS - MÉTODO DE NEWTON ###########################
###############################################################################

"""
RECORDAMOS QUE CONTAMOS CON:
#Función 1.3 - u**2+2*v**2+2*sin(2*pi*u)*sin(2*pi*v)
def F(u,v):
    return u**2+2*v**2+2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v)

#Derivada parcial de F con respecto a u - 4*pi*sin(2*pi*v)*cos(2*pi*u)+2*u
def dFu(u,v):
    return 4*np.pi*np.sin(2*np.pi*v)*np.cos(2*np.pi*u)+2*u

#Derivada parcial de F con respecto a v - 4*pi*sin(2*pi*u)*cos(2*pi*v)+4*v
def dFv(u,v):
    return 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v)+4*v

#Gradiente de F
def gradF(u,v):
    return np.array([dFu(u,v), dFv(u,v)])
"""
def dFuu(u,v):
	# Segunda derivada respecto a u de F
	return 2-8*np.pi**2*np.sin(2*np.pi*v)*np.sin(2*np.pi*u)

def dFuv(u,v):
	# Segunda derivada de F primero respecto a u y luego respecto a v de F
	return 8*np.pi**2*np.cos(2*np.pi*u)*np.cos(2*np.pi*v)

def dFvv(u,v):
	# Segunda derivada respecto a v de F
	return 4-8*np.pi**2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v)

def dFvu(u,v):
	# Segunda derivada de F primero respecto a v y luego respecto a u de F
	return 8*np.pi**2*np.cos(2*np.pi*v)*np.cos(2*np.pi*u)

def Hessian(u,v):
	"""
	Función que devuelve la matriz Hessiana de orden dos
	segun las coordenadas (u, v) pasadas como argumentos.
	"""
	aux1 = np.array([dFuu(u,v), dFuv(u,v)])
	aux2 = np.array([dFvu(u,v), dFvv(u,v)])
	H = np.array([aux1, aux2])

	return H
    
def NewtonsMethod(func,grad,u,v,maxIter,epsilon=1e-14, learning_rate = 0.1):
	"""
	Aceptamos como parámetros:
	La fución sobre la que calcularemos el gradiente
	Las coordenadas con las que evaluaremos la función (u,v)
	El número máximo de iteraciones a realizar
	Un valor de Z mínimo (epsilon)
	Un learning-rate que por defecto será 0.1
	"""
	#Creamos un contador de iteraciones
	it = 0
	#Creamos una lista donde guardar todas las aproximaciones que realiza el algoritmo
	points2min = []
	"""
	Creamos una variable donde guardaremos el último valor de Z obtenido
		con el fin de acabar el algoritmo si es necesario
	Creamos también un booleano para indicar la salida del bucle
	Creamos una tupla donde almacenar las coordenadas de nuestro mínimo local alcanzado
	"""
	continuar = True
	last_z=np.Inf
	w=[u,v]

	"""
	Realizamos el cálculo de un nuevo punto
		hasta alcanzar nuestro mínimo objetivo(epsilon)
		o hasta superar el máximo de iteraciones
	"""
	while it < maxIter and continuar:
		# Calculamos las pendientes respecto a u y v
		_pend = grad(w[0],w[1])
		# Montamos la matriz Hessiana y calculamos su inversa
		H_inv = np.linalg.inv( Hessian(w[0],w[1]) )

		# Calculamos el nuevo punto más próximo al mínimo local con el método de Newton
		w = w - learning_rate*(np.dot(H_inv, _pend))

		#points2min almacena todas las coordenadas (u,v) de los puntos que se han ido calculando
		points2min.append( [ w[0],w[1] ] )
		# Calculamso la "altura" del nuevo punto
		new_z = func(w[0],w[1])

		# Comprobamos que la diferencia entre los puntos calculados sea mayor que epsilon para seguir con el algoritmo
		if last_z - new_z > epsilon:
			last_z = new_z
		else:
			continuar = False

		# Aumentamos el número de iteraciones realizadas
		it = it+1

	# Devolvemos las coordenadas (x,y) del punto mínimo alcanzado
	# junto con el nº de iteraciones y todos los valores que se han ido recorriendo
	return w, it, points2min


# Creamos una tabla donde almacenaremos los distintos resultados del algoritmo dependiendo de nuestro punto de partida
# La crearemos como un objeto 'pandas' al que le pasaremos las columnas en el siguiente orden:
# punto incial - u - v - lr - f(u,v) - it
columna1 = [[0.1,0.1],[1.0,1.0],[-0.5,-0.5],[-1,-1]]
columna2 = []
columna3 = []
columna4 = [0.1, 0.1, 0.1, 0.1]
columna5 = []
columna6 = []

# Realizamos el algoritmo para una lista de puntos iniciales
for initial_point_F in ([0.1,0.1],[1.0,1.0],[-0.5,-0.5],[-1,-1]):

	"""
	Realizamos el algoritmo del Gradiente Descendiente para la función F
		partiendo desde los puntos ([0.1,0.1],[1,1],[-0.5,-0.5],[-1,-1])
	Como tope de iteraciones indicamos 50

	En w guardamos las coordenadas (x,y) del punto con z mínimo alcanzado
	En it almacenamos el número de iteraciones que han sido necesarias para calcular w
	En points2min guardamos la secuencia de (x,y) que se ha ido generando hasta llegar a w
	"""
	w, it, points2min = NewtonsMethod(F,gradF,initial_point_F[0], initial_point_F[1],50)

	# Incluimos en la tabla los resultados obtenidos
	####tabla.append([tuple(initial_point_F), w[0],w[1],F(w[0], w[1])])
	columna2.append(w[0])
	columna3.append(w[1])
	columna5.append(F(w[0],w[1]))
	columna6.append(it)

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
	figura = 'Ejercicio 3. Valor de Z en las distintas iteraciones del algoritmo'
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
	Creamos una figura 3D donde pintaremos la función para un conjunto de valores
	"""
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
	surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet', alpha=0.5)
	"""
	Dibujamos el punto mínimo encontrado como una estrella roja,
		los puntos intermedios como puntos verdes
		y el punto inicial como una estrella negra
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


	input("\n--- Pulsar intro para continuar ---\n")

dict_tabla = {'Initial Point':columna1, 'u':columna2, 'v':columna3, 'lr': columna4,
			'F(u,v)':columna5, 'iteraciones':columna6}
dataframe = pd.DataFrame(dict_tabla)

print('   Tabla de datos con función F\n')
print(dataframe)

input("\n--- Finalizar ---\n")
