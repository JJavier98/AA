# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Estudiante: JJavier Alonso Ramos

"""

# Importamos módulo para trabajar con datos matemáticos
import numpy as np
# Importamos módulo para gráficos 2D
import matplotlib.pyplot as plt
# Importamos el mñodulo para formater tablas
import pandas as pd
# Importamos el módulo para hacer el gráfico 3D
from mpl_toolkits.mplot3d import Axes3D

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
	Aceptamos como parámetros:
	La fución sobre la que calcularemos el gradiente
	Las coordenadas con las que evaluaremos la función (u,v)
	El número máximo de iteraciones a realizar
	Un valor de Z mínimo (epsilon)
	Un learning-rate que por defecto será 0.01
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

input("\n--- Pulsar intro para continuar con el ejercicio 1.3 ---\n")

################################################################################################
######################################## 1.3 ###################################################
################################################################################################

# Creamos una tabla donde almacenaremos los distintos resultados del algoritmo dependiendo de nuestro punto de partida
# La crearemos como un objeto 'pandas' al que le pasaremos las columnas en el siguiente orden:
# punto incial - u - v - f(u,v)

columna1 = [[0.1,0.1],[2.1,-2.1],[-0.5,-0.5],[-1,-1],[22.0,22.0]]
columna2 = []
columna3 = []
columna4 = []

# Realizamos el algoritmo para una lista de puntos iniciales
for initial_point_F in ([0.1,0.1],[2.1,-2.1],[-0.5,-0.5],[-1,-1],[22.0,22.0]):

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
	columna4.append(F(w[0],w[1]))

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


	input("\n--- Pulsar intro para continuar ---\n")

dict_tabla = {'Initial Point':columna1, 'u':columna2, 'v':columna3, 'F(u,v)':columna4}
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
	numerador = (x@w-y)**2 # Diferencia cuadrática
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
	w = px @ Y # Calculamos W multiplicando vectorialmente la pseudoinversa de X por los valores Y
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
"""
w = sgd(x,y)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
"""

# Calculamos el vector de pesos W por medio de la pseudoinversa de X
w = pseudoinverse(x,y)
print ('Bondad del resultado para pseudoinversa de X:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))


"""
#Ponemos un título a la figura
figura = 'Ejercicio 2.1. Representacion 3D de los datos según su target (1 ó -1)'
# cremaos la figura
fig = plt.figure(figura)
ax = Axes3D(fig)

#Preparamos los datos para poder representarlos
x_1_ = np.array(x[:,1])
x_2_ = np.array(x[:,2])
y_ = np.array(y)

ax.plot(x_1_, x_2_, y_, '.', c='c')
# Ponemos título y nombre a los ejes de la gráfica
ax.set(title='Data 3D')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')
# Imprimimos por pantalla el resultado
plt.show()
"""

"""
Dibujamos en un diagrama de puntos la muestra y la separamos por medio de la recta calculada
	por regresión lineal
Los puntos que coincidan con una etiqueta igual a 1 los pintaremos de rojo mientras que los
	que tengan una etiqueta = -1 serán azul cian
Aplicaremos transparencia a estos puntos para que sea más fácil comprender porqué la recta está
	más arriba de lo que en un principio esperamos. Esto es porque la densidad de puntos de etiqueta 1
	es muy alta.
"""
plt.scatter(x[np.where(y==1),1], x[np.where(y==1),2], c='r', alpha=0.5)
plt.scatter(x[np.where(y==-1),1], x[np.where(y==-1),2], c='c', alpha=0.5)
plt.plot([0.0,0.6],[w[0]+w[1]*0.1+w[2]*0.1, w[0]+w[1]*0.6+w[2]*0.6])
# Esrablecemos un título a la gráfica
plt.title(u'Gráfica de regresión lineal')
# La imprimimos
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2 ---\n")



###############################################################################
############################### EJERCICIO 2.2 #################################
###############################################################################


print('Ejercicio 2.2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...
