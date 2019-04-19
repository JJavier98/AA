# -*- coding: utf-8 -*-
"""
TRABAJO 2. 
Nombre Estudiante: 
"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.optimize as sp
from sklearn.preprocessing import normalize


# Fijamos la semilla
np.random.seed(8)

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

################################################################################################
######################################## 1.1 ###################################################
################################################################################################

# Obtenemos la nube de puntos según la función simula_unif
lista_puntos_unif = simula_unif(50,2,[-50,50])
# Obtenemos la nube de puntos según la función simula_gaus
lista_puntos_gaus = simula_gaus(50,2,[5,7])

#a) Mostramos puntos de simula_unif
titulo = 'Puntos de simula_unif'
plt.title(titulo)
plt.scatter(lista_puntos_unif[:,0], lista_puntos_unif[:,1], c='r', label='uniform')
plt.legend()
plt.show()

# b) Mostramos puntos de simula_gaus
titulo = 'Puntos de simula_gaus'
plt.title(titulo)
plt.scatter(lista_puntos_gaus[:,0], lista_puntos_gaus[:,1], c='c', label='gaus')
plt.legend()
plt.show()

# Comparamos puntos de simula_unif y gaus
titulo = 'Puntos de simula_unif y simula_gaus'
plt.title(titulo)
plt.scatter(lista_puntos_gaus[:,0], lista_puntos_gaus[:,1], c='c', label='gaus')
plt.scatter(lista_puntos_unif[:,0], lista_puntos_unif[:,1], c='r', label='unif')
plt.legend()
plt.show()

input("\n--- Pulsar Intro para continuar con el ejercicio 1.2 ---\n")

################################################################################################
######################################## 1.2 ###################################################
################################################################################################

# Coordenada y de los puntos de una recta
recta_y = lambda a,b,x: a*x + b
# Signo de la distancia de un punto a una recta
distancia_a_recta = lambda a,b,x,y: np.sign(y-a*x-b)

# a)
# Generamos la muestra de puntos mediante simula_unif
muestra_de_puntos = simula_unif(100,2,(-50,50))
# Generamos los coeficientes a,b de la recta y = ax + b
a,b = simula_recta((-50,50))
# Generamos dos puntos en el intervalo [-50,50] para generar la recta
eje_x_recta = np.linspace(-50,50,2)
# Calculamos las coordenadas del eje y para los dos puntos anteriores de la recta
eje_y_recta = recta_y(a,b,eje_x_recta)
# Calculamos las etiquetas asociadas a los puntos 2D
lista_etiquetas = distancia_a_recta(a,b,muestra_de_puntos[:,0],muestra_de_puntos[:,1])
# Hacemos corresponder cada etiqueta con su punto
datos_completos = np.c_[muestra_de_puntos, lista_etiquetas]
# Imprimimos los resultados
titulo = 'Puntos según etiqueta original'
plt.title(titulo)
plt.scatter(datos_completos[datos_completos[:,2]<0,0], datos_completos[datos_completos[:,2]<0,1], c='c', label='negativos')
plt.scatter(datos_completos[datos_completos[:,2]>0,0], datos_completos[datos_completos[:,2]>0,1], c='r', label='positivos')
plt.plot(eje_x_recta, eje_y_recta, 'k-',label='ax+b')
plt.legend()
plt.show()

# b)
# Dividimos los datos según sus etiquetas
_negativos = datos_completos[ datos_completos[:,2]<0, :]
_positivos = datos_completos[ datos_completos[:,2]>0, :]
# Mezclamos los conjuntos para asegurarnos la introducción de ruido aleatoria
np.random.shuffle(_negativos)
np.random.shuffle(_positivos)
# Aplicamos el 10% de ruido a ambos conjuntos (redondeamos cuánto es el 10% de cada grupo)
tope = round(_negativos.shape[0]*0.1)
for i in range(0,tope):
	_negativos[i,2] = -_negativos[i,2];

tope = round(_positivos.shape[0]*0.1)
for i in range(0,tope):
	_positivos[i,2] = -_positivos[i,2];
# Volvemos a juntarlos en un único conjunto y los volvemos a mezclar
datos_completos_ruido = np.r_[_negativos, _positivos]
np.random.shuffle(datos_completos_ruido)
# Imprimimos el resultado
titulo = 'Puntos según etiqueta con 10% de ruido'
plt.title(titulo)
plt.scatter(datos_completos_ruido[datos_completos_ruido[:,2]<0,0], datos_completos_ruido[datos_completos_ruido[:,2]<0,1], c='c', label='negativos')
plt.scatter(datos_completos_ruido[datos_completos_ruido[:,2]>0,0], datos_completos_ruido[datos_completos_ruido[:,2]>0,1], c='r', label='positivos')
plt.plot(eje_x_recta, eje_y_recta, 'k-',label='ax+b')
plt.legend()
plt.show()

input("\n--- Pulsar Intro para continuar con el ejercicio 1.3 ---\n")

################################################################################################
######################################## 1.3 ###################################################
################################################################################################

# Declaramos las distintas funciones
f1 = lambda x,y: (x-10)*(x-10)+(y-20)*(y-20)-400
f2 = lambda x,y: 0.5*(x+10)*(x+10)+(y-20)*(y-20)-400
f3 = lambda x,y: 0.5*(x-10)*(x-10)-(y+20)*(y+20)-400
f4 = lambda x,y: y-20*x*x-5*x+3
# Las incorporamos a una lista para hacer más sencillo su uso en un bucle
f = []
f.append(f1)
f.append(f2)
f.append(f3)
f.append(f4)

# Dibujamos las funciones con el etiquetado anterior
# Vemos que da igual cómo de compleja sea la función que no divide bien los datos
titulo = 'Puntos según etiqueta 2b - 10% de ruido'
plt.title(titulo)
i=1
for func in f:
	# Generamos 100 puntos (x,y) en el intervalo [-50,50] para generar la función
	eje_x_recta = np.linspace(-50,50,100)
	eje_y_recta = np.linspace(-50,50,100)
	X,Y = np.meshgrid(eje_x_recta, eje_y_recta)
	# Calculamos las coordenadas Z
	Z = func(X,Y)

	plt.subplot(2,2,i)
	plt.scatter(datos_completos_ruido[datos_completos_ruido[:,2]<0,0], datos_completos_ruido[datos_completos_ruido[:,2]<0,1], c='c')
	plt.scatter(datos_completos_ruido[datos_completos_ruido[:,2]>0,0], datos_completos_ruido[datos_completos_ruido[:,2]>0,1], c='r')
	plt.contour(X,Y,Z,[0])
	# Aumentamos el índice donde se colocará el gráfico en el subplot
	i+=1

plt.show()

# Vamos a dibujar de nuevo las funciones pero etiquetando los puntos según su distancia a la función que se evalúa en ese momento
nuevos_datos = datos_completos
titulo = 'Puntos según la etiqueta asignada por f'
plt.title(titulo)
i=1
for func in f:
	# Generamos 100 puntos (x,y) en el intervalo [-50,50] para generar la función
	eje_x_recta = np.linspace(-50,50,100)
	eje_y_recta = np.linspace(-50,50,100)
	X,Y = np.meshgrid(eje_x_recta, eje_y_recta)
	Z = func(X,Y)
	# Calculamos la etiqueta del punto
	nuevos_datos[:,2] = np.sign( func(nuevos_datos[:,0], nuevos_datos[:,1]) )

	plt.subplot(2,2,i)
	plt.scatter(nuevos_datos[nuevos_datos[:,2]<0,0], nuevos_datos[nuevos_datos[:,2]<0,1], c='c')
	plt.scatter(nuevos_datos[nuevos_datos[:,2]>0,0], nuevos_datos[nuevos_datos[:,2]>0,1], c='r')
	plt.contour(X,Y,Z,[0])
	# Aumentamos el índice donde se colocará el gráfico en el subplot
	i+=1

plt.show()

input("\n--- Pulsar Intro para continuar con el ejercicio 2.1 a) ---\n")

################################################################################################
######################################## 2.1 ###################################################
################################################################################################

# Función que nos indicará el error obtenido en el cálculo de etiquetas
def Err(x,y,w):
	error = 0
	for i in range(x.shape[0]):
		# Calculamos la etiqueta con el w pasado
		y_calculated = np.sign( np.dot(w.T, x[i]) )
		# Si es distinta aumentamos error
		if y_calculated != y[i]:
			error += 1

	return error

# Función de Perceptrón devolverá los coeficientes del hiperplano que divide los datos según sus etiquetas
def ajusta_PLA(datos, label, max_iter, v_ini):
	fin = False
	# Añadimos las columnas de unos a w y a los datos para obtener los términos independientes
	w = np.append(np.array([1]), v_ini)
	datos_copy = np.c_[np.ones(datos.shape[0]), datos]
	it = 0
	error = []

	# Iteramos hasta un máximo de iteraciones o hasta converger en error=0
	while it < max_iter and not fin:
		# Contamos como iteración el recorrer por completo el conjunto de datos
		it += 1
		for i in range(datos_copy.shape[0]):
			# Calculamos su etiqueta
			y_calculated = np.sign( np.dot(w.T, datos_copy[i]) )
			# Si es distinta corregimos w
			if y_calculated != label[i]:
				w = w+label[i]*datos_copy[i]
				# Comprobamos el error con el nuevo w
				e = Err(datos_copy, label, w)
				error.append(e)
				# Si el error es 0 hemos acabado
				if e == 0:
					fin = True
					break

	# Normalizamos los parámetros de w
	w_max = np.max(w)
	w[ w < 0.0] = 0.0
	w /= w_max

	return w, it, error

# Utilizamos muestra_de_puntos y lista_etiquetas
# a) w=0
w=np.zeros(muestra_de_puntos.shape[1])
# Calculamos w, el número de iteraciones necesario y la lista de errores que hemos ido obteniendo
# Indicamos como máximas iteraciones infinito porque los puntos son linealmente separables y sin ruido por lo que
#	va a acabar convergiendo
print('w inicial')
print(w)
w, i, e= ajusta_PLA(muestra_de_puntos, lista_etiquetas, np.Inf, w)

print('W alcanzado en ', i, 'iteraciones con w_inicial=[0,...,0]:\n', w)
titulo = 'Evolución del error'
plt.figure(titulo)
plt.plot(e, label='error')
plt.legend()

plt.show()

it = 0
for i in range(10):
	# Repetimos lo anterior pero con un w inicializado aleatoriamente 
	w=np.random.rand(muestra_de_puntos.shape[1])
	print('w inicial')
	print(w)
	# Indicamos como máximas iteraciones infinito porque los puntos son linealmente separables y sin ruido por lo que
	#	va a acabar convergiendo
	w, i, e= ajusta_PLA(muestra_de_puntos, lista_etiquetas, np.Inf, w)
	it += i

	print('W alcanzado en ', i, 'iteraciones con w_inicial=rand():\n', w)
	titulo = 'Evolución del error'
	plt.figure(titulo)
	plt.plot(e, label='error')
	plt.legend()
	plt.show()
print('Iteraciones medias hasta converger: ', it/10)

input("\n--- Pulsar Intro para continuar con el ejercicio 2.1 b) ---\n")

# b) Utilizamos datos_completos_ruido
# Separamos el conjunto datos y el conjunto etiquetas
datos = datos_completos_ruido[:,:-1]
etiquetas = datos_completos_ruido[:,-1]

# Repetimos lo que en el apartado anterior con los nuevos datos con ruido
w=np.zeros(datos.shape[1])
# Indicamos como máximas iteraciones 300 porque los puntos puede que NO sean linealmente separables ya que hay ruido y puede que
#	no acabe convergiendo

print('w inicial')
print(w)
w, i, e= ajusta_PLA(datos, etiquetas, 300, w)

print('W alcanzado en ', i, 'iteraciones con w_inicial=[0,...,0]:\n', w)
titulo = 'Evolución del error'
plt.figure(titulo)
plt.plot(e, label='error')
plt.legend()
plt.show()

it = 0
for i in range(10):
	# Repetimos lo anterior pero con un w inicializado aleatoriamente 
	w=np.random.rand(datos.shape[1])
	print('w inicial')
	print(w)
	# Indicamos como máximas iteraciones 300 porque los puntos puede que NO sean linealmente separables ya que hay ruido y puede que
	#	no acabe convergiendo
	w, i , e= ajusta_PLA(datos, etiquetas, 300, w)
	it += i

	print('W alcanzado en ', i, 'iteraciones con w_inicial=rand():\n', w)
	titulo = 'Evolución del error'
	plt.figure(titulo)
	plt.plot(e, label='error')
	plt.legend()
	plt.show()
print('Iteraciones medias hasta converger: ', it/10)

input("\n--- Pulsar Intro para continuar con el ejercicio 2.2 ---\n")

################################################################################################
######################################## 2.2 ###################################################
################################################################################################

# Funcione sigmoide
def sigmoid(x):
	return 1/(1+np.exp(-x))

# Aplica la función sigmoide al producto vectorial de pesos por datos
def h(x,w):
	return sigmoid(np.dot(x,w))

# Función de pérdida (error)
def loss(w, x, y):
	H = h(x, w)
	return -np.mean(y*np.log( H ) + (1-y)*np.log(1 - H ))

"""
def loss(w, X, Y):
	sumatoria = 0
	for i in range(len(Y)):
		sumatoria += np.log(1 + np.exp(-Y[i] * np.dot(w, X[i].T)))
	res = sumatoria / len(Y)
	return res
"""

# Función gradiente, derivada de función pérdida
def gradient(w, x, y):
	return (1/y.shape[0])*x.T.dot( h(x,w)-y )

# Regresión Logística con gradiente descendente
def lgr(X,Y,epsilon = 0.01, lr = 0.01, epocas = 500, minibatch_size = 1):
	# Creamos vector de pesos inicializado a cero
	w = np.zeros( (X.shape[1]) )
	# Juntamos datos y etiquetas
	matriz_completa = np.c_[ X,Y ]
	itera = 0
	errores = []

	# Realizamos tantas iteraciones como épocas indiquemos
	for i in range(epocas):
		# Guardamos una copia de w antes de modificarlo en esta época
		w_epoca_anterior = np.copy(w)
		# Mezclamos los datos
		np.random.shuffle(matriz_completa)
		etiquetas = np.reshape( matriz_completa[:,-1] ,(matriz_completa.shape[0]) )
		datos = matriz_completa[:,0:-1]
		for j in range(0, X.shape[0], minibatch_size):
			# modificamos los pesos con el gradiente
			grad = gradient(w, datos[j : j+minibatch_size, :], etiquetas[j : j+minibatch_size])
			w = w - lr*grad

		# Guardamos el error obtenido por cada época
		errores.append(loss(w,datos,etiquetas))
		# Si cumplimos la condición de parada salimos
		if(np.linalg.norm(w_epoca_anterior - w) < epsilon):
			break

		itera = itera+1

	return w_epoca_anterior, errores, itera

# Generamos los coeficientes a,b de la recta y = ax + b
a,b = simula_recta((0,2))
# Generamos la muestra de puntos mediante simula_unif
muestra_de_puntos = simula_unif(100,2,(0,2))
# Generamos dos puntos en el intervalo [0,2] para generar la recta
puntos_recta_x = [0,2]
puntos_recta_y = a*puntos_recta_x[0]+b, a*puntos_recta_x[1]+b
# Generamos las etiquetas
etiquetas_recta = distancia_a_recta(a,b,muestra_de_puntos[:,0], muestra_de_puntos[:,1])
etiquetas_recta[etiquetas_recta == -1] = 0

# Imprimimos los resultados
titulo = 'Puntos según etiqueta recta (0/1)'
plt.title(titulo)
plt.scatter(muestra_de_puntos[etiquetas_recta==0,0], muestra_de_puntos[etiquetas_recta==0,1], c='c', label='negativos')
plt.scatter(muestra_de_puntos[etiquetas_recta==1,0], muestra_de_puntos[etiquetas_recta==1,1], c='r', label='positivos')
plt.plot(puntos_recta_x, puntos_recta_y, 'k-',label='ax+b')
plt.legend()
plt.show()

# Añadimos una columna de unos al conjunto de entrenamiento
train = np.c_[np.ones( (muestra_de_puntos.shape[0]) ), muestra_de_puntos]
# Calculamos los pesos y obtenemos los errores y las iteraciones necesarias
w, e, i = lgr(train, etiquetas_recta)
print('w: ', w)

# Imprimimos los datos anteriores con la recta calculada con w
titulo = 'Puntos según etiqueta función (0/1)'
plt.title(titulo)
plt.scatter(muestra_de_puntos[etiquetas_recta==0,0], muestra_de_puntos[etiquetas_recta==0,1], c='c', label='negativos')
plt.scatter(muestra_de_puntos[etiquetas_recta==1,0], muestra_de_puntos[etiquetas_recta==1,1], c='r', label='positivos')
x = muestra_de_puntos[:,0]
y = (-w[0]-w[1]*x)/w[2]
plt.plot(x, y, 'k-',label='recta de regresión')
plt.legend()
plt.show()

# Imprimimos la evolución del error
titulo = 'Evolución del error al calcular los pesos tras ' + str(i) + ' iteraciones'
plt.title(titulo)
plt.plot(e)
plt.xlabel('iteraciones')
plt.ylabel('error')
plt.show()

# Calculamos el error dentro de la muestra con el vector de pesos hallado
print('Ein -> ', loss(w, train, etiquetas_recta))

# Calculamos el error medio obtenido en 1000 conjuntos test de 100 datos cada uno
losses = []
for i in range(1000):
	X = simula_unif(100, 2, (0, 2))
	Y = distancia_a_recta(a,b,X[:,0], X[:,1])
	Y[Y == -1] = 0
	X = np.c_[np.ones( (X.shape[0]) ), X]
	losses.append(loss(w, X, Y))

plt.title('Loss Measures')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(losses)
plt.show()
print('Loss Mean:', np.mean(losses))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO BONUS\n')

label4 = 1
label8 = -1

# Función para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 4 o la 8
	for i in range(0,datay.size):
		if datay[i] == 4 or datay[i] == 8:
			if datay[i] == 4:
				y.append(label4)
			else:
				y.append(label8)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y