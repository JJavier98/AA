# -*- coding: utf-8 -*-
"""
TRABAJO 2. 
Nombre Estudiante: 
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)

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

# Declaramos funciones auxiliares para el cálculo de parámetros.
recta_y = lambda a,b,x: a*x + b
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
titulo = 'Puntos según etiqueta'
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
titulo = 'Puntos según etiqueta - 10% de ruido'
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

f = []
f.append(f1)
f.append(f2)
f.append(f3)
f.append(f4)

titulo = 'Puntos según etiqueta fi'
plt.title(titulo)
i=1
for func in f:
	# Generamos 100 puntos en el intervalo [-50,50] para generar la función
	eje_x_recta = np.linspace(-50,50,100)
	eje_y_recta = np.linspace(-50,50,100)
	X,Y = np.meshgrid(eje_x_recta, eje_y_recta)
	Z = func(X,Y)

	plt.subplot(2,2,i)
	plt.scatter(datos_completos_ruido[datos_completos_ruido[:,2]<0,0], datos_completos_ruido[datos_completos_ruido[:,2]<0,1], c='c')
	plt.scatter(datos_completos_ruido[datos_completos_ruido[:,2]>0,0], datos_completos_ruido[datos_completos_ruido[:,2]>0,1], c='r')
	plt.contour(X,Y,Z,[0])
	i+=1

plt.show()

nuevos_datos = datos_completos
titulo = 'Puntos según etiqueta 2b - 10% de ruido'
plt.title(titulo)
i=1
for func in f:
	# Generamos 100 puntos en el intervalo [-50,50] para generar la función
	eje_x_recta = np.linspace(-50,50,100)
	eje_y_recta = np.linspace(-50,50,100)
	X,Y = np.meshgrid(eje_x_recta, eje_y_recta)
	Z = func(X,Y)

	nuevos_datos[:,2] = np.sign( func(nuevos_datos[:,0], nuevos_datos[:,1]) )

	plt.subplot(2,2,i)
	plt.scatter(nuevos_datos[nuevos_datos[:,2]<0,0], nuevos_datos[nuevos_datos[:,2]<0,1], c='c')
	plt.scatter(nuevos_datos[nuevos_datos[:,2]>0,0], nuevos_datos[nuevos_datos[:,2]>0,1], c='r')
	plt.contour(X,Y,Z,[0])
	i+=1

plt.show()

input("\n--- Pulsar Intro para continuar con el ejercicio 2.1 ---\n")

################################################################################################
######################################## 2.1 ###################################################
################################################################################################

def Err(x,y,w):
	error = 0
	for i in range(x.shape[0]):
		y_calculated = np.sign( np.dot(w.T, x[i]) )
		if y_calculated != y[i]:
			error += 1

	return error

def ajusta_PLA(datos, label, max_iter, v_ini):
	fin = False
	w = np.append(np.array([1]), v_ini)
	datos_copy = np.c_[np.ones(datos.shape[0]), datos]
	it = 0
	error = []

	while it < max_iter and not fin:
		it += 1
		for i in range(datos_copy.shape[0]):
			y_calculated = np.sign( np.dot(w.T, datos_copy[i]) )
			if y_calculated != label[i]:
				w = w+label[i]*datos_copy[i]
			e = Err(datos_copy, label, w)
			error.append(e)
			if e == 0:
				fin = True
				break

	w_max = np.max(w)
	w[ w < 0.0] = 0.0
	w /= w_max

	return w, it, error

# Utilizamos muestra_de_puntos y lista_etiquetas
# a) w=0
w=np.zeros(muestra_de_puntos.shape[1])
w, i, e= ajusta_PLA(muestra_de_puntos, lista_etiquetas, np.Inf, w)

print('W alcanzado en ', i, 'iteraciones:\n', w)
titulo = 'Evolución del error'
plt.figure(titulo)
plt.plot(e)
plt.show()

it = 0
for i in range(10):
	w=np.random.rand(muestra_de_puntos.shape[1])
	w, i , e= ajusta_PLA(muestra_de_puntos, lista_etiquetas, np.Inf, w)
	it += i

	print('W alcanzado en ', i, 'iteraciones:\n', w)
	titulo = 'Evolución del error'
	plt.figure(titulo)
	plt.plot(e)
	plt.show()
print('Iteraciones medias hasta converger: ', it/10)

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