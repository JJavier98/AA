#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:52:21 2019

@author: jjavier98
"""

#Ejemplo de problema de aprendizaje automatico
from sklearn import datasets
iris=datasets.load_iris()

#Se comienza analizando los datos disponibles
iris


#Consulta de datos
"""
iris.future_names
iris.data
iris.target
"""
# Contamos cuantos elementos de cada tipo hay
import collections as coll
contadores = coll.Counter(iris.target)
# Mostramos el resultado de la operacion anterior
contadores

# Guardamos el numero de elementos segun su tipo en distintas variables
num_of_0 = contadores[0]
num_of_1 = contadores[1]
num_of_2 = contadores[2]
#Cogemos el 80% de cada tipo para formar nuestro grupo de entrnamiento
entr_0 = round(0.8*num_of_0)
entr_1 = round(0.8*num_of_1)
entr_2 = round(0.8*num_of_2)
# Cogemos los elementos restantes (20%) como grupo de prueba
prueb_0 = num_of_0-entr_0
prueb_1 = num_of_1-entr_1
prueb_2 = num_of_2-entr_2

entrenamiento = []
prueba = []

ceros = 0
unos = 0
doses = 0

for i in iris:
    if i.target == 0 and ceros < entr_0:
        entrenamiento.append(i)
        ceros += 1
    elif  i.target == 1 and unos < entr_1:
        entrenamiento.append(i)
        unos += 1
    elif i.target == 2 and doses < entr_2:
        entrenamiento.append(i)
        doses += 1


entrenamiento