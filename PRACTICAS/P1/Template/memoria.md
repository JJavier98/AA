###### José Javier Alonso Ramos  
# Aprendizaje Automático  
## Práctica 1  
#### Ejercicio 1.- BÚSQUEDA ITERATIVA DE ÓPTIMOS

___  

1. ___Implementar algoritmo de gradiente Descendente___  

**Descripción:** con este algoritmo tratamos de avanzar hacia el mínimo local de una función más próximo a un punto inicial indicado.  

**Parámetros:**  

- __func:__ indicamos la función sobre la que se aplicará el gradiente descendiente.  
- __grad:__ es un array de dos componentes de las cuales, la primera contiene la derivada parcial respecto a la primera variable de _func_, y la segunda la derivada parcial respecto a la segunda variable de _func_.  
- __u:__ primera coordenada del punto inicial a partir del cual se buscará el mínimo local.  
- __v:__ segunda coordenada del punto inicial a partir del cual se buscará el mínimo local.  
- __maxIter:__ número máximo de iteraciones que realizará el algoritmo. Funciona como condición de parada en caso de no encontrar antes el mínimo local.  
- __epsilon:__ número suficientemente pequeño según el cual consideramos que no ha habido mejora de una iteración a otra del algoritmo. Si la diferencia entre la interpretación de la función _func_ en unas coordenas _(x,y)_ y la interpretación de otras coordenadas _(x',y')_ correspondientes a la siguiente iteración del algoritmo es menor que __epsilon__ paramos el cálculo del mínimo. Por defecto le asignamos un valor de $10^{-14}$.  
$$f(x,y)-f(x',y') < epsilon$$
- __learning_rate:__ el gradiente descendente nos indica hacia qué dirección debemos dirigirnos para tomar nuestra siguiente coordenada a considerar como mínimo. El learning_rate _(lr)_ indica en qué medida tomamos como válida esa dirección y, por tanto, cuánto avanzamos en esa dirección. Por defecto le asignamos un valor de $0.01$.


**Funcionamiento:** mientras no cumplamos el límite de iteraciones y por cada iteración obtengamos una mejora notable, actualizaremos nuestras coordenadas _(x,y)_ restándole el _gradiente_ calculado multiplicado por el _learning-rate_. $$(x',y')=(x,y)-lr*grad$$.
Al finalizar, el algoritmo devolverá:  

- __w:__ dupla con las coordenadas _(x,y)_ del punto mínimo alcanzado.  
- __it:__ número de iteraciones que han sido necesarias para alcanzar el mínimo.  
- __points2min:__ conjunto de puntos _(x,y)_ que hemos ido obteniendo en cada iteración del algoritmo.  

___  

2. ___Considerar la función $E(u,v)=(u^2e^v-2v^2e^{-u})^2$ . Usar gradiente descendente para encontrar un mínimo de esta función, comenzando desde el punto $(u,v)=(1,1)$ usando una tasa de aprendizaje $lr=0.01$.___  


- a) ___Calcular analíticamente y mostrar la expresión del gradiente de la función $E(u,v)$.___  

_Derivada de E respecto a 'u'_  

$${d\over du}[(u^2e^v-2v^2e^{-u})^2]$$ Aplicamos la regla de la potencia.  
$$2(u^2e^v-2v^2e^{-u}){d\over du}[u^2e^v-2v^2e^{-u}]$$ Derivamos los términos por eseparado.  
$$2(u^2e^v-2v^2e^{-u})(e^v {d\over du}[u^2] - 2v^2 {d\over du}[e^{-u}])$$ Resultado de la derivada parcial respecto _u_:  
$$2(u^2e^v-2v^2e^{-u})(2e^v u + 2v^2 e^{-u})$$  

---  

_Derivada de E respecto a 'v'_

$${d\over du}[(u^2e^v-2v^2e^{-u})^2]$$ Aplicamos la regla de la potencia.  
$$2(u^2e^v-2v^2e^{-u}){d\over du}[u^2e^v-2v^2e^{-u}]$$ Derivamos los términos por eseparado.  
$$2(u^2e^v-2v^2e^{-u})(u^2 {d\over du}[e^v] - 2e^{-u} {d\over du}[v^2])$$ Resultado de la derivada parcial respecto _v_:  
$$2(u^2e^v-2v^2e^{-u})(e^v u^2 - 4v e^{-u})$$  

---  

