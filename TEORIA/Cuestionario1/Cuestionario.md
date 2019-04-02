---
author: José Javier Alonso Ramos
date: \today
geometry: margin=2cm
title: "Aprendizaje Automático : Cuestionario 1"
header-includes: |
  \usepackage{stmaryrd}
---

# Preguntas

1. Identificar, para cada una de las siguientes tareas, cual es el problema, que tipo de
aprendizaje es el adecuado (supervisado, no supervisado, por refuerzo) y los elementos de
aprendizaje ($X , f, Y$) que deberíamos usar en cada caso. Si una tarea se ajusta a más de
un tipo, explicar como y describir los elementos para cada tipo.

a) Clasificación automática de cartas por distrito postal:



b) Decidir si un determinado índice del mercado de valores subirá o bajará dentro de un
periodo de tiempo fijado.

c) Hacer que un dron sea capaz de rodear un obstáculo.

d) Dada una colección de fotos de perros, posiblemente de distintas razas, establecer
cuantas razas distintas hay representadas en la colección.



2. ¿Cuales de los siguientes problemas son más adecuados para una aproximación por
aprendizaje y cuales más adecuados para una aproximación por diseño? Justificar la decisión

a) Determinar si un vertebrado es mamífero, reptil, ave, anfibio o pez.

b) Determinar si se debe aplicar una campaña de vacunación contra una enfermedad.

c) Determinar perfiles de consumidor en una cadena de supermercados.

d) Determinar el estado anímico de una persona a partir de una foto de su cara.

e) Determinar el ciclo óptimo para las luces de los semáforos en un cruce con mucho
tráfico.



3. Construir un problema de aprendizaje desde datos para un problema de clasificación de
fruta en una explotación agraria que produce mangos, papayas y guayabas. Identificar los
siguientes elementos formales $X,Y,D,f$ del problema. Dar una descripción de los mismos
que pueda ser usada por un computador. ¿Considera que en este problema estamos ante
un caso de etiquetas con ruido o sin ruido? Justificar las respuestas.



4. Suponga una matriz cuadrada A que admita la descomposición $A = X^TX$ para alguna
matriz $X$ de números reales. Establezca una relación entre los valores singulares de las
matriz $A$ y los valores singulares de $X$.



5. Sean x e y dos vectores de características de dimensión M × 1. La expresión

$$\operatorname{cov}(\mathbf{x}, \mathbf{y})=\frac{1}{M} \sum_{i=1}^{M}\left(x_{i}-\overline{x}\right)\left(y_{i}-\overline{y}\right)$$

define la covarianza entre dichos vectores, donde $\overline{z}$ representa el valor medio de los elementos
de $z$. Considere ahora una matriz X cuyas columnas representan vectores de características.
La matriz de covarianzas asociada a la matriz $X = (x_1, x_2, \dots , x_N )$ es el conjunto de
covarianzas definidas por cada dos de sus vectores columnas. Es decir,

$$ \operatorname{cov}(\mathrm{X})=\left( \begin{array}{cccc}{\operatorname{cov}\left(\mathbf{x}_{1}, \mathbf{x}_{1}\right)} & {\operatorname{cov}\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right)} & {\cdots} & {\operatorname{cov}\left(\mathbf{x}_{1}, \mathbf{x}_{N}\right)} \\ {\operatorname{cov}\left(\mathbf{x}_{2}, \mathbf{x}_{1}\right)} & {\operatorname{cov}\left(\mathbf{x}_{2}, \mathbf{x}_{2}\right)} & {\cdots} & {\operatorname{cov}\left(\mathbf{x}_{2}, \mathbf{x}_{N}\right)} \\ {\cdots} & {\cdots} & {\cdots} & {\cdots} \\ {\operatorname{cov}\left(\mathbf{x}_{N}, \mathbf{x}_{1}\right)} & {\operatorname{cov}\left(\mathbf{x}_{N}, \mathbf{x}_{2}\right)} & {\cdots} & {\operatorname{cov}\left(\mathbf{x}_{N}, \mathbf{x}_{N}\right)}\end{array}\right) $$


Sea ${1_M}^T$ = $(1, 1, \dots , 1)$ un vector $M \times 1$ de unos. Mostrar que representan las siguientes
expresiones:

a) $E 1=11^{T} \mathrm{X}$

b) $E 2=\left(\mathrm{X}-\frac{1}{M} E 1\right)^{T}\left(\mathrm{X}-\frac{1}{M} E 1\right)$



6. Considerar la matriz hat definida en regresión,
$\hat{\mathrm{H}}=\mathrm{X}\left(\mathrm{X}^{\mathrm{T}} \mathrm{X}\right)^{-1} \mathrm{X}^{\mathrm{T}}$
donde X es la matriz de observaciones de dimensión $N \times (d + 1)$, y $\mathrm{X}^T\mathrm{X}$ es invertible.

a) ¿Que representa la matriz $\hat{\mathrm{H}}$ en un modelo de regresión?

b) Identifique la propiedad más relevante de dicha matriz en relación con regresión lineal.



7. La regla de adaptación de los pesos del Perceptrón $(w_{new} = w_{old} + y_x)$ tiene la interesante
propiedad de que mueve el vector de pesos en la dirección adecuada para clasificar x de
forma correcta. Suponga el vector de pesos $w$ de un modelo y un dato $x(t)$ mal clasificado
respecto de dicho modelo. Probar matematicámente que el movimiento de la regla de
adaptación de pesos siempre produce un movimiento de $w$ en la dirección correcta para
clasificar bien $x(t)$.



8. Sea un problema probabilístico de clasificación binaria con etiquetas $\{0,1\}$, es decir
$P(Y = 1) = h(x)$ y $P(Y = 0) = 1 − h(x)$, para una función $h()$ dependiente de la muestra

a) Considere una muestra i.i.d. de tamaño N $(x_1, \dots , x_N)$. Mostrar que la función $h$
que maximiza la verosimilitud de la muestra es la misma que minimiza.

$$E_{\mathrm{in}}(\mathbf{w})=\sum_{n=1}^{N}\llbracket y_{n}=1 \rrbracket \ln \frac{1}{h\left(\mathbf{x}_{n}\right)}+\llbracket y_{n}=0 \rrbracket \ln \frac{1}{1-h\left(\mathbf{x}_{n}\right)}$$

donde $\llbracket \cdot \rrbracket$ vale 1 o 0 según que sea verdad o falso respectivamente
la expresión en su interior.


b) Para el caso $h(x) = \sigma(w^Tx)$ mostrar que minimizar el error de la muestra en el
apartado anterior es equivalente a minimizar el error muestral

$$E_{\mathrm{in}}(\mathbf{w})=\frac{1}{N} \sum_{n=1}^{N} \ln \left(1+e^{-y_{n} \mathbf{w}^{T} \mathbf{x}_{n}}\right)$$



9. Derivar el error $E_{in}$ para mostrar que en regresión logística se verifica:

$$\nabla E_{\mathrm{in}}(\mathbf{w})=-\frac{1}{N} \sum_{n=1}^{N} \frac{y_{n} \mathbf{x}_{n}}{1+e^{y_{n} \mathbf{w}^{T} \mathbf{x}_{n}}}=\frac{1}{N} \sum_{n=1}^{N}-y_{n} \mathbf{x}_{n} \sigma\left(-y_{n} \mathbf{w}^{T} \mathbf{x}_{n}\right)$$

Argumentar sobre si un ejemplo mal clasificado contribuye al gradiente más que un ejemplo
bien clasificado.



10. Definamos el error en un punto $(x_n, y_n)$ por

$$\mathbf{e}_{n}(\mathbf{w})=\max \left(0,-y_{n} \mathbf{w}^{T} \mathbf{x}_{n}\right)$$


Argumentar si con esta función de error el algoritmo PLA puede interpretarse como SGD
sobre $\mathbf{e}_n$ con tasa de aprendizaje $ν = 1$.



# BONUS

1. (2 puntos) En regresión lineal con ruido en las etiquetas, el error fuera de la muestra
para una $h$ dada puede expresarse como

$$E_{\mathrm{out}}(h)=\mathbb{E}_{\mathbf{x}, y}\left[(h(\mathbf{x})-y)^{2}\right]=\iint(h(\mathbf{x})-y)^{2} p(\mathbf{x}, y) d \mathbf{x} d y$$

a) Desarrollar la expresión y mostrar que

$$E_{\mathrm{out}}(h)=\int\left(h(\mathbf{x})^{2} \int p(y | \mathbf{x}) d y-2 h(\mathbf{x}) \int y \cdot p(y | \mathbf{x}) d y+\int y^{2} p(y | \mathbf{x}) d y\right) p(\mathbf{x}) d \mathbf{x}$$


b) El término entre paréntesis en $E_{out}$ corresponde al desarrollo de la expresión

$$\int(h(\mathbf{x})-y)^{2} p(y | \mathbf{x}) d y$$

¿Que mide este término para una $h$ dada?.


c) El objetivo que se persigue en Regression Lineal es encontrar la función $h \in \mathcal{H}$ que
minimiza $E_{out}(h)$. Verificar que si la distribución de probabilidad $p(x, y)$ con la que
extraemos las muestras es conocida, entonces la hipótesis óptima $h^{*}$ que minimiza
$E_{out}(h)$ está dada por


$$h^{*}(\mathbf{x})=\mathbb{E}_{y}[y | \mathbf{x}]=\int y \cdot p(y | \mathbf{x}) d y$$


d) ¿Cuál es el valor de $E_{out}(h^{*})$?


e) Dar una interpretación, en términos de una muestra de datos, de la definición de la
hipótesis óptima.



2. (1 punto) Una modificación del algoritmo perceptrón denominada ADALINE, incorpora en
la regla de adaptación una ponderación sobre la cantidad de movimiento necesaria. En PLA se
aplica $w_{new} = w_{old} + y_n x_n$ y en ADALINE se aplica la regla $w_{new} = w_{old} + \eta (y_n − w^T x_n) x_n$.
Considerar la función de error $En(w) = (max(0, 1 − y_n w^T x_n))^2$
. Argumentar que la regla
de adaptación de ADALINE es equivalente a gradiente descendente estocástico (SGD)
sobre $\frac{1}{N} \sum_{n=1}^{N} E_{n}(\mathbf{w})$.
