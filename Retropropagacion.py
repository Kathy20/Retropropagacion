"""
Instituto Tecnologico de Costa Rica
Centro Academico de San Jose
Ingenieria en Computacion

Inteligencia Artificial

Estudiante: Kathy Brenes G.
Profesor: Jose Castro.
Fecha: 22 de abril del 2016
"""
######################################################
#              CONJUNTO DE APRENDIZAJE               #
##### La base de datos tiene 3 numeros al inicio:
# 1-> Cantidad de pares de entrenamiento
# 2-> Dimension de la entrada
# 3-> Dimension de la salida
##### Cada imagen esta separada por un vector que indica su posicion

"""S-> Pares de entrenamiento """
import random
import math


def newWeights(n,m):
    "Genera una nueva matriz de pesos nxm que conecta con la capa siguiente"
    weights= []
    for i in range(n):
        weight=[]
        for j in range(m):
            # Genera pesos random
            weight.append(random.random())
        weights.append(weight)        
    return weights

def newNetwork (archivo, inputP, oculta, output):
    """ Genera una nueva red neural con pesos aleatorios.
    Entradas:
        -inputP: numero de neuronas en la primer capa
        -oculta: numero de neuronas en la capa oculta
        -output: numero de neuronas en la capa de salida
        -archivo: nombre del archivo donde se almacena la nueva red.
    """
    net1= newWeights(inputP,oculta)
    net2= newWeights(oculta,output)
    newFile = open(archivo, "w")
    newFile.truncate()
    #newFile.write(net1)
    newFile.write("\n")
    #newFile.write(net2)
    newFile.close()
    return newFile

def readNetwork(archivo):
    """Carga una red neural que ha sido guardada en un archivo.
    Entrada:
        -archivo: Nombre del archivo que almacena la red neural.
    Salida:
        - Par de matrices correspondientes a los pesos
        entre la capa de input y la capa oculta, y entre la capa
        oculta y la salida (los pesos incluyen a la neurona de sesgo).
    """
    fileRead= open(archivo, 'w')
    print fileRead
    return 0

def saveNetwork (archivo, W1, W2):
    """
    Guarda una red neural en un archivo de texto, la red
    se asume que tiene tres capas, por lo tanto, esta definida
    por las dos matrices de pesos entre las capas 1 y 2 y la 2 y 3.

    Entradas:
        -W1: Matriz de peso entre la capa 1 y 2.
        -W2: Matriz de peso entre la capa 2 y 3.
    Salida:
        Una red neuronal.
    """
    return 0

def g(x):
    """ Funcion de achatamiento"""
    total = 1 / (1+ (math.e **-x)) 
    return total

def forward (X,W1,W2):
    """ Calcula la propagacion hacia adelante y deja el
    resultado en tres variables net1, out1, net2, out2.
    Entradas:
        - X: Patron de entrada, salidas de las neuronas de la primer capa.
        - W1:
        - W2:
    Salida:
        - net1
    """
    out1= []
    neth1= []
    M= [1,2,3]
    out1[1]= X[1]
    for k in range(2,len(M)-1):
        for j in range(0,len(W1)):
            net1[k]= W1[k-1][j] * out1[k-1][j]
        if k== len(M)-1:
            out1[k]= g(net1[k])
    ########################################
    out2= []
    neth2= []
    out2[1]= X[1]
    for k in range(2,len(M)-1):
        net2[k+1]= W2[k] * out2[k]
        if k== len(M)-1:
            out2[k+1]= g(net2[k])        
    return net1,out1,net2,out2

def backward (out1, out2, W1, W2):
    """
    Calcula los valores delta1 y delta2, necesarios para
    la retropropagacion.
    """
    return 0

def update (delta1, delta2, out1, out2, W1,W2):
    """
    Actualiza los pesos de la red y retorna los valores
    de W1 y W2.
    """
    return 0

###########################################################
#                 FUNCION DE ENTRENAMIENTO                #
###########################################################

def train (inputP, red, output, eta, error,max_iter):
    """
    Funcion de entrenamiento de la red neuronal
    Entradas:
        - inputP: Nombre del archivo de entrada que contiene
        los pares de entrenamiento.
        - red: Nombre del archivo que contiene los pesos de
        la red neuronal.
        - output: Nombre del archivo de salida donde guarda los
        pesos de la red entrenada.
        - eta: valor n que corresponde al ritmo de aprendizaje.
        - error: error tolerado, cuando su error este por debajo de este
        monto puede dar por terminado el entrenamiento. El error lo debe
        de calcular con la formula de minimos cuadrados.
        - max_iter: Numero maximo de iteraciones al set de entrenamiento,
        cuando llega a esta cantidad aborta el entrenamiento de su red.

    Salida:
        Un archivo de la red neural
    """
    return 0

##############################################################
#            FUNCION DE EJECUCION / PRUEBA                   #
##############################################################

def test (red, casos):
    """
    Funcion para la ejecucion del algoritmo.
    Entradas:
        - red: Es el archivo que contiene los pesos de la red.
        - casos: archivos con igual formato al de entrenamiento.
    Salida:
        -Se debe generar un reporte en pantalla de cuantos de los casos
        en el archivo de prueba fueron clasificados correctamente y cuantos no.
    """
    print "Funcion test"
    newNetwork ("prueba.txt", 64, 32, 10)
    readNetwork("prueba.txt")
    return 0


test("red","casos")

