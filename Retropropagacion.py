"""
Instituto Tecnologico de Costa Rica
Centro Academico de San Jose
Ingenieria en Computacion

Inteligencia Artificial

Estudiante: Kathy Brenes G.
Profesor: Jose Castro.
Fecha: 22 de abril del 2016
"""

def newWeights(n,m):
    "Genera una nueva matriz de pesos nxm que conecta con la capa siguiente"
    return 0

def newNetwork (archivo, inputP, oculta, output):
    """ Genera una nueva red neural con pesos aleatorios.
    Entradas:
        -inputP: numero de neuronas en la primer capa
        -oculta: numero de neuronas en la capa oculta
        -output: numero de neuronas en la capa de salida
        -archivo: nombre del archivo donde se almacena la nueva red.
    """
    return 0

def readNetwork(archivo):
    """Carga una red neural que ha sido guardada en un archivo.
    Entrada:
        -archivo: Nombre del archivo que almacena la red neural.
    Salida:
        - Es un par de matrices correspondientes a los pesos
        entre la capa de input y la capa oculta, y entre la capa
        oculta y la salida (los pesos incluyen a la neurona de sesgo).
    """

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

def forward (X,W1,W2):
    """ Calcula la propagacion hacia adelante y deja el
    resultado en tres variables net1, out1, net2, out2.
    """
    return 0

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
    return 0


test("red","casos")

