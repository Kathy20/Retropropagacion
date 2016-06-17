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
from random import randint
import random
import math
import copy


def newWeights(n,m):
    "Genera una nueva matriz de pesos nxm que conecta con la capa siguiente"
    weights= []
    for i in range(n):
        weight=[]
        for j in range(m):
            # Genera pesos random de -3 a 3
            weight.append(randint(-3, 3))
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
    W1= newWeights(oculta,inputP+1)
    W2= newWeights(output,oculta+1)
    newFile= saveNetwork(archivo,W1,W2)
    return W1,W2

def readNetwork(archivo):
   fileR = open("BaseDatos.txt", "r")
   lineas = fileR.readline().split('\n')[0].split(" ")
   S= []
   pares = lineas[0]
   pixeles = lineas[1]
   salida = lineas[2]
   for i in range(int(pares)):
       x=[]
       for j in range (int(math.sqrt(int(pixeles)))):
           read =fileR.readline().split('\n')[0].split(" ")
           for k in range(len(read)):
               x.append(int(read[k]))
       y=[]
       y=[int(i) for i in (fileR.readline().split('\n')[0].split(" "))]
       tupleA = (x,y)
       S.append(tupleA)
   return S


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
    newFile = open(archivo, "w")
    newFile.truncate()
    oculta= len(W1)
    inputP= len(W1[0])
    #Recorre y escribe W1.
    for i in range(oculta):
        for j in range(inputP):
            # Escribe los pesos en el file
            newFile.write(" "+ str(W1[i][j]))
        newFile.write("\n")        
    newFile.write("###################################")
    newFile.write("\n")
    output=len(W2)
    oculta= len(W2[0])
    #Recorre y escribe W2.
    for i in range(output):
        for j in range(oculta):
            # Escribe los pesos en el file
            newFile.write(" "+ str(W2[i][j]))
        newFile.write("\n")   
    newFile.close()
    return newFile

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
    out1= copy.copy(X)
    out2= [] 
    net2= [0] #Seria el proceso de la segunda capa ya que la primera es la misma a la de las entradas
    out3= []
    net3= [0] #Seria el proceso de la tercera capa ya que la primera es la misma a la de las entradas
    out1.insert(0,1)
    for i in range(len(W1)):     
        sumatoria = 0
        for j in range(len(W1[0])):
            sumatoria += out1[j] * W1[i][j]
        net2.append(sumatoria)
        out2.append(g(sumatoria))  
        
    out2.insert(0,1)        
    ########################### TERCERA CAPA
    for i in range(len(W2)):
        sumatoria = 0
        for j in range(len(W2[0])):
            sumatoria += out2[j] * W2[i][j]
        net3.append(sumatoria)
        out3.append(g(sumatoria))
    out3.insert(0,1)          
    return (out1,net2,out2,net3,out3)

def backward (out1,out2, out3, W1, W2,y):
    """
    Calcula los valores delta1 y delta2, necesarios para
    la retropropagacion.
    """
    delta1= []
    delta2= []
    delta3= []
    for i in range(len(W2)):
        delta3.append(out3[i]*(1-out3[i])*(y[i]-out3[i]))

    for j in range(len(W2[0])): #######Segunda capa
        sumatoria= 0
        for i in range(len(W2)):
            sumatoria+= delta3[i] * W2[i][j]
        sumatoria*= out2[j] * (1-out2[j])
        delta2.append(sumatoria)
    ###################### Estructura original
    for j in range(len(W1[0])): 
        sumatoria= 0
        for i in range(len(W1)):
            sumatoria+= delta2[i] * W1[i][j]
        sumatoria*= out1[j] * (1-out1[j])
        delta1.append(sumatoria)
    return (delta1,delta2, delta3)

def update (delta1,delta2, delta3, out1, out2,out3, W1,W2,N):
    """
    Actualiza los pesos de la red y retorna los valores
    de W1 y W2.
    """
    updateW1=copy.copy(W1)
    updateW2=copy.copy(W2)
    
    ###################### Estructura original
    for i in range(len(W1)): 
        for j in range(1,len(W1[0])):
            trianguloPesos= N * delta2[i] *out1[j]
            updateW1[i][j]+= trianguloPesos
    for i in range(len(W2)): #######Segunda capa
        for j in range(1,len(W2[0])):
            trianguloPesos2= N * delta3[i] *out2[j]
            updateW2[i][j]+= trianguloPesos2
    return updateW1,updateW2

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
    S= readNetwork(inputP)
    (w1,w2)= red
    for iteracion in range(max_iter):
        #random.shuffle(S)
        for(x,y) in S:
            (out1,net2,out2,net3,out3) = forward(x,w1,w2)
            #net2,out2,net3,out3
            (delta1,delta2,delta3) = backward(out1,out2,out3,w1,w2,y)
            #out1,out2, out3, W2, W3,y
            #delta2, delta3, out1, out2, W1,W2,N
            (updateW1,updateW2)= update(delta1,delta2,delta3,out1,out2,out3,w1,w2,0.5)
            w1= updateW1
            w2= updateW2

        prediccionGeneral= forward(S[0][0],w1,w2)
        print prediccionGeneral[3]
            
    
    

##############################################################
#            FUNCION DE EJECUCION / PRUEBA                   #
##############################################################

def retropropagacion (archivo, salida):
    """
    Funcion para la ejecucion del algoritmo.
    Entradas:
        - red: Es el archivo que contiene los pesos de la red.
        - casos: archivos con igual formato al de entrenamiento.
    Salida:
        -Se debe generar un reporte en pantalla de cuantos de los casos
        en el archivo de prueba fueron clasificados correctamente y cuantos no.
    """
    net = newNetwork (salida, 64, 30, 10)
    train(archivo,net,salida,10,20,50)
  


retropropagacion("BaseDatos.txt","output.txt")

