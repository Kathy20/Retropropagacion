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

import numpy as np
import math

def g(x):
    """ Funcion de achatamiento"""
    return 1.0/(1.0 + np.exp(-x))

def g_prime(x):
    return g(x)*(1.0-g(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'g':
            self.activation = g
            self.activation_prime = g_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def train(self, X, y, learning_rate=0.2, epochs=100000):
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
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()
        

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            

    def update(self, x):
        """
        Actualiza los pesos de la red y retorna los valores
        de W1 y W2.
        """
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
    
def readNetwork(archivo):
   fileR = open(archivo, "r")
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

if __name__ == '__main__':

    network= readNetwork("BaseDatos.txt")
    neurona1 = []
    neurona2 = []

    for (i,j) in network:
        neurona1.append(i)
        neurona2.append(j)
        
    neuronas = NeuralNetwork([64,20,10])
    ICapa = np.array(neurona1)
    IICapa = np.array(neurona2)
    neuronas.train(ICapa, IICapa)
    for num in ICapa:
        print("Numero")
        print(num)
        print("Probabilidades")
        print(neuronas.update(num))
        print("\n")
