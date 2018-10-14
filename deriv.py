import math
import numpy as np
import random

def sigmoid( x ):
    value = 1/(1 + np.exp(-x) )
    return value

def for_prop(theta, x): #theta to lista wektorow
    a =[]
    a.append(x)
    for th in theta:
        x = [1] + x
        x = sigmoid( np.dot( th, x ) )
        a.append( x )

pder_theta = [] #pochodne czastkowe to lista list macierzy [output X warstwa X z ... X do ... ]
pder_z = [] #pochodne czastkowe to lista macierzy [output X warstwa X ktory neuron ]

def par_der_z( theta, values, x, a):
    layers = len( theta )
    for i in range( len(values) ):
        pder_z[i][layers][i] = 1

    for v in range( len(values) ):

        lay = layers - 1
        while lay >= 0:
            z = a[lay]
            z_next = a[ lay + 1 ]
            for i in range( len( z ) ):
                suma = 0
                for j in range( len( z_next ) ):
                    suma += pder_z[v][lay + 1][j] * theta[v][lay][i][j] * z[v][lay+1][j] * ( 1 - z[v][lay + 1][j])
                pder_z[v][lay][i] = suma
            lay -= 1

def par_der_th( theta, values, x, a ):
    layers = len(theta)
    nets = layers-1
    th = theta[ nets ]
    for v in range( len(value) ):
        for i in range( len(th) ):
            z = a[nets][i]
            pder_theta[v][nets][i][v] = z*values[v] * (1 - values[v] )

        lay = nets - 1
        while lay >= 0:
            z = a[lay]
            z_next = a[lay+1]
            z_succ = a[lay+2]
            for i in range( len(z) ):
                for j in range( len(z_next) ):
                    pder_theta[v][lay][i][j] = pder_z[v][lay+1][j] * z[i] * z_next[j] * ( 1- z_next[j] )
            lay -= 1