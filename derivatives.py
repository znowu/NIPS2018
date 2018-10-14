import random
import numpy as np
v_layers = 6
ac_layers = 6
s_v = [409, 500, 500, 500, 500, 1]
s_ac = [409, 500, 500, 500, 500, 19]
der_th_ac = []
der_z_ac = []
der_th_v = []
der_z_v = []
theta_ac = []
theta_v = []

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(v_layers-1):
    theta_v.append(np.random.rand(s_v[i]+1, s_v[i+1])-0.5)
    der_th_v.append(np.random.rand(s_v[i]+1, s_v[i+1])-0.5)
    der_z_v.append( np.matrix( [0]*(s_v[i] + 1) ).transpose() )

for i in range(ac_layers-1):
    theta_ac.append(np.random.rand(s_ac[i]+1, s_ac[i+1])-0.5)


for outs in range(19):
    temp_th = []
    temp_z = []
    for i in range(ac_layers - 1):
        temp_th.append(np.zeros((s_ac[i] + 1, s_ac[i + 1])))
        temp_z.append( np.matrix([0] * (s_ac[i] + 1)).transpose() )
    der_th_ac.append(temp_th)
    der_z_ac.append(temp_z)

def par_der_th( theta, pder_theta, pder_z, neurons, output_size ): #pder_'y,  x i output_size zaleza od sieci ktorej uzywasz
    #pder_theta   pochodne czastkowe to lista list macierzy [ warstwa X z ... X do ... X output]
    #pder_z   pochodne czastkowe to lista macierzy [ warstwa X ktory neuron X output ]
    layers = len(theta)

    for v in range( output_size ):
        lay = layers - 1
        while lay >= 0:
            z = np.matrix( neurons[lay] )
            if lay == layers - 1:
                z_next = np.array( neurons[lay + 1] )
                pder_z[v][lay] = np.multiply( theta[v][lay], np.matrix( [z_next*(1-z_next)]*len(z) ) )
                pder_theta[v][lay] = np.dot(z, (z_next * (1 - z_next) ))
            else:
                z_next = neurons[lay+1][1:]
                pder_z[v][lay] = np.multiply(np.multiply(theta[v][lay],np.matrix( [z_next * (1 - z_next)] * len(z)) ), [pder_z[v][lay]] * len(z) )
                pder_theta[v][lay] = np.dot(z, pder_z[lay+1] * z_next * (1 - z_next))

            print( "size theta[v][lay], z_next, pder_z")
            print( np.shape( theta[v][lay])," ", np.shape(z_next), " ", np.shape(pder_z[v][lay+1]) )

            pder_z[v][lay] = np.dot( theta[v][lay] * [z_next * (1 - z_next)]*len(z_next), pder_z[v][lay+1] )
            pder_theta[v][lay] = np.dot(z, (z_next * (1 - z_next)* pder_z[v][lay+1] ).transpose())
            lay -= 1

    return pder_theta

state = np.random.rand(409)
neurons_ac = []
x = np.matrix(state)
for i in range(ac_layers - 1):
    z = np.concatenate( ( x, [1]), axis=1 )
    neurons_ac.append( z )
    x = sigmoid(np.dot(z, theta_ac[i]))
neurons_ac.append( x )
print(np.size(neurons_ac))
par_der_th( theta_ac, der_th_ac, der_z_ac, neurons_ac, output_size=19)