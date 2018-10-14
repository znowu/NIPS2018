import random
import numpy as np
v_layers = 7
ac_layers = 7
s_v = [409, 500, 500, 500, 500, 500, 1]
s_ac = [409, 500, 500, 500, 500, 500, 19]
der_th_ac = []
der_z_ac = []
der_th_v = []
der_z_v = []
theta_ac = []
theta_v = []

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range( v_layers - 1):
    theta_v.append( np.random.randn( s_v[i] + 1, s_v[i+1] ) )
    der_th_v.append( np.random.randn( s_v[i] +1, s_v[i+1] ) )
    der_z_v.append( np.random.randn( s_v[i] + 1, 1 ))

for i in range(ac_layers-1):
    theta_ac.append(np.random.randn( s_ac[i] + 1, s_ac[i+1]) )

for v in range(19):
    temp_th = []
    temp_z = []
    for i in range( ac_layers - 1):
        temp_th.append(np.random.randn(s_ac[i] + 1, s_ac[i+1]) )
        temp_z.append( np.random.randn(s_ac[i] + 1, 1))
    der_th_ac.append(temp_th)
    der_z_ac.append(temp_z)

def for_prop(theta, x, siec): #x to macierz ileś X 1
    for th in theta:
        x = ( np.concatenate( (np.matrix([1]), x.transpose()), axis= 1).transpose() ).copy()
        siec.append(x)
        z = ( sigmoid( x.transpose() * th ) ).copy()
        x = ( z.transpose() ).copy()
    siec.append(x)
    return siec

def der_theta_ac(theta, neurons, der_th, der_z, outs):
    for v in range( outs ):
        limit = random.randint(0, 5)
        lay = 5
        z = neurons[lay]
        z_next = neurons[lay+1][v, 0] #liczba
        th = ( theta[lay].transpose()[v] ).copy()
        der_z[v][lay] = ( np.matrix(th * z_next * (1-z_next) ) ).transpose().copy()
        der_th[v][lay] = ( np.matrix(z) * z_next * (1 - z_next) ).copy() #macierz
        lay -= 1
        while lay >= limit:
            z = neurons[lay].copy()
            z_next = neurons[lay+1][1:].copy()

            t = ( np.multiply(z_next, 1-z_next) ).copy()
            t = ( np.multiply(t, der_z[v][lay+1][1:]) ).copy()
            der_z[v][lay] = (( t.transpose()*theta[lay].transpose()).transpose() ).copy()
            der_th[v][lay] = ( z * t.transpose() ).copy()

            lay -= 1
    return der_th


def der_theta_v(theta, neurons, der_th, der_z, outs):
    limit = random.randint(0,5)
    lay = 5
    z = neurons[lay].copy()
    z_next = neurons[lay+1][0,0] #liczba
    th = ( theta[lay].transpose() ).copy()
    der_z[lay] = ( np.matrix(th * z_next * (1-z_next) ).transpose() ).copy()
    der_th[lay] = ( np.matrix(z) * z_next * (1 - z_next) ).copy() #macierz
    lay -= 1
    while lay >= limit:
        z = neurons[lay].copy()
        z_next = neurons[lay+1][1:].copy()
        t = ( np.multiply(z_next, 1-z_next) ).copy()
        t = ( np.multiply(t, der_z[lay+1][1:]) ).copy()
        der_z[lay] = ( (t.transpose()*theta[lay].transpose()).transpose() ).copy()
        der_th[lay] = ( z * t.transpose() ).copy()

        lay -= 1
    return der_th

x = np.random.randn(409, 1)
'''
siecv = []
siecv = for_prop(theta_v, x, siec)
der_th_v = der_theta_v( theta_v, siec, der_th_v, der_z_v, 1)
theta_v[3][3, 4] += 1/1e12
siecv1 = []
siecv1 = for_prop(theta_v, x, siec1)
print( ( siecv1[-1] - siecv[-1]) * 1e12 )
'''
siecac = []
siecac = for_prop( theta_ac, x, siecac)
der_th_ac = der_theta_ac( theta_ac, siecac, der_th_ac, der_z_ac, 19)
print( der_th_ac[2][4][1, 2])
theta_ac[4][1, 2] += 1/1e13
siecac1 = []
siecac1 = for_prop( theta_ac, x, siecac1 )
print( (siecac1[-1][2] - siecac[-1][2])*1e13)
