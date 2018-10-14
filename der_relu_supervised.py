import random
import numpy as np
v_layers = 3
ac_layers = 4
s_v = [2, 2, 1]
s_ac = [409, 214, 214, 19]
der_th_ac = []
der_z_ac = []
der_th_v = []
der_z_v = []
theta_ac = []
theta_v = []

def relu(x):
    s = np.multiply( x,  (x > 0) )
    return s

def sigmoid(x):
    s = 1/(1 + np.exp(-x) )
    return s

for i in range( v_layers - 1):
    theta_v.append( np.random.randn( s_v[i] + 1, s_v[i+1] )/100 )
    der_th_v.append( np.random.randn( s_v[i] +1, s_v[i+1] ) )
    der_z_v.append( np.random.randn( s_v[i] + 1, 1) )

der_z_v.append( np.random.randn( s_v[i] + 1, 1) )

for i in range(ac_layers-1):
    theta_ac.append(np.random.randn( s_ac[i] + 1, s_ac[i+1])/100)

for v in range(19):
    temp_th = []
    temp_z = []
    for i in range( ac_layers - 1):
        temp_th.append(np.random.randn(s_ac[i] + 1, s_ac[i+1])/100 )
        temp_z.append( np.random.randn(s_ac[i] + 1, 1)/100)
    der_th_ac.append(temp_th)
    der_z_ac.append(temp_z)

def for_prop_ac(theta, x, siec): #x to macierz ileś X 1
    for i in range(len(theta)):
        x = np.concatenate( (np.matrix([1]), x.transpose()), axis= 1).transpose().copy()
        siec.append(x)
        th = theta[i].copy()
        if i == len(theta) - 1 :
            z = ( sigmoid( x.transpose() * th) ).copy()
        else:
            z = ( relu( x.transpose() * th ) ).copy()
        x = ( z.transpose() ).copy()
    siec.append(x)
    return siec

def for_prop_v( theta, x, siec):
    for i in range(len(theta_v)):
        x = np.concatenate( (np.matrix([1]), x.transpose()), axis= 1).transpose().copy()
        siec.append(x)
        th = theta[i].copy()
        if i + 1 == len(theta):
            z = (x.transpose() * th).copy()
        else:
            z = ( relu( x.transpose() * th ) ).copy()
        x = ( z.transpose() ).copy()
    siec.append(x)
    return siec

def der_theta_ac(theta, neurons, der_th, der_z, outs):
    for v in range( outs ):
        lay = ac_layers - 2
        z = neurons[lay]
        z_next = neurons[lay + 1][v, 0]  # liczba
        th = (theta[lay].transpose()[v]).copy()
        der_z[v][lay] = (np.matrix(th * z_next * (1 - z_next))).transpose().copy()
        der_th[v][lay] = (np.matrix(z) * z_next * (1 - z_next)).copy()  # macierz
        lay -= 1
        while lay >= 0:
            z = neurons[lay].copy()
            z_next = neurons[lay+1][1:].copy()

            t = ( z_next > 0 ).copy()
            t = ( np.multiply(t, der_z[v][lay+1][1:]) ).copy()
            der_z[v][lay] = (( t.transpose()*theta[lay].transpose()).transpose() ).copy()
            der_th[v][lay] = ( z * t.transpose() ).copy()

            lay -= 1
    return der_th


def der_theta_v(theta, neurons, der_th, der_z, results, outs):
    lay = v_layers - 1
    z = neurons[lay][0, 0].copy()
    print(results, z)
    der_z_v[lay] = 2 * (z - results)
    lay -= 1
    while lay >= 0:
        z = neurons[lay].copy()
        pochodne = der_z[lay + 1].copy()
        if lay == v_layers -2:
            z_next = neurons[lay + 1].copy()
            pochodne = der_z[lay+1].copy()
        else:
            z_next = neurons[lay+1][1:].copy()
            pochodne = der_z[lay + 1][1:].copy()

        t = ( z_next > 0 ).copy()
        t = ( np.multiply(t, pochodne) ).copy()
        der_z[lay] = ( (t.transpose()*theta[lay].transpose()).transpose() ).copy()
        der_th[lay] = ( z * t.transpose() ).copy()

        lay -= 1
    return der_th

x = np.random.randn(2, 1)/10
y = np.random.uniform(0, 1)

siecv = []
siecv = for_prop_v(theta_v, x, siecv)
der_th_v = der_theta_v(theta_v, siecv, der_th_v, der_z_v, y, 1)
print( der_th_v[1][1, 4] )
theta_v[1][1, 4] += 1/1e8
siecv1 = []
for_prop_v( theta_v, x, siecv1)
print( 1e8 *( (siecv1[-1] - y)**2 - (siecv[-1] - y)**2) )
'''
print("wypisuje pochodne")
print(der_th_ac[2][-2][2, 1])
print("wypisuje neurony")
print(siecac[-3][2,0])
print( siecac[-2][1, 0])

print( der_th_ac[3][-1])
print("siec", siecac[-2])
'''