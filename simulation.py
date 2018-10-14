from osim.env import ProstheticsEnv
import random
import numpy as np
import math
import pickle
import time
from der import der_theta_ac
from der import der_theta_v
gamma = 1
alpha_v = 0.1
alpha = 0.1
beta = 0.0001
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
delta = 0.00001
sigma = 1
epsilon = 0.3
vis = True
reset_experience = True

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range( v_layers - 1):
    theta_v.append( np.random.randn( s_v[i] + 1, s_v[i+1] )/10 )
    der_th_v.append( np.random.randn( s_v[i] +1, s_v[i+1] )/10 )
    der_z_v.append( np.random.randn( s_v[i] + 1, 1 )/10)

for i in range(ac_layers-1):
    theta_ac.append(np.random.randn( s_ac[i] + 1, s_ac[i+1]) )

for v in range(19):
    temp_th = []
    temp_z = []
    for i in range( ac_layers - 1):
        temp_th.append(np.random.randn(s_ac[i] + 1, s_ac[i+1])/10 )
        temp_z.append( np.random.randn(s_ac[i] + 1, 1)/10)
    der_th_ac.append(temp_th)
    der_z_ac.append(temp_z)

if not reset_experience:
    with open('variables.pkl', 'rb') as f:
        var, theta_v, theta_ac, rewards, maximum = pickle.load(f)
else:
    maximum = [0]*19
    theta_v = []
    theta_ac = []
    rewards = []
    var = 1
    for i in range(v_layers-1):
        theta_v.append(np.random.randn(s_v[i]+1, s_v[i+1]))
    for i in range(ac_layers-1):
        macierz = np.random.randn( s_ac[i]+1, s_ac[i+1])/10
        #print( np.max(macierz) )
        theta_ac.append(macierz)



def for_prop(theta, x):
    siec = []
    x = np.matrix(x).transpose()
    for th in theta:
        x = np.concatenate( (np.matrix([1]), x.transpose()), axis= 1).transpose()
        siec.append(x)
        z = sigmoid(  x.transpose() * th )
        x = z.transpose()
    siec.append(x)
    return siec


def density(x, mu, sigma2):
    return (1/(math.sqrt(2*math.pi*sigma2)))*np.exp(-(x-mu)**2/(2*sigma2))


def density_simple(x, sigma2):
    return (1/(math.sqrt(2*math.pi*sigma2)))*np.exp(-x**2/(2*sigma2))


def my_rewards(st):
    r2 = 0.
    r3 = 0.
    r4 = 0.
    r5 = 0.
    r6 = 0.
    if 0.4 > st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2]:
        r2 = (st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2])
    else:
        r2 = -(st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2])
    r3 = st['body_pos']['head'][0]
    f1 = abs(st['body_pos']['pelvis'][0] - st['body_pos']['head'][0])
    f2 = abs(st['body_pos']['pelvis'][2] - st['body_pos']['head'][2])
    r4 = -np.log( f1 + f2 + 0.001)
    r5 = ( st['body_pos']['toes_l'][0]  - st['body_pos']['pros_foot_r'][0])
    r6 = st['body_pos']['toes_l'][0]
    #r6 = ( st['body_pos']['pelvis'][0] - st['body_pos']['toes_l'][0] )* 100000

    return r2 + 0*r3 + 0*r4 + 0*r5 + 0*r6


def compute_phi(sigma2, delta_prime, iterations):
    phi_prime = []
    sum = 0.
    x_prev = 0

    for i in range(iterations):
        x1 = x_prev
        x2 = x1 + delta_prime
        x_prev = x2

        val1 = density_simple(x1, sigma2)
        val2 = density_simple(x2, sigma2)

        sum += (val1 + val2) * delta_prime / 2
        phi_prime.append(sum)

    return phi_prime


def map_number(a, mu, phi_prime, delta_prime):
    if a >= 0.5:
        sign = 1.
        a -= 0.5
    else:
        sign = -1.

    b = 0
    e = len(phi_prime) - 1

    while b < e:
        s = int((b+e)/2)
        if phi_prime[s] < a:
            b = s+1
        else:
            e = s

    return mu + sign * delta_prime * s


def dictionary_to_list(dictionary):
    l = []
    for i in dictionary.values():
        if type(i) == dict:
            l = l + dictionary_to_list(i)
        elif type(i) == list:
            l = l + i
        else:
            l.append(i)
    return l


def random_action(mu):
    ret = [0]*19
    for i in range(len(mu)):
        temp = map_number(random.uniform(0, 1), mu[i], phi, delta)
        if 0 <= temp <= 1:
            ret[i] = temp[0,0]
        else:
            ret[i] = mu[i, 0]
    return np.array(ret).reshape(19, 1)


def update_v_derivative(neurons, error_delta):
    global theta_v, der_th_v, der_z_v, alpha
    der_th_v = der_theta_v(theta_v, neurons, der_th_v, der_z_v, 1)
    for i in range(0, v_layers-1):
        logarytm = np.round( np.log(abs(der_th_v[i]) + 1/1e30) )
        skalar =  np.exp(- logarytm )
        #print("Update V:", skalar * der_th_v[i][1,0])
        alpha0 = 0.001
        err = error_delta[0,0]
        theta_v[i] +=   alpha0 * err *np.multiply( der_th_v[i],  skalar)


def update_ac_derivative(neurons, a):
    global theta_ac, der_th_ac, der_z_ac, alpha
    der_th_ac = der_theta_ac(theta_ac, neurons, der_th_ac, der_z_ac, 19 )
    for v in range(19):
        for i in range(0, ac_layers - 1):
            logarytm = np.round( np.log( abs(der_th_ac[v][i])  + 1/1e30 ) )
            skalar = np.exp(- logarytm  )
            #print("Update AC:", skalar * alpha * (a[v, 0] - neurons[-1][v, 0]) * der_th_ac[v][i][1, 0] )
            #print('AC:', np.min( abs(der_th_ac[v][i]) ), 'Logarytm AC', logarytm/np.log(10))
            #print("UPDATE")
            #print( a[v, 0] - neurons[-1][v, 0])
            theta_ac[i] += alpha * (a[v, 0] - neurons[-1][v, 0]) * np.multiply( skalar, der_th_ac[v][i])


def random_action_vector():
    action_vector = [0]*19
    for i in range(19):
        if random.uniform(0, 1) < 0.5:
            action_vector[i] = 0
        else:
            action_vector[i] = random.uniform(0.5, 1)
    return action_vector


def epsilon_greedy(optimal_vector, random_vector):
    global epsilon
    epsilon_greedy_vector = [0]*19
    for i in range(19):
        if random.uniform(0, 1) < epsilon:
            epsilon_greedy_vector[i] = optimal_vector[i, 0]
        else:
            epsilon_greedy_vector[i] = random_vector[i]
    return np.array(epsilon_greedy_vector).reshape(19,1)


phi = compute_phi(sigma, delta, 500000)
env = ProstheticsEnv(visualize=vis)
counter = 0
start = time.time()
print("theta _vvvvv",theta_v[-1][2, 0])

while True:

    state = env.reset(project=False)
    state = dictionary_to_list(state)
    for i in range(19):
        if abs( state[i] ) > maximum[i]:
            maximum[i] = state[i]

    done = False
    total_reward = 0
    std_reward = 0
    const_random_action_vector = random_action_vector()
    #action = np.array(random_action_vector()).reshape(19, 1)

    while not done:

        #print('value function dla randomowego inputa: ', (for_prop(theta_v, (np.random.rand(1,409)-0.5)*random.randint(1, 18363))[-1]-0.5)*1000000)
        #print('dla innego randomowego inputa: ', (for_prop(theta_v, (np.random.rand(1,409)-0.5)*random.randint(1, 18363))[-1]-0.5)*1000000)

        siec_ac = for_prop(theta_ac, state)
        #for i in siec_ac[-1]:
        #    print(i[0, 0], end=' ')
        #print()
        action = epsilon_greedy(siec_ac[-1].copy(), const_random_action_vector)

        state_next, reward, done, info = env.step(action, project=False)
        std_reward += reward
        reward += my_rewards(state_next)
        total_reward += reward
        state_next = dictionary_to_list(state_next)
        for i in range(19):
            if abs(state_next[i]) > maximum[i]:
                maximum[i] = state_next[i]

        siec_v = for_prop(theta_v, state)
        siec_v_next = for_prop(theta_v, state_next)
        #print("values:", siec_v[-1], siec_v_next[-1])
        '''
        for v in range(ac_layers-2):
            print("warstwa ", v)
            print("v i pochodna: ", theta_v[v][1, 3], der_th_v[v][1, 3])
            print("ac i pochodna: ", theta_ac[v][1, 3], der_th_ac[2][v][1, 3])
        '''

        td_error = reward + gamma * (siec_v_next[-1]-0.5)*10 - (siec_v[-1]-0.5)*10
        #print('values: ', reward, (siec_v_next[-1]-0.5)*1000000, (siec_v[-1]-0.5)*1000000)
        #print('td error: ', td_error)

        #print("prownuje")
        #print( theta_v[3][2, 4])
        print("TD ERROR", td_error)
        print( "randomowa waga v", theta_v[-1][2, 0])
        #print("randomowa waga ac", theta_ac[-1][2, 0], " i pochodna", der_th_ac[3][-1][2,0], "i jej logarytm", np.log( abs(der_th_ac[3][-1][2, 0]) )/ np.log(10) )
        print("randomowa waga ac", theta_ac[-2][2, 0], " i pochodna ", der_th_ac[3][-2][2,0])

        update_v_derivative(siec_v, td_error)
        #print( theta_v[3][2, 4])

        if counter > 0:
            for i in range(math.ceil(td_error/math.sqrt(var))):
                update_ac_derivative(siec_ac, action)

        var = (1-beta)*var + beta*td_error**2
        state = state_next
        with open('variables.pkl', 'wb') as f:
            pickle.dump((var, theta_v, theta_ac, rewards, maximum), f)


    counter += 1
    print("episode "+str(counter))

    print("Total reward: "+str(total_reward))
    rewards.append(total_reward)

    print("Total std reward: "+str(std_reward))

    print("Average time: "+str((time.time()-start)/counter))

    #with open('variables.pkl', 'wb') as f:
    #    pickle.dump((var, theta_v, theta_ac, rewards), f)
