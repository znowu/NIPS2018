from osim.env import ProstheticsEnv
import random
import numpy as np
import math
import pickle
import time

gamma = 0.9
alpha = 0.01
beta = 0.0001
v_layers = 6
ac_layers = 6
s_v = [409, 500, 500, 500, 500, 1]
s_ac = [409, 500, 500, 500, 500, 19]
delta = 0.00001
sigma = 0.001
maks = 0
der_th_ac = []
der_z_ac = []
der_th_v = []
der_z_v = []
reset_experience = True


# WYJAŚNIENIE ZMIENNYCH:
#
# gamma - discount factor
# v_layers to liczba warstw w sieci neuronowej value function (wliczając input layer i output layer)
# ac_layers to liczba warstw w sieci neuronowej actor-critic function
# s_v[i] zawiera liczbe neuronów w i-tej warstwie dla value function
# s_ac[i] zawiera liczbe neuronów w i-tej warstwie dla actor-critic function
# delta dokładność losowania (wysokość trapezów, przy liczeniu całki)
# sigma - variance
#
# maks - wartosc najwiekszego cumulative rewarda, jaki udalo sie uzyskac,
#        jeśli uda nam sie przekroczyc ta wartosc to zapisujemy parametry
#        sieci neuronowej ktorej udalo sie tego dokonac i podmieniamy maksa
#
# reset_experience - True, jesli program ma sie uczyc od zera
#                    False, jesli ma wczytac parametry sieci neuronowej z pliku variables2.pkl
# przy pierwszym odpaleniu powinna byc wartosc True



def update_theta_v( theta, neurons, pder_z, pder_theta):
    lay = v_layers -2
    z = neurons[lay]
    z_next = neurons[lay+1]
    pder_z[lay] = theta[lay] * z * (1 - z)
    pder_theta[lay] = z * z_next * (1 - z_next)
    lay -= 1
    while lay >= 0:
        z = neurons[lay]
        z_next = neurons[lay+1][1:]
        print( np.shape( theta[lay]) ," ", np.shape(z), " ", np.shape(z_next)," ", np.shape(pder_z[lay+1]))
        pder_z[lay] = np.dot( theta[lay], pder_z[lay+1] * z_next * (1 - z_next) )
        pder_theta[lay] = np.dot( z, (z_next*(1-z_next)*pder_z[lay+1]).transpose() )
    return pder_theta

def update_theta_ac( theta, neurons, pder_z, pder_theta):
    lay = ac_layers - 2
    z = neurons[lay]
    z_next = neurons[lay + 1]
    outs = 19
    for v in range( outs ):
        pder_z[v][lay] = theta[v][lay] * z * (1 - z)
        pder_theta[v][lay] = z * z_next * (1 - z_next)
        lay -= 1
        while lay >= 0:
            z = neurons[lay]
            z_next = neurons[lay + 1]
            pder_z[v][lay] = np.dot(theta[v][lay], pder_z[v][lay + 1] * z_next * (1 - z_next))
            pder_theta[v][lay] = np.dot(z, (z_next * (1 - z_next) * pder_z[v][lay + 1]).transpose())
    return pder_theta

if not reset_experience:
    with open('variables2.pkl', 'rb') as f:
        var, theta_v, theta_ac, rewards = pickle.load(f)
else:
    theta_v = []
    theta_ac = []
    rewards = []
    var = 1
    for i in range(v_layers-1):
        theta_v.append(np.random.rand(s_v[i]+1, s_v[i+1])-0.5)
        der_th_v.append(np.random.rand(s_v[i]+1, s_v[i+1])-0.5)
        der_z_v.append(np.array( [0]*(s_v[i] + 1) ))

    for i in range(ac_layers-1):
        theta_ac.append(np.random.rand(s_ac[i]+1, s_ac[i+1])-0.5)

    for outs in range(19):
        temp_th = []
        temp_z = []
        for i in range(ac_layers - 1):
            temp_th.append(np.zeros((s_ac[i] + 1, s_ac[i + 1])))
            temp_z.append(np.array([0] * (s_ac[i] + 1)))
        der_th_ac.append(temp_th)
        der_z_ac.append(temp_z)

# theta_v lista macierzy sieci neuronowej, theta_v[i] jest wymiarów s_v[i]+1 na s_v[i+1]
# theta_ac analogicznie
# rewards becie zawieral historie wszystkich total_rewardow
# var - variance TD-erroru


# density liczy normal distribution probability density od x
def density(x, mu, sigma2):
    return (1/(math.sqrt(2*math.pi*sigma2)))*np.exp(-(x-mu)**2/(2*sigma2))


# density_simple liczy normal distribution probability density od x, zakładając że mu=0
def density_simple(x, sigma2):
    return (1/(math.sqrt(2*math.pi*sigma2)))*np.exp(-x**2/(2*sigma2))


# funkcja zwraca reward, w zaleznosci od podanego statu, mozna opracowac wlasne rewardy
def my_rewards(st):
    r1 = min(0, st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2])*30
    r2 = 0
    if st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2] > 0.5:
        r2 = -((st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2])*2)**2
    r3 = st['body_pos']['head'][0] * 5

    return r1 + r2 + r3


# zwraca tablice PHI, gdzie PHI[i] = pole pod probability density function od 0 do i*delta
def compute_phi(sigma2, delta, iter):
    phi = []
    sum = 0.
    x_prev = 0

    for i in range(iter):
        x1 = x_prev
        x2 = x1 + delta
        x_prev = x2

        val1 = density_simple(x1, sigma2)
        val2 = density_simple(x2, sigma2)

        sum += (val1 + val2) * delta / 2
        phi.append(sum)

    return phi


phi = compute_phi(sigma, delta, 500000)


# przyporzadkowuje liczbie a z przedzialu (0, 1) liczbe z przedzialu (-inf, +inf), z prawdopodobienstwem
# odpowiadajacym normal distribution
def map_number(a, mu, phi, delta):
    if a >= 0.5:
        sign = 1.
        a -= 0.5
    else:
        sign = -1.

    b = 0
    e = len(phi) - 1

    while b < e:
        s = int((b+e)/2)
        # print(b, e, s, phi[s], a)
        if phi[s] < a:
            b = s+1
        else:
            e = s

    return mu + sign * delta * s


# funkcja sigmoidalna
def sigmoid(x):
    return 1/(1+np.exp(-x))


# jako argument przyjmuje dictionary, zwraca wektor, ktory posiada wszystkie wartosci wystepujace w tym dictionary
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


# bierze wektor mu, i dla kazda wartosc w tym wektorze zastepuje
# liczbą map_number(random.uniform(0,1), mu[i], phi, delta)
def random_action(mu):
    for i in range(len(mu)):
        temp = map_number(random.uniform(0,1), mu[i], phi, delta)
        if 0 <= mu[i] <= 1:
            mu[i] = temp
    return mu


# wykonuje forward propagation na sieci neuronowej actor-critic zwracajac wartosci uzyskane
# na output layer

def actor_critic(state, neurons_ac):
    x = np.array(state)
    for i in range(ac_layers - 1):
        z = np.concatenate( ([1], x), axis= 0)
        neurons_ac.append( z )
        x = sigmoid(np.dot(z, theta_ac[i]))
    neurons_ac.append( x )
    return [x, neurons_ac]


# wykonuje forward propagation na sieci neuronowej value function zwracajac wartosci uzyskane
# na output layer
def value_function(state, neurons_v):
    x = np.array(state)
    for i in range(v_layers-1):
        z = np.concatenate( ([1], x), axis= 0)
        neurons_v.append(z)
        x = sigmoid(np.dot(z, theta_v[i]))
    neurons_v.append(x)
    return x[0], neurons_v


# updateuje value function
# zamysł: dla każdego theta_v[i]
# theta_v[i] := theta_v[i] + alpha * TD-error * (pochodne cząstkowe value function od state)


# updateuje actor-critic functio
# zamysł: dla każdego theta_ac[i]
# theta_ac[i] := theta_ac[i] + alpha * (action - actor_critic(state)) * (pochodne cząstkowe actor-critic od state)

env = ProstheticsEnv(visualize=False)

counter = 0

while True:

    print("theta_ac ",np.shape(theta_ac))
    print("theta_v ", np.shape(theta_v) )
    print("der_th_ac ", np.shape(der_th_ac) )
    print("der_z_ac ", np.shape(der_z_ac) )
    print("der_th_v ", np.shape(der_th_v) )
    print("der_z_v ", np.shape(der_z_v))
    counter += 1
    print("episode "+str(counter))

    state = env.reset(project=False)
    r = my_rewards(state)
    state = dictionary_to_list(state)

    done = False

    # total_reward - cumulative reward wliczajac wlasne rewardy
    # std_reward - cumulative reward nie wliczajac wlasnych rewardów
    total_reward = 0
    std_reward = 0
    start = time.time()

    while not done:

        siec_ac = actor_critic( state, neurons_ac = [])
        action = random_action(siec_ac[0])
        neus_ac = siec_ac[1]
        state_next, reward, done, info = env.step(action, project=False)
        std_reward += reward
        reward += r
        total_reward += reward
        r = my_rewards(state_next)
        state_next = dictionary_to_list(state_next)

        siec_v = value_function( state, neurons_v= [])
        ValueFunction_s = siec_v[0]
        neus_v = siec_v[1]

        td_error = reward + gamma*value_function(state_next, neurons_v=[])[0] - siec_v[0]

        update_theta_v(theta_v, neus_v, der_z_v, der_th_v) #update sieci value function

        for i in range(math.ceil(td_error/math.sqrt(var))):
            update_theta_ac(theta_ac, neus_ac, der_z_ac, der_th_ac)

        var = (1-beta)*var + beta*td_error**2
        state = state_next

    if std_reward > maks:
        maks = std_reward
        with open('best_ac.pkl', 'wb') as f:
            pickle.dump((maks, theta_ac), f)

    print("Total reward: "+str(total_reward))
    rewards.append(total_reward)

    end = time.time()
    print("Time: "+str(end-start))

    with open('variables2.pkl', 'wb') as f:
        pickle.dump((var, theta_v, theta_ac, rewards), f)