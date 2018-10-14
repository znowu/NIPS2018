from osim.env import ProstheticsEnv
import random
import numpy as np
import math
import pickle
import time
from der import der_theta_ac
from der import der_theta_v
#skalar = 10**6
gamma = 0.99
alpha = 0.03
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
sigma = 0.04
maks = 0
vis = True
reset_experience = True

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(v_layers-1):
    der_th_v.append(np.random.randn(s_v[i] + 1, s_v[i + 1]))
    der_z_v.append(np.zeros((s_v[i] + 1, 1)) + 0.01)

for v in range(19):
    temp_th = []
    temp_z = []
    for i in range(ac_layers - 1):
        temp_th.append(np.random.randn(s_ac[i] + 1, s_ac[i + 1]) -0.5 )
        temp_z.append(np.zeros((s_ac[i] + 1, 1)))
    der_th_ac.append(temp_th)
    der_z_ac.append(temp_z)

if not reset_experience:
    with open('variables.pkl', 'rb') as f:
        var, theta_v, theta_ac, rewards = pickle.load(f)
else:
    theta_v = []
    theta_ac = []
    rewards = []
    var = 1
    for i in range(v_layers-1):
        theta_v.append((np.random.rand(s_v[i]+1, s_v[i+1]))*0 + 0.01)
    for i in range(ac_layers-1):
        theta_ac.append((np.random.rand(s_ac[i]+1, s_ac[i+1])-0.5)/2)

def for_prop(theta, x, siec): #x to macierz ileś X 1
    x = (np.matrix(x).transpose()).copy()
    for th in theta:
        x = (np.concatenate( (np.matrix([1]), x.transpose()), axis= 1).transpose()).copy()
        siec.append(x)
        z = ( sigmoid( x.transpose() * th ) ).copy()
        x = z.transpose()
    siec.append(x)
    return siec

# density liczy normal distribution probability density od x
def density(x, mu, sigma2):
    return (1/(math.sqrt(2*math.pi*sigma2)))*np.exp(-(x-mu)**2/(2*sigma2))


# density_simple liczy normal distribution probability density od x, zakładając że mu=0
def density_simple(x, sigma2):
    return (1/(math.sqrt(2*math.pi*sigma2)))*np.exp(-x**2/(2*sigma2))

# funkcja zwraca reward, w zaleznosci od podanego statu, mozna opracowac wlasne rewardy
def my_rewards(st):
    r2 = 0.
    r3 = 0.
    r4 = 0.
    r5 = 0.
    r6 = 0.
    r7 = 0.
    r8 = 0.
    r9 = 0.
    r10 = 0.
    r11 = 0.
    r12 = 0.
    r13 = 0.
    if st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2] >= 0.2: #nagroda/kara za przeplatanie nog
        r2 = 100 * ( st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2])
    elif 0.2 > st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2] > 0:
        r2 = -100
    else:
        r2 = ( st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2] ) * 100000
    #r13 = st['body_pos']['head'][0] * 1000 #nagroda z ruszenie do przodu

    #r12 = - (st['body_pos']['pelvis'][2])*1000 #kara za upadek do tylu
    #r3 = 10 * ( (20 * (st['body_pos']['head'][1] - 1.4) )**3)
    #r4 = 100 * ( st['body_pos']['head'][0] - st['body_pos']['pelvis'][0] )**2
    #r5  = 100 * ( st['body_pos']['head'][2] - st['body_pos']['pelvis'][2] )**2
    #r6 = r2 * (r3 > 1.4) * st['body_pos']['pros_foot_r'][0]
    #r7 = r2 * (r3 > 1.4) * st['body_pos']['toes_l'][0]
    #r8 = 100 * (np.mean( st['joint_pos']['ground_pelvis']) - 0.2)
    #if st['body_pos']['pros_foot_r'][1] < 0.4 and st['body_pos']['toes_l'][1] < 0.4:
        #r9 = st['body_pos']['pros_foot_r'][1]*1000
        #r10 = st['body_pos']['toes_l'][1]*100
    #else:
        #r9 = -st['body_pos']['pros_foot_r'][1]*1000
        #r10 = -st['body_pos']['toes_l'][1] * 100
    #r11 = ( st['body_pos']['head'][1] > 1.4) *  100 * st['body_pos']['head'][0] ** 3

    return r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13

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
    for i in range(len(mu)):
        temp = map_number(random.uniform(0,1), mu[i], phi, delta)
        if 0 <= temp <= 1:
            mu[i] = temp
    return mu

def update_v_derivative(neurons, error_delta):
    global theta_v, der_th_v, der_z_v, alpha
    der_th_v = ( der_theta_v(theta_v, neurons, der_th_v, der_z_v, 1) ).copy()
    for i in range(0, v_layers-1):
        alpha = 0.3
        theta_v[i] += alpha * error_delta[0,0] * der_th_v[i]

def update_ac_derivative(neurons, action):
    global theta_ac, der_th_ac, der_z_ac, alpha
    der_th_ac = ( der_theta_ac(theta_ac, neurons, der_th_ac, der_z_ac, 19 ) ).copy()
    zmiana = 0.
    for v in range(19):
        for i in range(0, ac_layers - 1):
            alpha = 0.3
            print( action[v,0], neurons[-1, 0])
            zmiana = alpha * (action[v, 0] - neurons[-1][v, 0]) * der_th_ac[v][i]
            theta_ac[i] += zmiana

env = ProstheticsEnv(visualize=vis)
counter = 0
start = time.time()

while True:
    state = env.reset(project=False)
    #print("feet", state['body_pos']['toes_l'], state['body_pos']['pros_foot_r'])
    #print("head", state['body_pos']['head'])
    #print("pelvis", state['body_pos']['pelvis'])
    r = my_rewards(state)
    state = dictionary_to_list(state)
    #print("state ", state)
    done = False

    # total_reward - cumulative reward wliczajac wlasne rewardy
    # std_reward - cumulative reward nie wliczajac wlasnych rewardów
    total_reward = 0
    std_reward = 0
    los = random.uniform(0,1)
    if los > 0.6:
        sigma = 0.04
    else:
        sigma = 0.4

    compute_phi(sigma, delta, 500000)

    licze = 0
    while not done:
        #print('hyc')
        siec_ac = []
        x = state
        siec_ac = for_prop(theta_ac, x, siec_ac).copy()
        action = random_action(siec_ac[-1]).copy()
        state_next, reward, done, info = env.step(action, project=False)
        #print("ground pelvis", np.mean( state_next['joint_pos']['ground_pelvis']) )
        #print("feet", state_next['body_pos']['toes_l'][2], state_next['body_pos']['pros_foot_r'][2])
        #print("head", state_next['body_pos']['head'])
        std_reward += reward
        r = 1000 * my_rewards(state_next)
        reward += r
        total_reward += reward
        state_next = dictionary_to_list(state_next)
        x_next = state_next
        siec_v = []
        siec_v_next = []
        siec_v = for_prop(theta_v, x, siec_v).copy()
        siec_v_next = for_prop(theta_v, x_next, siec_v_next).copy()
        val = siec_v[-1] - 0.5
        val_n = siec_v_next[-1] - 0.5
        td_error = reward + gamma * val_n * 1000000 - val*1000000
        update_v_derivative(siec_v, td_error)
        #print( "randomowa waga sieci v ", theta_v[4][1, 2], "i jej pochodna ", der_th_v[4][1, 2])
        if td_error > 0 and licze > 0:
            '''
            print("randomowy neuron ", siec_v[4][1,0])
            print( "randomowa waga ", theta_v[3][2,1])
            print("randomowa pochodna ", der_th_v[3][2,1])
            '''
            for i in range(math.ceil(td_error/math.sqrt(var))):
                update_ac_derivative(siec_ac, action)

        var = (1-beta)*var + beta*td_error**2
        state = state_next
        licze += 0

    counter += 1
    print("episode "+str(counter))

    print("Total reward: "+str(total_reward))
    rewards.append(total_reward)

    print("Total std reward: "+str(std_reward))

    print("Average time: "+str((time.time()-start)/counter))

    with open('variables.pkl', 'wb') as f:
        pickle.dump((var, theta_v, theta_ac, rewards), f)