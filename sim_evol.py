from osim.env import ProstheticsEnv
import random
import numpy as np
import math
import pickle
gamma = 1
alpha = 0.1
ac_layers = 4
s_ac = [409, 500, 500, 19]
theta_ac = []
vis = True
reset_experience = True

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.multiply(x, (x>0))

with open('normalization.pkl', 'rb') as f:
    maximum, minimum, srednia = pickle.load(f)

def for_prop_ac(theta, x): #x to macierz ileś X 1
    siec = []
    x = np.matrix(x).transpose()
    for i in range(len(theta)):
        x = np.concatenate( (np.matrix([1]), x.transpose()), axis= 1).transpose()
        siec.append(x)
        th = theta[i]
        if i == len(theta) - 1 :
            z = ( sigmoid( x.transpose() * th) ).copy()
        else:
            z = ( relu( x.transpose() * th ) ).copy()
        x = ( z.transpose() ).copy()
    siec.append(x)
    return siec

def my_rewards(st):
    r2 = 0.
    r3 = 0.
    r4 = 0.
    r5 = 0.
    r6 = 0.
    if 0.3 > st['body_pos']['pros_foot_r'][2] - st['body_pos']['toes_l'][2]:
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

    return r2 + r3 + r4 + 0*r5 + 0*r6

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

env = ProstheticsEnv(visualize=vis)
counter = 1

baza = []
for i in range(ac_layers-1):
    macierz = np.zeros( (s_ac[i]+1, s_ac[i+1]) )
    baza.append(macierz)
best = baza.copy()
rekord = 0.
rekord = -1e6
skalar = 7.
while counter < 10:
    baza = best.copy()
    skalar *= 1.6
    with open('evolution.pkl', 'wb') as f:
        pickle.dump((baza), f)
    for k in range(50):
        theta_ac = []
        for i in range(ac_layers - 1):
            theta_ac.append( baza[i] + np.random.randn(s_ac[i] + 1, s_ac[i + 1])/skalar)
        state = env.reset(project=False)
        state = dictionary_to_list(state)
        for i in range(409):
            if maximum[i] - minimum[i] != 0:
                state[i] = (state[i] - srednia[i])/( maximum[i] - minimum[i])
        done = False
        total_reward = 0
        std_reward = 0
        while not done:
            siec_ac = for_prop_ac(theta_ac, state)
            action = siec_ac[-1].copy()
            state_next, reward, done, info = env.step(action, project=False)
            std_reward += reward
            reward += 0 * my_rewards(state_next)
            total_reward += reward
            state_next = dictionary_to_list(state_next)
            for i in range(409):
                if maximum[i] - minimum[i] != 0:
                    state_next[i] = (state_next[i] - srednia[i]) / (maximum[i] - minimum[i])
            state = state_next.copy()
        if total_reward > rekord:
            best = theta_ac.copy()
            rekord = total_reward

        print(k, " Total reward: "+str(total_reward))
        #print("Total std reward: "+str(std_reward))
    print("Wybrałem ", counter, "poziom z wynikiem ", rekord)

    counter += 1