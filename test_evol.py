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

with open('evolution.pkl', 'rb') as f:
    baza = pickle.load(f)

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


while counter < 30:
    for k in range(40):
        state = env.reset(project=False)
        state = dictionary_to_list(state)
        for i in range(409):
            if maximum[i] - minimum[i] != 0:
                state[i] = (state[i] - srednia[i])/( maximum[i] - minimum[i])
        done = False
        std_reward = 0
        while not done:
            siec_ac = for_prop_ac(baza, state)
            action = siec_ac[-1].copy()
            state_next, reward, done, info = env.step(action, project=False)
            std_reward += reward
            state_next = dictionary_to_list(state_next)
            state = state_next

        print("Total std reward: "+str(std_reward))
    with open('evolution.pkl', 'wb') as f:
        pickle.dump((baza), f)
    counter += 1