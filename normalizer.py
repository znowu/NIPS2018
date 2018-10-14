import math
import numpy as np
import pickle
from osim.env import ProstheticsEnv
import random
vis = True
env = ProstheticsEnv(visualize=vis)

maximum = []
minimum = []
srednia = []
for i in range(409):
    maximum.append(-1000)
    minimum.append(1000)
    srednia.append(-100000)


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


licze = 1
while True:
    state_next = env.reset(project=False)
    pelvis = state_next['body_pos']['pelvis']
    dictionary = state_next.copy()
    for key in dictionary.keys():
        if key == 'body_pos':
            for key1 in dictionary[key].keys():
                for i in range(3):
                    dictionary[key][key1][i] = dictionary[key][key1][i] - pelvis[i]
    state_next = dictionary_to_list(state_next)
    for i in range(len(state_next)):
        if state_next[i] > maximum[i]:
            maximum[i] = state_next[i]
        if state_next[i] < minimum[i]:
            minimum[i] = state_next[i]
        srednia[i] = (srednia[i] * licze + state_next[i]) / (licze + 1)
    state = state_next.copy()
    licze += 1

    done = False
    while not done:

        action = np.random.rand(19)
        state_next, reward, done, info = env.step(action, project=False)
        pelvis = state_next['body_pos']['pelvis']
        dictionary = state_next.copy()
        for key in dictionary.keys():
            if key == 'body_pos':
                for key1 in dictionary[key].keys():
                    for i in range(3):
                        dictionary[key][key1][i] = dictionary[key][key1][i] - pelvis[i]
        state_next = dictionary_to_list(state_next)
        for i in range( len(state_next) ):
            if state_next[i] > maximum[i]:
                maximum[i] = state_next[i]
            if state_next[i] < minimum[i]:
                minimum[i] = state_next[i]
            srednia[i] = (srednia[i] * licze + state_next[i]) / (licze+1)
        licze += 1
        state = state_next
        with open('normalization.pkl', 'wb') as f:
            pickle.dump((maximum, minimum, srednia), f)