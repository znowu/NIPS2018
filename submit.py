import opensim as osim
from osim.http.client import Client
import random
import numpy as np
import math
import pickle
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "d140e3d9a1bf0e4aa9b4efba4d862460"
client = Client(remote_base)
observation = client.env_create(crowdai_token, env_id="ProstheticsEnv")

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

def my_controller( observation ):
    global baza
    state = dictionary_to_list(observation)
    for i in range( len(state) ):
        if maximum[i] - minimum[i] != 0:
            state[i] = (state[i] - srednia[i]) / (maximum[i] - minimum[i])
    siec = for_prop_ac(baza, state)
    lista = []
    for i in range( len( siec[-1] ) ):
        lista.append(siec[-1][i, 0])
    return lista

while True:
    [observation, reward, done, info] = client.env_step(my_controller(observation), True)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()