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
alpha = 0.00000001
beta = 0.0001
v_layers = 7
ac_layers = 7
s_v = [409, 500, 500, 500, 500, 500, 1]
s_ac = [409, 500,  500, 500, 500, 500, 19]
der_th_ac = []
der_z_ac = []
der_th_v = []
der_z_v = []
theta_ac = []
theta_v = []
delta = 0.00001
sigma = 0.02
maks = 0
vis = True
reset_experience = True

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(v_layers-1):
    der_th_v.append(np.random.randn(s_v[i] + 1, s_v[i + 1]))
    der_z_v.append(np.zeros((s_v[i] + 1, 1)))

for v in range(19):
    temp_th = []
    temp_z = []
    for i in range(ac_layers - 1):
        temp_th.append(np.random.randn(s_ac[i] + 1, s_ac[i + 1]) )
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
    var = 100
    for i in range(v_layers-1):
        theta_v.append((np.random.rand(s_v[i]+1, s_v[i+1])-0.5)/2)
    for i in range(ac_layers-1):
        theta_ac.append((np.random.rand(s_ac[i]+1, s_ac[i+1])-0.5)/2)

def for_prop(theta, x, siec): #x to macierz ileś X 1
    x = np.matrix(x).transpose()
    print(x)
    for th in theta:
        x = ( np.concatenate( (np.matrix([1]), x.transpose()), axis= 1).transpose() )
        siec.append(x)
        print(np.shape(x.transpose()) , np.shape(th))
        z = ( sigmoid( x.transpose() * th ) ).copy()
        x = ( z.transpose() ).copy()
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
    r2 = 0
    if st['body_pos']['toes_l'][2] - st['body_pos']['pros_foot_r'][2] > 0:
        r2 = -((st['body_pos']['toes_l'][2] - st['body_pos']['pros_foot_r'][2])*30)

    return r2

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

def random_action(mu):
    for i in range(len(mu)):
        temp = map_number(random.uniform(0,1), mu[i], phi, delta)
        if 0 <= temp <= 1:
            mu[i] = temp
    return mu

def update_v_derivative(neurons, error_delta):
    global theta_v, der_th_v, der_z_v, alpha
    der_th_v = ( der_theta_v(theta_v, neurons, der_th_v, der_z_v, 1) ).copy()
    #print("witamy z Kazimierza")
    for i in range(0, v_layers-1):
        theta_v[i] += alpha * error_delta[0,0] * der_th_v[i]
    #print("czas na piwko w Zabrzu!")

def update_ac_derivative(neurons, action):
    global theta_ac, der_th_ac, der_z_ac, alpha
    der_th_ac = ( der_theta_ac(theta_ac, neurons, der_th_ac, der_z_ac, 19 ) ).copy()
    #print("Jest plaża na Suwałkach!")
    for v in range(19):
        for i in range(0, ac_layers - 1):
            theta_ac[i] += alpha * (action[v, 0] - neurons[-1][v, 0]) * der_th_ac[v][i]
    #print("Musi być i kiełba!")



env = ProstheticsEnv(visualize=vis)
counter = 0
start = time.time()

while True:
    state = ( env.reset(project=False) ).copy()
    r = my_rewards(state)
    #print("state ", state)
    done = False

    # total_reward - cumulative reward wliczajac wlasne rewardy
    # std_reward - cumulative reward nie wliczajac wlasnych rewardów
    total_reward = 0
    std_reward = 0

    while not done:
        #print('hyc')
        siec_ac = []
        x = dictionary_to_list(state).copy()
        siec_ac = for_prop(theta_ac, x, siec_ac).copy()
        #print("siec ", siec_ac[-1])
        action = random_action(siec_ac[-1]).copy()
        state_next, reward, done, info = env.step(action, project=False)
        #print(reward)
        std_reward += reward
        reward += r
        #print(r)
        total_reward += reward
        r = my_rewards(state_next)
        state_next = dictionary_to_list(state_next)
        #print("state ", state_next)
        x_next = state_next.copy()
        siec_v = []
        #print(siec_v)
        siec_v_next = []
        siec_v = for_prop(theta_v, x, siec_v).copy()
        siec_v_next = for_prop(theta_v, x_next, siec_v_next).copy()
        val = 1000000 * (siec_v[-1] - 0.5)
        val_n = 1000000 * (siec_v_next[-1] - 0.5)
        td_error = reward + gamma * val- val_n
        #print("TD ERROR ", td_error)
        #print(val_n)

        update_v_derivative(siec_v, td_error)

        if counter > 1:
            for i in range(math.ceil(td_error/math.sqrt(var))):
                update_ac_derivative(siec_ac, action)

        var = (1-beta)*var + beta*td_error**2
        print("randomowa waga ", der_th_ac[2][3][1, 3])
        state = state_next.copy()

    counter += 1
    print("episode "+str(counter))

    print("Total reward: "+str(total_reward))
    rewards.append(total_reward)

    print("Total std reward: "+str(std_reward))

    print("Average time: "+str((time.time()-start)/counter))

    with open('variables.pkl', 'wb') as f:
        pickle.dump((var, theta_v, theta_ac, rewards), f)