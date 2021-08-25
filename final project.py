import numpy as np
import gym
import random

# defult
n = 10000
max = 100000

alfa = 0.1
gamma = 0.99

e = 1
maxe = 1
mine = 0.0001
ed = 0.0001


def new_q(state, Rnow, action):
    global q
    newstate, reward, done, info = env.step(action)

    q[state, action] = q[state, action] * (1 - alfa) + alfa * (reward + gamma * np.max(q[newstate, :]))

    state = newstate
    Rnow += reward

    return state, Rnow, done


# input


print("type 0 for  8*8 \ntype 1 for  4*4")
while True:
    a = input()
    if a == '1' or a == '0':
        break
    else:
        print("\nWTF\n\n|:  |:  |: \n|:  |:  |: \n|:  |:  |: \n|:  |:  |: \n|:  |:  |: \n")
a = int(a)
if a:
    env = gym.make("FrozenLake-v0", map_name="4x4")
else:
    env = gym.make("FrozenLake-v0", map_name="8x8")

a = env.action_space.n
s = env.observation_space.n

q = np.zeros((s, a))

R_all = []

for episode in range(n):
    state = env.reset()
    done = False
    Rnow = 0

    for step in range(max):

        t = random.uniform(0, 1)
        if t > e:
            action = np.argmax(q[state, :])
        else:
            action = env.action_space.sample()

        state, Rnow, done = new_q(state, Rnow, action)

        if done == True:
            break

    e = mine + (maxe - mine) * np.exp(-ed * episode)

    R_all.append(Rnow)

Rp = np.split(np.array(R_all), n / 1000)
count = 1000

print("Average reward : ")
print()
for r in Rp:
    print(count, " ===> ", str(sum(r / 1000)))
    count += 1000
print()
print()
print("Q table")
print()
print(q)
print()
print()
print()
print()
input("press enter  for play")

for episode in range(10):
    state = env.reset()
    done = False
    print()
    print()
    print()
    print("episode : ", end=" ")
    print(episode + 1)
    print()
    print()

    for step in range(max):

        env.render()

        action = np.argmax(q[state, :])
        newstate, reward, done, info = env.step(action)

        if done:

            env.render()
            if reward == 1:
                print("    Goal")

            else:
                print("    Hole")

            break
        state = newstate

env.close()
