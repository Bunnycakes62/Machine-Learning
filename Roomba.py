# Task 1: (Create the roomba vaccuum game)
# Consider a 3x3 grid space of discrete coordinates (0, 0), (1, 0), ... (3, 2), (3, 3).
# Each coordinate location can be either dirty (1) or clean (0).
# The initial state is that every coordinate is dirty and the location of the agent is randomly located at one of the
# discrete coordinates.
# The agent has five actions:
# Similar to the mouse maze game, the agent can move up (0,1), down (0,-1), left (-1,0) and right(0,1), where moving up
# in state (3,3) would update to state (3,3) instead of (3,4).
# The fifth action is to suck the dirt in the space, where the agent is, that is the coordinate status will update to
# clean.
# The agent gets a positive reward (10 say) if it sucks up dirt. If there is no dirt in the location of the agent before
# sucking then it won't get a reward.
# The game is finished when every coordinate is clean.
# Code the framework of this game, using the template from lecture. (Links to an external site.)
# Task 2:
# Run Q-learning on the game to get the agent to clean the room

import random
import numpy as np
import itertools
from collections import Counter


states = [
    (0,0, 'dirt'),  # (x,y,clean)
    (0,1, 'dirt'),
    (0,2, 'dirt'),
    (1,0, 'dirt'),
    (1,1, 'dirt'),
    (1,2, 'dirt'),
    (2,0, 'dirt'),
    (2,1, 'dirt'),
    (2,2, 'dirt'),
]

cleanMatrix = np.ones((3,3), dtype=int).tolist()


actions = [
      'suck',
      (1,0),#right
      (-1,0),#left
      (0,1),#up
      (0,-1)#down
]


def perform_action(state, action):
    newstate = np.array(state)+np.array(action[1:5])
    if tuple(newstate) in states and cleanMatrix[newstate[0]][newstate[1]][:] == 'dirt':
        cleanMatrix[newstate[0]][newstate[1]][:] = 'clean'
       # rewards.update({tuple(newstate):0})
        return tuple(newstate)
    return tuple(state)


q_values = {
    state:{
        action:0
        for action in actions
    }
    for state in states+list(rewards.keys())
}

alpha = 0.1 #learning rate
gamma = 0.99 #discount factor
rho = 0.3 #exploration factor


def get_max_action(state):
    state_qs = q_values[state]
    ind = np.argmax(state_qs.values())
    return list(state_qs.keys())[ind]


def choose_action(state, rho=0.3):
    if np.random.random() < rho:
        #exploration
        return random.choice(actions)
    else:
        #exploitation
        return get_max_action(state)


def update_q_values(state, action, reward, new_state):
    first_term = (1-alpha)*q_values[state][action]
    new_action = get_max_action(new_state)
    second_term = alpha*(reward + gamma*q_values[tuple(new_state)][new_action])
    # print(first_term, second_term)
    q_values[state][action] = np.round(first_term + second_term,1)


def play_game(rho=0.3):
    check_endgame_condition(cleanMatrix)
    current_state = random.choice(states)#(0,0)
    while current_state[2] not in rewards.keys():
        action = choose_action(current_state, rho)
        new_state = perform_action(current_state, action)
        reward = 10 if dirt in current_state and action =='suck' else 0
        # reward = terminals.get(new_state, 0)
        if rho > 0:
            update_q_values(current_state, action, reward, new_state)
        current_state = new_state

# def play_game():
#     current_state = random.choice(states) #(0,0)
#     action = choose_action(current_state)
#     new_state = perform_action(current_state, action)
#     reward = get_reward(tuple(current_state))
#     update_q_values(current_state, action, reward, new_state)
#     #current_state = new_state

# main() play game until finished
# q_values_trained={}
# for i in range(1000):
#     round =0
#     while sum(rewards.values())!=0:
#         play_game()
#         round+=1
#         print("round: ", round)
#     q_values_trained = Counter(q_values)
#
#

######################################################################################################################
# Task 2: Clean the room.
######################################################################################################################
def reset_matrix():
    matrix = np.ones((3, 3), dtype=int).tolist()
    return matrix


def play_optimal_policy(init_state):
    round = 0
    current_state=init_state
    #for i in range(50):
    while check_endgame_condition(cleanMatrix):
        action = choose_action_optimized(current_state)
        new_state = perform_action_optimized(current_state, action)
        current_state = new_state
        round += 1
        print("round:", round)


def choose_action_optimized(state):
    if sum(q_values[state].values()) == 0:
        # q-values don't have data for end cases, handling it this way
        return random.choice(states)
    # else exploitation
    return get_max_action_optimized(state)


def perform_action_optimized(state, action):
    newstate = np.array(state)+np.array(action)
    if tuple(newstate) in states and cleanMatrix[newstate[0]][newstate[1]] != 0:
        cleanMatrix[newstate[0]][newstate[1]] = 0
        return tuple(newstate)
    return tuple(state)


def check_endgame_condition(cleanMatrix):
    out = [item for t in cleanMatrix for item in t]
    return sum(out) != 0

def get_max_action_optimized(state):
    state_qs = q_values_trained[state]
    ind = np.argmax(state_qs.values())
    return list(state_qs.keys())[ind]


#training
for i in range(500):play_game()
cleanMatrix = reset_matrix()
#testing
play_game(0)
# init_state = random.choice(states)
# play_optimal_policy(init_state)

