# Task 2 Multi Armed Bandit
# Task 2: Run the UCB1 algorithm on the bandits with return probabilities 0.4, 0.45, 0.5, 0.55 and 0.6. The code block
# underneath the definition of the class shows how to choose an arm randomly.
#
# Display the UCB1 outputs for each arm after 50, 100 and 500 trials.


import numpy as np
import math
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, return_probability):
        self.return_probability = return_probability
    def play(self):
        return 10 if np.random.random() < self.return_probability else 0
        # return random.gauss(0, 2)


bandits = [Bandit(0.4), Bandit(0.45), Bandit(0.5), Bandit(0.55), Bandit(0.6)]
trial_outputs = [0, 50, 100, 500]
counts = np.zeros(5, dtype=int)
values = np.zeros(5, dtype=float)
reward_total = np.zeros(5, dtype=float)


def ind_max(x):
    m = max(x)
    return x.index(m)


def select_arm(n_arms):
    # play each arm once
    for arm in range(n_arms):
        if counts[arm] == 0:
            return arm
    return ind_max(list(values))
    # select arm by calculating ucb1 values
    # ucb_values = [0.0 for arm in range(n_arms)]
    # total_counts = sum(counts)
    # for arm in range(n_arms):
    #     bonus = math.sqrt((2 * math.log(total_counts)) / float(counts[arm]))
    #     ucb_values[arm] = values[arm] + bonus
    # return ind_max(ucb_values)


def update(chosen_arm, reward):
    # Update Counts
    counts[chosen_arm] = counts[chosen_arm] + 1
    total_counts = sum(counts)
    # Update Values
    reward_total[chosen_arm] = reward_total[chosen_arm] + reward
    new_value = reward_total[chosen_arm]/counts[chosen_arm] + math.sqrt((2 * math.log(total_counts)) / float(counts[chosen_arm]))
    values[chosen_arm] = new_value

    return


total_values = []
for time_step in range(501):
    if time_step in trial_outputs:
        print(time_step, values[:])
    temp = values[:]
    # select an arm
    bandit_index = select_arm(len(bandits))
    bandit = bandits[bandit_index]
    # get random reward
    reward = bandit.play()
    # update values
    update(bandit_index, reward)
    running_total = values[:] + temp[:] if time_step > 0 else values[:]
    total_values.append(running_total)

total_values = np.asarray(total_values)
plt.plot(range(len(total_values)), total_values[:])
plt.title("Cumulative Probability vs. Step")
plt.show()