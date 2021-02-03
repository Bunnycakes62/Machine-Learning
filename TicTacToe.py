# Task 1 Tic Tac Toe
# Task 1: Train an agent to play Tic Tac Toe using Q-learning and show that the agent can always beat/draw with the
# computer, i.e. after training the agent with exploration factor > 0 set exploration factor to zero and tabulate the
# win (reward=10), draws (reward=0) and losses (reward=-10). The code block underneath the class definition shows how
# to play a single game with random plays.

import random

lr = 0.1
gamma = 0.9
q_values = {}

class TicTacToe:
    def __init__(self):
        self.locations = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2)
        ]
        self.winning_groups = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6)
        ]
        self.restart_game()

    def restart_game(self):
        self.plays = {}

    def legal_plays(self):
        return list(set(self.locations).difference(self.plays.keys()))

    def opponent_play(self):
        new_location = random.choice(self.legal_plays())
        self.plays[new_location] = 'O'

    def perform_action(self, location):
        if location in self.plays:
            print('location already taken by {}'.format(self.plays[location]))
            #self.display_board()
            #return
        self.plays[location] = 'X'
        if not self.end_of_game()[0]:
            self.opponent_play()
            self.end_of_game()
        self.display_board()

    def end_of_game(self):
        player_states = [loc for loc, player in self.plays.items() if player == 'X']
        opponent_states = [loc for loc, player in self.plays.items() if player == 'O']
        finished = (False, 0)
        for group in self.winning_groups:
            if all(self.locations[g] in player_states for g in group):
                print('Game over. Player wins')
                return True, 10
            if all(self.locations[g] in opponent_states for g in group):
                print('Game over. Computer wins')
                return True, -10
            if len(self.plays) == 9:
                finished = (True, 0)
        if finished[0]:
            print('Game over. Draw')
        return finished

    def display_board(self):
        for i in range(3):
            row = [self.plays.get((j, i), ' ') for j in range(3)]
            print('{}|{}|{}'.format(*row))
            if i < 2:
                print('------')
        print('      ')
        print('      ')

    def update_q_values(self, reward):
        for st in self.plays.keys():
            if st in q_values.keys():
                new_value = {st: q_values[st] + lr * ((gamma * reward) - q_values[st])}
                q_values.update(new_value)
            else:
                q_values[st] = 0


def play_game(ttt, epsilon=.1):
    while not ttt.end_of_game()[0]:
        if random.gauss(.5, 1) < epsilon:
            action = random.choice(ttt.legal_plays())
            ttt.perform_action(action)
        else:
            q_list = []
            for lp in ttt.plays.keys():
                q_list.append(q_values.get((lp)))
            if len(q_list) > 0:
                action = max(q_list)
            else:
                action = random.choice(ttt.legal_plays())
            ttt.perform_action(action)
        _, reward = ttt.end_of_game()
        ttt.update_q_values(reward)


ttt = TicTacToe()
for i in range(50):
    play_game(ttt)
    ttt.restart_game()

# win, loss, draw tabulation
counter = [0, 0, 0]
for i in range(1000):
    play_game(ttt, 0)
    _, reward = ttt.end_of_game()
    if reward == 10:
        print("Win")
        counter[0] += 1
    elif reward == 0:
        print("Draw")
        counter[1] += 1
    elif reward == -10:
        print("Loss")
        counter[2] += 1

print(" Win: {} \n Loss: {} \n Draw: {}".format(counter[0]/sum(counter), counter[2]/sum(counter), counter[1]/sum(counter)))
print(q_values)
