import random
from typing import List, Tuple

import numpy as np
from sympy import false
# TODO simplify. The plan 1 is as follows:
# For pick 0 there are 2 "optimal" choices: 
#   1. Pick the least value die
#   2. Pick the least value die that has 2+ appearances
# For pick 1+ there are 3 "optimal" choices:
#   1. Pick nothing (cannot do twice in a row)
#   2. Pick least value
#   3. IFF all die are the same, pick same die.


# Plan 2: Instead of making 1 pick be a round make 2 picks be a round. This results in the following for rounds 1+:
#   1. Pick two least dice
#   2. Pick one least dice
#   3. IFF all hand dice are the same, pick two of same die
#   4. IFF all hand dice are the same, pick one of same die
# For round 0 the rules change as follows:
#   1. Pick two least dice
#   2. Pick one least dice
#   3. Pick two of same least dice
# Therefore, the sum of available choices are:
#   0. Pick one least die
#   1. Pick two least dice (IF two dice are available)
#   2. Pick two of same least dice (IF hand is empty AND IF two dice are available)
#   3. Pick one dice that matches hand (IF hand only has one unique value AND IF 4 is unavailable.)
#   4. Pick two dice that matches hand (IF hand only has one unique value AND IF two dice are available)
# Goal: Primary goal is to win. Secondary goal is to win by the most.

class DieGame():
    def __init__(self) -> None:
        self.field = np.zeros(5)
        self.hand = np.zeros(5)
        self.action_mask = np.zeros(5)
        self.to_beat = 0
        self.selected_dice = 0
        self.roll()
        self.create_action_mask()
    
    def reset(self, to_beat:int):
        self.field = np.zeros(5)
        self.hand = np.zeros(5)
        self.action_mask = np.zeros(5)
        self.to_beat = to_beat
        self.selected_dice = 0
        self.roll()
        self.create_action_mask()

    def hand_is_same(self) -> int:
        if self.selected_dice == 0:
            return -1
        if 1 != len(np.unique(self.hand)):
            return -1
        else:
            return self.hand[0]

    def create_action_mask(self):
        self.action_mask[0] = 1
        if len(self.field) > 1:
            self.action_mask[1] = 1
            if len(self.hand) == 0:
                self.action_mask[2] = 1
            if self.hand_is_same() != -1:
                count = np.count_nonzero(self.field == self.hand[0])
                if count >= 2:
                    self.action_mask[4] = 1
                elif count == 1:
                    self.action_mask[3] = 1
    def roll(self):
        for i in range(5 - self.selected_dice):
            self.field[i] = random.randint(1,6)
        np.sort(self.field)
    
    def pick_least(self):
        self.hand.append(self.field.pop(0))
    
    def pick_common(self):
        if len(set(self.hand)) != 1:
            raise Exception('Invalid choice')
        
        for i in range(len(self.field)):
            if self.field[i] == self.hand[0]:
                self.hand.append(self.field.pop())
                self.hand.sort()
                return
            
    def return_sum_of_least(self) -> int:
        return self.field[0]
            
    def return_sum_of_least_two(self) -> int:
        if len(self.field) >= 2:
            return self.field[0] + self.field[1]
        else:
            return self.return_sum_of_least()
        
    def return_value_of_least_double(self) -> int:
        last_value = -1
        for value in self.field:
            if value == last_value:
                return value
            last_value = value
        return last_value

game = DieGame()
print(game.field)
print(game.action_mask)

class Environment:
    def __init__(self) -> None:
        self.field = [0,0,0,0,0]
        self.hand = [0,0,0,0,0]
        self.to_beat = 0
        self.complete = False
        self.selected_dice = 0
        self.pick_number = 0

    
    def reset(self, to_beat=None):
        self.selected_dice = 0
        self.roll()
        self.hand = [0,0,0,0,0]
        self.complete = False
        if to_beat is None:
            self.random_to_beat()
        else:
            self.to_beat = to_beat
        self.pick_number = 0
        
        return self.state()

    def step(self, action):
        if not self.select(action):
            return self.state(), -.1, self.complete
        return self.state(), self.reward(), self.complete

    def state(self):
        return self.hand.copy() + self.field.copy() + [self.to_beat] + [self.pick_number]

    def random_to_beat(self):
        #self.to_beat = max(round(np.random.normal(8,3)),0)
        if self.to_beat is None:
            self.to_beat = 0
        else:
            self.to_beat += 1
        if self.to_beat > 10:
            self.to_beat = 0
    
    def roll(self):
        self.field = []
        for _ in range(5-self.selected_dice):
            rand_append = 3
            while rand_append == 3:
                rand_append = random.randint(0,6)
            self.field.append(rand_append)
        self.field.sort()
        for _ in range(self.selected_dice):
            self.field.append(0)

    def score(self) -> int:
        total = 0
        if self.selected_dice == 5 and self.get_len_hand_set() == 1:
            return 0
        for die in self.hand:
            if die != 3:
                total += die
        return total

    def get_len_hand_set(self) -> int:
        return len(set(self.hand[0:self.selected_dice]))
    
    def lost(self) -> bool:
        return self.score() > self.to_beat and self.get_len_hand_set() != 1
    
    def won(self) -> bool:
        return self.selected_dice == 5 and (self.score() < self.to_beat or self.get_len_hand_set() == 1)
    
    def tied(self) -> bool:
        return self.selected_dice == 5 and (self.score() == self.to_beat or self.to_beat == 0 and self.get_len_hand_set() == 1)
    
    def flush_win(self) -> bool:
        return (self.tied() or self.won()) and self.get_len_hand_set() == 1

    def action_space(self) -> int:
        return 5-self.selected_dice
    
    def select(self, action) -> bool:
        if self.pick_number == 1 and action >= self.action_space():
            return False
        self.pick_number += 1
        if action >= self.action_space():
            return True
        self.hand[self.selected_dice] = self.field.pop(action)
        relevant_dice = self.hand[0:self.selected_dice+1]
        relevant_dice.sort()
        self.hand = relevant_dice + self.hand[self.selected_dice+1:]
        self.field.append(0)
        self.selected_dice += 1
        if self.lost() or self.selected_dice == 5:
            self.complete = True
            self.roll()
        if self.pick_number > 1:
            self.pick_number = 0
            self.roll()
        return True


    def reward(self) -> int:
        # Bigger reward for low value hands and hands that have nothing but duplicates
        if self.lost():
            return -1 + self.selected_dice/10
        if self.won():
            return 0.8 + (20 - self.score())/100
        if self.tied():
            return 0.4 + (20 - self.score())/100
        same_mod = 0 
        if self.get_len_hand_set() == 1:
            same_mod = self.selected_dice**2
        return (7-self.score())/1000+self.selected_dice/1000+same_mod/1000
