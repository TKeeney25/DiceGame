import random
from typing import List, Tuple

import numpy as np
from sympy import false


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
