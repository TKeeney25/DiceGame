import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from numpy import average, mean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment import Environment

env = Environment()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        mid_layer = 22
        self.layer1 = nn.Linear(n_observations, mid_layer)
        self.layer2 = nn.Linear(mid_layer, mid_layer)
        self.layer3 = nn.Linear(mid_layer, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 1.5e-2

# Get number of actions from gym action space
n_actions = env.action_space()+1
# Get the number of state observations
state = env.reset()
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, env.action_space())]], device=device, dtype=torch.long)


episode_scores = []
episode_by_opponent_scores = {}

def plot_durations(show_result=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    plt.plot(running_winrate)
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

wins = []
running_winrate = []
episodes = 0
flush_wins = 0
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 1000
while True:
    selection = int(input('1: Training\n2: Competition\n3: Assistance\n4: Exit\n'))
    win_num = None
    log = False

    if selection == 1:
        num_episodes = int(input('Num episodes: '))
    elif selection == 4:
        break
    else:
        win_num = int(input('Enter winning number: '))
        log = True
        num_episodes = 1
    for i_episode in range(num_episodes):
        episodes += 1
        # Initialize the environment and get it's state
        state = env.reset(win_num)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            if selection == 3 and env.pick_number == 0:
                rolls = input('Enter Comma Seperated Rolls: ')
                roll_list = rolls.split(',')
                roll_int_list = []
                for roll in roll_list:
                    roll_int_list.append(int(roll))
                roll_int_list.sort()
                for i in range(5-len(roll_int_list)):
                    roll_int_list.append(0)
                env.field = roll_int_list
                state = env.state()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action(state)
            observation, reward, terminated = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated

            if terminated:
                if env.score() < env.to_beat:
                    wins.append(1)
                    if env.flush_win():
                        flush_wins += 1
                else:
                    wins.append(0)
                if len(wins) > 100:
                    wins.pop(0)
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            if log:
                print(state, action.item(), next_state, reward.item(), terminated)
                if terminated:
                    print(env.score())
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                if env.to_beat not in episode_by_opponent_scores:
                    episode_by_opponent_scores[env.to_beat] = [env.score()]
                else:
                    episode_by_opponent_scores[env.to_beat].append(env.score())
                running_winrate.append(sum(wins)/10)
                episode_scores.append(env.score())
                plot_durations()
                break
        if i_episode % 100 == 0:
            print(sum(wins), flush_wins)

    print('SCORES:')
    print(running_winrate[-1])
    dict_keys = list(episode_by_opponent_scores.keys())
    dict_keys.sort()
    for opponent_score in dict_keys:
        print(opponent_score, mean(episode_by_opponent_scores[opponent_score]), 1-(sum(i > opponent_score for i in episode_by_opponent_scores[opponent_score])/len(episode_by_opponent_scores[opponent_score])))
    print('')

torch.save(target_net.state_dict(),'torchfile.onnx')
print(target_net.state_dict())
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()