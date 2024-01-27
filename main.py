import datetime
from pathlib import Path
from agent import Player
from environment import Environment
from metrics import MetricLogger
# https://github.com/yfeng997/MadMario/blob/master/main.py

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
env = Environment()
checkpoint = None

player = Player(state_dim=16, action_dim=25, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 400

for e in range(episodes):
    env.reset()
    while not env.complete:
        state = env.state()
        action = player.act(state)
        env.select(action)
        player.cache(state, env.state(), action, env.reward(), env.complete)
        q, loss = player.learn()
        logger.log_step(env.reward(), loss, q)

    logger.log_episode()
    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=player.exploration_rate,
            step=player.curr_step
        )
