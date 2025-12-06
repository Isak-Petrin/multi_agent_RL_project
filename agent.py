from gym_env.clash_royale_env import ClashRoyaleEnv
import time

env = ClashRoyaleEnv()
obs, _ = env.reset()
# Seed ego observations for both sides (bottom is direct, top is flipped)
obs_bottom = obs
obs_top = env.get_ego_observation(friendly=False)
done = False

while not done:
    # Sample random actions for both agents (bottom and top). Action 0 is no-op.
    bottom_action = env.action_space.sample()
    top_action = env.action_space.sample()

    (obs_bottom, obs_top), (reward_bottom, reward_top), done, _, info = env.step_self_play(
        bottom_action, top_action
    )
    env.render()

env.close()
