import gym
import random
import time
# make in human mode
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", frameskip=4)

#launch the game
env.reset()
done = False
while not done:
    action = random.choice([0,1,2,3,4,5,6,7,8,9,10,11])
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)
    env.render()
env.close()