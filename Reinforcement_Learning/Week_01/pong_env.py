'''
@Author: WANG Maonan
@Date: 2021-11-02 14:33:14
@Description: 对环境「pong-ram-v0」和「pong-v0」环境的理解
@LastEditTime: 2021-11-02 16:54:32
'''
import gym

# need to install the following package
# pip install 'gym[atari]'
# pip install 'gym[accept-rom-license]'

env = gym.make("Pong-ram-v0") # Pong-v0 or Pong-ram-v0

# 查看 obs
print(env.observation_space)

# 查看 action
print(env.action_space)

env.reset()
env.render()
import time;time.sleep(10)
# 随机策略
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print('obs: {}; reward: {}'.format(observation.shape, reward))
    env.render()
