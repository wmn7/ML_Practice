'''
@Author: WANG Maonan
@Date: 2021-11-02 11:21:09
@Description: LunarLander-v2 环境测试
@LastEditTime: 2021-11-02 11:51:44
'''
import gym

env = gym.make('LunarLander-v2')

# 查看 obs
print(env.observation_space)

# 查看 action
print(env.action_space)

# 随机策略
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    env.render(mode='rgb_array')