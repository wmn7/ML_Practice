'''
@Author: WANG Maonan
@Date: 2023-03-13 20:13:50
@Description: 
@LastEditTime: 2023-03-13 22:06:53
'''
import cv2
import gymnasium as gym
from utils.get_abs_path import getAbsPath

if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)
    env = gym.make("CarRacing-v2", render_mode='rgb_array') # rgb_array or human
    observation, info = env.reset(seed=42)
    
    done = False
    while not done:
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation.shape, terminated, truncated)
        
        # Save Image
        img_path = pathConvert('./save_obs.png')
        cv2.imwrite(img_path, observation)

        if terminated or truncated:
            done = True
    env.close()