'''
@Author: WANG Maonan
@Date: 2023-03-13 21:26:31
@Description: 测试 crop image 的效果
@LastEditTime: 2023-03-13 22:17:59
'''
import cv2
import numpy as np
import gymnasium as gym
from utils.get_abs_path import getAbsPath

from utils.utils import center_crop_image

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
        observation = np.transpose(observation, (2,0,1))
        _center_crop = center_crop_image(observation, 48)
        cv2.imwrite(pathConvert('./save_obs.png'), np.transpose(observation, (1,2,0)))
        cv2.imwrite(pathConvert('./save_obs_center.png'), np.transpose(_center_crop, (1,2,0)))

        if terminated or truncated:
            done = True
    env.close()