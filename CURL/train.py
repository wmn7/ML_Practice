'''
@Author: WANG Maonan
@Date: 2023-03-13 19:58:24
@Description: 主文件，训练 agent
- 关注 sample_cpc, update, update_cpc, compute_logits
@LastEditTime: 2023-03-13 23:30:43
'''
import numpy as np
import torch
import os
import time

import gymnasium as gym

from utils.utils import set_seed_everywhere, make_dir, center_crop_image, eval_mode, FrameStack, ReplayBuffer
from utils.logger import Logger
from utils.get_abs_path import getAbsPath

from models.curl_sac import CurlSacAgent


def evaluate(env, agent, num_episodes, L, step):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset(seed=1)
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                obs = center_crop_image(obs, output_size=84) # 中心裁减
                with eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(
        obs_shape, action_shape, device,
        # sac
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_beta=0.5,
        # actor
        actor_lr=1e-3,
        actor_beta=0.9, # Adam 参数
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        # critic
        critic_lr=1e-3,
        critic_beta=0.9, 
        critic_tau=0.01,
        critic_target_update_freq=2,
        # encoder
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4, # 卷机层数
        num_filters=32, # output channel
        curl_latent_dim=128,
        hidden_dim=128, 
        discount=0.99
    ):
    return CurlSacAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=hidden_dim,
        discount=discount,
        init_temperature=init_temperature,
        alpha_lr=alpha_lr,
        alpha_beta=alpha_beta,
        actor_lr=actor_lr,
        actor_beta=actor_beta,
        actor_log_std_min=actor_log_std_min,
        actor_log_std_max=actor_log_std_max,
        actor_update_freq=actor_update_freq,
        critic_lr=critic_lr,
        critic_beta=critic_beta,
        critic_tau=critic_tau,
        critic_target_update_freq=critic_target_update_freq,
        encoder_type=encoder_type,
        encoder_feature_dim=encoder_feature_dim,
        encoder_lr=encoder_lr,
        encoder_tau=encoder_tau,
        num_layers=num_layers,
        num_filters=num_filters,
        log_interval=True,
        detach_encoder=True,
        curl_latent_dim=curl_latent_dim

    )

def main(
        image_size:int=84, # 进行转换后的图片大小
        pre_transform_image_size=96, # 原始图片大小
        # reply buffer
        replay_buffer_capacity=100000,
        # train
        num_train_steps=1e6,
        init_steps=100,
        # eval
        eval_freq=1000,
        num_eval_episodes=10, # 测试的次数
        log_interval=100,
        frame_stack:int=3, 
        batch_size:int=32, 
        encoder_type='pixel'
    ):
    num_train_steps = int(num_train_steps)
    pathConvert = getAbsPath(__file__)
    set_seed_everywhere(1) # 设置随机数种子
    env = gym.make("CarRacing-v2")
    
    # stack several consecutive frames together
    env = FrameStack(env, k=frame_stack)
    
    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = 'CarRacing' # 环境的名字
    exp_name = 'Results-' + env_name + '-' + ts + '-im' + str(image_size) +'-b' + str(batch_size) + '-' + encoder_type
    work_dir = pathConvert(f'./{exp_name}')

    make_dir(work_dir)
    model_dir = make_dir(os.path.join(work_dir, 'model'))
    buffer_dir = make_dir(os.path.join(work_dir, 'buffer'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape # 连续动作空间

    if encoder_type == 'pixel':
        obs_shape = (3*frame_stack, image_size, image_size)
        pre_aug_obs_shape = (3*frame_stack, pre_transform_image_size, pre_transform_image_size)

    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=replay_buffer_capacity,
        batch_size=batch_size,
        device=device,
        image_size=image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device
    )

    L = Logger(work_dir, use_tb=True) # logger

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(num_train_steps):
        # evaluate agent periodically
        if step % eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, num_eval_episodes, L, step)
            agent.save_curl(model_dir, step)
            replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset(seed=1)
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < init_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step) # 更新网络

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
