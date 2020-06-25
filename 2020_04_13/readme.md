# 2020_04_13

## 强化学习, Reinforcement Learning

### 第一课时部分
- 00_Reinforcement_Learning_Gym介绍 (关于工具gym的详细介绍)

### 第三课时部分
- 01_Example_of_PolicyEvaluation (一个介绍Policy Evaluation的文件, 也就是给定策略, 对策略进行评价)
- 02_Example_of_PolicyIterative.ipynb (从给定policy, 进行迭代)
- 03_Example_of_ValueIteration.ipynb (value iteration的例子)

### 第四课时部分
- 04_BlackJack_Playground.ipynb (21点的环境的介绍)
- 04_Monte-Carlo_normalMean.ipynb (使用MC方法来估计value function, 这里使用普通的求平均的方式)
- 04_Monte-Carlo_IncrementalMean.ipynb (使用MC方法来估计value function, 这里使用incremental mean)
- 04_Temporal-Difference_BlackJack.ipynb (使用TD方法来估计value function)
- 04_Eligibility_traces_BlackJack.ipynb (使用了Eligibility traces来进行更新)

### 第五课时部分

- 环境介绍
    - 05_Windy_Gridworld_Playground.ipynb (环境Windy Gridworld Playground的介绍);
- On Policy Learning介绍
    - MC
        - 05_GLIE_BlackJack.ipynb (使用MC方法来优化策略, 使用GLIE算法, 给出最优策略, 用在BlackJack上面);
        - 05_GLIE_Windy_Gridworld.ipynb (使用MC方法, 解决Windy Gridworld上面);
    - TD
        - 05_Sarsa_Windy_Gridworld.ipynb (使用最基础的Sarsa算法, 在windy gridworld中寻找最优路径);
        - 05_Sarsa_Lambda_Windy_Gridworld.ipynb (使用Sarsa($\lambda$)方法, 也就是使用eligibility traces);
- Off Policy Learning介绍
    - MC
        - 05_Importance_Sampling_Random_Eligibility_Traces_Windy_Gridworld.ipynb (使用MC+OFF policy的方式, 但是没有收敛)
    - TD
        - 05_Importance_Sampling_Random_Windy_Gridworld.ipynb (使用importance sampling, 并且执行的policy为随机policy, 即每个action执行的概率相同)
        - 05_Importance_Sampling_Q_Windy_Gridworld.ipynb (q-learning在先前版本)
        - 05_Q-Learning_Windy_Gridworld.ipynb (q-learning介绍)
