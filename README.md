# Reinforcement Learning

Below is the reinforcement Learning taxonomy as defined by OpenAI ([source](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)):

![test](./taxonomy.jpg)

In the model-based RL approach, the agent learns both the reward function and the transition probabilities, enabling it to simulate future states and plan accordingly. The added complexity and improved efficiency make it a more advanced topic than model-free approaches, which we will discuss in the future instead.

For Model-Free RL problems, Policy Optimization learns the policy itself (probability distribution over actions) through iterating over policy parameters to directly optimize action selection, and is ideal for continuous or large state/action spaces, often used with neural networds. A-learns the value of actions by iterating over Q-values (state-action values) to estimate the best action for each state, and is best for discrete state and action spaces.

Below, we start with the model-free RL Q-learning approach on classic problems: (1) Gridworld and (2) Acrobot.

## GridWorld
### Problem Statement:
In an n x n grid, there exists an agent, blocks, and bombs. The agent has four possible actions in each state (up, down, left, right), but the actions are unreliable. If the direction of the movement is blocked, the agent remains in the same grid square. The grid squares with the gold and the bomb are terminal states. Compute the best policy for finding the gold in as few steps as possible while avoiding the bomb. Compare the results from MDP vs. Q-learning.


### Formula for MDP and Q-Learning
This reinfrocemnt learning problem can be defined by `(S, A, P, R, γ)`, where `S` is the set of states, `A` as the set of actions, `P(s' | s, a)` the transition probability function, `R(s, a)` the reward function, and `γ` the discount factor of future reward. 

Other symbols used below include `π(a|s)` as the policy (the probability of choosing action a in states s), `V(s)` as the function value (expected long-term reward after taking action a in state s), and `Q(s,a)` as the action-value function (expected long-term reward after taking action a in state s), and `α` as the learning rate.

In the two approaches suggested in the prompt, the **MDP** (Markov Decision Process; off-line learning) solve the problem using Value Iteration or Policy Iteration, while the **Q-learning** (model-free; on-line learning) learn the policy through exploration and reward.

More specifically, in **MDP**, the agent iteratively updates the value of each state `V(s)` by:

$$
V(s) \leftarrow \max_a \sum_{s'} P(s' | s, a)[R(s,a) + \gamma V(s')].
$$

This update continues until the value function converges to the optimal values, after which the optimal action at each state is that which maximize the expected return. Alternatively, MDP can also be solved with policy iteration, where the policy updated with:

$$
\pi'(s) \leftarrow \arg \max_a \sum_{s'} P(s'|s,a)(R(s,a,s') + \gamma V^\pi (s')),
$$

until the policy converges. 

Generally, value iteration is preferred when the state space is relatively small, and direct updates to the value function are simpler and more intuitive, while policy iteration is preferred when the policy evaluation can be performed efficiently, or when the state space is large and value iteration is costly. 

**Q-learning**, on the other hand, focuses on learning the action-value function `Q(s,a)` and updates it by:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R(s_t, a_t) +\gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)],
$$

usually for a fixed number of episodes.

Unlike MDPs, **<span style="color:#C05780">Q-learning doesn't need the transition probabilities `P(s'|s,a)`, as it directly learns from interactions.</span>** Additionally, MDPs use a convergence threshold since they operate on a known model and update values until they no longer change significantly. In contrast, Q-learning typically uses a fixed number of episodes because it learns from interactions and does not have a clear convergence criterion due to the stochastic nature of exploration. The number of episodes is sufficient when Q-learning converges, meaning all Q-values for every state-action pair stabilize. To check this, plot the maximum change in Q-values over iterations. If this value approaches zero, it indicates convergence.

## Classic Control with Gymnasium

Gymnasium is a toolkit for building and testing reinforcement learning algorithms. **It provides various environments**, such as games and simulations, including the observation and action spaces, where we can train reinforcement learning agents. The environments are standardized, making it easy to experiment with and compare algorithms across tasks. In the scripts, we go through the [classic environments](https://gymnasium.farama.org/environments/classic_control/)—Acrobot as starting points for experimenting with reinforcement learning algorithms like Q-learning, Policy Gradients, and Deep Deterministic Policy Gradient (DDPG).

### Acrobot

In the acrobot environment, the system consists of two links connected linearly to form a chain, with one end of the chain fixed. The joint between the two links is actuated. The goal is to apply toques on the actuated ojint to swing the free end of the linear chain above a given height while starting from the initial state of hanging downwards.

The **action space** is discrete, deterministic, and represents the torque applied on the actuated joint between the two links. 

- 0: apply -1 torque to the acuated joint
- 1: apply 1 torque to the actuated joint
- 2: apply 1 torque to the actuated joint.

The **observation space** is a ndarray with shape (6,) that provides information about the two rotational joint angles as well as their angular velocities.

- 0: cosine of theta1
- 1: sine of theta1
- 2: cosine of theta2
- 3: sine of theta 2
- 4: angular velocity of theta1
- 5: angular velocity of theta2,

where theta1 is the angle of the first join, and theta2 is relative to the angle of the first link.

The free end achiving the target height results in termination with a **reward** of 0, and all steps that do not reach the goal incur a reward of -1. An agent is considered sucessful if it achieves an average cumulative reward of -100.

At the **starting state**, each parameter is initialized uniformly between -0.1 and 0.1. The **episode ends** if the free end reaches the target height, or the episode length is greater than 500.

The script uses two different approaches, the **Deep Q-Network (DQN)** algorithm aims to minimize the Bellman equation error, and the **Proximal Policy Optimization (PPO)** which aims to optimize the policy by constraining large ipdates.

For **DQN**, the neural network is updated by minimizing the difference between the predicted Q-value (from the network) and the target Q-value (which is computed using the reward plus the estimated value of the next state). This difference (loss) is minimized using gradient descent, and the network learns to predict better Q-values.

For **PPO**, the network has two outputs: policy (action probabilities) and value function (state values). The policy is updated by maximizing the clipped objective function, ensuring small updates, while the value function is updated by minimizing the error between predicted and actual returns.