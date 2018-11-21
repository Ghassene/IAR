import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# world height
WORLD_HEIGHT = 12

# world width
WORLD_WIDTH = 18

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# gamma for Q-Learning and Expected Sarsa
GAMMA = 1

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
START = [0, 4]
GOAL = [17, 0]

def step(state, action):
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        assert False

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
        action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward

# reward for each action in each state
# actionRewards = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
# actionRewards[:, :, :] = -1.0
# actionRewards[2, 1:11, ACTION_DOWN] = -100.0
# actionRewards[3, 0, ACTION_RIGHT] = -100.0

# set up destinations for each action in each state
# actionDestination = []
# for i in range(0, WORLD_HEIGHT):
#     actionDestination.append([])
#     for j in range(0, WORLD_WIDTH):
#         destinaion = dict()
#         destinaion[ACTION_UP] = [max(i - 1, 0), j]
#         destinaion[ACTION_LEFT] = [i, max(j - 1, 0)]
#         destinaion[ACTION_RIGHT] = [i, min(j + 1, WORLD_WIDTH - 1)]
#         if i == 2 and 1 <= j <= 10:
#             destinaion[ACTION_DOWN] = START
#         else:
#             destinaion[ACTION_DOWN] = [min(i + 1, WORLD_HEIGHT - 1), j]
#         actionDestination[-1].append(destinaion)
# actionDestination[3][0][ACTION_RIGHT] = START

# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, step_size=ALPHA):
    state = START
    rewards = 0.0
    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        # Q-Learning update
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
    return rewards

# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)

# Use multiple runs instead of a single run and a sliding window
# With a single run I failed to present a smooth curve
# However the optimal policy converges well with a single run
# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def figure_6_4():
    # episodes of each run
    episodes = 500

    # perform 40 independent runs
    runs = 50

    rewards_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        q_q_learning =np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        for i in range(0, episodes):
            # cut off the value by -100 to draw the figure more elegantly
            # rewards_sarsa[i] += max(sarsa(q_sarsa), -100)
            # rewards_q_learning[i] += max(q_learning(q_q_learning), -100)
            rewards_q_learning[i] += q_learning(q_q_learning)

    # averaging over independt runs
    rewards_q_learning /= runs

    # draw reward curves
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('figure_6_4.png')
    plt.close()


    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)

# Due to limited capacity of calculation of my machine, I can't complete this experiment
# with 100,000 episodes and 50,000 runs to get the fully averaged performance
# However even I only play for 1,000 episodes and 10 runs, the curves looks still good.
def figure_6_6():
    step_sizes = np.arange(0.1, 1.1, 0.1)
    episodes = 1000
    runs = 10


    ASY_QLEARNING = 2

    INT_QLEARNING = 5
    methods = range(0, 6)

    performace = np.zeros((6, len(step_sizes)))
    for run in range(runs):
        for ind, step_size in tqdm(list(zip(range(0, len(step_sizes)), step_sizes))):
            q_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
            for ep in range(episodes):
                q_learning_reward = q_learning(q_q_learning, step_size=step_size)

                performace[ASY_QLEARNING, ind] += q_learning_reward

                if ep < 100:
                    performace[INT_QLEARNING, ind] += q_learning_reward

    performace[:3, :] /= episodes * runs
    performace[3:, :] /= 100 * runs
    labels = [ 'Asymptotic Q-Learning', 'Interim Q-Learning']

    for method, label in zip(methods, labels):
        plt.plot(step_sizes, performace[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()

    plt.savefig('figure_6_6.png')
    plt.close()

if __name__ == '__main__':
    figure_6_4()
    figure_6_6()