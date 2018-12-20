#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang                                           #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy
from scipy.spatial import distance
from collections import defaultdict


class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder


# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:
    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    # extend a state to a higher resolution maze
    # @state: state in lower resoultion maze
    # @factor: extension factor, one state will become factor^2 states after extension
    def extend_state(self, state, factor):
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states

    # extend a state into higher resolution
    # one state in original maze will become @factor^2 states in @return new maze
    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        new_maze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions))
        # new_maze.stateActionValues = np.zeros((new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions)))
        new_maze.resolution = factor
        return new_maze

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward

    def stepSt(self, state, action):
        x, y = state
        nb = np.random.rand()
        if action == self.ACTION_UP:
            if (nb < 2 / 3):
                x = max(x - 1, 0)
            elif (nb < 5 / 6):
                y = max(y - 1, 0)
            else:
                y = min(y + 1, self.WORLD_WIDTH - 1)

        elif action == self.ACTION_DOWN:
            if (nb < 2 / 3):
                x = min(x + 1, self.WORLD_HEIGHT - 1)
            elif (nb < 5 / 6):
                y = max(y - 1, 0)
            else:
                y = min(y + 1, self.WORLD_WIDTH - 1)

        elif action == self.ACTION_LEFT:
            if (nb < 2 / 3):
                y = max(y - 1, 0)
            elif (nb < 5 / 6):
                x = max(x - 1, 0)
            else:
                x = min(x + 1, self.WORLD_HEIGHT - 1)

        elif action == self.ACTION_RIGHT:
            if (nb < 2 / 3):
                y = min(y + 1, self.WORLD_WIDTH - 1)
            elif (nb < 5 / 6):
                x = max(x - 1, 0)
            else:
                x = min(x + 1, self.WORLD_HEIGHT - 1)

        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward


# a wrapper class for parameters of dyna algorithms
class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q']

        # threshold for priority queue
        self.theta = 0


# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward

# Heuristic model for planning in Dyna-H
class HeuristicModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward

    #to verif if an action still in model
    def verif(self, state, action):
        if tuple(state) not in self.model.keys() and action not in self.model[state]:
            return True
        return False

#calculate the eucleudean distance between current state and goal state
def H(state, action, maze):
    next, reward = maze.step(state, action)
    return distance.euclidean(next, maze.GOAL_STATES[0])


# choose an action based on heuristic algorithm
def action_h(state, maze):
    tableof = []
    maxval = 0
    for i in maze.actions:
        ha = H(state, i, maze)
        if (ha >= 0):
            if (ha > maxval):
                maxval = ha
                tableof.append(i)
            elif (ha == maxval):
                tableof.append(i)
    if len(tableof) > 0:
        return np.random.choice(tableof)
    else:
        return np.random.choice(maze.actions)


# Model containing a priority queue for Prioritized Sweeping
class PriorityModel(TrivialModel):
    def __init__(self, rand=np.random):
        TrivialModel.__init__(self, rand)
        # maintain a priority queue
        self.priority_queue = PriorityQueue()
        # track predecessors for every state
        self.predecessors = dict()

    # add a @state-@action pair into the priority queue with priority @priority
    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        self.priority_queue.add_item((tuple(state), action), -priority)

    # @return: whether the priority queue is empty
    def empty(self):
        return self.priority_queue.empty()

    # get the first item in the priority queue
    def sample(self):
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return -priority, list(state), action, list(next_state), reward

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        TrivialModel.feed(self, state, action, next_state, reward)
        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        self.predecessors[tuple(next_state)].add((tuple(state), action))

    # get all seen predecessors of a state @state
    def predecessor(self, state):
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[tuple(state)]):
            predecessors.append([list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors


# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_q(q_value, model, maze, dyna_params):
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # Q-Learning update
        q_value[state[0], state[1], action] += dyna_params.alpha * (
                reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) - q_value[
            state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * (
                    reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                    q_value[state_[0], state_[1], action_])
        state = next_state

        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break

    return steps

# play for an episode for Dyna-H algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_h(q_value, model, maze, dyna_params):
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # Q-Learning update
        q_value[state[0], state[1], action] += dyna_params.alpha * (
                reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) - q_value[
            state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            action = action_h(state, maze)
            if (model.verif(state, action)):
                state_, action_, next_state_, reward_ = model.sample()
            else:
                model.feed(state, action, next_state, reward)
                state_, action_, next_state_, reward_ = model.sample()

            q_value[state_[0], state_[1], action_] += dyna_params.alpha * (
                    reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - q_value[
                state_[0], state_[1], action_])
        state = next_state

        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break

    return steps


# play for an episode for prioritized sweeping algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
# @return: # of backups during this episode
def prioritized_sweeping(q_value, model, maze, dyna_params):
    state = maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while state not in maze.GOAL_STATES:
        steps += 1

        # get action
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # feed the model with experience
        model.feed(state, action, next_state, reward)
        # get the priority for current state action pair
        priority = np.abs(reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                          q_value[state[0], state[1], action])

        if priority > dyna_params.theta:
            model.insert(priority, state, action)

        # start planning
        planning_step = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_step < dyna_params.planning_steps and not model.empty():
            # get a sample with highest priority from the model
            priority, state_, action_, next_state_, reward_ = model.sample()

            # update the state action value for the sample
            delta = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - \
                    q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta

            # deal with all the predecessors of the sample state
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + dyna_params.gamma * np.max(q_value[state_[0], state_[1], :]) -
                                  q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_step += 1

        state = next_state

        # update the # of backups
        backups += planning_step + 1

    return backups


# Figure 9, 30*30 Maze
class Maze_figure9():
    def __init__(self):
        # set up a blocking maze instance
        self.WORLD_HEIGHT = 30
        self.WORLD_WIDTH = 30
        self.START_STATE = [0, 16]
        self.GOAL_STATES = [[0, 28], [0, 29]]
        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        # new obstalces will block the optimal path
        self.obstacles = [[3, 2], [3, 3], [4, 2], [11, 5], [12, 5], [13, 5], [16, 27]]
        self.obstacles.extend([25, i] for i in np.arange(17))
        self.obstacles.extend([26, i] for i in np.arange(17))
        self.obstacles.extend([27, i] for i in np.arange(17))
        self.obstacles.extend([i, 12] for i in np.arange(12, 21))
        self.obstacles.extend([i, 13] for i in np.arange(12, 21))
        self.obstacles.extend([i, 14] for i in np.arange(8, 24))
        self.obstacles.extend([i, 15] for i in np.arange(8, 24))
        self.obstacles.extend([i, j] for i in np.arange(7, 25) for j in np.arange(18, 27))

        # step limit
        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

        # extend a state to a higher resolution maze
        # @state: state in lower resoultion maze
        # @factor: extension factor, one state will become factor^2 states after extension

    def extend_state(self, state, factor):
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states

        # extend a state into higher resolution
        # one state in original maze will become @factor^2 states in @return new maze

    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        new_maze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions))
        # new_maze.stateActionValues = np.zeros((new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions)))
        new_maze.resolution = factor
        return new_maze

        # take @action in @state
        # @return: [new state, reward]

    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward

    def stepSt(self, state, action):
        x, y = state
        nb = np.random.rand()
        if action == self.ACTION_UP:
            if (nb < 2 / 3):
                x = max(x - 1, 0)
            elif (nb < 5 / 6):
                y = max(y - 1, 0)
            else:
                y = min(y + 1, self.WORLD_WIDTH - 1)

        elif action == self.ACTION_DOWN:
            if (nb < 2 / 3):
                x = min(x + 1, self.WORLD_HEIGHT - 1)
            elif (nb < 5 / 6):
                y = max(y - 1, 0)
            else:
                y = min(y + 1, self.WORLD_WIDTH - 1)

        elif action == self.ACTION_LEFT:
            if (nb < 2 / 3):
                y = max(y - 1, 0)
            elif (nb < 5 / 6):
                x = max(x - 1, 0)
            else:
                x = min(x + 1, self.WORLD_HEIGHT - 1)

        elif action == self.ACTION_RIGHT:
            if (nb < 2 / 3):
                y = min(y + 1, self.WORLD_WIDTH - 1)
            elif (nb < 5 / 6):
                x = max(x - 1, 0)
            else:
                x = min(x + 1, self.WORLD_HEIGHT - 1)

        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward


# Check whether state-action values are already optimal
def check_path(q_values, maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    max_steps = 14 * maze.resolution * 1.2
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        state, _ = maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False
    return True

# figure 7, mazes with different resolution
def figure7():
    # get the original 6 * 9 maze
    original_maze = Maze()

    # set up the parameters for each algorithm
    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95

    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, params_dyna, params_dyna]

    # set up models for planning
    models = [PriorityModel, TrivialModel, HeuristicModel]
    method_names = ['Largest 1st Dyna', 'Random Dyna', 'Focused Dyna']

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # assuming the 1st maze has w * h states, then k-th maze has w * h * k * k states
    num_of_mazes = 8
    # build all the mazes
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q, dyna_h]

    # My machine cannot afford too many runs...
    runs = 100

    # track the # of backups
    backups = np.zeros((runs, 3, num_of_mazes))

    for run in tqdm(range(0, runs)):
        for i in range(0, len(method_names)):
            for mazeIndex, maze in zip(range(0, len(mazes)), mazes):
                # print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * (maze.WORLD_WIDTH - 1)))

                # initialize the state action values
                q_value = np.zeros(maze.q_size)

                # track steps / backups for each episode
                steps = []

                # generate the model
                model = models[i]()
                # play for an episode
                while True:
                    steps.append(methods[i](q_value, model, maze, params[i]))

                    # print best actions w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)
                    # check whether the (relaxed) optimal path is found
                    if check_path(q_value, maze):
                        break

                # update the total steps / backups for this maze
                backups[run, i, mazeIndex] = np.sum(steps)

    backups = backups.mean(axis=0)

    # Dyna-Q performs several backups per step
    backups[1, :] *= params_dyna.planning_steps + 1

    for i in range(0, len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.savefig('figure7.png')
    plt.close()

# figure 8, performance of maze figure 6
def figure8():
    original_maze = Maze()
    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95
    # track the # of backups
    backups = defaultdict(list)

    runs = 100
    episodes = 100
    steps = []
    params = [params_prioritized, params_dyna]

    # set up models for planning
    models = [PriorityModel, TrivialModel]
    method_names = ['Largest 1st Dyna', 'Random Dyna']

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # assuming the 1st maze has w * h states, then k-th maze has w * h * k * k states
    num_of_mazes = 1

    # build all the mazes
    mazes = [original_maze.extend_maze(i * 4) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q]

    for run in tqdm(range(runs)):
        for planning_step in range(params_dyna.planning_steps):
            params_dyna.planning_steps = planning_step
            q_value = np.zeros(mazes[0].q_size)

            # generate an instance of Dyna-H model
            model = HeuristicModel()
            while True:  # for ep in range(episodes):
                # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                steps.append(dyna_h(q_value, model, mazes[0], params_dyna))
                if check_path(q_value, mazes[0]):
                    break

    for run in tqdm(range(0, runs)):
        for i in range(0, len(method_names)):
            for mazeIndex, maze in zip(range(0, len(mazes)), mazes):
                # print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * maze.WORLD_WIDTH))

                # initialize the state action values
                q_value = np.zeros(maze.q_size)

                # track steps / backups for each episode
                stepss = []

                # generate the model
                model = models[i]()

                # play for an episode
                while True:
                    stepss.append(methods[i](q_value, model, maze, params[i]))

                    # print best actions w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)
                    # check whether the (relaxed) optimal path is found
                    if check_path(q_value, maze):
                        break
                # print("steps", steps)

                # update the total steps / backups for this maze
                backups[method_names[i]].extend(stepss)

    # Dyna-Q performs several backups per step
    for i in method_names:
        # backups[i] *= dyna_params.planning_steps + 1
        # print(backups[i])
        plt.plot(sorted(backups[i], reverse=True), label=i)

    plt.plot(sorted(steps, reverse=True), label='Focused Dyna')
    plt.xlabel('Backups')
    plt.ylabel('Steps to Goal')
    plt.ylim((0, 10000))
    plt.xscale('log')
    plt.legend()

    plt.savefig('figure8.png')
    plt.close()


def figure10():
    original_maze = Maze_figure9()
    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95
    # track the # of backups
    backups = defaultdict(list)

    runs = 100
    episodes = 100
    steps = []
    params = [params_prioritized, params_dyna]

    # set up models for planning
    models = [PriorityModel, TrivialModel]
    method_names = ['Largest 1st Dyna', 'Random Dyna']

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # assuming the 1st maze has w * h states, then k-th maze has w * h * k * k states
    num_of_mazes = 1

    # build all the mazes
    methods = [prioritized_sweeping, dyna_q]

    for run in tqdm(range(runs)):
        for planning_step in range(params_dyna.planning_steps):
            params_dyna.planning_steps = planning_step
            q_value = np.zeros(original_maze.q_size)

            # generate an instance of Dyna-H model
            model = HeuristicModel()
            while True:  # for ep in range(episodes):
                # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                steps.append(dyna_h(q_value, model, original_maze, params_dyna))
                if check_path(q_value, original_maze):
                    break

    for run in tqdm(range(0, runs)):
        for i in range(0, len(method_names)):
            # print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * maze.WORLD_WIDTH))

            # initialize the state action values
            q_value = np.zeros(original_maze.q_size)

            # track steps / backups for each episode
            stepss = []

            # generate the model
            model = models[i]()

            # play for an episode
            while True:
                stepss.append(methods[i](q_value, model, original_maze, params[i]))

                # print best actions w.r.t. current state-action values
                # printActions(currentStateActionValues, maze)
                # check whether the (relaxed) optimal path is found
                if check_path(q_value, original_maze):
                    break
            # print("steps", steps)

            # update the total steps / backups for this maze
            backups[method_names[i]].extend(stepss)

    # Dyna-Q performs several backups per step
    for i in method_names:
        plt.plot(sorted(backups[i], reverse=True), label=i)

    plt.plot(sorted(steps, reverse=True), label='Focused Dyna')
    plt.xlabel('Backups')
    plt.ylabel('Steps to Goal')
    plt.ylim((0, 10000))
    plt.xscale('log')
    plt.legend()

    plt.savefig('figure10.png')
    plt.close()


def figure11():
    original_maze = Maze()
    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95
    # track the # of backups
    backups = defaultdict(list)

    runs = 10

    params = [params_dyna,params_dyna]

    # set up models for planning
    models = [HeuristicModel,TrivialModel]
    method_names = ['Focused Dyna','Random Dyna']

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # assuming the 1st maze has w * h states, then k-th maze has w * h * k * k states
    num_of_mazes = 1

    # build all the mazes
    mazes = [original_maze.extend_maze(i*2) for i in range(1, num_of_mazes + 1)]
    methods = [dyna_h,dyna_q]

    for run in tqdm(range(0, runs)):
        for i in range(0, len(method_names)):
            for mazeIndex, maze in zip(range(0, len(mazes)), mazes):
                #print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * maze.WORLD_WIDTH))

                # initialize the state action values
                q_value = np.zeros(maze.q_size)

                # track steps / backups for each episode
                stepss = []

                # generate the model
                model = models[i]()

                # play for an episode
                while True:
                    stepss.append(methods[i](q_value, model, maze, params[i]))

                    # print best actions w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)
                    # check whether the (relaxed) optimal path is found
                    if check_path(q_value, maze):
                        break

                # update the total steps / backups for this maze
                backups[method_names[i]].extend(stepss)

    # Dyna-Q performs several backups per step
    for i in method_names:
        plt.plot(sorted(backups[i], reverse=True), label=i)
    plt.xlabel('Backups')
    plt.ylabel('Steps to Goal')
    plt.ylim((0, 10000))
    plt.xscale('log')
    plt.legend()

    plt.savefig('figure11.png')
    plt.close()

#cette procédure teste les différents planings_steps
#param : name (nom du l'algo)
def parametrePlSt(name):
    # set up an instance for DynaMaze
    dyna_maze = Maze()
    dyna_params = DynaParams()

    runs = 10
    episodes = 50
    planning_steps = [0, 5, 10, 50]
    steps = np.zeros((len(planning_steps), episodes))

    # set up the parameters for each algorithm
    params_dyna = DynaParams()
    params_dyna.alpha = 0.1
    params_dyna.gamma = 0.95

    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, params_dyna, params_dyna]

    # set up models for planning
    methods = {'Largest 1st Dyna':prioritized_sweeping, 'Random Dyna':dyna_q, 'Focused Dyna':dyna_h}

    for run in tqdm(range(runs)):
        for index, planning_step in zip(range(len(planning_steps)), planning_steps):
            q_value = np.zeros(dyna_maze.q_size)

            # generate an instance of Dyna-Q model
            if (name == 'Largest 1st Dyna'):
                params_prioritized.planning_steps = planning_step
                model = PriorityModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_prioritized)
            elif(name == 'Random Dyna'):
                params_dyna.planning_steps = planning_step
                model = TrivialModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_dyna)
            else:
                params_dyna.planning_steps = planning_step
                model = HeuristicModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_dyna)
    # averaging over runs
    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label='%d planning steps' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

    plt.savefig(name+'-parametrePlSt.png')
    plt.close()

#cette procédure teste les différents alpha
#param : name (nom du l'algo)
def parametreAlpha(name):
    # set up an instance for DynaMaze
    dyna_maze = Maze()
    dyna_params = DynaParams()

    runs = 10
    episodes = 50
    alphas = [0, 0.1, 0.5, 1]
    steps = np.zeros((len(alphas), episodes))

    # set up the parameters for each algorithm
    params_dyna = DynaParams()
    params_dyna.alpha = 0.1
    params_dyna.gamma = 0.95

    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, params_dyna, params_dyna]

    # set up models for planning
    methods = {'Largest 1st Dyna':prioritized_sweeping, 'Random Dyna':dyna_q, 'Focused Dyna':dyna_h}

    for run in tqdm(range(runs)):
        for index, alpha in zip(range(len(alphas)), alphas):
            q_value = np.zeros(dyna_maze.q_size)

            # generate an instance of Dyna-Q model
            if (name == 'Largest 1st Dyna'):
                params_prioritized.alpha = alpha
                model = PriorityModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_prioritized)
            elif(name == 'Random Dyna'):
                params_dyna.alpha = alpha
                model = TrivialModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_dyna)
            else:
                params_dyna.alpha = alpha
                model = HeuristicModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_dyna)
    # averaging over runs
    steps /= runs
    for i in range(len(alphas)):
        plt.plot(steps[i, :], label='%f alpha' % (alphas[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

    plt.savefig(name+'-parametreAlpha.png')
    plt.close()

#cette procédure teste les différents gamma
#param : name (nom du l'algo)
def parametreGamma(name):
    # set up an instance for DynaMaze
    dyna_maze = Maze()
    runs = 10
    episodes = 10
    gammas = [0.05, 0.15, 0.5,0.95, 1]
    steps = np.zeros((len(gammas), episodes))

    # set up the parameters for each algorithm
    params_dyna = DynaParams()
    params_dyna.alpha = 0.1
    params_dyna.gamma = 0.95

    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, params_dyna, params_dyna]

    # set up models for planning
    methods = {'Largest 1st Dyna':prioritized_sweeping, 'Random Dyna':dyna_q, 'Focused Dyna':dyna_h}

    for run in tqdm(range(runs)):
        for index, gamma in zip(range(len(gammas)), gammas):
            q_value = np.zeros(dyna_maze.q_size)

            # generate an instance of Dyna-Q model
            if (name == 'Largest 1st Dyna'):
                params_prioritized.gamma = gamma
                model = PriorityModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_prioritized)
            elif(name == 'Random Dyna'):
                params_dyna.gamma = gamma
                model = TrivialModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_dyna)
            else:
                params_dyna.gamma = gamma
                model = HeuristicModel()
                for ep in range(episodes):
                    # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                    steps[index, ep] += methods[name](q_value, model, dyna_maze, params_dyna)
    # averaging over runs
    steps /= runs

    for i in range(len(gammas)):
        plt.plot(steps[i, :], label='%f gamma' % (gammas[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

    plt.savefig(name+'-parametreGamma.png')
    plt.close()

if __name__ == '__main__':
    figure7()
    figure8()
    figure10()
    #figure11()  #pour la figure 11, la simulation doit avoir une grande mémoire pour l'éxécution qui peut durer plusieurs heures
    parametrePlSt('Largest 1st Dyna')       #pour chacun des agorithmes il faut suivre cette notation : "Random Dyna" , "Focused Dyna"
    parametreAlpha("Largest 1st Dyna")      #pour chacun des agorithmes il faut suivre cette notation : "Random Dyna" , "Focused Dyna"
    parametreGamma("Largest 1st Dyna")      #pour chacun des agorithmes il faut suivre cette notation : "Random Dyna" , "Focused Dyna"
