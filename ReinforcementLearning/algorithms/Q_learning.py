import numpy
import pandas

# global variiables
N_STATES = 6
ACTIONS = ['left', 'right']
EPSLON = 0.9        # greedy police
ALPHA = 0.1         # learning rate
LAMBDA = 0.9        # discount factor
MAX_EPISODES = 13
FRESH_TIME = 0.3

q_table = pandas.DataFrame(numpy.zeros([N_STATES, len(ACTIONS)]),columns=ACTIONS)
for episode in range(MAX_EPISODES):
    step_counter = 0
    state = 0
    is_terminated = False

    while not is_terminated:

        state_actions = q_table.iloc[state, :]
        if numpy.random.uniform() > EPSLON or state_actions.all() == 0:
            action = numpy.random.choice(ACTIONS)
        else:
            action = state_actions.argmax()

        if action == 'right':
            if state == N_STATES - 2:
                state_ = 'terminal'
                reward = 1
            else:
                state_ = state + 1
                reward = 0
        else:
            reward = 0
            if state == 0:
                state_ = state
            else:
                state_ = state - 1

        q_predict = q_table.ix[state, action]
        if state_ != 'terminal':
            q_target = reward + LAMBDA * q_table.iloc[state_,:].max()
        else:
            q_target = reward
            is_terminated = True

        q_table.ix[state, action] += ALPHA*(q_target - q_predict)
        state = state_
        step_counter += 1

print(q_table)
