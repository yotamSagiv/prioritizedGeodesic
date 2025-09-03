import numpy as np
import sys

from graph import DiGraph
from geodesic_agent import GeodesicAgent, SftmxGeodesicAgent
from RL_utils import softmax

np.random.seed(865612)

### Convenience functions
def action_seq(goal, arm_length, start_to_choice):
    """
    Given maze parameters, given the full sequence of actions that leads from the start state to the goal state.

    Args:
        goal (int): Goal state
        arm_length (int): Length of the T-maze arms
        start_to_choice (int): Length of the T-maze stem

    Returns:

    """
    actions = []

    # Get to choice point
    for _ in range(start_to_choice - 1):
        actions.append(1)

    # Get to goal arm
    actions.append(goal + 2)

    # Continue to the end
    for _ in range(arm_length - 1):
        actions.append(1)

    return actions

### Main script
## Set parameters
# Geometry
num_arms = 2
arm_length = 2
start_to_choice = 2

## Build underlying MDP
# Basics
num_states = num_arms * arm_length + start_to_choice
num_actions = 2 + num_arms
permit_backtrack = True  # If the agent makes a choice, can it go backwards to the choice point?

# Fill out edges
edges = np.zeros((num_states, num_actions))
for state in range(num_states):  # Convenient way of filling in null actions
    edges[state, :] = state

# Overwrite null actions with actual transitions
for state in range(start_to_choice):
    if not state == 0:
        edges[state, 0] = state - 1  # Go left

    if not state == start_to_choice - 1:
        edges[state, 1] = state + 1  # Go right

# Add choice transitions
for action in range(2, num_actions):
    edges[start_to_choice - 1, action] = start_to_choice + (action - 2) * arm_length

# Add within-arm transitions
for arm in range(num_arms):
    for rel_state in range(arm_length):
        state = start_to_choice + arm * arm_length + rel_state
        if rel_state == arm_length - 1:  # Goal states are terminal
            continue

        if not rel_state == 0:
            edges[state, 0] = state - 1  # Backwards through arm
        elif permit_backtrack:
            edges[state, 0] = start_to_choice - 1

        if not rel_state == arm_length - 1:
            edges[state, 1] = state + 1  # Forwards through arm


# Task
# num_mice = 30
num_mice = 200
num_sessions = 6
num_trials = 20

# Build graph object
s0_dist = np.zeros(num_states)
s0_dist[0] = 1
carey_maze = DiGraph(num_states, edges, init_state_dist=s0_dist)
T = carey_maze.transitions
all_experiences = carey_maze.get_all_transitions()

# Define MDP-related parameters
goal_states = start_to_choice + np.arange(num_arms) * arm_length + (arm_length - 1)

## Build agent
behav_lr = 1.0
behav_temp = 0.50  # Behaviour softmax temperature

num_replay_steps = 3
replay_lr = 0.04  # Per-arm Rescorla-Wagner learning rate for replay prioritization

decay_rate = 0.94  # GR forgetting decay
noise = 0.00  # Update noise

online_alpha = 1.0  # Online GR learning rate
replay_alpha = 0.7  # Replay GR learning rate
policy_temperature = 0.40  # Temperature of assumed behavioural policy (for gain computation)

pre_replay_forgetting = True
uniform_forgetting = True
use_softmax = True

## Run task
# Build storage variables
choices = np.zeros((num_mice, num_sessions, num_trials))
rewards = np.zeros((num_mice, num_sessions, num_trials))
sess_seq = np.zeros((num_mice, num_sessions, num_trials))

postoutcome_replays = np.zeros((num_mice, num_sessions, num_trials, num_replay_steps, 3)) - 1   # so obvious if some row is unfilled
states_visited = np.zeros((num_mice, num_sessions, num_trials, start_to_choice + arm_length))
posto_Gs = np.zeros((num_mice, num_sessions, num_trials, num_states, num_actions, num_states))

for mouse in range(num_mice):
    # Agent parameters
    behav_goal_vals = np.zeros(num_arms)
    replay_goal_vals = np.ones(num_arms) * (1 / num_arms)

    # Simulate
    sess_type = 0
    for session in range(num_sessions):
        print('mouse %d, session %d' % (mouse, session))

        # Instantiate agent
        if not use_softmax:
            ga = GeodesicAgent(num_states, num_actions, goal_states, alpha=online_alpha, goal_dist=None,
                               s0_dist=s0_dist, T=T)
        else:
            ga = SftmxGeodesicAgent(num_states, num_actions, goal_states, alpha=online_alpha, goal_dist=None,
                                    s0_dist=s0_dist, T=T, policy_temperature=policy_temperature)

        ga.remember(all_experiences)

        # Set basic task variables
        rvec = np.zeros(num_states)
        if sess_type == 0:
            best_arm_idx = 0
            rvec[goal_states[0]] = 1.5
            rvec[goal_states[1]] = 1

            behav_goal_vals[0] = 1.5
            behav_goal_vals[1] = 1
        else:
            best_arm_idx = 1
            rvec[goal_states[1]] = 1.5
            rvec[goal_states[0]] = 1

            behav_goal_vals[1] = 1.5
            behav_goal_vals[0] = 1

        for trial in range(num_trials):
            ga.curr_state = 0
            sess_seq[mouse, session, trial] = sess_type

            # Choose an arm
            chosen_arm = np.random.choice(num_arms, p=softmax(behav_goal_vals, behav_temp))
            choices[mouse, session, trial] = chosen_arm

            # Go to it
            act_seq = action_seq(chosen_arm, arm_length, start_to_choice)
            for adx, action in enumerate(act_seq):
                next_state, reward = carey_maze.step(ga.curr_state, action=action, reward_vector=rvec)

                # Update GR
                if uniform_forgetting:
                    ga.basic_learn((ga.curr_state, action, next_state), decay_rate=1, noise=noise)
                else:
                    ga.basic_learn((ga.curr_state, action, next_state), decay_rate=decay_rate, noise=noise)

                ga.curr_state = next_state
                states_visited[mouse, session, trial, adx + 1] = ga.curr_state

                # Check for rewards
                if adx == len(act_seq) - 1:
                    behav_goal_vals[chosen_arm] += behav_lr * (reward - behav_goal_vals[chosen_arm])

                    if chosen_arm == best_arm_idx:
                        target = np.zeros(num_arms)
                        target[chosen_arm] = 1
                    else:
                        target = np.ones(num_arms) * (1 / (num_arms - 1))
                        target[chosen_arm] = 0

                    replay_goal_vals = replay_lr * target + (1 - replay_lr) * replay_goal_vals
                    rewards[mouse, session, trial] = reward

            if uniform_forgetting or pre_replay_forgetting:
                ga.decay(decay_rate)

            # Post-outcome replay
            posto_Gs[mouse, session, trial, :] = ga.G
            replays, _, _ = ga.replay(num_replay_steps, goal_dist=replay_goal_vals, verbose=True,
                                      check_convergence=False, prospective=True, alpha=replay_alpha)
            postoutcome_replays[mouse, session, trial, :, :] = replays

        sess_type = int(1 - sess_type)

# Save everything
np.savez('./Data/carey/GR.npz', posto=postoutcome_replays, state_trajs=states_visited,
         choices=choices, rewards=rewards, sess_seq=sess_seq, allow_pickle=True)
