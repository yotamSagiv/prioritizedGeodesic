import random
import numpy as np
np.random.seed(865612)

from gridworld import GridWorld
from RL_utils import twod_oned, softmax
from geodesic_agent import SftmxGeodesicAgent

def action_seq(start, goal, true_GR, mdp):
    """
    Compute the optimal action sequence for reaching 'goal' from 'start'.

    Args:
        start (int): Start state
        goal (int): Goal state
        true_GR (np.ndarray): True GR for the environment
        mdp (MarkovDecisionProcess): The environment MDP object

    Returns:
        seq (list): list of steps to optimally navigate from start to goal
    """
    seq = []
    s = start
    while s != goal:
        best_action = np.argmax(true_GR[s, :, goal])
        successor, reward = mdp.perform_action(s, best_action)

        seq.append(best_action)
        s = successor

    return seq

def likelihood(well_seq, t0_type, home, num_wells):
    """
    Computes the likelihood over a given well sequence, assuming a known first trial type and Home location

    Args:
        well_seq (list): Sequence of observed well IDs
        t0_type (int): Indicates whether first trial was a Home (0) or a Random (1) trial
        home (int): Location of the Home well
        num_wells (int): Total number of wells

    Returns:
        p (float): Probability of this well sequence.
    """
    p = 1
    trial_type = t0_type
    for wdx, well in enumerate(well_seq):
        if well == home and trial_type == 0:
            p *= 1
        elif well != home and trial_type == 0:
            p *= 0
        elif well == home and trial_type == 1:
            p *= 0
        else:
            p *= 1 / (num_wells - 1)

        trial_type = 1 - trial_type

    return p

def unnorm_posterior(t0_type, home, well_seq, num_wells):
    """
    Computes the unnormalized posterior probability that the first trial type is t0_type, and the
    Home well is home.

    Args:
        t0_type (int): Indicates whether first trial was a Home (0) or a Random (1) trial
        home (int): Location of the Home well
        well_seq (list): Sequence of observed well IDs
        num_wells (int): Total number of wells

    Returns:
        p (float): the unnormalized posterior probability.
    """
    prior_phase = 1/2
    prior_home = 1 / num_wells

    return likelihood(well_seq, t0_type, home, num_wells) * prior_phase * prior_home

def joint_posterior(well_seq, num_wells):
    """
    Computes the joint posterior distribution over the first trial type and the location of Home
    given a sequence of observed wells.

    Args:
        well_seq (list): Sequence of observed well IDs
        num_wells (int): Total number of wells

    Returns:
        outcomes (np.ndarray): The joint posterior.
    """
    outcomes = np.zeros((2, num_wells))
    for t0_phase in range(2):
        for home in range(num_wells):
            outcomes[t0_phase, home] = unnorm_posterior(t0_phase, home, well_seq, num_wells)

    return outcomes / np.sum(outcomes)


height = 9
width = 9
num_states = height * width
num_actions = 4

num_mice = 25
num_sessions = 10
num_trials = 15

HOME_TRIAL_CODE = 0
AWAY_TRIAL_CODE = 1

# Define the possible goals
goal_states = []
for y in range(height):
    for x in range(width):
        if x % 2 == 0 or y % 2 == 0:
            continue

        state_id = twod_oned(x, y, width)
        goal_states.append(state_id)

num_goals = len(goal_states)

# Initialize agent
s0_dist = np.zeros(num_states)
s0_dist[0] = 1

policy_temp = 0.4
episode_disc_rate = 0.95
num_replay_steps = 10

decay_rate = 0.95  # GR decay rate on every step
noise = 0.00  # Update noise

behav_alpha = 1.0  # Behaviour (online) GR learning rate
replay_alpha = 0.7  # Replay (offline) GR learning rate

only_learn_home_trials = False
home_always_first = False

# Storage
postoutcome_replays = np.zeros((num_mice, num_sessions, num_trials, num_replay_steps, 3)) - 1  # so obvious if some row is unfilled
pretrial_replays = np.zeros((num_mice, num_sessions, num_replay_steps, 3)) - 1
predists = np.zeros((num_mice, num_sessions, int(num_trials / 2))) - 1
postdists = np.zeros((num_mice, num_sessions, int(num_trials / 2))) - 1
trial_types = np.zeros((num_mice, num_sessions, num_trials)) - 1
home_wells = np.zeros((num_mice, num_sessions)) - 1
sampled_goals = np.zeros((num_mice, num_sessions, num_trials)) - 1

pre_replay_forget = True
uniform_forget = True

# Simulate
for mouse in range(num_mice):
    # Build the corresponding grid world
    gw = GridWorld(width, height, semipermeable=False, init_state_distribution=s0_dist)
    true_GR = gw.solve_GR(num_iters=500, gamma=0.95)
    all_experiences = gw.get_all_transitions()
    T = gw.transitions

    # Initialize agent
    ga = SftmxGeodesicAgent(num_states, num_actions, goal_states, T, goal_dist=None, s0_dist=s0_dist,
                            policy_temperature=policy_temp, alpha=behav_alpha)

    ga.remember(all_experiences)

    # Choose random sequence of home wells without replacement
    home_well_seq = np.random.choice(goal_states, size=num_sessions, replace=False)

    for session in range(num_sessions):
        ga.curr_state = 0
        well_seq = []

        # Select home goal
        home_well = home_well_seq[session]
        random_wells = [well for well in range(num_states) if (well != home_well and well in goal_states)]

        home_wells[mouse, session] = home_well
        ttype = AWAY_TRIAL_CODE  # Makes first trial a home trial

        # Behave
        for trial in range(num_trials):
            print('mouse: %d, session: %d, trial: %d' % (mouse, session, trial))

            # Only on first trial: pre-trial replay (approximates replay as the rodent
            # is searching around for the Home well)
            if trial == 0:
                joint_post = joint_posterior([], num_goals)
                pretrial_replays[mouse, session, :, :], _, _ = ga.foster_bayes_replay(num_replay_steps, goal_states,
                                                       joint_post,
                                                       HOME_TRIAL_CODE,
                                                       verbose=True,
                                                       check_convergence=False,
                                                       prospective=False,
                                                       disc_rate=episode_disc_rate,
                                                       alpha=replay_alpha)

            # Pick trial goal and set corresponding reward vector
            ttype = 1 - ttype
            trial_types[mouse, session, trial] = ttype
            if ttype == 0:  # Home trials
                curr_goal = home_well
            else:  # Random trials
                curr_goal = np.random.choice(random_wells)

            sampled_goals[mouse, session, trial] = curr_goal
            rvec = np.zeros(num_states)
            rvec[curr_goal] = 1

            # Simulate agent
            act_seq = action_seq(ga.curr_state, curr_goal, true_GR, gw)
            for adx, action in enumerate(act_seq):
                successor, reward = gw.step(ga.curr_state, action=action, reward_vector=rvec)

                if uniform_forget:
                    ga.basic_learn((ga.curr_state, action, successor), decay_rate=1, noise=noise)  # Update GR
                else:
                    ga.basic_learn((ga.curr_state, action, successor), decay_rate=decay_rate, noise=noise)  # Update GR

                ga.curr_state = successor

            # Update observed goal well sequence
            goal_idx = goal_states.index(curr_goal)
            well_seq.append(goal_idx)
            joint_post = joint_posterior(well_seq, num_goals)

            if home_always_first:
                joint_post[1, :] = 0
                joint_post /= np.sum(joint_post)

            if pre_replay_forget:
                ga.decay(decay_rate)

            if trial % 2 == AWAY_TRIAL_CODE:  # Current trial is an away trial, next trial is a home trial
                # Compute expected pre-replay distance to Home well
                predists[mouse, session, int((trial - 1) / 2)] = ga.compute_expected_distance(home_well)

            # Post-outcome replay
            replays, _, _ = ga.foster_bayes_replay(num_replay_steps, goal_states,
                                                   joint_post,
                                                   1 - (trial % 2),
                                                   verbose=True,
                                                   check_convergence=False,
                                                   prospective=False,
                                                   disc_rate=episode_disc_rate,
                                                   alpha=replay_alpha)

            if trial % 2 == AWAY_TRIAL_CODE:
                # Compute expected post-replay distance to Home well
                postdists[mouse, session, int((trial - 1) / 2)] = ga.compute_expected_distance(home_well)

            postoutcome_replays[mouse, session, trial, :, :] = replays

    np.savez('./Data/widloski/pfeiffer/GR.npz', posto=postoutcome_replays, trial_types=trial_types,
                                                home_wells=home_wells, predists=predists, postdists=postdists,
                                                pretrial_replays=pretrial_replays, goal_states=goal_states, mdx=mouse,
                                                sdx=session, sampled_goals=sampled_goals)










