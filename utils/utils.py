'''
According to the number of demonstrations and the test set ratio,
we separate the demonstrations into the training and test sets
return training set, test set, observation index list for training set, the number of observations in training set
'''

import numpy as np
import random


def gen_training_and_test_sets(demos, test_set_ratio=0.1):
    training_set = {'states': [], 'actions': [], 'rewards': []}
    test_set = {'states': [], 'actions': [], 'rewards': []}
    training_set_idx_list = []

    demos_len = len(demos['states'])
    if demos_len < 2:
        raise NameError('Should have more than 2 trajectories to train your model!')
        return training_set, test_set
    test_set_num = int(max(np.ceil(demos_len * test_set_ratio), 1))
    test_idx = np.sort(np.random.choice(range(demos_len), test_set_num, replace=False)).tolist()
    starting_idx = 0
    for idx in range(demos_len):
        if idx in test_idx:
            test_set['states'].append(demos['states'][idx])
            test_set['actions'].append(demos['actions'][idx])
            test_set['rewards'].append(demos['rewards'][idx])
        else:
            training_set['states'].append(demos['states'][idx])
            training_set['actions'].append(demos['actions'][idx])
            training_set['rewards'].append(demos['rewards'][idx])
            training_set_idx_list.append([idx, starting_idx, starting_idx + len(demos['states'][idx]) - 1])
            starting_idx = starting_idx + len(demos['states'][idx])

    return training_set, test_set, training_set_idx_list, starting_idx


def gen_batch(training_set, batch_size, index_list):
    len_obs = index_list[-1][-1] + 1
    obs_index = [i for i in range(len_obs)]
    random.shuffle(obs_index)
    obs_len = len(training_set['states'][0][0])

    history_batches = []
    sub_history_batch = []
    state_batch = []
    sub_state_batch = []
    action_labels = []
    sub_action_labels = []
    reward_labels = []
    sub_reward_labels = []

    for oi in obs_index:
        for il in index_list:
            if il[1] <= oi <= il[2]:
                if oi == 0:
                    sa_pairs = np.zeros((1, obs_len + 1))
                else:
                    start_idx = 0
                    end_idx = oi - il[1]
                    states = np.array(training_set['states'][il[0]][start_idx:end_idx])
                    actions = np.reshape(np.array(training_set['actions'][il[0]][start_idx:end_idx]), (-1, 1))
                    sa_pairs = np.concatenate([states, actions], axis=1).astype(np.uint8)

                sub_history_batch.append(sa_pairs)
                sub_state_batch.append(training_set['states'][il[0]][end_idx])
                sub_action_labels.append(training_set['actions'][il[0]][end_idx])
                sub_reward_labels.append(training_set['rewards'][il[0]])

                if len(sub_history_batch) > batch_size - 1:
                    history_batches.append(sub_history_batch)
                    state_batch.append(sub_state_batch)
                    action_labels.append(sub_action_labels)
                    reward_labels.append(sub_reward_labels)
                    sub_history_batch = []
                    sub_state_batch = []
                    sub_action_labels = []
                    sub_reward_labels = []

                break

    if sub_history_batch != []:
        history_batches.append(sub_history_batch)

    return history_batches, np.array(state_batch, dtype=np.uint8), np.array(action_labels, dtype=np.uint8), np.array(
        reward_labels)


def gen_history_sequence(training_set):
    for i in range(len(training_set['states'])):
        states = np.array(training_set['states'][i])
        actions = np.reshape(np.array(training_set['actions'][i]), (-1, 1))
        sa_pairs = np.concatenate([states, actions], axis=1).astype(np.uint8)
        if i == 0:
            history = sa_pairs
        else:
            history = np.concatenate([history, sa_pairs], axis=0).astype(np.uint8)

    return history
