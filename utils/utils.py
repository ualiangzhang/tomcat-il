import numpy as np


# According to the number of demonstrations and the test set ratio,
# we seperate the demonstrations into the training and test sets
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
            training_set_idx_list.append([starting_idx, starting_idx + len(demos['states'][idx]) - 1])
            starting_idx = starting_idx + len(demos['states'][idx])

    return training_set, test_set, training_set_idx_list, starting_idx
