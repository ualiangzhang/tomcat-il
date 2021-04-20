import gym
from gym_minigrid.wrappers import HumanFOVWrapper
import pickle
import argparse
import numpy as np
import random
import torch

import gym_minigrid
import sys
import warnings
from utils.utils import gen_training_and_test_sets, gen_training_batch, gen_eval_batch, gen_history_sequence, epoch_time
import torch.optim as optim
import torch.nn as nn
from model import ToMnet
import optuna
from optuna.trial import TrialState
import optuna.visualization as vis

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Imitation Learning for Falcon')
parser.add_argument('--env_name', type=str, default='MiniGrid-MinimapForFalcon-v0',
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None,
                    help='path to load the saved model')
parser.add_argument('--clipped_gradient', type=int, default=1,
                    help='clipped gradient of the training (default: 1)')
parser.add_argument('--total_epochs', type=int, default=100,
                    help='total epochs to train the model (default: 100)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
# game settings
parser.add_argument('--level', type=str, default='easy',
                    help='game level')
parser.add_argument('--strategy', type=str, default='yellow',
                    help='game strategy')
parser.add_argument('--test_set_ratio', type=int, default=0.1,
                    help='test set ratio in the all demonstrations')

args = parser.parse_args()
# if gpu is to be used
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):
    env = gym.make(args.env_name, difficulty=args.level)
    env = HumanFOVWrapper(env)
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load demonstrations
    if args.level == 'easy':
        demos_name = 'e_'
    elif args.level == 'medium':
        demos_name = 'm_'
    else:
        demos_name = 'd_'

    if args.strategy == 'yellow':
        demos_name += 'y_'
    else:
        demos_name += 'o_'

    demos_name += 'demos'
    demos_file = demos_name + '.p'

    expert_demo = pickle.load(open('./expert_demos/' + demos_file, "rb"))
    training_set, test_set, demo_idx_list, num_demos = gen_training_and_test_sets(expert_demo, args.test_set_ratio)

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

    eval_pre_traj_batches, eval_state_batches, eval_action_labels, eval_reward_labels, eval_len_pre_traj = gen_eval_batch(
        test_set, batch_size)
    past_traj, len_past_traj = gen_history_sequence(training_set)

    hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 32, 64, 128, 256])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, 0.2, 0.5, 0.8])

    model = ToMnet(hidden_size=hidden_size, dropout=dropout_rate).to(args.device)

    if args.load_model:
        model.load_state_dict(torch.load('saved_model/' + args.load_model))

    learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.0005, 0.001, 0.005, 0.01])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    action_criterion = nn.CrossEntropyLoss()
    reward_criterion = nn.MSELoss()

    for epoch in range(args.total_epochs):
        pre_traj_batches, state_batches, action_labels, reward_labels, len_pre_traj = gen_training_batch(training_set,
                                                                                                         batch_size,
                                                                                                         demo_idx_list)
        reward_loss_weight = trial.suggest_categorical("reward_loss_weight", [1, 10, 50, 100, 500, 1000, 5000])

        training_loss, training_acc1, training_acc2, training_acc3, training_reward_loss = train(model, state_batches,
                                                                                                 pre_traj_batches,
                                                                                                 past_traj,
                                                                                                 action_labels,
                                                                                                 reward_labels,
                                                                                                 len_past_traj,
                                                                                                 len_pre_traj,
                                                                                                 optimizer,
                                                                                                 action_criterion,
                                                                                                 reward_criterion,
                                                                                                 reward_loss_weight)

        eval_loss, eval_acc1, eval_acc2, eval_acc3, eval_reward_loss = evaluate(model, eval_state_batches,
                                                                                eval_pre_traj_batches,
                                                                                past_traj,
                                                                                eval_action_labels,
                                                                                eval_reward_labels,
                                                                                len_past_traj,
                                                                                eval_len_pre_traj,
                                                                                action_criterion,
                                                                                reward_criterion)

        trial.report(eval_acc1, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return eval_acc1


def train(model, state_batches, pre_traj, past_traj, action_labels, reward_labels, len_past_traj, len_pre_traj,
          optimizer,
          action_criterion,
          reward_criterion, reward_loss_weight):
    epoch_loss = 0
    epoch_reward_loss = 0
    epoch_action_acc1 = 0
    epoch_action_acc2 = 0
    epoch_action_acc3 = 0

    len_batches = len(state_batches)

    model.train()

    for idx in range(len_batches):
        t_past_traj_batches = torch.tensor(past_traj / 255).to(args.device)
        t_pre_traj_batches = torch.tensor(pre_traj[idx] / 255).to(args.device)
        t_state_batches = torch.tensor(state_batches[idx] / 255).to(args.device)
        t_len_past_traj = torch.tensor(len_past_traj).to(args.device)
        t_len_pre_traj = torch.tensor(len_pre_traj[idx]).to(args.device)
        t_action_labels = torch.LongTensor(action_labels[idx]).to(args.device)
        t_reward_labels = torch.Tensor(reward_labels[idx] / 1000).to(args.device)

        optimizer.zero_grad()

        action_predictions, reward_predictions = model(t_past_traj_batches, t_pre_traj_batches, t_state_batches,
                                                       t_len_past_traj, t_len_pre_traj)

        action_loss = action_criterion(action_predictions, t_action_labels)
        reward_loss = reward_criterion(reward_predictions, t_reward_labels)

        loss = action_loss + reward_loss_weight * reward_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipped_gradient)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_reward_loss += reward_loss
        acc1, acc2, acc3 = action_accuracy(action_predictions, t_action_labels)
        epoch_action_acc1 += acc1
        epoch_action_acc2 += acc2
        epoch_action_acc3 += acc3

    return epoch_loss / len_batches, epoch_action_acc1 / len_batches, epoch_action_acc2 / len_batches, epoch_action_acc3 / len_batches, epoch_reward_loss / len_batches


def evaluate(model, state_batches, pre_traj, past_traj, action_labels, reward_labels, len_past_traj, len_pre_traj,
             action_criterion,
             reward_criterion,
             reward_loss_weight):
    epoch_loss = 0
    epoch_reward_loss = 0
    epoch_action_acc1 = 0
    epoch_action_acc2 = 0
    epoch_action_acc3 = 0

    len_batches = len(state_batches)

    model.eval()

    with torch.no_grad():
        for idx in range(len_batches):
            t_past_traj_batches = torch.tensor(past_traj / 255).to(args.device)
            t_pre_traj_batches = torch.tensor(pre_traj[idx] / 255).to(args.device)
            t_state_batches = torch.tensor(state_batches[idx] / 255).to(args.device)
            t_len_past_traj = torch.tensor(len_past_traj).to(args.device)
            t_len_pre_traj = torch.tensor(len_pre_traj[idx]).to(args.device)
            t_action_labels = torch.LongTensor(action_labels[idx]).to(args.device)
            t_reward_labels = torch.Tensor(reward_labels[idx] / 1000).to(args.device)

            action_predictions, reward_predictions = model(t_past_traj_batches, t_pre_traj_batches, t_state_batches,
                                                           t_len_past_traj, t_len_pre_traj)

            action_loss = action_criterion(action_predictions, t_action_labels)
            reward_loss = reward_criterion(reward_predictions, t_reward_labels)

            real_reward_predictions = reward_predictions.to('cpu').numpy() * 1000
            mse_reward = ((real_reward_predictions - reward_labels[idx]) ** 2).mean()

            loss = action_loss + reward_loss_weight * reward_loss

            epoch_loss += loss.item()
            epoch_reward_loss += mse_reward
            acc1, acc2, acc3 = action_accuracy(action_predictions, t_action_labels)
            epoch_action_acc1 += acc1
            epoch_action_acc2 += acc2
            epoch_action_acc3 += acc3

    return epoch_loss / len_batches, epoch_action_acc1 / len_batches, epoch_action_acc2 / len_batches, epoch_action_acc3 / len_batches, epoch_reward_loss / len_batches


def action_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top1 = torch.topk(preds, 1)[1]
    top2 = torch.topk(preds, 2)[1]
    top3 = torch.topk(preds, 3)[1]

    correct1 = torch.eq(torch.unsqueeze(y, 1), top1).any(dim=1)
    correct2 = torch.eq(torch.unsqueeze(y, 1), top2).any(dim=1)
    correct3 = torch.eq(torch.unsqueeze(y, 1), top3).any(dim=1)

    acc1 = correct1.sum() / len(correct1)
    acc2 = correct2.sum() / len(correct2)
    acc3 = correct3.sum() / len(correct3)
    return acc1.item(), acc2.item(), acc3.item()


if __name__ == "__main__":
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=0, seed=args.seed),
                                direction="maximize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=1000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    from sklearn.externals import joblib

    tuning_name = 'tuning-' + args.level + '-' + args.strategy

    joblib.dump(study, 'saved_model/' + tuning_name + '.pkl')

    df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'system_attrs'], axis=1)
    df.head(10)

    vis.plot_contour(study)
    vis.plot_parallel_coordinates(study)
    vis.plot_optimization_history(study)
