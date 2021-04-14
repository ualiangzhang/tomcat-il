import os
import gym
from gym_minigrid.wrappers import HumanFOVWrapper
import pickle
import argparse
import numpy as np
from collections import deque
import random
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import gym_minigrid
import sys
import warnings
from utils.utils import gen_training_and_test_sets as gen_training_and_test_sets

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Imitation Learning for Falcon')
parser.add_argument('--env_name', type=str, default='MiniGrid-MinimapForFalcon-v0',
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None,
                    help='path to load the saved model')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='learning rate of training (default: 1e-4)')
parser.add_argument('--total_epochs', type=int, default=100,
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size to update the model (default: 32)')
parser.add_argument('--save_frequency', type=int, default=10,
                    help='save model freqeuncy (epochs) (default: 10)')
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


def main():
    env = gym.make(args.env_name, difficulty=args.level)
    env = HumanFOVWrapper(env)
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num_inputs = env.observation_space.shape
    num_actions = env.action_space.n

    print('state size:', num_inputs)
    print('action size:', num_actions)

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

    demos_name += 'demos.p'

    expert_demo = pickle.load(open('./expert_demos/' + demos_name, "rb"))
    training_set, test_set, demo_idx_list, num_demos = gen_training_and_test_sets(expert_demo, args.test_set_ratio)

    print('training set number: ' + str(len(training_set['states'])))
    print('test set number: ' + str(len(test_set['states'])))
    saved_file_name = './expert_demo/' + demos_name + '_training.p'
    pickle.dump(training_set, open(saved_file_name, "wb"))
    saved_file_name = './expert_demo/' + demos_name + '_test.p'
    pickle.dump(test_set, open(saved_file_name, "wb"))

    demonstrations = []
    for demo in training_set:
        for d in demo:
            demonstrations.append(d)
    demonstrations = np.array(demonstrations)
    demonstrations_list = demonstrations.tolist()  # to check if state in demonstrations
    print("demonstrations shape: ", demonstrations.shape)

    # replay_buffer = deque(maxlen=demonstrations.shape[0])  # share the same size of demonstration
    writer = SummaryWriter(args.logdir)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        discrim.load_state_dict(ckpt['discrim'])

    episodes = 0
    total_steps = 0
    saved_times = 0
    scores = deque(maxlen=100)
    iteration_times = 0
    state_counts = {}

    while total_steps < args.max_total_steps + args.total_sample_size:
        actor.eval(), critic.eval(), discrim.eval()

        memory = deque()
        steps = 0

        while steps < args.total_sample_size:
            state = env.reset()
            score = 0

            if args.zfilter:
                state = running_state(state)

            for _ in range(env.max_steps):
                if args.render:
                    env.render()

                policy = actor(torch.Tensor(state / 255).unsqueeze(0).to(args.device))
                action = get_action(policy)
                next_state, reward, done, _ = env.step(action)
                irl_reward = get_reward(discrim, state, action, args)

                if done:
                    mask = 0
                else:
                    mask = 1

                if args.add_exploration_bonus:
                    explore_bonus = get_exploration_bonus(state_counts, state, action)
                    irl_reward += explore_bonus

                memory.append([state, action, irl_reward, mask])
                # check if state in demonstrations
                # state_list = np.expand_dims(np.append(state, action).astype('uint8'), axis=0).tolist()
                state_list = np.append(state, action).astype('uint8').tolist()
                if state_list not in demonstrations_list:
                    replay_buffer.append(np.append(state, action).astype('uint8'))
                # else:
                #     print('duplicates!')


                if args.zfilter:
                    next_state = running_state(next_state)

                state = next_state
                score += reward
                steps += 1

                if done:
                    break

            episodes += 1
            scores.append(score)

        score_avg = np.mean(scores)
        total_steps += steps
        iteration_times += 1

        print(
            '{} steps :: {} episodes :: average score for 100 episodes is {:.2f}'.format(total_steps, episodes,
                                                                                         score_avg))
        writer.add_scalar('log/score', float(score_avg), total_steps)

        actor.train(), critic.train(), discrim.train()

        # L = {array.tostring(): array for array in replay_buffer}
        # L = L.values()
        # L = list(L)

        #  train discriminator
        if iteration_times > args.discrim_leanring_starts and iteration_times % args.discrim_training_frequency == 0:
            expert_acc, learner_acc, well_trained = train_discrim(discrim, replay_buffer, discrim_optim, demonstrations, args)
            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))

        # if discriminator is well-trained, we decrease the training times for it to make the policy training stable
        # if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen and well_trained:
        #     discrim_update_num = copy.deepcopy(args.discrim_update_num)
        #     args.discrim_update_num = int(0.5 * discrim_update_num)
        #     print('well trained discriminator')

        #  train actor and critic
        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        if total_steps >= saved_times * args.save_frequency:
            saved_times += 1
            saved_steps = int(total_steps // args.save_frequency * args.save_frequency)

            model_path = os.path.join(os.getcwd(), 'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, args.level + '_' + args.strategy + '_ckpt_' + str(saved_times) + 'M.pth.tar')

            if args.zfilter:
                save_checkpoint({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'discrim': discrim.state_dict(),
                    'z_filter_n': running_state.rs.n,
                    'z_filter_m': running_state.rs.mean,
                    'z_filter_s': running_state.rs.sum_square,
                    'args': args,
                    'score': score_avg
                }, filename=ckpt_path)
            else:
                save_checkpoint({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'discrim': discrim.state_dict(),
                    'args': args,
                    'score': score_avg
                }, filename=ckpt_path)


if __name__ == "__main__":
    main()