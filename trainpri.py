import argparse
import os
import shutil
from random import random, randint
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from src.deep_q_network import DeepQNetwork  # assuming this is your DQN implementation
from src.flappy_bird import FlappyBird  # assuming this is your Flappy Bird environment implementation
from src.utils import pre_processing  # assuming this is your preprocessing function
from prioritized_replay_buffer import PrioritizedReplayBuffer  # import your replay buffer

def get_args():
    parser = argparse.ArgumentParser("Implementation of Double Deep Q Network with Prioritized Experience Replay to play Flappy Bird")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=500000)
    parser.add_argument("--replay_memory_size", type=int, default=50000)
    parser.add_argument("--log_path", type=str, default="tensorboard_prioritized")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument('--update_target_frequency', type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha value for prioritized replay buffer")
    parser.add_argument("--beta_start", type=float, default=0.4, help="Starting value of beta for prioritized replay")
    parser.add_argument("--beta_frames", type=int, default=100000, help="Number of frames over which beta is annealed")

    args = parser.parse_args()
    return args

def train(opt):
    losses=[]
    rewards=[]
    Q_values=[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepQNetwork().to(device)
    target_model = DeepQNetwork().to(device)
    target_model.load_state_dict(model.state_dict())
    
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image).to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_buffer = PrioritizedReplayBuffer(opt.replay_memory_size, opt.alpha)
    iter = 0
    beta = opt.beta_start

    while iter < opt.num_iters:
        # print(state.shape)
        prediction = model(state)[0]
        epsilon = opt.final_epsilon + (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters
        u = random()
        random_action = u <= epsilon
        
        if random_action:
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
        next_image = torch.from_numpy(next_image).to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        
        replay_buffer.add(state, action, reward, next_state, terminal)

        state = next_state
        episode_reward = reward

        if len(replay_buffer.buffer) > opt.batch_size:
            batch, indices, weights = replay_buffer.sample(opt.batch_size, beta)
            # print(indices.shape)
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

            state_batch = torch.cat(tuple(state for state in state_batch))
            action_batch = torch.from_numpy(
                np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = torch.cat(tuple(state for state in next_state_batch))
            # print(state_batch.shape)
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)
            next_state_batch = next_state_batch.to(device)
            weights = torch.FloatTensor(weights).to(device)
            # print(state_batch.shape)
            # print(action_batch.shape)
            
            q_values = model(state_batch)
            next_q_values = model(next_state_batch)
            next_q_state_values = target_model(next_state_batch)

            
            y_batch=[]
            for i,(reward, terminal,prediction) in enumerate(zip(reward_batch,terminal_batch,next_q_values)):
                
                if terminal:
                    y_batch.append(reward)
                else:
                    next_action=torch.argmax(prediction)
                    y_batch.append(reward+opt.gamma*next_q_state_values[i][next_action])
            y_batch=torch.cat(tuple(y_batch))
            q_value=torch.sum(q_values*action_batch,dim=1)
            optimizer.zero_grad()
            # y_batch = y_batch.detach()

            loss = criterion(q_value, y_batch)*weights
            # print(loss)
            loss = loss.mean()
            # print(loss)
            
            loss.backward()
            optimizer.step()
            
            

            
            
            loss=loss.detach().cpu().numpy()
            loss=np.atleast_1d(loss)
            # print(loss.shape)
            replay_buffer.update_priorities(indices, loss)

        if iter % opt.update_target_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        iter += 1

        if iter % 1000 == 0:
            print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                iter, opt.num_iters, action, loss.item(), epsilon, reward, prediction.max().item()))
            writer.add_scalar('Train/Loss', loss.item(), iter)
            writer.add_scalar('Train/Epsilon', epsilon, iter)
            writer.add_scalar('Train/Reward', reward, iter)
            writer.add_scalar('Train/Q-value', prediction.max().item(), iter)
        losses.append(loss.item())
        rewards.append(reward)
        Q_values.append(torch.max(prediction))
        if (iter + 1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_prio{}".format(opt.saved_path, iter + 1))
    
    torch.save(model, "{}/flappy_bird_prio".format(opt.saved_path))
    writer.close()
    data={'loss':losses,
          'reward':rewards,
          'Q value':Q_values}
    df=pd.DataFrame(data)
    df.to_csv('trainpri.csv')
if __name__ == "__main__":
    opt = get_args()
    train(opt)
