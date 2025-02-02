
import argparse
import os
import shutil
from random import random, randint, sample
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Double Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=1000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard_ddqn")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument('--update_target_frequency', type=int, default=10000)
    parser.add_argument("--log_path", type=str, default="tensorboard_ddqn")
    parser.add_argument("--checkpoint_path",type=str,default="checkpointddqn.pth")
    
    args = parser.parse_args()
    return args

def save_checkpoint(main_model,target_model,optimizer,replay_memory,iter,opt,state):
    checkpoint={
        'main_model_state_dict':main_model.state_dict(),
        'target_model_state_dict':target_model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'replay_memory':replay_memory,
        "iter":iter,
        "state":state,
    }
    torch.save(checkpoint,opt.checkpoint_path)
    print(f'checkpoint saved at iteration {iter}')
def load_checkpoint(opt,main_model,target_model,optimizer):
    if os.path.isfile(opt.checkpoint_path):
        checkpoint=torch.load(opt.checkpoint_path)
        main_model.load_state_dict(checkpoint['main_model_state_dict'])
        target_model.load_state_dict(checkpoint['target_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        replay_memory=checkpoint['replay_memory']
        iter=checkpoint['iter']
        state=checkpoint['state']
        print(f'checkpoint loaded from iteration {iter}')
        return main_model,target_model,optimizer,replay_memory,iter,state
    else:
        print("No checkpoint found")
        return main_model,target_model,optimizer,[],0,None

def train(opt):
    losses=[]
    rewards=[]
    Q_values=[]
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork().to(device)
    target_model=DeepQNetwork().to(device)
    # target_model.load_state_dict(model.state_dict())
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path) #xóa các mục con và thư mục con trong tệp nếu thư mục có tồn tại
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    # load checkpoint if exists
    model,target_model,optimizer,replay_memory,start_iter,state=load_checkpoint(opt,model,target_model,optimizer)
    if state is None:
        game_state = FlappyBird()
        image, reward, terminal = game_state.next_frame(0)
        image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            model.to(device)
            image = image.to(device)
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    # replay_memory = []
    # iter = 0
    iter=start_iter
    while iter < opt.num_iters:
        prediction = model(state)[0]
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:

            action = torch.argmax(prediction)

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))
        print(state_batch.shape)
        # if torch.cuda.is_available():
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)
        next_states_values=target_model(next_state_batch)
        # y_batch = torch.cat(
        #     tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
        #           zip(reward_batch, terminal_batch, next_prediction_batch)))

        # current_prediction_batch = model(state_batch).gather(1, action_batch.type(torch.int64))
        # next_prediction_batch = model(next_state_batch)
        # next_action_batch = torch.argmax(next_prediction_batch, dim=1, keepdim=True)
        # next_target_batch = target_model(next_state_batch).gather(1, next_action_batch)
        
        # y_batch = reward_batch + (1 - terminal_batch) * opt.gamma * next_target_batch
        # y_batch=y_batch.detach()
        y_batch=[]
        for i,(reward, terminal,prediction) in enumerate(zip(reward_batch,terminal_batch,next_prediction_batch)):
            
            if terminal:
                y_batch.append(reward)
            else:
                next_action=torch.argmax(prediction)
                y_batch.append(reward+opt.gamma*next_states_values[i][next_action])
        y_batch=torch.cat(tuple(y_batch))
        # y_batch = torch.cat(
        #     tuple(reward if terminal else reward + opt.gamma * next_states_values[torch.argmax(prediction)] for reward, terminal, prediction in
        #           zip(reward_batch, terminal_batch, next_prediction_batch)))

        current_prediction_batch = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(current_prediction_batch, y_batch)
        loss.backward()
        optimizer.step()

        if iter % opt.update_target_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        state = next_state
        iter += 1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            opt.num_iters,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
        losses.append(loss.item())
        rewards.append(reward)
        Q_values.append(torch.max(prediction))
        if (iter+1) %50000==0:
            save_checkpoint(model,target_model,optimizer,replay_memory,iter,opt,state)
        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_ddqn{}".format(opt.saved_path, iter+1))
    torch.save(model, "{}/flappy_bird_ddqn".format(opt.saved_path))
    save_checkpoint(model,target_model,optimizer,replay_memory,iter,opt,state)
    writer.close()
    data={'loss':losses,
          'reward':rewards,
          'Q value':Q_values}
    df=pd.DataFrame(data)
    df.to_csv('trainddqn.csv')

if __name__ == "__main__":
    opt = get_args()
    train(opt)
