import os
import yaml
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from env_wrapper import KoreGymEnv
from dataset.dataset import Dataset
from models.model_factory import model_build
from utils import build_loss_func, build_optim, build_scheduler
from utils import plot_progress, log_progress, get_agent_ratio, get_env_info

def train(cfg, args):
    """train model
        if args.mode == sl, then proceed supervised learning
        if args.mode == rl, then proceed reinforcement learning
    """
    mode = args.mode
    if mode == 'sl':
        sl_train(cfg, args)
    elif mode == 'rl':
        rl_train(cfg, args)
    else:
        raise ValueError('train mode must one of ["sl", "rl"]')

def sl_train(cfg):
    """train model by supervised learning

    """
    ##############################
    #       DataLoader           #
    ##############################
    dataset = Dataset(cfg['sl']['dataset']['txt_path'])
    dataset_size = len(dataset)
    train_size = int(dataset_size*cfg['sl']['dataset']['train_val_ratio'])
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")

    batch = cfg['sl']['dataset']['batch_size']

    train_dataloader = DataLoader(train_dataset, batch_size=batch, \
        shuffle=True, num_workers=cfg['sl']['dataset']['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=batch, \
        shuffle=False, num_workers=cfg['sl']['dataset']['num_workers'])
    
    ##############################
    #       BUILD MODEL          #
    ##############################
    device = cfg['train']['device']
    if device == 'cuda':
        if not torch.cuda.is_available():
            print('Cuda is unavailable. Replay the device with cpu.')
            device = 'cpu'
            
    model = model_build().to(device)
    if args.resume:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        os.makedirs(cfg['train']['save_path'], exist_ok=True)
        start_epoch = 1

    ##############################
    #       Training SetUp       #
    ##############################
    
    # loss / optim / scheduler / ...
    loss_func = build_loss_func(cfg, args)
    optim = build_optim(cfg, args, model)
    scheduler = build_scheduler(cfg, args, [optim])
    epochs = cfg['train']['sl']['epochs']

    history = {"T_loss": [], "V_loss": []}

    ##############################
    #       START TRAINING !!    #
    ##############################

    for epoch in range(start_epoch, epochs+1):
        model.train()
        training_loss = 0.0
        for i, (state, action) in enumerate(train_dataloader):
            # TODO : get state and action from dataloader
            # state, action = ....
            optim.zero_grad()
            prediction = model(state)
            loss = loss_func(prediction, action)
            training_loss += loss.item()
            loss.backward()
            optim.step()
            log_progress()
        history["T_loss"].append(training_loss/len(train_dataloader))

        model.eval()
        with torch.no_grad():
            validating_loss = 0.0
            for (state, action) in val_dataloader:
                # TODO : get state and action from dataloader
                # state, action = ....
                # state, action = state.to(device), action.to(device)
                prediction = model(state)
                loss = loss_func(prediction, action)
                validating_loss += loss.item()

            print(f"(Finish) Epoch : {epoch}/{epochs} >>>> Validation loss : {validating_loss/len(val_dataloader):.6f}")
            history["V_loss"].append(validating_loss/len(val_dataloader))

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
            }
            , f"{cfg['train']['sl']['save_path']}/checkpoint_{epoch}.ckpt")

def rl_train(cfg, args):
    """train model by reinforcement learning
    
        
    """    
    ##############################
    #       BUILD MODEL          #
    ##############################
    if device == 'cuda':
        if not torch.cuda.is_available():
            print('Cuda is unavailable. Replay the device with cpu.')
            device = 'cpu'
    model = model_build(cfg, args).to(device)


    ##############################
    #       Training SetUp       #
    ##############################
    
    # loss / optim / scheduler / ...
    actor_loss_func, critic_loss_func = build_loss_func(cfg, args)
    actor_optim, critic_optim = build_optim(cfg, args, model)
    actor_scheduler, critic_scheduler = build_scheduler(cfg, args, [actor_optim, critic_optim])
    episodes = cfg['train']['rl']['episodes']
    device = cfg['train']['device']

    # set other agents for training
    play_history = defaultdict(lambda: [0, 0])
    for agent_file in os.listdir('./other_agents'):
        agent_name, file_extension = os.path.splitext(agent_file)
        if file_extension == '.py':
            play_history[agent_name]
    print(f'other agents : {play_history.keys()}')
    agents_ratio = get_agent_ratio(play_history)

    if args.resume:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        actor_optim.load_state_dict(checkpoint['optimizer_state_dict']['actor'])
        critic_optim.load_state_dict(checkpoint['optimizer_state_dict']['critic'])
        actor_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['actor'])
        critic_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['critic'])
        episode= checkpoint['episode']
    else:
        os.makedirs(cfg['train']['save_path'], exist_ok=True)
        episode = 1

    # set environment
    env = KoreGymEnv()
    model.env = env
    model.episode = episode

    ##############################
    #       START TRAINING !!    #
    ##############################
    print(f"Learning... Running {model.max_timesteps_per_episode} timesteps per episode, ")
    print(f"{model.timesteps_per_batch} timesteps per batch for a total of {episodes} episodes")
    while model.episode < episodes:
        other_agent = np.random.choice(agents_ratio.keys(), 1, p=agents_ratio.values())
        if np.random.random() < 0.5:
            agents = ['./other_agents'+other_agent+'.py', None]
        else:
            agents = [None, './other_agents'+other_agent+'.py']
        
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, play_history = get_env_info(model, agents, play_history)

        # Calculate advantage at k-th iteration
        V, _ = model.evaluate(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

        # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
        # isn't theoretically necessary, but in practice it decreases the variance of 
        # our advantages and makes convergence much more stable and faster. I added this because
        # solving some environments was too unstable without it.
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # This is the loop where we update our network for some n epochs
        for _ in range(model.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = model.evaluate(batch_obs, batch_acts)

            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            # NOTE: we just subtract the logs, which is the same as
            # dividing the values and then canceling the log with e^log.
            # For why we use log probabilities instead of actual probabilities,
            # here's a great explanation: 
            # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
            # TL;DR makes gradient ascent easier behind the scenes.
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - model.clip, 1 + model.clip) * A_k

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
            # the performance function, but Adam minimizes the loss. So minimizing the negative
            # performance function maximizes it.
            actor_loss = actor_loss_func(surr1, surr2)
            critic_loss = critic_loss_func(V, batch_rtgs)

            # Calculate gradients and perform backward propagation for actor network
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

        # Print a summary of our training so far
        log_progress()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {'actor' : actor_optim.state_dict(),
                                        'critic' : critic_optim.state_dict()},
                "scheduler_state_dict": {'actor' : actor_scheduler.state_dict(),
                                        'critic' : critic_scheduler.state_dict()},
                "episode": model.episode
            }
            , f"{cfg['train']['rl']['save_path']}/checkpoint_{model.episode}.ckpt")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.yaml', \
        help='Path to config file')
    parser.add_argument('--mode', type=str, default='sl', \
        help='Option to train mode. sl(supervised learning), rl(reinforcement learning) is available')
    parser.add_argument('--resume', type=bool, default=False, \
        help='Option to resume train')
    parser.add_argument('--resume_path', type=str, default='./pretrained/supervised_learning/checkpoint_1.ckpt', \
        help='Path to saved checkpoint file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    train(cfg, args)