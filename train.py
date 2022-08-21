import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset.dataset import Dataset

import yaml
import argparse
from models.model_factory import model_build

def train(cfg, args):
    '''train model
        if args.mode == sl, then proceed supervised learning
        if args.mode == rl, then proceed reinforcement learning
    '''
    mode = args.mode
    if mode == 'sl':
        sl_train(cfg, args)
    elif mode == 'rl':
        rl_train(cfg, args)
    else:
        raise ValueError('train mode must one of ["sl", "rl"]')

def sl_train(cfg):
    '''train model by supervised learning

    '''
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
    #       Training SetUp       #
    ##############################
    
    # loss / optim / scheduler / ...
    loss_func = build_loss_func(cfg, args)
    optimizer = build_optim(cfg, args, model)
    scheduler = build_scheduler(cfg, args, optimizer)
    epochs = cfg['train']['sl']['epochs']
    device = cfg['train']['device']
    if device == 'cuda':
        if not torch.cuda.is_available():
            print('Cuda is unavailable. Replay the device with cpu.')
            device = 'cpu'
    
    ##############################
    #       BUILD MODEL          #
    ##############################
    model = model_build().to(device)
    if args.resume:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        os.makedirs(cfg['train']['save_path'], exist_ok=True)
        start_epoch = 1

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
            optimizer.zero_grad()
            prediction = model(state)
            loss = loss_func(prediction, action)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
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
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
            }
            , f"{cfg['train']['sl']['save_path']}/checkpoint_{epoch}.ckpt")

def rl_train(cfg, args):
    '''train model by reinforcement learning

        
    '''
    pass

def build_loss_func(cfg, args):
    '''build loss function depends on train type

    Returns:
        torch loss function
    '''
    mode = args.mode
    if mode == 'sl':
        pass
    elif mode == 'rl':
        pass
    
    # raise NotImplementedError()
    return None



def build_optim(cfg, args, model):
    '''train model by supervised learning

    Returns:
        torch optimizer
    '''
    mode = args.mode
    if mode == 'sl':
        optim = cfg['train'][mode]['optim']
        lr = cfg['train'][mode]['learning_rate']
    elif mode == 'rl':
        optim = cfg['train']['rl']['optim']
        lr = cfg['train']['rl']['learning_rate']
    
    optim = optim.lower()    

    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    
    # TODO : add optimizer


def build_scheduler(cfg, args, optim):
    '''train model by supervised learning

    Returns:
        torch learning rate scheduler
    '''
    mode = args.mode
    if mode == 'sl':
        scheduler_dict = cfg['train']['sl']['scheduler']
    elif mode == 'rl':
        scheduler_dict = cfg['train']['rl']['scheduler']

    scheduler, spec = scheduler_dict.items()
    scheduler = scheduler.lower()
    
    if scheduler == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[spec["milestones"]], gamma=spec["gamma"])

    if scheduler == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(optim, base_lr=spec["base_lr"], max_lr=spec["max_lr"])
    
    # TODO : add leraning rate scheduler

def plot_progress():
    """
    
    """
    pass

def log_progress():
    '''

    '''
    pass

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