import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import math
import copy
import pickle

from utils.datasets import get_dataloader, plot_images_stacked
from utils.networks import SimCLR, build_resnet18, build_resnet50
from utils.contrastive import InfoNCELoss, knn_evaluation, top_k_accuracy, eval_loop, get_avg_loss
from utils.transforms import get_transforms_list, NUM_DISCREATE, transformations_dict
from utils.logs import init_neptune, get_model_save_path
from utils.hardness_estimation import get_hardness_estimator, parts2vector
import argparse
    

def fix_seed(seed=None):
    
    if seed is None:
        seed = random.randint(0, 100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed
    

def mean_last_percentage(lst, P):

    # Calculate the number of elements to consider based on the percentage
    num_elements = int(len(lst) * P)
    
    # Extract the last percentage of elements
    last_percentage_elements = lst[-num_elements:]

    # Calculate the mean
    mean_value = sum(last_percentage_elements) / len(last_percentage_elements)
    
    return mean_value


def contrastive_init(args, device):
    
    if args.encoder_backbone == 'resnet18':
        encoder = build_resnet18(args.reduce_resnet)
    elif args.encoder_backbone == 'resnet50':
        encoder = build_resnet50(args.reduce_resnet)
    
    encoder = encoder.to(device)

    criterion = InfoNCELoss()
    
    optimizer = torch.optim.SGD(
        encoder.parameters(),
        momentum=0.9,
        lr=args.lr * args.simclr_bs / 256,
        weight_decay=0.0005
    )
    
    return encoder, optimizer, criterion


def adjust_learning_rate(
        epochs: int,
        warmup_epochs: int,
        base_lr: float,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        step: int
    ):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def init(args, neptune_run, device):
    
    start_epoch = 1
    
    encoder, simclr_optimizer, simclr_criterion = contrastive_init(args, device)
    
    if args.checkpoint_id:
        
        checkpoint_params = args.checkpoint_params
        
        encoder.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder.pt'))
        simclr_optimizer.load_state_dict(torch.load(f'params/{checkpoint_params}/encoder_opt.pt'))
        
        # if USE_NEPTUNE:
        #     prev_run = neptune.init_run(
        #         project="nazim-bendib/simclr-rl",
        #         api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDVjNWJkYi1mMTIwLTRmNDItODk3Mi03NTZiNzIzZGNhYzMifQ==",
                
        #         # project="nazimbendib1/SIMCLR",
        #         # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGIxMzhlZS00MzhhLTQ0ZDktYTU2Yy0yZDk3MjE4MmU4MDgifQ==",
                
        #         with_id=args.checkpoint_id
        #     )
                        
        #     test_acc = prev_run['linear_eval/test_acc'].fetch_values().value.tolist()
        #     start_epoch = len(test_acc) + 1
            
        #     for acc in test_acc:
        #         neptune_run["linear_eval/test_acc"].append(acc)

        #     prev_run.stop()
    
    return (
        encoder, simclr_optimizer, simclr_criterion, start_epoch
    )

def contrastive_round(
        encoder: SimCLR,
        train_loader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        args,
        epoch: int,
        neptune_run,
        device,
        estimator=None
    ):
    
    batch_size = args.simclr_bs
    
    print('[contrastive_round]')
    tqdm_train_loader = tqdm(enumerate(train_loader), total=len(train_loader), desc='[contrastive_round]')
    # tqdm_train_loader = enumerate(train_loader)
    
    collected_data = []
    
    encoder.train()
    for it, (x1, x2, y, details) in tqdm_train_loader:
        
        # points = []
        # for i in range(len(details)):
        #     box = [ details[i][1][j][1] for j in range(len(details[i][1])) if 'crop' in details[i][1][j] ][0]
        #     print(box)
        #     points.append(box)
        # plot_images_stacked(x1, x2, points)
        # plot_images_stacked(x1, x2)
        
        lr = adjust_learning_rate(epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr * args.simclr_bs / 256,
            optimizer=optimizer,
            loader=train_loader,
            step=it+(epoch-1)*len(train_loader)
        )
        
        # Simclr:
        _, z1 = encoder(x1.to(device))
        _, z2 = encoder(x2.to(device))
        
        
        weights = None
        if estimator is not None:
            vectors = list(map(parts2vector, details))
            estimated_cos = estimator.predict(vectors)
            estimated_cos = torch.tensor(estimated_cos)
            weights = (1-estimated_cos)
            min, max, eps = weights.min(), weights.max(), 1e-5
            weights = (weights - min) / (max - min) + eps
            weights = weights / weights.sum()

        _, _, simclr_loss, positives = criterion(z1, z2, temperature=0.5, return_positives=True, weights=weights)
        
        optimizer.zero_grad()
        simclr_loss.backward()
        optimizer.step()

        # logs:            
        if (len(train_loader) < 10) or (it % (len(train_loader) // 10) == 0):
            neptune_run["simclr/loss"].append(simclr_loss.item())
        
        
        positives = positives.reshape(-1).tolist()
        positives = [round(n,2) for n in positives]
        collected_data.append( (details, positives) )
        
        
    with open(f'{args.model_save_path}/collected_data.txt', 'a') as file:
        for details, positives in collected_data:
            for (details_1, details_2), sim in zip(details, positives):
                details_1 = ' '.join(f'{trans} {val}' for trans, val in details_1)
                details_2 = ' '.join(f'{trans} {val}' for trans, val in details_2)
                file.write(f'{details_1};{details_2};{sim}\n')
        file.write('\n\n\n\n\n\n\n')
    
    x1_x2, sims = [], []
    for details, positives in collected_data:
        for (x1, x2), sim in zip(details, positives):
            x1_x2.append((x1, x2))
            sims.append(sim)
    
    return x1_x2, sims
                        

def main(args):
    
    
    args.reduce_resnet = True
    
    if args.epochs == None:
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            args.epochs = 800
        elif args.dataset == 'svhn':
            args.epochs = 400
        elif args.dataset == 'TinyImagenet':
            args.epochs = 400
        elif args.dataset == 'stl10':
            args.epochs = 400
            
    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        args.reduce_resnet = True
    elif args.dataset in ['stl10', 'TinyImagenet']:
        args.reduce_resnet = False
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    
    model_save_path = args.model_save_path
    
    neptune_run = init_neptune(
        tags=[
            f'seed={args.seed}',
            f'dataset={args.dataset}',
            f'simclr_bs={args.simclr_bs}',
            f'model_save_path={args.model_save_path}', 
            f'encoder_backbone={args.encoder_backbone}', 
            f'lr={args.lr}',
        ],
        mode=args.mode,
    )
    
    for k, v in vars(args).items():
        print(k, ':', v)
        
    neptune_run["scripts"].upload_files(["./utils/*.py", "./*.py"])

    encoder, simclr_optimizer, simclr_criterion, start_epoch = init(args, neptune_run, device)

    train_loader = get_dataloader(
        args=args,
        batch_size=args.simclr_bs,
    )
    
    estimator = None

    for epoch in tqdm(range(start_epoch, args.epochs+1), desc='[Main Loop]'):

        print(f'EPOCH:{epoch}')
        
        x1_x2, sims = contrastive_round(
            encoder=encoder,
            train_loader=train_loader,
            epoch=epoch,
            args=args,
            optimizer=simclr_optimizer, 
            criterion=simclr_criterion, 
            neptune_run=neptune_run,
            device=device,
            estimator=estimator
        )
    
        if args.weights:
            vectors = list(map(parts2vector, x1_x2))
            print(len(vectors))
            estimator = get_hardness_estimator(vectors, sims, plot=False)
            print('Estimator built!')
                

        if  ((args.dataset in ['cifar10', 'svhn']) and epoch % 1 == 0) or \
            ((args.dataset in ['cifar100', 'TinyImagenet', 'stl10']) and epoch % 5 == 0):
            test_acc = knn_evaluation(encoder, args)
            neptune_run["linear_eval/test_acc"].append(test_acc)

        
        
        if  (args.dataset == 'cifar10'  and epoch in [200, 400, 600, 800]) or \
            (args.dataset == 'cifar100' and epoch in [200, 400, 600, 800]) or \
            (args.dataset == 'svhn'     and epoch in [100, 200, 300, 400]) or \
            (args.dataset == 'stl10'    and epoch in [100, 200, 300, 400]) or \
            (args.dataset == 'TinyImagenet' and epoch in [50, 100, 150, 200]):
            
            os.mkdir(f'{model_save_path}/epoch_{epoch}/')
            torch.save(encoder.state_dict(), f'{model_save_path}/epoch_{epoch}/encoder.pt')
            torch.save(simclr_optimizer.state_dict(), f'{model_save_path}/epoch_{epoch}/encoder_opt.pt')
            
            neptune_run[f"params/epoch_{epoch}/encoder"].upload(f'{model_save_path}/epoch_{epoch}/encoder.pt')
            neptune_run[f"params/epoch_{epoch}/encoder_opt"].upload(f'{model_save_path}/epoch_{epoch}/encoder_opt.pt')
            neptune_run[f"params/epoch_{epoch}/collected_data"].upload(f'{model_save_path}/epoch_{epoch}/collected_data.txt')
        
        
        if (epoch % 10 == 0) or (epoch == args.epochs):
            torch.save(encoder.state_dict(), f'{model_save_path}/encoder.pt')
            torch.save(simclr_optimizer.state_dict(), f'{model_save_path}/encoder_opt.pt')
        
        if  (epoch % (args.epochs // 4) == 0) or (epoch == args.epochs):
            neptune_run["params/encoder"].upload(f'{model_save_path}/encoder.pt')
            neptune_run["params/encoder_opt"].upload(f'{model_save_path}/encoder_opt.pt')
            neptune_run["params/collected_data"].upload(f'{model_save_path}/collected_data.txt')
    
    
    if isinstance(encoder, torch.nn.DataParallel):
        enc = encoder.module.enc
        enc = torch.nn.DataParallel(enc)
    else:
        enc = encoder.enc

    print('\n\n\nLinear evaluation:')
    accs = []
    for i in range(3):
        accs.append(eval_loop(copy.deepcopy(enc), args, i))
        line_to_print = f'aggregated linear probe: {np.mean(accs):.3f} +- {np.std(accs):.3f}'
        print(line_to_print)

        

    neptune_run.stop()
    
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Argument Parser for Training Configuration')
        
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warm-up epochs')

    parser.add_argument('--simclr_iterations', type=str, default='all', help='Iterations for SimCLR training')
    parser.add_argument('--simclr_bs', type=int, default=512, help='Batch size for SimCLR training')
    parser.add_argument('--encoder_backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50'], help='Encoder backbone architecture')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn', 'TinyImagenet', 'cifar100', 'stl10'], help='Pretraining dataset')
    parser.add_argument('--weights', action='store_true')
    
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate for contrastive training')

    parser.add_argument('--mode', type=str, default='debug', choices=['sync', 'async', 'debug'], help='Training mode')

    parser.add_argument('--model_save_path', type=str, default="", help='Path to save the model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')

    parser.add_argument('--checkpoint_id', type=str, default="", help='Checkpoint ID')
    parser.add_argument('--checkpoint_params', type=str, default="", help='Checkpoint parameters')

    args = parser.parse_args()
    
    seed = fix_seed(args.seed)
    # seed = fix_seed(None)
    args.seed = seed
    
    if not args.model_save_path:
        model_save_path = get_model_save_path()
        args.model_save_path = model_save_path

    main(args)