import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import torchvision.transforms as T
from torch.utils.data import  DataLoader
from tqdm import tqdm
import math

import numpy as np

from utils.datasets import get_dataloader, get_knn_evaluation_loader, get_linear_evaluation_loader
from utils.transforms import get_policy_distribution

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device



class InfoNCELoss(nn.Module):
    def __init__(self, reduction:str='mean'):
        super(InfoNCELoss, self).__init__()
        self.CE = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, z1: Tensor, z2: Tensor, temperature: float, return_positives:bool=False, weights:Tensor=None):
        
        batch_size = z1.shape[0]
        
        features = torch.cat((z1, z2), dim=0)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0).to(device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        full_similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = full_similarity_matrix[~mask].view(full_similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        # loss = self.CE(logits, labels)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -log_probs[:, 0]
        
        if weights is not None:
            weights = torch.cat([weights, weights]).to(device)
            loss = loss * weights
        loss = loss.mean()

        if return_positives:
            return full_similarity_matrix, logits, loss, positives[:len(positives)//2]
        else:
            return full_similarity_matrix, logits, loss
    



            
            
            
# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, targets=None):
    if not targets:
        if 'targets' in memory_data_loader.dataset.__dir__():
            targets = memory_data_loader.dataset.targets
        elif 'labels' in memory_data_loader.dataset.__dir__():
            targets = memory_data_loader.dataset.labels
        else:
            raise NotImplementedError
        
    net.eval()
    classes = len(np.unique(targets))
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        # for data, target in tqdm(memory_data_loader, total=len(memory_data_loader)):
        for data, target in memory_data_loader:
            feature = net(data.to(device=device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device, dtype=torch.int64)
        # loop test data to predict the label by weighted knn search
        # for data, target in tqdm(test_data_loader, total=len(test_data_loader)):
        for data, target in test_data_loader:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)
            
            total_num += data.size(0)

            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    feature and feature_bank are normalized
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # print(one_hot_label)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels         


def knn_evaluation(encoder, args):
    
    memory_loader, test_loader = get_knn_evaluation_loader(args.dataset, batch_size=512)
    
    if isinstance(encoder, torch.nn.DataParallel):
        enc = encoder.module.enc
    else:
        enc = encoder.enc

    acc = knn_monitor(
        enc,
        memory_loader,
        test_loader,
    )
    
    return acc


def top_k_accuracy(sim, k):
    n_samples = sim.shape[0] // 2
    sim[range(sim.shape[0]), range(sim.shape[0])] = -1
    y_index = torch.tensor(list(range(n_samples, sim.shape[0]))).reshape(-1, 1)
    acc = (sim.argsort()[:n_samples, -k:].detach().cpu() == y_index).any(-1).sum() / n_samples
    return acc.item()











def eval_loop(encoder, args, ind=None):
    
    batch_size = 256
    epochs = 100
    
    if args.dataset == 'TinyImagenet':
        num_classes = 200
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
        
    train_loader, test_loader = get_linear_evaluation_loader(args.dataset, batch_size=batch_size)
        
    
    feature_dim = 2048
    
    # if len(encoder.layer4) == 2: # resnet18
    #     feature_dim = 512
    # elif len(encoder.layer4) == 3: # resnet50
    #     feature_dim = 2048
    # else:
    #     raise NotImplementedError
    
    classifier = nn.Linear(feature_dim, num_classes).cuda()
    # optimization
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=0.9,
        lr=30,
        weight_decay=0
    )
    
    def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
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

    # training
    for e in tqdm(range(1, epochs+1)):
        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=epochs,
                                 warmup_epochs=0,
                                 base_lr=30,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                with torch.no_grad():
                    b = encoder(inputs.cuda())
                logits = classifier(b)
                loss = F.cross_entropy(logits, y.cuda())
                return loss

            # optimization step
            loss = forward_step()
            loss.backward()
            optimizer.step()

        if e % 10 == 0:
            accs = []
            classifier.eval()
            for idx, (images, labels) in enumerate(test_loader):
                with torch.no_grad():
                    b = encoder(images.cuda())
                    preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.cuda()).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs) * 100
            # final report of the accuracy
            line_to_print = (
                f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
            )
            print(line_to_print)

    return accuracy



def get_avg_loss(encoder, policies, criterion, batch_size, args, num_steps=10):
    
    dist = get_policy_distribution(N=min(len(policies), 4), p=0.6)
    train_loader = get_dataloader(
        args=args,
        batch_size=batch_size,
    )
    
    tqdm_train_loader = tqdm(enumerate(train_loader), total=len(train_loader), desc='[get_average_infoNCE_loss]')
    avg_infoNCE_loss = []
    encoder.train()
    
    for it, (x, x1, x2, y) in tqdm_train_loader:

        # Simclr:
        with torch.no_grad():
            _, z1 = encoder(x1.to(device))
            _, z2 = encoder(x2.to(device))

            _, _, simclr_loss = criterion(z1, z2, temperature=0.5)
            
            avg_infoNCE_loss.append(simclr_loss.item())
            
            if it == num_steps-1:
                break
     
    return sum(avg_infoNCE_loss) / len(avg_infoNCE_loss)