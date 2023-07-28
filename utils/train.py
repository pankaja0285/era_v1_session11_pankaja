import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.helper import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# def train(model, device, train_loader, optimizer, epoch, train_acc, train_loss,
# lambda_l1, scheduler, criterion, lrs, writer, grad_clip=None):

#   model.train()
#   pbar = tqdm(train_loader)
  
#   correct = 0
#   processed = 0

#   for batch_idx, (data, target) in enumerate(pbar):
#     # get samples
#     data, target = data.to(device), target.to(device)

#     # Init
#     optimizer.zero_grad()
#     # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
#     # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

#     # Predict
#     y_pred = model(data)

#     # Calculate loss
#     loss = criterion(y_pred, target)
    
#     #L1 Regularization
#     if lambda_l1 > 0:
#       l1 = 0
#       for p in model.parameters():
#         l1 = l1 + p.abs().sum()
#       loss = loss + lambda_l1*l1

#     train_loss.append(loss.data.cpu().numpy().item())
#     #t_loss += loss.data.cpu().numpy().item()
    
#     writer.add_scalar(
#                 'Batch/Train/train_loss', loss.data.cpu().numpy().item(), epoch*len(pbar) + batch_idx)

#     # Backpropagation
#     loss.backward()
    
#     # Gradient clipping
#     if grad_clip: 
#         nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
#     optimizer.step()   
#     if "OneCycleLR" in str(scheduler):
#         scheduler.step()
        
#     lrs.append(get_lr(optimizer))

#     # Update pbar-tqdm
    
#     pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#     correct += pred.eq(target.view_as(pred)).sum().item()
#     processed += len(data)

#     pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]:0.5f} Accuracy={100*correct/processed:0.2f}')
#     train_acc.append(100*correct/processed)


# Training
def train(net, device, train_loader, optimizer, epoch, train_acc, train_losses,
    lambda_l1, scheduler, criterion, lrs, writer, grad_clip=None): 

    # Start the training
    net.train()
    correct = 0
    processed = 0
    train_loss = 0
    total = 0
    pbar = tqdm(train_loader)

    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        #L1 Regularization
        if lambda_l1 > 0:
            l1 = 0
            for p in net.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1*l1

        # train_loss.append(loss.data.cpu().numpy().item())
        # train_loss += loss.item()
        # t_loss += loss.data.cpu().numpy().item()
        
        writer.add_scalar(
                    'Batch/Train/train_loss', loss.data.cpu().numpy().item(), epoch*len(pbar) + batch_idx)

        # Backpropagation
        loss.backward()
        # Gradient clipping
        if grad_clip: 
            nn.utils.clip_grad_value_(net.parameters(), grad_clip)
            
        optimizer.step()   
        if "OneCycleLR" in str(scheduler):
            scheduler.step()
        lrs.append(get_lr(optimizer))
        
        # train_loss.append(loss.data.cpu().numpy().item())
        train_loss = loss.data.cpu().numpy().item()
        train_losses.append(train_loss)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        processed += len(inputs)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]:0.5f} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
