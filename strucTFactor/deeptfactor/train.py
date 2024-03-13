import torch
import torch.nn as nn
import numpy as np

# training function for trainings loop
def train_model(model, device, train_loader, optimizer, criterion,  cutoff):
    model.train()
    correct = 0
    
    for batch_idx, (x, label) in enumerate(train_loader): # maybe change tqdm to stuff from saliency

        
        x = x.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad() 
        output = model(x)
        loss = criterion(output, label)
        pred = (output > cutoff).float()
        correct += pred.eq(label).sum().item()
        loss.backward()
        optimizer.step()

    print("Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        loss, correct, len(train_loader.dataset),
        100.0 * correct / len(train_loader.dataset)))

    return loss.item(), (correct/len(train_loader.dataset) * 100.0)



