import torch
import torch.nn as nn
import numpy as np

# validation function for trainings loop
def validate_model(model, vali_loader, device, best_vali_loss, batch_size, criterion,  cutoff):
    model.eval()
    vali_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x , label) in enumerate(vali_loader):
            x = x.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            x, label = x.to(device), label.to(device)
            output = model(x)
            vali_loss += criterion(output, label).item()  # item function extract the loss as a float 
            pred = (output > cutoff).float()
            correct += pred.eq(label).sum().item()

    vali_loss /= len(vali_loader.dataset)
    vali_loss *= batch_size

    print("Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        vali_loss, correct, len(vali_loader.dataset),
        100.0 * correct / len(vali_loader.dataset)))

    # save model
    if vali_loss < best_vali_loss:
        print("Saving model...")
        torch.save(model.state_dict(), "model.pt")
        best_vali_loss = vali_loss
    print(f"best vali_loss: {best_vali_loss} vs current vali_loss: {vali_loss}")
    return best_vali_loss, vali_loss, (correct/len(vali_loader.dataset) * 100.0)

