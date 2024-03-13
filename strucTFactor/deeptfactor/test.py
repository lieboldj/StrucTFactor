import torch
import torch.nn as nn
import numpy as np

# test function after the training loop
def test_model(model, test_loader, device, x_test):
    y_pred = torch.zeros([len(x_test), 1])
    with torch.no_grad():
        model.eval()
        cnt = 0
        for x, _ in test_loader:
            x = x.type(torch.FloatTensor)
            x_length = x.shape[0]
            output = model(x.to(device))
            prediction = output.cpu()
            y_pred[cnt:cnt+x_length] = prediction
            cnt += x_length

    return y_pred[:,0]