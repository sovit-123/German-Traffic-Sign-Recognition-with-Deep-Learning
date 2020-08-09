from model import Net
from tqdm import tqdm
from dataset import train_data, val_data
from dataset import train_data_loader, val_data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np

matplotlib.style.use('ggplot')

# learning parameters
epochs = 20
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training function
def fit(model, dataloader, optimizer, criterion, train_data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        image, target = data['image'].to(device), data['label'].to(device)
        plt.show()
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/len(dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(dataloader.dataset)    
    return train_loss, train_accuracy

# validation function
def validate(model, dataloader, optimizer, criterion, val_data):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            image, target = data['image'].to(device), data['label'].to(device)
            outputs = model(image)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss / len(dataloader.dataset)
        val_accuracy = 100. * (val_running_correct / len(dataloader.dataset))
        return val_loss, val_accuracy

# initialize the model
model = Net()
model = model.to(device)
# optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 
                       eps=1e-8, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(model, train_data_loader, 
                                                 optimizer, criterion, 
                                                 train_data)
    val_epoch_loss, val_epoch_accuracy = validate(model, val_data_loader, 
                                                 optimizer, criterion, 
                                                 val_data)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../outputs/accuracy.png')
plt.show()
 
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()

# save model checkpoint
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, '../outputs/model.pth')