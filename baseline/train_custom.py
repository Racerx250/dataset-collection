import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
from matplotlib import pyplot as plt
import torchvision.models as models
from collections import defaultdict
import custom

# xavier initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

# utility function for plotting
def graph(val_loss, train_loss, label):
    min_epoch = len(val_loss)

    x = [[i + 1] for i in range(min_epoch)]

    if label == "loss":
        plt.plot(x, train_loss, alpha=0.9, linewidth=2.5, color='red', label="train_loss")
        plt.plot(x, val_loss, alpha=0.8, color='blue', linewidth=1.5, label="validate loss")
        plt.ylabel("loss")
    if label == "acc":
        plt.plot(x, train_loss, alpha=0.9, linewidth=2.5, color='red', label="train_acc")
        plt.plot(x, val_loss, alpha=0.8, color='blue', linewidth=1.5, label="validate_acc")
        plt.ylabel("acc")
    plt.legend(loc="upper right")
    plt.xlabel("epoch size")
    plt.show()

def train_model(model, dataloaders, epochs, optimizer, criterion, patience) :
    model.apply(weights_init)
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['validate']
    test_dataloader = dataloaders['test']
    loss_increase = 0
    train_loss_total = []
    val_loss_total = []
    train_acc_total = []
    valid_acc_total = []


    for epoch in range(epochs):
        print(epoch)
        train_loss = 0
        val_loss = 0
        batch = 0
        batch_valid = 0
        train_acc = 0
        valid_acc = 0

        # train
        for i, sample in enumerate(train_dataloader) :
            batch += 1
            image, label = sample
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            # forward pass
            output = model(image)
            # get train accuracy
            _, predicted = torch.max(output.data, 1)
            train_acc += torch.sum(predicted == label)
            # get train loss
            loss = criterion(output, label)
            # backprop
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validate 
        model = model.eval()
        with torch.no_grad():
            for i, sample in enumerate(val_dataloader) :
                batch_valid += 1
                image, label = sample
                image = image.cuda()
                label = label.cuda()
                # validate accuracy, loss
                output_valid = model(image)
                _, predicted = torch.max(output_valid.data, 1)
                valid_acc += torch.sum(predicted == label)
                loss = criterion(output_valid, label)
                val_loss += loss.item()
        model = model.train()

        train_loss_total.append(train_loss / batch)
        val_loss_total.append(val_loss / batch_valid)

        train_acc_total.append(train_acc.cpu().numpy() / len(train_dataloader.dataset))
        valid_acc_total.append(valid_acc.cpu().numpy() / len(val_dataloader.dataset))

        if (epoch != 0) and (val_loss_total[epoch] >= val_loss_total[epoch - 1]):
            loss_increase += 1
            if loss_increase >= patience :
                torch.save(model, './saved_model')
                break
        else :
            loss_increase = 0

    # graph train/validation loss and accuracy
    graph(val_loss_total, train_loss_total, "loss")
    graph(valid_acc_total, train_acc_total, "acc")
    torch.save(model, './saved_model')

    test_acc = 0
    model = model.eval()
    with torch.no_grad():
        for i, test in enumerate(test_dataloader):
            images, labels = test
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_acc += torch.sum(predicted == labels)


    print("Accuracy of the network: " + str(test_acc.cpu().numpy()/len(test_dataset)))


model = custom.Custom_Net(classes = 20)

# load the datasets
root = 'birds_dataset/'
train_dataset = datasets.bird_dataset(root,'train_list.txt')
test_dataset = datasets.bird_dataset(root,'test_list.txt')
valid_dataset = datasets.bird_dataset(root,'valid_list.txt')

# dataloaders for the datasets
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size = 8)
test_dataloader = DataLoader(test_dataset, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False)

dataloaders = defaultdict(DataLoader)
dataloaders['train'] =  train_dataloader
dataloaders['validate'] =  valid_dataloader
dataloaders['test'] =  test_dataloader

epochs = 2
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
model = model.cuda()
train_model(model, dataloaders, epochs, optimizer, criterion, 5)
