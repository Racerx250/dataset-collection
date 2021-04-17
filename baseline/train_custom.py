import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import custom
import datasets
from matplotlib import pyplot as plt
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# load the datasets
root = 'birds_dataset/'
train_dataset = datasets.bird_dataset(root,'train_list.txt')
train_dataset2 = datasets.bird_dataset(root,'train_list2.txt')
test_dataset = datasets.bird_dataset(root,'test_list.txt')
valid_dataset = datasets.bird_dataset(root,'valid_list.txt')
valid_dataset2 = datasets.bird_dataset(root,'valid_list2.txt')


# dataloaders for the datasets
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size = 8)
test_dataloader = DataLoader(test_dataset, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, shuffle=False)
train_dataloader2 = DataLoader(train_dataset2, shuffle=True, batch_size = 8)
valid_dataloader2 = DataLoader(valid_dataset2, shuffle=False)

# lists of dataloaders for the cross validation sets
train_loaders = [train_dataloader, train_dataloader2]
valid_loaders = [valid_dataloader, valid_dataloader2]

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

# train
# vairable for saving best model
best_model = custom.Custom_Net(classes = 20)
# print architecture
print(best_model)
best_loss = 10000
for i in range(len(train_loaders)) :
    nn_model = custom.Custom_Net(classes = 20)
    nn_model.to(device)
    nn_model.apply(weights_init)
    # Create loss functions, optimizers
    # For baseline model use this
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=3e-4)
    
    print(i)
    print("---")
    cur_train_loader = train_loaders[i]
    cur_valid_loader = valid_loaders[i]
    
    train_loss_total = []
    val_loss_total = []
    train_acc_total = []
    valid_acc_total = []
    for epoch in range(25):
        print(epoch)
        train_loss = 0
        val_loss = 0
        batch = 0
        batch_valid = 0
        train_correct = 0
        train_acc = 0
        valid_acc = 0
        
        # train
        for i, sample in enumerate(cur_train_loader) :
            batch += 1
            image, label = sample
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # forward pass
            output = nn_model(image)
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
        nn_model = nn_model.eval()
        with torch.no_grad():
            for i, v in enumerate(cur_valid_loader) :
                batch_valid += 1
                image, label = v
                image = image.to(device)
                label = label.to(device)
                # validate accuracy, loss
                output_valid = nn_model(image)
                _, predicted = torch.max(output_valid.data, 1)
                valid_acc += torch.sum(predicted == label)
                loss = criterion(output_valid, label)
                val_loss += loss.item()
        nn_model = nn_model.train()

        train_loss_total.append(train_loss / batch)
        val_loss_total.append(val_loss / batch_valid)
        train_acc_total.append(train_acc.cpu().numpy() / len(cur_train_loader.dataset))
        valid_acc_total.append(valid_acc.cpu().numpy() / len(cur_valid_loader.dataset))
        
    graph(val_loss_total, train_loss_total, "loss")
    graph(valid_acc_total, train_acc_total, "acc")
    
    # determine which model is best based on the validation loss
    if (valid_acc_total[len(valid_acc_total) - 1] < best_loss) :
        best_model = copy.deepcopy(nn_model)
   

# get test accuracy of the best model
test_acc = 0
best_model = best_model.eval()
with torch.no_grad():
    for i, test in enumerate(test_dataloader):
        images, labels = test
        images = images.to(device)
        labels = labels.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_acc += torch.sum(predicted == labels)

print("Accuracy of the network: " + str(test_acc.cpu().numpy()/len(test_dataset)))