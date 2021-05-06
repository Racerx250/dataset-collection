import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import dogData
from matplotlib import pyplot as plt
import torchvision.models as models
from collections import defaultdict
from torchvision import transforms as transforms
from torchvision import datasets as datasets

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

        print("epoch train complete")

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

        print("epoch validate complete")

        print("save check point")
        save_dir = "check_point/"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, filename=os.path.join(save_dir, 'checkpoint_{}.tar'.format(epoch)))

        train_loss_total.append(train_loss / batch)
        val_loss_total.append(val_loss / batch_valid)

        train_acc_total.append(train_acc.cpu().numpy() / len(train_dataloader.dataset))
        valid_acc_total.append(valid_acc.cpu().numpy() / len(val_dataloader.dataset))

        if (epoch != 0) and (val_loss_total[epoch] >= val_loss_total[epoch - 1]):
            loss_increase += 1
            if loss_increase >= patience :
                torch.save(model, './saved_model')
                print ("early stopping")
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

if __name__ == '__main__': 
    model = models.vgg16(pretrained=True)
   # model.fc = nn.Linear(512,120)
   # model.fc.apply(weights_init)

    model.classifier[6] = nn.Linear(4096,120)
    model.classifier[6].apply(weights_init)

    
    # load the datasets
    
    train_dataset = dogData.Dog_Train_Dataset('train_list.txt')
    valid_dataset = dogData.Dog_Test_Dataset('val_list.txt')
    test_dataset = dogData.Dog_Test_Dataset('test_list.txt')
    
    train_transform = transforms.Compose([ 
        transforms.Resize(256), 
        transforms.RandomCrop(224), 
        transforms.ToTensor()])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([ 
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #train_dataset= datasets.ImageFolder('TrainImages/', transform=train_transform)
    #valid_dataset= datasets.ImageFolder('ValImages/', transform=test_transform)
    #test_dataset= datasets.ImageFolder('TestImages/', transform=test_transform)

    #train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [14406, 3087, 3087])

    # dataloaders for the datasets
    dataloaders = defaultdict(DataLoader)
    dataloaders['train'] =  DataLoader(train_dataset, shuffle=True, batch_size = 64, num_workers=4)
    dataloaders['validate'] =  DataLoader(valid_dataset, shuffle=False, num_workers=4)
    dataloaders['test'] =  DataLoader(test_dataset, shuffle=False, num_workers=4)

    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    train_model(model, dataloaders, epochs, optimizer, criterion, 2)
