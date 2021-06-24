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
import ic_dataset

TEST_MODEL = True
EPOCHS = 50
PATIENCE = 3

# xavier initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

# utility function for plotting
def graph(val_loss, train_loss, label, loopNum):
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
    #plt.show()
    plt.savefig('loop' + str(loopNum) + '_' + label)

def train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, optimizer, criterion, patience, test_model, loopNum) :
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
            outputs, aux_outputs = model(image)
            loss1 = criterion(outputs, label)
            loss2 = criterion(aux_outputs, label)
            loss = loss1 + 0.4*loss2
            # get train accuracy
            _, predicted = torch.max(outputs, 1)
            train_acc += torch.sum(predicted == label)
            # backprop
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch % 10 == 0) :
            print("epoch" + epoch + " train complete")

        # validate 
        model = model.eval()
        with torch.no_grad():
            for i, sample in enumerate(val_dataloader) :
                batch_valid += 1
                image, label = sample
                image = image.cuda()
                label = label.cuda()
                # validate accuracy, loss
    
                outputs = model(image)
                loss = criterion(outputs, label)
                #output_valid = model(image)
                
                _, predicted = torch.max(outputs, 1)
                valid_acc += torch.sum(predicted == label)
                val_loss += loss.item()
        model = model.train()

        if (epoch % 10 == 0) :
            print("epoch" + epoch + " train complete")

        train_loss_total.append(train_loss / batch)
        val_loss_total.append(val_loss / batch_valid)

        train_acc_total.append(train_acc.cpu().numpy() / len(train_dataloader.dataset))
        valid_acc_total.append(valid_acc.cpu().numpy() / len(val_dataloader.dataset))

        if (epoch != 0) and (val_loss_total[epoch] >= val_loss_total[epoch - 1]):
            loss_increase += 1
            if loss_increase >= patience :
                #torch.save(model.state_dict(), './')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, "oracle1.pt")

                print ("early stopping")
                break
                
        else :
            loss_increase = 0

    # graph train/validation loss and accuracy
    graph(val_loss_total, train_loss_total, "loss", loopNum)
    graph(valid_acc_total, train_acc_total, "acc", loopNum)
    
    if (test_model) :
        model = model.eval()
        batch_test = 0
        test_acc = 0
        test_loss = 0
        with torch.no_grad():
            for i, sample in enumerate(test_dataloader) :
                batch_test += 1
                image, label = sample
                image = image.cuda()
                label = label.cuda()
                # validate accuracy, loss
    
                outputs = model(image)
                loss = criterion(outputs, label)
                #output_valid = model(image)
                
                _, predicted = torch.max(outputs, 1)
                test_acc += torch.sum(predicted == label)
                test_loss += loss.item()
        model = model.train()
        f = open("loopTable.txt", "a")
        f.write("Test accuracy: " + str(test_acc.cpu().numpy() / len(test_dataloader.dataset)))
        f.write("Test loss: " + str(test_loss / batch_test))
        f.close()

    # save model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, "oracle1.pt")

'''
if __name__ == '__main__': 
    torch.cuda.empty_cache()
    model = models.inception_v3(pretrained=True, init_weights=True, aux_logits=True)
    model.fc = nn.Linear(2048, 120)
    model.AuxLogits.fc = nn.Linear(768, 120)
   # model.fc = nn.Linear(512,120)
   # model.fc.apply(weights_init)

   # model.classifier[6] = nn.Linear(4096,120)
   # model.classifier[6].apply(weights_init)

    
    # load the datasets
    
    #train_dataset, valid_dataset = ic_dataset.get_icdataset_train_test('/data/dataset_stanford_dog_recreation', train_perc=.8)
    #train_dataset.save_label_map()
    #valid_dataset = dogData.Dog_Test_Dataset('val_list.txt')
    #test_dataset = datasets.ImageFolder()
    

    train_transform = transforms.Compose([ 
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([ 
        transforms.Resize(299),
        transforms.CenterCrop(299), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    #train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [14406, 3087, 3087])

    # dataloaders for the datasets
    
    train_dataset= datasets.ImageFolder('/data/classifier/TrainImages/', transform=train_transform)
    valid_dataset= datasets.ImageFolder('/data/classifier/ValImages/', transform=test_transform)
    test_dataset= datasets.ImageFolder('/data/classifier/TestImages/', transform=test_transform)

    dataloaders = defaultdict(DataLoader)
    dataloaders['train'] =  DataLoader(train_dataset, shuffle=True, batch_size = 64, num_workers=4)
    dataloaders['validate'] =  DataLoader(valid_dataset, shuffle=False, num_workers=4)
    if (TEST_MODEL):
        dataloaders['test'] =  DataLoader(test_dataset, shuffle=False, num_workers=4)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = 64, num_workers=4)
    val_loader =  DataLoader(valid_dataset, shuffle=False, num_workers=4)
    test_loader = DataLoader(valid_dataset, shuffle=False, num_workers=4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    train_model(model, train_loader, val_loader,test_loader, EPOCHS, optimizer, criterion, PATIENCE, TEST_MODEL)
'''