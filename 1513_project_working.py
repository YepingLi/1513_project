import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("ml_train.csv")
test_data = pd.read_csv("ml_test.csv")
val_data = pd.read_csv("ml_val.csv")
train_data_stat = train_data.drop(columns=['target']).values
train_data_label = train_data['target'].values
test_data_stat = test_data.drop(columns=['target']).values
test_data_label = test_data['target'].values
val_data_stat = val_data.drop(columns=['target']).values
val_data_label = val_data['target'].values
#removing mean and scaling to unit varience
scaler = StandardScaler()
train_data_stat = scaler.fit_transform(train_data_stat)
test_data_stat = scaler.fit_transform(test_data_stat)
val_data_stat = scaler.fit_transform(val_data_stat)

train_data_stat_tensor = torch.tensor(train_data_stat, dtype=torch.float32)
train_data_label_tensor = torch.tensor(train_data_label, dtype=torch.long)
test_data_stat_tensor = torch.tensor(test_data_stat, dtype=torch.float32)
test_data_label_tensor = torch.tensor(test_data_label, dtype=torch.long)
val_data_stat_tensor = torch.tensor(val_data_stat, dtype=torch.float32)
val_data_label_tensor = torch.tensor(val_data_label, dtype=torch.long)

train_dataset = TensorDataset(train_data_stat_tensor, train_data_label_tensor)
test_dataset = TensorDataset(test_data_stat_tensor, test_data_label_tensor)
val_dataset = TensorDataset(val_data_stat_tensor, val_data_label_tensor)


val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


#The model similiar to assignment 4
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #3 channels
        self.fc1 = torch.nn.Linear(41, 150)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(150, 46)
        self.fc3 = torch.nn.Linear(46, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x
    
def train_model(model, optimizer, train_loader, val_loader, test_loader):
    #This makes the trainloder balance in terms of sample
    weight_loss = torch.tensor([1.0, 6758/1093], dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_loss)
    train_accuracy = []
    training_losses = []
    validation_losses = []
    validation_accuracy = []
    for epoch in range(35): 
        running_loss = 0.0
        correct_predicts = 0
        total_predicts = 0
        running_val_loss = 0.0
        running_val_acc = 0.0
        model.train()
        for _, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predicts += labels.size(0)
            correct_predicts += (predicted == labels).sum().item()
        running_loss = running_loss / len(train_loader)
        training_losses.append(running_loss)
        train_accuracy.append(100 * correct_predicts / total_predicts)
        print('[%d] training loss: %.3f | accuracy: %.2f%%' % (epoch + 1, running_loss, 100 * correct_predicts / total_predicts))
        model.eval()
        correct_predicts = 0
        total_predicts = 0
        #Compute the valudation accuracy and loss
        with torch.no_grad():
            criterion_val = torch.nn.CrossEntropyLoss()
            for data in val_loader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion_val(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predicts += labels.size(0)
                correct_predicts += (predicted == labels).sum().item()
        running_val_acc = 100* correct_predicts / total_predicts
        running_val_loss = running_val_loss / len(val_loader)
        validation_losses.append(running_val_loss)
        validation_accuracy.append(running_val_acc)
    print("Training done")
    return model, train_accuracy, training_losses, validation_losses, validation_accuracy

def test(model, test_loader):
    correct_predicts = 0
    total_predicts = 0
    running_test_loss = 0.0
    with torch.no_grad():
        criterion_val = torch.nn.CrossEntropyLoss()
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion_val(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predicts += labels.size(0)
            correct_predicts += (predicted == labels).sum().item()
    acc = 100* correct_predicts / total_predicts
    loss = running_test_loss / len(val_loader)
    print("Test result display")
    print('test loss: %.3f | test accuracy: %.2f%%' % (loss, acc))
           
        

model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model, train_accuracy, training_losses, validation_losses, validation_accuracy = train_model(model, optimizer, train_loader, val_loader, test_loader)
#print(train_data['target'].value_counts())
test(model, test_loader)
plt.figure()
plt.title('losses vs number of epoch')
x = range(1,len(training_losses)+1)
plt.plot(x, training_losses, label='the training loss')
plt.plot(x, validation_losses, label='the validation loss')
plt.xticks(range(1,len(training_losses)+1))
plt.xlabel('number of epoch')
plt.ylabel('losses')
plt.legend()


plt.figure()
plt.title('accuracy vs number of epoch')
x = range(1,len(train_accuracy)+1)
plt.plot(x, train_accuracy, label='the training accuracy')
plt.plot(x, validation_accuracy, label='the validation accuracy')
plt.xticks(range(1,len(training_losses)+1))
plt.xlabel('number of epoch')
plt.ylabel('accuracies')
plt.legend()
plt.show()