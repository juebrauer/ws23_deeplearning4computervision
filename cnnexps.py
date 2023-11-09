import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn

print("Willkommen zum cnnexps Modul! V3.0")

def prepare_dataset(folder):
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()    
    ])
    
    # Trainings- und Testdatensatz vorbereiten
    train_dataset = datasets.ImageFolder(root=f"{folder}/train",
                                         transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True)
    
    test_dataset = datasets.ImageFolder(root=f"{folder}/val",
                                        transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=False)

    return train_loader, test_loader



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)       
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # Benutzen Sie einen Dummy-Eingabedatensatz, um die Größe automatisch zu ermitteln
        with torch.no_grad():
            self._initialize_fc_layers(torch.rand(1, 3, 224, 224))

    def _initialize_fc_layers(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        print("x.size=",x.size())
        n_size = x.size(1)
        print("n_size=",n_size)
        self.fc1   = nn.Linear(n_size, 512)
        self.relu4 = nn.ReLU()
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Übergang zum MLP
        x = self.flatten(x)

        # MLP        
        x = self.relu4(self.fc1(x))        
        x = self.fc2(x)

        # Output-Tensor zurückliefern
        return x




def train_model(model, device, train_loader, test_loader, num_epochs=2):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.train()
    accuracy = test_model(model, device, test_loader)
    testaccs = [accuracy]
   
    
    for epoch in range(1,num_epochs+1):
        import time
        start_time = time.time()
    
        print(f"Trainings-Epoche: {epoch}/{num_epochs}")

        batch_nr = 1
        for images, labels in train_loader:    
            images, labels = images.to(device), labels.to(device)
            
            if batch_nr % 25 == 0:
                print(f"\tBatch: {batch_nr}")
            outputs = model(images)
    
            loss = criterion(outputs, labels)
      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_nr += 1

        # Am Ende einer Epoche ...

        # ... Genauigkeit des Modells bestimmen
        test_accuracy = test_model(model, device, test_loader)
        testaccs.append( test_accuracy )

        # ... Modell und Genauigkeiten pro Epoche speichern
        save_model(epoch, model, testaccs)

        print(f"Epoch {epoch}/{num_epochs}, TestAcc: {test_accuracy*100:.4f}%")

        # Wie lange hat die Epoche gedauert?
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration:.2f} seconds")

    import matplotlib.pyplot as plt
    plt.plot(testaccs)
    plt.show()



def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # Deaktiviert Gradientenberechnung
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def save_model(epoch, model, testaccs):
    import pickle
    datei = open(f"models/model_{epoch:04}.pkl", "wb")
    pickle.dump(model, datei)
    pickle.dump(testaccs, datei)
    datei.close()







            