import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import os

print("Willkommen zum cnnexps Modul! V2023-11-19")

imagenette2_class_names = ["tench",
                           "English springer",
                           "cassette player",
                           "chain saw",
                           "church",
                           "French horn",
                           "garbage truck",
                           "gas pump",
                           "golf ball",
                           "parachute"]

def prepare_dataset(folder):

    transform = get_image_transforms()
    
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

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # Benutzen Sie einen Dummy-Eingabedatensatz, um die Größe automatisch zu ermitteln
        with torch.no_grad():
            self._initialize_fc_layers(torch.rand(1, 3, 224, 224))

    
    def feature_hierarchy(self, x):

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.pool6(self.relu6(self.conv6(x)))
        return x
        

    def _initialize_fc_layers(self, x):

        x = self.feature_hierarchy(x)        
        
        x = self.flatten(x)
        print("x.size=",x.size())
        n_size = x.size(1)
        print("n_size=",n_size)
        self.fc1   = nn.Linear(n_size, 512)
        self.relufc1 = nn.ReLU()
        self.fc2   = nn.Linear(512, 10)
        

    def forward(self, x):
        # Bild --> Merkmalstensor
        x = self.feature_hierarchy(x) 

        # Merkmalstensor flach machen
        x = self.flatten(x)

        # Merkmalstensor mit MLP klassifizieren
        x = self.relufc1(self.fc1(x))        
        x = self.fc2(x)

        # Output-Tensor zurückliefern
        return x




def train_model(model_dir,
                model,
                device,
                train_loader,
                test_loader,
                num_epochs=2):

    from torchsummary import summary
    summary(model, (3, 224, 224), device=str(device))
    
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
        save_model(model_dir, epoch, model, testaccs)

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


def save_model(model_dir, epoch, model, testaccs):
    

    if not os.path.exists(model_dir):        
        os.makedirs(model_dir)
        
    import pickle
    datei = open(f"{model_dir}/model_{epoch:04}.pkl", "wb")
    pickle.dump(model, datei)
    pickle.dump(testaccs, datei)
    datei.close()


def get_image_transforms():

    transform = transforms.Compose([        
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()    
    ])

    return transform


def classify_image(img, model, device):

    # OpenCV image --> PIL image
    from PIL import Image
    img = Image.fromarray(img)

    transform = get_image_transforms()

    img_tensor = transform(img).unsqueeze(0)

    img_tensor = img_tensor.to(device)

    model.eval()

    outputs = model(img_tensor)

    _, idx = torch.max(outputs, 1)
        
    return outputs, imagenette2_class_names[idx.item()]


def visualize_predictions(outputs):
    import matplotlib.pyplot as plt
    plt.bar(imagenette2_class_names, outputs.detach().cpu().flatten())
    plt.xticks(rotation=90)
    plt.show()



            