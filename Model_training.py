import loadMedicalMNIST as load_data
import Model as neural_network
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path
from torchsummary import summary

root_dir = 'Documents/datasets/MedicalMNIST'
df = load_data.get_labels_df(root_dir)
dataset = load_data.MedicalMNIST(df, root_dir, load_data.data_transform())

print(len(dataset))

train_set, test_set = torch.utils.data.random_split(dataset,
                                                   [48954,10000])
train_loader = DataLoader(train_set, batch_size=(batch_size), shuffle=True)
test_loader = DataLoader(test_set, batch_size=(batch_size), shuffle=True)

#Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Set the hyperparameters
n_epochs = 50
lr = 0.001
in_channels = 3
output_classes = 6
batch_size = 64

model = neural_network.CNN_MNIST(in_channels, output_classes, device)
print(summary(model,(in_channels))

#Loss and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

#Train the network
def train(model, n_epochs, train_loader):
    for epoch in range(num_epochs):
        for batch, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            #Forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            #Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient descent
            optimizer.step()
            
        print(epoch, "Current Loss:", loss)
            
train(model, num_epochs)

def evaluate(loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, pred = scores.max(1)
            correct += (pred == y).sum()
            total += pred.size(0)
        print("Accuracy:", correct/total*100, "%")
    
evaluate(train_loader, model)
evaluate(test_loader, model)


