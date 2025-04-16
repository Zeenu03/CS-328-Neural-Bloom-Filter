import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import NeuralBloomFilter  # This imports your module from model.py

# Simple classifier that maps the read vector to 10 classes
class NFClassifier(nn.Module):
    def __init__(self, word_size=32, num_classes=10):
        super(NFClassifier, self).__init__()
        self.fc = nn.Linear(word_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Hyperparameters
batch_size = 64
num_epochs = 5
learning_rate = 0.001

# MNIST transforms and dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Set computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Neural Bloom Filter module and the classifier
nbf = NeuralBloomFilter(memory_slots=10, word_size=32).to(device)
classifier = NFClassifier(word_size=32, num_classes=10).to(device)

# Combine parameters of both modules for the optimizer
optimizer = optim.Adam(list(nbf.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Loss function: CrossEntropyLoss expects logits and target class indices
criterion = nn.CrossEntropyLoss()

# Training loop
nbf.train()
classifier.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients and clear the memory buffer at the start of each batch
        optimizer.zero_grad()
        nbf.memory.zero_()  # Reset memory for each batch

        # Write phase: update memory with all images in the batch.
        # Here, we simply perform a write pass on the entire batch.
        _ = nbf.forward(images, mode='write')
        
        # Read phase: get the feature vector from the memory using the same images.
        read_vector, _ = nbf.forward(images, mode='read')  # shape: (batch_size, word_size)
        
        # Classifier: predict digit classes from the read feature vector.
        logits = classifier(read_vector)  # shape: (batch_size, 10)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

# Evaluation on test set
nbf.eval()
classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        # Reset memory for the test batch
        nbf.memory.zero_()
        _ = nbf.forward(images, mode='write')
        features, _ = nbf.forward(images, mode='read')
        outputs = classifier(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f"Test Accuracy: {100 * correct / total:.2f}%")
