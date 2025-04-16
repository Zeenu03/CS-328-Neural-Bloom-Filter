import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import NeuralBloomFilter  

# For simplicity, we create a dummy task function that simulates a set-membership task.
# In each task, we create a small "storage set" S and a "query set" Q.
# Items in S have label 1; items not in S have label 0.
def sample_task(num_storage=10, num_queries=10, channels=1, height=28, width=28):
    # Create dummy storage images (e.g. random noise simulating MNIST images)
    storage_images = torch.randn(num_storage, channels, height, width)
    # Queries: first num_storage images (in S) and then num_queries images (not in S)
    queries_in = storage_images.clone()  # these should be recognized (label=1)
    queries_out = torch.randn(num_queries, channels, height, width)  # not stored (label=0)
    
    queries = torch.cat([queries_in, queries_out], dim=0)
    # Labels: ones for in-set, zeros for out-of-set
    labels = torch.cat([torch.ones(num_storage), torch.zeros(num_queries)], dim=0)
    return storage_images, queries, labels

# A simple classifier that maps the read vector (from NeuralBloomFilter) to a membership probability.
class MembershipClassifier(nn.Module):
    def __init__(self, word_size):
        super(MembershipClassifier, self).__init__()
        self.fc = nn.Linear(word_size, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Training loop for meta-learning tasks.
def train_model(model, classifier, optimizer, clf_optimizer, num_tasks, device):
    model.train()
    classifier.train()
    criterion = nn.BCELoss()
    total_loss = 0.0
    for task in range(num_tasks):
        # Reset the model's memory for each task
        model.memory.zero_()
        
        # Sample a task: get storage set S and query set Q with labels
        storage_images, queries, labels = sample_task(num_storage=10, num_queries=10)
        storage_images = storage_images.to(device)
        queries = queries.to(device)
        labels = labels.to(device).unsqueeze(1)  # shape (N, 1)

        # Write phase: write each storage image into the neural memory.
        # We assume batch size 1 per write for simplicity.
        for img in storage_images:
            img = img.unsqueeze(0)  # add batch dimension
            # Write mode: update memory with this image.
            model.forward(img, mode='write')
        
        # Read phase: process the query set to get the read vector.
        read_vector, _ = model.forward(queries, mode='read')  # read_vector: (num_queries_total, word_size)
        
        # Use the classifier to predict membership from the read vector.
        outputs = classifier(read_vector)
        
        # Compute binary cross-entropy loss.
        loss = criterion(outputs, labels)
        
        # Backpropagation and parameter update.
        optimizer.zero_grad()
        clf_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clf_optimizer.step()
        
        total_loss += loss.item()
        if (task + 1) % 10 == 0:
            print(f"Task {task + 1}/{num_tasks}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / num_tasks
    print(f"Average Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == '__main__':
    # Set the computation device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the Neural Bloom Filter model.
    # For example, here we use 10 memory slots and a word size of 32.
    model = NeuralBloomFilter(memory_slots=10, word_size=32).to(device)
    
    # Initialize the membership classifier.
    classifier = MembershipClassifier(word_size=32).to(device)
    
    # Define separate optimizers for the Neural Bloom Filter and the classifier.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    # Train the model on a number of meta-learning tasks (episodes).
    num_tasks = 100  # You can adjust the number of training episodes.
    train_model(model, classifier, optimizer, clf_optimizer, num_tasks, device)
