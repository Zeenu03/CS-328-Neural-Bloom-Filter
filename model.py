import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import mmh3 
from tensorflow.keras.datasets import mnist

class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        # A simple CNN encoder: input (batch_size,channels,height,width) -> output vector (batch_size,128)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 7 * 7, 128)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x  # output shape: (batch_size, 128)

###############################################
# 2. Neural Bloom Filter Module
###############################################
class NeuralBloomFilter(nn.Module):
    def __init__(self, memory_slots=10, word_size=32, class_num=10):
        """
        memory_slots: Number of memory slots (columns in memory matrix)
        word_size: Dimension of the write word and query vector
        """
        super(NeuralBloomFilter, self).__init__()
        self.encoder = SimpleEncoder()
        
        # f_w: maps encoder output to a write word of dimension word_size
        self.fc_w = nn.Linear(128, word_size)
        
        # f_q: maps encoder output to a query vector of dimension word_size
        self.fc_q = nn.Linear(128, word_size)
        
        # Learnable addressing matrix A: shape (word_size, memory_slots)
        self.A = nn.Parameter(torch.randn(word_size, memory_slots), requires_grad=True)
        
        # Memory matrix M: shape (memory_slots, word_size), stored as a buffer (non-trainable)
        self.register_buffer('M', torch.zeros(memory_slots, word_size))

        # MLP for final output: maps read vector to class_num
        self.mlp = nn.Sequential(
            nn.Linear(word_size, 128), # (batch_size, word_size) -> (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, class_num) # (batch_size, 128) -> (batch_size, class_num)
        )
 
    def write(self, x):
        """
        Write operation: Encode input x, generate write word w and query vector q,
        compute addressing weights a via softmax(q @ A), then update memory: M = M + outer(w, a)
        """
        z = self.encoder(x)         # (batch_size, 128)
        w = self.fc_w(z)            # (batch_size, word_size)
        q = self.fc_q(z)            # (batch_size, word_size)

        a_logits = torch.matmul(q, self.A)  # (batch_size, memory_slots)
        a = F.softmax(a_logits, dim=1)                   # (batch_size, memory_slots)
        
        
        # a -> (batch_size, memory_slots) and w -> (batch_size, word_size)
        # a.unsqueeze(2) -> (batch_size, memory_slots, 1)
        # w.unsqueeze(1) -> (batch_size, 1, word_size)
        update = torch.matmul(a.unsqueeze(2), w.unsqueeze(1))  # (batch_size, memory_slots, word_size)
        
        self.M = self.M + update.sum(dim=0) # (memory_slots, word_size)
        
        return a 

    def read(self, x):
        """
        Read operation: Given input x, generate query vector q, compute addressing weights a, 
        then read from memory via weighted sum: read_vector = a @ M, resulting in a vector of dimension word_size.
        """
        z = self.encoder(x)         # (batch_size, 128)
        q = self.fc_q(z)            # (batch_size, word_size)
        a_logits = torch.matmul(q, self.A)  # (batch_size, memory_slots)
        a = F.softmax(a_logits, dim=1)                   # (batch_size, memory_slots)
        # Read: weighted sum over memory slots:
        read_vector = torch.matmul(a, self.M)       # (batch_size, word_size)
        
        logits = self.mlp(read_vector) # (batch_size, class_num)
        return logits

    def forward(self, x, mode='read'):
        if mode == 'write':
            return self.write(x)
        else:
            return self.read(x)
        
        
# Example usage:
model = NeuralBloomFilter(memory_slots=3, word_size=2, class_num=2)
x = torch.randn(1, 1, 28, 28)

model.write(x)

# Print the memory matrix M after writing
print("Memory matrix M after write:")
print(model.M.shape)

# Read from the memory
logits = model.read(x)
print("Logits after read:")
print(logits.shape)
print(logits)

# Convert logits to probabilities
probs = F.softmax(logits, dim=1)
print("Probabilities:")
print(probs.shape)
print(probs)

# Final class predictions
_, predicted_class = torch.max(probs, 1)
print("Predicted classes:")
print(predicted_class)

# ###############################################
# # 3. Backup Bloom Filter (classical implementation)
# ###############################################
# class BackupBloomFilter:
#     def __init__(self, capacity, error_rate):
#         """
#         capacity: Expected number of elements
#         error_rate: Desired false positive rate (e.g., 0.01)
#         """
#         self.capacity = capacity
#         self.error_rate = error_rate
#         self.num_bits = self.get_num_bits(capacity, error_rate)
#         self.num_hashes = self.get_num_hashes(self.num_bits, capacity)
#         # Using a simple Python list for bit array (0 or 1)
#         self.bit_array = [0] * self.num_bits

#     def add(self, item):
#         """
#         Adds an item to the Bloom filter by hashing it with num_hashes and setting bits.
#         """
#         for i in range(self.num_hashes):
#             hash_val = mmh3.hash(str(item), i) % self.num_bits
#             self.bit_array[hash_val] = 1

#     def __contains__(self, item):
#         """
#         Returns True if the item is probably in the filter (all corresponding bits are 1),
#         or False if the item is definitely not in the filter.
#         """
#         for i in range(self.num_hashes):
#             hash_val = mmh3.hash(str(item), i) % self.num_bits
#             if self.bit_array[hash_val] == 0:
#                 return False
#         return True

#     @staticmethod
#     def get_num_bits(capacity, error_rate):
#         """
#         Calculates the number of bits (m) needed using: m = - (n * ln(p)) / (ln(2)^2)
#         """
#         m = - (capacity * math.log(error_rate)) / (math.log(2) ** 2)
#         return int(m)

#     @staticmethod
#     def get_num_hashes(num_bits, capacity):
#         """
#         Calculates the number of hash functions (k) using: k = (m / n) * ln(2)
#         """
#         k = (num_bits / capacity) * math.log(2)
#         return int(k)

# if __name__ == '__main__':
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     print(x_train.shape, y_train.shape)
    