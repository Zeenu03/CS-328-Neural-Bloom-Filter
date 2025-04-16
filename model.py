import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import mmh3  # Install via pip: pip install mmh3

###############################################
# 1. Encoder for MNIST-like images
###############################################
class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        # A simple CNN encoder: input (1, 28, 28) -> output vector of size 128
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
    def __init__(self, memory_slots=10, word_size=32):
        """
        memory_slots: Number of memory slots (columns in memory matrix)
        word_size: Dimension of the write word and query vector
        """
        super(NeuralBloomFilter, self).__init__()
        # Encoder: converts input image to a 128-dim vector
        self.encoder = SimpleEncoder()
        # f_w: maps encoder output to a write word of dimension word_size
        self.fc_w = nn.Linear(128, word_size)
        # f_q: maps encoder output to a query vector of dimension word_size
        self.fc_q = nn.Linear(128, word_size)
        # Learnable addressing matrix A: shape (word_size, memory_slots)
        self.address_matrix = nn.Parameter(torch.randn(word_size, memory_slots))
        # Memory matrix M: shape (memory_slots, word_size), stored as a buffer (non-trainable)
        self.register_buffer('memory', torch.zeros(memory_slots, word_size))
    
    def write(self, x):
        """
        Write operation: Encode input x, generate write word w and query vector q,
        compute addressing weights a via softmax(q @ A), then update memory: M = M + outer(w, a)
        """
        z = self.encoder(x)         # (batch_size, 128)
        w = self.fc_w(z)            # (batch_size, word_size)
        q = self.fc_q(z)            # (batch_size, word_size)
        # Compute addressing: for each sample, a = softmax(q @ A)
        a_logits = torch.matmul(q, self.address_matrix)  # (batch_size, memory_slots)
        a = F.softmax(a_logits, dim=1)                   # (batch_size, memory_slots)
        
        # For simplicity, assume batch_size = 1; otherwise, you may average the update or process each sample separately.
        # Compute outer product for the first (or each) sample:
        # Outer product: (word_size x 1) and (1 x memory_slots) gives (word_size, memory_slots).
        # We then transpose to match memory shape (memory_slots, word_size).
        update = torch.matmul(a.unsqueeze(2), w.unsqueeze(1))  # (batch_size, memory_slots, word_size)
        # Update memory: add the update from each sample in the batch (here, summing over batch dimension)
        self.memory = self.memory + update.sum(dim=0)
        return a  # optionally return the addressing weights

    def read(self, x):
        """
        Read operation: Given input x, generate query vector q, compute addressing weights a, 
        then read from memory via weighted sum: read_vector = a @ M, resulting in a vector of dimension word_size.
        """
        z = self.encoder(x)         # (batch_size, 128)
        q = self.fc_q(z)            # (batch_size, word_size)
        a_logits = torch.matmul(q, self.address_matrix)  # (batch_size, memory_slots)
        a = F.softmax(a_logits, dim=1)                   # (batch_size, memory_slots)
        # Read: weighted sum over memory slots:
        read_vector = torch.matmul(a, self.memory)       # (batch_size, word_size)
        return read_vector, a

    def forward(self, x, mode='read'):
        if mode == 'write':
            return self.write(x)
        else:
            return self.read(x)

###############################################
# 3. Backup Bloom Filter (classical implementation)
###############################################
class BackupBloomFilter:
    def __init__(self, capacity, error_rate):
        """
        capacity: Expected number of elements
        error_rate: Desired false positive rate (e.g., 0.01)
        """
        self.capacity = capacity
        self.error_rate = error_rate
        self.num_bits = self.get_num_bits(capacity, error_rate)
        self.num_hashes = self.get_num_hashes(self.num_bits, capacity)
        # Using a simple Python list for bit array (0 or 1)
        self.bit_array = [0] * self.num_bits

    def add(self, item):
        """
        Adds an item to the Bloom filter by hashing it with num_hashes and setting bits.
        """
        for i in range(self.num_hashes):
            hash_val = mmh3.hash(str(item), i) % self.num_bits
            self.bit_array[hash_val] = 1

    def __contains__(self, item):
        """
        Returns True if the item is probably in the filter (all corresponding bits are 1),
        or False if the item is definitely not in the filter.
        """
        for i in range(self.num_hashes):
            hash_val = mmh3.hash(str(item), i) % self.num_bits
            if self.bit_array[hash_val] == 0:
                return False
        return True

    @staticmethod
    def get_num_bits(capacity, error_rate):
        """
        Calculates the number of bits (m) needed using: m = - (n * ln(p)) / (ln(2)^2)
        """
        m = - (capacity * math.log(error_rate)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def get_num_hashes(num_bits, capacity):
        """
        Calculates the number of hash functions (k) using: k = (m / n) * ln(2)
        """
        k = (num_bits / capacity) * math.log(2)
        return int(k)

