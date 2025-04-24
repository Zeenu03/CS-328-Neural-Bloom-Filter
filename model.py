import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import random
import mmh3
from bitarray import bitarray
import torch.optim as optim
import seaborn as sns
from latex import latexify, format_axes
latexify(columns=2)

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
    
class NeuralBloomFilter(nn.Module):
    def __init__(self, memory_slots=10, word_size=32, class_num=1):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.fc_q    = nn.Linear(128, word_size)
        self.fc_w    = nn.Linear(128, word_size)
        self.A       = nn.Parameter(torch.randn(word_size, memory_slots))
        self.register_buffer('M', torch.zeros(memory_slots, word_size))
        inp_dim = memory_slots*word_size + word_size + 128
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, class_num)
        )

    def controller(self, x):
        """
        x -> (B, 1, 28, 28): input image
        Runs the encoder, computes query q, write word w, and normalized address a.
        Returns (a, w, z).
        """
        z = self.encoder(x)                   # (B,128)
        q = self.fc_q(z)                      # (B,word_size)
        w = self.fc_w(z)                      # (B,word_size)
        a_logits = q @ self.A                 # (B,slots)
        a = F.softmax(a_logits, dim=1)        # (B,slots)
        return a, w, z

    def write(self, x):
        """
        Write all x in one shot:
          M ← M + ∑_i w_i a_i^T
        """
        a, w, _ = self.controller(x)          # a:(B,slots), w:(B,word_size)
        # Outer product per sample: update_i[k,p] = a[i,k] * w[i,p]
        # Stack them and sum over batch:
        # shape: (B, slots, word_size)
        update = torch.einsum('bk,bp->bkp', a, w)
        # Add to memory and detach so writes don't backprop through time:
        self.M = self.M + update.sum(dim=0).detach()

    def read(self, x):
        """
        Read operation:
          r_i = flatten( M ⊙ a_i )  (componentwise)
          logits = f_out([r_i, w_i, z_i])
        """
        a, w, z = self.controller(x)          # (B,slots), (B,word_size), (B,128)
        # M ⊙ a_i: scale each row of M by a_i:
        # shape before flatten: (B, slots, word_size)
        r = (a.unsqueeze(2) * self.M.unsqueeze(0)).reshape(x.size(0), -1)  # (B, slots*word_size)
        # Concatenate r, w, z:
        concat = torch.cat([r, w, z], dim=1)                              # (B, inp_dim)
        logits = self.mlp(concat)                                          # (B, class_num)
        return logits,a