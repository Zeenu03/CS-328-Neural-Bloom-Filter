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
from model import NeuralBloomFilter, SimpleEncoder
latexify(columns=2)


def sample_task(labels, storage_set_size, num_queries, class_num):
    class_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    storage_indices = random.sample(class_to_indices[class_num], storage_set_size)

    num_in = num_queries // 2
    num_out = num_queries - num_in
    query_in_indices = random.sample(class_to_indices[class_num], num_in)
    other_classes = [c for c in class_to_indices if c != class_num]
    query_out_indices = []
    for _ in range(num_out):
        other_class = random.choice(other_classes)
        query_out_indices.append(random.choice(class_to_indices[other_class]))
    
    query_indices = query_in_indices + query_out_indices
    # define target as [1,0] for in-class and [0,1] for out-of-class
    targets = []
    for i in range(num_in):
        targets.append(1)
    for i in range(num_out):
        targets.append(0)
    
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) 
    return storage_indices, query_indices, targets

def meta_train(model, dataset, labels, optimizer, criterion, device, meta_epochs=10, storage_set_size=60, num_queries=10, classes=[0,8,9,6]):  
    model.train()
    total_loss = 0.0
    for epoch in range(meta_epochs):
        class_num = random.choice(classes)
        storage_indices, query_indices, targets= sample_task(labels, storage_set_size, num_queries, class_num)
        storage_images = dataset[storage_indices]
        storage_images = torch.tensor(storage_images, dtype=torch.float32).unsqueeze(1)
        query_images = dataset[query_indices]
        query_images = torch.tensor(query_images, dtype=torch.float32).unsqueeze(1)
        
        model.M.zero_()
        _ = model.write(storage_images)  # Expected shape: (storage_set_size, word_size)
        logits,_ = model.read(query_images)  # Expected shape: (num_queries, class_num)
        probs = torch.sigmoid(logits) 
        # Convert probabilities to binary predictions
        predictions = (probs > 0.5).float()
        fnr = 0
        fpr = 0
        for i in range(len(predictions)):
            if predictions[i] == 0 and targets[i] == 1:
                fnr += 1
            elif predictions[i] == 1 and targets[i] == 0:
                fpr += 1
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / (epoch + 1)
            print(f"Epoch [{epoch+1}/{meta_epochs}], Loss: {loss.item():.4f}, False Positive Rate: {fpr}, False Negative Rate: {fnr}")
    return total_loss / meta_epochs


if __name__ == '__main__':
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # Hyperparameters
    meta_epochs = 2000
    storage_set_size = 300
    num_queries = 300
    learning_rate = 1e-6
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NeuralBloomFilter(memory_slots=1024, word_size=64, class_num=1).to(device)
    
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Start meta-training
    avg_loss = meta_train(model, x_test, y_test,optimizer, criterion, device,
                          meta_epochs=meta_epochs,
                          storage_set_size=storage_set_size,
                          num_queries=num_queries)
    print(f"Meta-training completed. Average Loss: {avg_loss:.4f}")  