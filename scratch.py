import torch
import torch.nn as nn
import torch.optim as optim

class GroupEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(GroupEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim + 2, 64)  # 2 for speed and SNR
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, group_indices, speed, snr):
        # Embed the group sequence
        embedded_group = self.embedding(group_indices)
        # Process the sequence with RNN
        _, (hidden, _) = self.rnn(embedded_group)
        # Concatenate the final hidden state with speed and SNR
        x = torch.cat((hidden[-1], speed, snr), dim=1)
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Define parameters
vocab_size = 37  # 26 letters + 10 digits + 1 for '/'
embedding_dim = 16
hidden_dim = 32

# Initialize the model, loss function, and optimizer
model = GroupEmbeddingModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
for epoch in range(100):  # Number of epochs
    for group_indices, speed, snr, label in training_data:
        optimizer.zero_grad()
        # Normalize speed and SNR
        speed = (speed - 10) / (50 - 10)
        snr = snr / 30
        speed = torch.tensor([speed])
        snr = torch.tensor([snr])
        # Forward pass
        output = model(group_indices, speed, snr)
        # Compute loss
        loss = criterion(output, label)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
