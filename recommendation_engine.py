import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Load and preprocess the data
ratings_df = pd.read_csv('ratings_df_svd.csv')
ratings = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
ratings_tensor = torch.tensor(ratings.values)

# Step 2: Define the model architecture
class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=100):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, movie_indices):
        # Clip indices to ensure they are within valid range
        user_indices = torch.clamp(user_indices, 0, self.user_embedding.num_embeddings - 1)
        movie_indices = torch.clamp(movie_indices, 0, self.movie_embedding.num_embeddings - 1)

        user_embedded = self.user_embedding(user_indices)
        movie_embedded = self.movie_embedding(movie_indices)
        combined = torch.cat([user_embedded, movie_embedded], dim=1)
        output = self.fc(combined)
        output = self.sigmoid(output)
        return output

# Step 3: Prepare the data for training and testing
class MovieDataset(Dataset):
    def __init__(self, ratings_tensor):
        self.ratings_tensor = ratings_tensor

    def __len__(self):
        return len(self.ratings_tensor)

    def __getitem__(self, idx):
        return torch.LongTensor([idx // len(self.ratings_tensor), idx % len(self.ratings_tensor)]), self.ratings_tensor[idx // len(self.ratings_tensor), idx % len(self.ratings_tensor)]

dataset = MovieDataset(ratings_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 4: Train the model
num_users, num_movies = ratings_tensor.shape
model = RecommenderModel(num_users, num_movies)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        user_indices, movie_indices = batch_inputs[:, 0], batch_inputs[:, 1]
        ratings_pred = model(user_indices, movie_indices).squeeze()
        # Define threshold for binary classification
        threshold = 0.5  # Adjust threshold for binary classification

        # Convert ratings to binary values using the threshold
        binary_targets = (batch_targets > threshold).float()

        # Calculate loss using binary_targets instead of batch_targets
        loss = criterion(ratings_pred, binary_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Step 5: Make predictions and evaluate the model
def predict_rating(user_id, movie_id):
    user_idx = torch.LongTensor([user_id])
    movie_idx = torch.LongTensor([movie_id])
    # Clip indices to ensure they are within valid range
    user_idx = torch.clamp(user_idx, 0, model.user_embedding.num_embeddings - 1)
    movie_idx = torch.clamp(movie_idx, 0, model.movie_embedding.num_embeddings - 1)

    rating_pred = model(user_idx, movie_idx).item()
    return rating_pred

# Example usage
user_id = 87980
movie_id = 123552
predicted_rating = predict_rating(user_id, movie_id)
print(f'Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating}')
