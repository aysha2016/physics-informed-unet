
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        return self.decoder(x2)

# Physics-informed loss function (example: scattering-based)
def physics_loss(pred, true):
    return torch.mean((pred - true)**2)

# Dataset for training
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Load your dataset
def load_dataset():
    # Example of random data for the sake of the template
    images = np.random.rand(100, 1, 128, 128)  # 100 samples, 1 channel, 128x128 size
    labels = np.random.rand(100, 1, 128, 128)
    return images, labels

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) + physics_loss(outputs, targets)  # Combine physics loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Main training loop
def main():
    images, labels = load_dataset()
    images_train, images_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(images_train, labels_train)
    val_dataset = CustomDataset(images_val, labels_val)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(10):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/10], Loss: {train_loss:.4f}')

        # Optionally save the model
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    # Visualize a sample result
    model.eval()
    with torch.no_grad():
        sample_input = torch.tensor(images[0:1], dtype=torch.float32).to(device)
        prediction = model(sample_input).cpu().numpy()
        plt.imshow(prediction[0, 0], cmap='gray')
        plt.show()

if __name__ == '__main__':
    main()
