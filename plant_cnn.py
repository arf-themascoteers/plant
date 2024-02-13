import torch
import torch.nn as nn


class PlantCNN(nn.Module):
    def __init__(self):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=16, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=16, stride=1)
        self.flatter = nn.Flatten()
        self.fc1 = nn.Linear(224, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatter(x)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    random_tensor = torch.rand((10,1,125,75), dtype=torch.float32)
    print(random_tensor.shape)
    model = PlantCNN()
    out = model(random_tensor)
    print(out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of learnable parameters: {total_params}")