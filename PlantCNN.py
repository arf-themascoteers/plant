import torch
import torch.nn as nn


class PlantCNN(nn.Module):
    def __init__(self):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.flatter = nn.Flatten()
        self.fc1 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatter(x)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    ds = PlantDataset(is_train=True)
    dl = DataLoader(ds, batch_size=20, shuffle=True)

    for image, group in dl:
        print(image.shape)
        print(group.shape)

        image = image[0]
        tensor_image_display = image.squeeze().numpy()
        tensor_image_display = (tensor_image_display + 1) / 2.0

        plt.imshow(tensor_image_display, cmap='gray')
        plt.axis('off')
        plt.show()

        break