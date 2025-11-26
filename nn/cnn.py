import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def data_loader(path) -> [torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    batch_size = 4
    # Download training data from open datasets.
    training_data = torchvision.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transform,
    )

    # Download test data from open datasets.
    test_data = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=transform,
    )

    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    return training_dataloader, test_dataloader

def show_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define layers.
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16,5 )
        self.fc1 = torch.nn.Linear(16 * 5* 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_cnn(model, train_data, fn_loss, optimizer,epochs=2):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = fn_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[Epoch {epoch + 1}, Mini-batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

def test_cnn(model, test_data):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data, 0):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def main():
    train_data, test_data = data_loader("..\\data")
    cnn = CNN()
    fn_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    train_cnn(cnn, train_data, fn_loss, optimizer, epochs=1)
    test_cnn(cnn, test_data)

    # imgiter = iter(train_data)
    # images, labels = next(imgiter)
    # matplotlib.use("QT5Agg")
    # show_image(torchvision.utils.make_grid(images))
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


if __name__ == "__main__":

    main()