import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(torch.nn.Linear(28 * 28, 512),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(512, 512),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(512, 10))
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def data_loader(path) -> [DataLoader, DataLoader]:
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root=path,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=path,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    training_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    return training_dataloader, test_dataloader

def train_one_epoch(nnmodel, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    nnmodel.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = nnmodel(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train(epochs, nnmodel, dataloader):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(nnmodel.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_one_epoch(nnmodel, dataloader, loss_fn, optimizer)



def save(nnmodel, path):
    torch.save(nnmodel.state_dict(), path)

def load(path) -> NeuralNetwork:
    model = NeuralNetwork().to("cpu")
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def test(nnmodel, dataloader):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    nnmodel.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = nnmodel(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print((f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"))

def main():
    train_data, test_data = data_loader("..\\data")
    nn_model = NeuralNetwork().to("cpu")
    train(7, nn_model, train_data)
    save(nn_model, "..\\models\\nn_model.pth")
    test_model = load("..\\models\\nn_model.pth")
    test(test_model, test_data)

    print("Done!")


if __name__ == "__main__":

    main()
    # X = torch.rand(3, 28, 28)
    # logits = model(X)
    # print(logits)

    # softmax = torch.nn.Softmax(dim=1)
    # probs = softmax(logits)
    # print(probs)
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # import matplotlib
    # matplotlib.use("QT5Agg")
    # import matplotlib.pyplot as plt
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
