import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Defining the model architecture
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def save_checkpoint(model, optimizer, epoch, path):
    # save model checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")


def load_checkpoint(path, model, optimizer):
    # load checkpoint
    if os.path.isfile(path):
        print("Loading checkpoint")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoint found at {path}")
        return 1


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # Training Loop

    model.train()
    for batch_id, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_id % args.log_interval == 0:
            print(
                " Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".format(
                    epoch,
                    batch_id * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_id / len(data_loader),
                    loss.item(),
                )
            )

            if args.dry_run:
                break


def test_epoch(model, device, data_loader):
    # Testing Loop

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(data_loader.dataset),
            100.0 * correct / len(data_loader.dataset),
        )
    )


def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description="MNIST Training Script")

    # Defining command line arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume Training from checkpoint",
    )

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the MNIST dataset for training and testing
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    training_data = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = DataLoader(training_data, **train_kwargs)
    test_loader = DataLoader(test_data, **test_kwargs)

    model = Net().to(device)
    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    model_checkpoint_path = "model_checkpoint.pth"
    # Defining a way to load the model checkpoint if 'resume' argument is True
    if args.resume:
        start_epoch = load_checkpoint(model_checkpoint_path, model, optimizer)

    # Training and testing cycles
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)
        test_epoch(model, device, test_loader)
        scheduler.step()

        # Save the model after each epoch
        if args.save_model:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=model_checkpoint_path,
            )


if __name__ == "__main__":
    main()
