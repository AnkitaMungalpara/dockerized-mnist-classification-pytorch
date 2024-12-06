**Introduction to Docker and Containerization**  

Containers are portable, lightweight, and efficient tools for application deployment. Unlike virtual machines, they allow multiple isolated environments to run on a single host operating system (OS), often supporting hundreds or thousands of containers simultaneously. By decoupling software from its runtime environment, containers enable developers to build applications on one OS, such as Linux, and deploy them seamlessly on another, like Windows, without facing configuration issues.  

Docker is a platform that simplifies the creation, provisioning, and execution of containers. A container bundles an application with everything it needs to run, including libraries, configuration files, and dependencies. Instead of requiring separate operating systems for each application, containers share the underlying OS services of the host system, making them highly resource-efficient.  


**How Containers Differ from Virtual Machines?**  
Unlike virtual machines (VMs), which include a full operating system along with the application and its dependencies, containers share the host OS kernel. This makes containers much lighter and faster to start compared to VMs, which require hardware-level virtualization and more resources. Containers focus on isolating applications, while VMs isolate entire operating systems.



**Docker Installation Guide**  

To get started with Docker, follow the installation instructions based on your operating system:  

- **macOS and Windows**: Install Docker Desktop using the official guide [here](https://docs.docker.com/engine/install/).  
- **Linux**: Follow the instructions for your distribution, such as [Ubuntu](https://docs.docker.com/engine/install/ubuntu/).  

If you're using Windows, it's recommended to enable **Windows Subsystem for Linux (WSL)** for better performance and compatibility. Learn more about WSL integration [here](https://docs.docker.com/desktop/windows/wsl/).  

**Fixing Permissions on Linux**  
To avoid permission issues when running Docker commands on Linux, add your user to the Docker group:  

```bash
sudo usermod -aG docker $USER
```  

**Testing Docker with "Hello World"**  
Run the following command to verify your Docker installation: 
 
```bash
docker run hello-world
```  

This will download and execute a simple Docker image, confirming that Docker is set up correctly.  

**Experiment with Docker Online**  
You can try Docker without installing it by using the **Play With Docker** platform: [Play With Docker](https://labs.play-with-docker.com/).  

**Running an Ubuntu Container**  
To launch an interactive Ubuntu container, use:  
```bash
docker run -it ubuntu bash
```  
The `-it` flag enables interactive mode, allowing you to access the Ubuntu container's command line directly.

**Docker from Scratch**  

To deeply understand Docker and containers, you can explore how to build containers from scratch. Check out this resource: [Containers From Scratch](https://github.com/satyajitghana/containers-from-scratch).  


**What Are Containers?**  

A **container** is a lightweight, standalone, and executable unit of software that includes everything needed to run an application: the code, runtime, libraries, and dependencies. Containers are created from images and can be managed using the Docker API or CLI.  

With containers, you can:  
- Create, start, stop, move, or delete instances.  
- Connect containers to networks or attach storage volumes.  
- Build new images based on a container's current state.  

Containers are isolated by default, meaning their network, storage, and subsystems are separate from the host machine and other containers. However, you can configure the level of isolation based on your needs.  



**Docker Architecture**  

Docker operates using a **client-server architecture**:  

1. **Docker Client**: This is the interface used to interact with Docker. Commands like `docker run` or `docker build` are sent from the client to the daemon.  
2. **Docker Daemon**: The daemon handles the heavy lifting of building, running, and managing containers.  
3. **Communication**: The client and daemon communicate using a REST API over UNIX sockets or network interfaces.  


**Docker Compose** is another client that helps manage multi-container applications. It allows you to define and run applications consisting of multiple interconnected containers.


### **Understanding the Dockerfile**  

Let’s break down the following Dockerfile and understand its key concepts:

```dockerfile  
# Our base image  
FROM python:3.10.5-alpine  

# Set working directory inside the image  
WORKDIR /app  

# Copy our requirements  
COPY requirements.txt requirements.txt  

# Install dependencies  
RUN pip3 install -r requirements.txt  

# Copy this folder's contents to the image  
COPY . .  

# Tell the port number the container should expose  
EXPOSE 5000  
```  


#### **Dockerfile Layers**  

Every instruction in the Dockerfile creates a **layer**. Layers are intermediate images that store changes compared to the previous state of the image.  

1. **`FROM python:3.10.5-alpine`**:  
   - This is the **base image** layer. It provides a lightweight Python 3.10.5 environment optimized for Alpine Linux.  
2. **`WORKDIR /app`**:  
   - Sets the working directory inside the container to `/app`. Any subsequent commands like `COPY` or `RUN` will be executed relative to this directory.  
3. **`COPY requirements.txt requirements.txt`**:  
   - Adds the `requirements.txt` file from the local system to the container’s `/app` directory.  
4. **`RUN pip3 install -r requirements.txt`**:  
   - Installs the Python dependencies listed in the `requirements.txt` file. This forms another layer storing the installed packages.  
5. **`COPY . .`**:  
   - Copies all the files from the current directory on the host machine into the container’s `/app` directory.  
6. **`EXPOSE 5000`**:  
   - Informs Docker that the container will listen on port `5000`. This doesn’t automatically map the port but acts as documentation for users.  



### **Key Concepts**  

#### **Layers in Dockerfile**  

Each instruction (e.g., `FROM`, `COPY`, `RUN`) creates a layer. Layers optimize the build process by reusing unchanged layers when the Dockerfile is re-built. Think of it like saving "checkpoints" during a build process.  



#### **`ADD` vs `COPY`**  

- **`COPY`**: Used for basic file copying from the local machine to the container.  
   - Example:  
     ```dockerfile  
     COPY requirements.txt requirements.txt  
     ```  

- **`ADD`**: Provides extra functionality, such as extracting `.tar` files or downloading files from a URL.  
   - Example:  
     ```dockerfile  
     ADD myfiles.tar.xz /app  
     ```  

   **Best Practice**: Use `COPY` for simple file operations and `ADD` only when additional features are required.  



#### **`CMD` vs `ENTRYPOINT`**  

- **`CMD`**:  
   - Specifies the **default command** to execute when the container starts.  
   - Example:  
     ```dockerfile  
     CMD ["python", "app.py"]  
     ```  
   - This executes the Python script `app.py` as the default.  

- **`ENTRYPOINT`**:  
   - Specifies the **command that will always run** when the container starts.  
   - Example:  
     ```dockerfile  
     ENTRYPOINT ["python"]  
     CMD ["app.py"]  
     ```  
   - This sets `python` as the main executable, with `app.py` as the default argument.  

   **Best Practice**: Use `ENTRYPOINT` for fixed commands and `CMD` for configurable arguments.  



#### **Exec Form vs Shell Form**  

- **Exec Form**: Directly specifies the executable and its arguments as a JSON array.  
   - Example:  
     ```dockerfile  
     CMD ["python", "app.py"]  
     ```  
   - Advantages: Signals like `CTRL-C` (SIGINT) are correctly passed to the running process, ensuring graceful termination.  

- **Shell Form**: Runs commands through a shell (e.g., `/bin/sh -c`).  
   - Example:  
     ```dockerfile  
     CMD python app.py  
     ```  
   - Limitation: Shells often don’t forward signals, causing issues with process management.  

   **Best Practice**: Always use **exec form** to ensure proper signal handling.  


#### **`docker stop` vs `docker kill`**  

- **`docker stop`**:  
   - Sends a `SIGTERM` signal to the process, allowing it to shut down gracefully.  
   - Example: Python applications can catch a `KeyboardInterrupt` and clean up resources.  

- **`docker kill`**:  
   - Sends a `SIGKILL` signal, immediately terminating the process without cleanup.  

   **Best Practice**: Use `stop` whenever possible to allow the application to exit cleanly.  
  
 

This Dockerfile demonstrates how to set up a Python application in a lightweight container. By understanding concepts like layers, `ADD` vs `COPY`, and `CMD` vs `ENTRYPOINT`, you can build efficient, reusable Docker images while following best practices.  

## **Dockerfile with Signal Handling**  

Let’s break down this Dockerfile and the Python script `main.py` that handles signals gracefully, along with key concepts about Docker signals.  

### **Dockerfile**  
```dockerfile  
FROM python:3.7.13-alpine  

# Copy the Python script into the container  
COPY main.py main.py  

# Define the default command to execute  
CMD ./main.py  

# Send SIGINT instead of SIGTERM when stopping the container  
STOPSIGNAL SIGINT  
```  



### **`main.py`**  
The `main.py` script is designed to handle system signals like `SIGTERM` and `SIGINT`.  

```python  
#!/usr/local/bin/python3 -u  
import sys  
import signal  
import time  

# Define the signal handler function  
def signal_handler(signum, frame):  
    print(f"Gracefully shutting down after receiving signal {signum}")  
    sys.exit(0)  

if __name__ == "__main__":  
    # Attach signal handlers for SIGTERM and SIGINT  
    signal.signal(signal.SIGTERM, signal_handler)  
    signal.signal(signal.SIGINT, signal_handler)  

    # Simulate work in a loop  
    while True:  
        time.sleep(0.5)  # Simulating some task  
        print("Interrupt me")  
```  



### **Running and Handling Signals**  

1. **Building the Docker Image**:  
   Run the following command to build the image:  
   ```bash  
   docker build -t signal-handling-example .  
   ```  

2. **Running the Container**:  
   Start the container:  
   ```bash  
   docker run signal-handling-example  
   ```  

3. **Stopping the Container Gracefully**:  
   Use `docker stop` to send a **`SIGTERM`** signal (or `SIGINT` as defined by `STOPSIGNAL` in the Dockerfile):  
   ```bash  
   docker stop <container_id>  
   ```  
   The container will terminate gracefully, and you’ll see the message:  
   ```
   Gracefully shutting down after receiving signal 15  
   ```  

4. **Forcefully Killing the Container**:  
   Use `docker kill` to send a **`SIGKILL`** signal, which terminates the container immediately without cleanup:  
   ```bash  
   docker kill <container_id>  
   ```  
   The exit status will be `137` (128 + 9, where `9` is the `SIGKILL` signal).  


### **`STOPSIGNAL` in Docker**  

The `STOPSIGNAL` instruction in the Dockerfile allows you to customize the signal sent when stopping the container.  

- **Default Behavior**:  
   By default, `docker stop` sends a `SIGTERM` signal.  
- **Customizing with `STOPSIGNAL`**:  
   Adding the following line in the Dockerfile changes the default signal to `SIGINT`:  
   ```dockerfile  
   STOPSIGNAL SIGINT  
   ```  
   Now, when stopping the container, it will send `SIGINT` instead of `SIGTERM`, ensuring proper handling by Python.  



### **Sending Custom Signals**  

You can send any signal to a container using the `docker kill` command with the `--signal` flag.  

- Send `SIGTERM`:  
   ```bash  
   docker kill --signal=SIGTERM <container_id>  
   ```  

- Send `SIGINT`:  
   ```bash  
   docker kill --signal=SIGINT <container_id>  
   ```  



### **Common Signals in Docker**  

<table>
  <thead>
    <tr>
      <th>Signal Name</th>
      <th>Signal Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SIGHUP</td>
      <td>1</td>
      <td>Hang up detected on the controlling terminal.</td>
    </tr>
    <tr>
      <td>SIGINT</td>
      <td>2</td>
      <td>Issued when the user sends an interrupt (Ctrl + C).</td>
    </tr>
    <tr>
      <td>SIGQUIT</td>
      <td>3</td>
      <td>Issued when the user sends a quit signal (Ctrl + D).</td>
    </tr>
    <tr>
      <td>SIGFPE</td>
      <td>8</td>
      <td>Issued for illegal mathematical operations.</td>
    </tr>
    <tr>
      <td>SIGKILL</td>
      <td>9</td>
      <td>Immediately terminates the process without cleanup.</td>
    </tr>
    <tr>
      <td>SIGALRM</td>
      <td>14</td>
      <td>Alarm clock signal (used for timers).</td>
    </tr>
    <tr>
      <td>SIGTERM</td>
      <td>15</td>
      <td>Default termination signal (sent by `docker stop`).</td>
    </tr>
  </tbody>
</table>


### **Best Practices for Signal Handling in Docker**  

1. Graceful Shutdown:  
   - Python applications should handle `SIGTERM` or `SIGINT` gracefully to clean up resources and exit properly.  

2. Use `STOPSIGNAL`:  
   - Customize the default stop signal in the Dockerfile to align with your application’s requirements.  

3. Avoid Forceful Termination (`SIGKILL`):  
   - Only use `docker kill` when absolutely necessary, as it doesn’t allow the application to perform cleanup.  

4. Version Pinning:  
   - Always specify exact versions in the Dockerfile (e.g., `python:3.7.13-alpine`) to ensure reproducibility.  


refer to the [Dockerfile complete syntax guide](https://docs.docker.com/engine/reference/builder/).  



# Dockerized Deep Learning for MNIST Digit Classification with PyTorch


This project implements a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) on the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) using PyTorch. The project is containerized using Docker to ensure easy setup and consistent environments across different machines. The script allows training the model from scratch, resuming training from a checkpoint, and evaluating the model's performance.


## Table of Contents
- [Overview](#overview)
- [What is Docker?](#what-is-docker)
- [Why use Docker?](#why-use-docker)
- [Requirements](#requirements)
- [Docker Setup](#docker-setup)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running the Docker Container](#running-the-docker-container)
  - [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Training Script Arguments](#training-script-arguments)
- [Model Architecture](#model-architecture)
- [Data Loading and Transformations](#data-loading-and-transformations)
- [Model Initialization](#model-initialization)
- [Checkpoint Loading and Saving](#checkpoint-loading-and-saving)
- [Training and Evaluation Loop](#training-and-evaluation-loop)
- [Results](#results)

## Overview
The goal of this project is to classify handwritten digits (0-9) from the MNIST dataset using a Convolutional Neural Network (CNN). The project uses [PyTorch](https://pytorch.org/) for the model implementation, and [Docker](https://www.docker.com/) is used to containerize the application for ease of use and portability.

### MNIST Dataset
The MNIST (Modified National Institute of Standards and Technology) dataset is a database of handwritten digits that is usually used for training multiple image processing systems. Here are some key details about the dataset:

- **Content**: 28x28 grayscale images of handwritten digits (0-9)
- **Size**:
  - 60,000 training images
  - 10,000 test images
- **Format**: Each image is represented as a 2D PyTorch tensor
- **Labels**: Each image is associated with a label (0-9)
- **Source**: The dataset is built into PyTorch and can be easily downloaded using `torchvision.datasets.MNIST`

In this project, we use PyTorch's `torchvision.datasets.MNIST` to download and load the MNIST dataset. The data is normalized and transformed into PyTorch tensors for training and testing.

## What is Docker?

Docker is an open-source platform that automates the deployment of applications in lightweight, portable containers. These containers package an application and all of its dependencies, ensuring it runs the same regardless of the environment. Docker provides a way to isolate applications from the underlying system, preventing dependency conflicts and making it easier to manage and deploy applications across different systems.

## Why use Docker?

Setting up environments for machine learning and deep learning projects can be challenging because of dependencies on hardware (such as CUDA for GPUs) and incompatibilities across Python versions and libraries. Docker offers a self-contained environment that resolves such issues.

For this project, Docker is especially useful because of the following:

- **Environment Consistency**: Every user runs the project in exactly the same environment. This solves the "`it works on my machine`" conundrum.
- **Easy Setup**: PyTorch, torchvision, and other dependencies don't need to be manually installed when using Docker.
- **Reproducibility**: By specifying dependencies in a `Dockerfile`, you can duplicate the environment required to perform the training pipeline.

## Requirements
To run this project, you need to have Docker installed on your system. The installation process varies depending on your operating system. Once installed, verify by running:

```bash
docker --version
```

## Docker Setup

### Dockerfile
Here's the Dockerfile for containerizing the MNIST training:

```dockerfile
FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 --no-cache-dir install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 --no-cache-dir install numpy==1.23.4

COPY train.py /workspace/ 

CMD ["python", "train.py"]
```

### Building the Docker Image
To build the Docker image for this project, navigate to the root directory of your project and run:

```bash
docker build --tag mnist-classifier .
```

### Running the Docker Container

To run the container for training, use the following command:

```bash
docker run --name mnist-container --rm -v $(pwd):/workspace mnist-classifier python /workspace/train.py
```

### Resuming from Checkpoint
To resume training from a saved checkpoint, mount the directory where the checkpoint is stored and pass the `--resume` argument:

```bash
docker run --name mnist-container --rm -v $(pwd):/workspace mnist-classifier python /workspace/train.py --resume
```


## Training Script Arguments

You can specify the following command-line arguments while running the training script:

<table>
  <tr>
    <th>Argument</th>
    <th>Default</th>
    <th>Type</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>--batch-size</code></td>
    <td>64</td>
    <td><code>int</code></td>
    <td>Input batch size for training.</td>
  </tr>
  <tr>
    <td><code>--test-batch-size</code></td>
    <td>1000</td>
    <td><code>int</code></td>
    <td>Input batch size for testing.</td>
  </tr>
  <tr>
    <td><code>--epochs</code></td>
    <td>15</td>
    <td><code>int</code></td>
    <td>Number of epochs to train.</td>
  </tr>
  <tr>
    <td><code>--lr</code></td>
    <td>0.001</td>
    <td><code>float</code></td>
    <td>Learning rate for the optimizer.</td>
  </tr>
  <tr>
    <td><code>--gamma</code></td>
    <td>0.7</td>
    <td><code>float</code></td>
    <td>Learning rate step gamma for the learning rate scheduler.</td>
  </tr>
  <tr>
    <td><code>--no-cuda</code></td>
    <td><code>False</code></td>
    <td><code>bool</code></td>
    <td>Disables CUDA (GPU) training.</td>
  </tr>
  <tr>
    <td><code>--no-mps</code></td>
    <td><code>False</code></td>
    <td><code>bool</code></td>
    <td>Disables macOS GPU training (MPS backend).</td>
  </tr>
  <tr>
    <td><code>--dry-run</code></td>
    <td><code>False</code></td>
    <td><code>bool</code></td>
    <td>Quickly check a single pass for debugging purposes.</td>
  </tr>
  <tr>
    <td><code>--seed</code></td>
    <td>1</td>
    <td><code>int</code></td>
    <td>Random seed for reproducibility.</td>
  </tr>
  <tr>
    <td><code>--log-interval</code></td>
    <td>10</td>
    <td><code>int</code></td>
    <td>Number of batches to wait before logging training status.</td>
  </tr>
  <tr>
    <td><code>--save-model</code></td>
    <td><code>True</code></td>
    <td><code>bool</code></td>
    <td>Save the model after each epoch.</td>
  </tr>
  <tr>
    <td><code>--resume</code></td>
    <td><code>True</code></td>
    <td><code>bool</code></td>
    <td>Resume training from the last checkpoint if available.</td>
  </tr>
</table>


## Model Architecture

Here’s the CNN architecture for MNIST digit classification:

```python

import torch.nn as nn
import torch.nn.functional as F

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
```

### Data Loading and Transformations

The dataset undergoes normalization and is converted into tensors for PyTorch training and testing. Here’s the data-loading and transformation setup:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

training_data = datasets.MNIST(
    "../data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST("../data", train=False, transform=transform)

train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
```

### Model Initialization

Initializing the model and optimizer with parameters for training:

```python
import torch.optim as optim
from model import Net  # Assuming Net is your CNN architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
```

### Checkpoint Loading and Saving

To resume training from a checkpoint, the `--resume` argument is provided:

```python
import torch

model_checkpoint_path = "model_checkpoint.pth"
start_epoch = 1

# Loading the model checkpoint if 'resume' argument is True
if args.resume:
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

# Saving the model after each epoch
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
```

### Training and Evaluation Loop

The main training and evaluation loop, including checkpoint saving:

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

for epoch in range(start_epoch, args.epochs + 1):
    train_epoch(epoch, args, model, device, train_loader, optimizer)  # Training for one epoch
    test_epoch(model, device, test_loader)  # Testing on validation data
    scheduler.step()

    if args.save_model:
        save_checkpoint(model, optimizer, epoch, model_checkpoint_path)
```

## Results

After training the model for 15 epochs, the results are:

```
Train Epoch: 15 [59520/60000 (99%)]     
Loss: 0.000001

Test set: 
Average loss: 0.0306, 
Accuracy: 9927/10000 (99%)
```
