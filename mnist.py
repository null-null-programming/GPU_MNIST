import os
import time
import torch

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F


import torch.multiprocessing as mp
import torch.distributed as dist


# Declare 3-layer MLP for MNIST dataset
class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=10, layers=[120, 84]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


writer = SummaryWriter()

# Global constants
EPOCHS = 100
WARM_UP_STEPS = 5
BATCH_SIZE = 32

# Load MNIST train dataset
train_dataset = mnist.MNIST(
    root="/tmp/MNIST_DATA_train", train=True, download=True, transform=ToTensor()
)


def main(rank, world_size):
    print("rank : {}".format(rank))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the GPU id corresponding to the process rank
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")

    # Prepare device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = MLP().to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss()

    # Prepare data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Run the training loop
    print("----------Training ---------------")
    model.train()
    step = 0
    t1 = time.time()
    for epoch in range(EPOCHS):
        start = time.time()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_x = train_x.view(train_x.size(0), -1)
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss", loss.item(), step)
            step += 1
            if idx < WARM_UP_STEPS:  # skip warmup iterations
                start = time.time()

    t2 = time.time()
    print("Total Time : {}".format(t2 - t1))
    writer.flush()

    # Compute statistics for the last epoch
    interval = idx - WARM_UP_STEPS  # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Train throughput (iter/sec): {}".format(throughput))
    print("Final loss is {:0.4f}".format(loss.detach().to("cpu")))

    # Save checkpoint for evaluation
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    torch.save(checkpoint, "checkpoints/checkpoint.pt")

    print("----------End Training ---------------")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("world_size : {}".format(world_size))
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
