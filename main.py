import torch
import argparse
from models.lenet import LeNet5
from utils.train import train
from utils.evaluate import evaluate
from utils.data_utils import get_loaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_loaders(args.batch_size)
    model = LeNet5()
    train(model, train_loader, val_loader, args)
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
