import logging
import argparse

import torch
from torch.utils import data

from model import DrugGRU
from data import DrugSideEffect
from utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # create dataset
    dataset = DrugSideEffect(root=args.data)
    # split dataset into train and val
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    train_set, val_set = data.random_split(
        dataset, [train_size, dataset_size - train_size])

    train_loader = data.DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=1,
        num_workers=4
    )
    val_loader = data.DataLoader(
        dataset=val_set,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )
    # get vocab size
    vocab_size = len(dataset.word2index)

    # create model & optimizer
    model = DrugGRU(hidden_size=args.hidden_size,
                    vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, model, optimizer, epoch)
        val_loss, val_acc = val(val_loader, model)
        # log metrics
        logging.info(
            "Epoch %d, Train Loss: %4.4f, Train Acc: %2.2f,"
            " Val Loss: %4.4f, Val Acc: %2.4f" %
            (epoch, train_loss, train_acc, val_loss, val_acc))


def train(loader, model, optim, e):
    model.train()
    train_loss, train_acc = AverageMeter(), AverageMeter()
    loader_size = loader.__len__()
    temp_loss = AverageMeter()
    # training step
    for i, (sentence, target) in enumerate(loader):
        batch_size, _ = sentence.size()

        sentence = sentence.to(device)
        target = target.to(device).view(-1, 1).type(torch.float)

        loss, logits = model(sentence, target)

        # optimization step
        optim.zero_grad()
        loss.backward()
        optim.step()

        # compute accuracy
        pred = torch.where(logits >= 0.5, 1., 0.)
        num_correct = (pred == target).sum()
        train_acc.update(num_correct, batch_size)

        train_loss.update(loss.item(), batch_size)
        temp_loss.update(loss.item(), batch_size)

        if i % 50:
            logging.info("Epoch %d|[%d/%d], Loss, : %4.4f" %
                         (e, i, loader_size, temp_loss.avg()))
            temp_loss.reset()

    return train_loss.avg(), train_acc.avg() * 100


def val(loader, model):
    model.eval()
    val_loss, val_acc = AverageMeter(), AverageMeter()
    # training step
    for sentence, target in loader:
        batch_size, _ = sentence.size()

        sentence = sentence.to(device)
        target = target.to(device).view(1, -1).type(torch.float)

        loss, logits = model(sentence, target)

        # compute accuracy
        pred = torch.where(logits >= 0.5, 1., 0.)
        num_correct = (pred == target).sum()
        val_acc.update(num_correct, batch_size)

        val_loss.update(loss.item(), batch_size)

    return val_loss.avg(), val_acc.avg() * 100


arg_parser = argparse.ArgumentParser(description="NN model to classifiy drugs"
                                                 "side effect")
# data
arg_parser.add_argument("--data", default="data/drug.xml", type=str,
                        help="path to data xml file")
# model
arg_parser.add_argument("--hidden_size", default=300, type=int,
                        help="hidden layer dimension")
arg_parser.add_argument("--lr", default=1e-4, type=float,
                        help="optimizer learning rate")
# training
arg_parser.add_argument("--epochs", default=100, type=int,
                        help="number of training epochs")
arguments = arg_parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    main(arguments)
