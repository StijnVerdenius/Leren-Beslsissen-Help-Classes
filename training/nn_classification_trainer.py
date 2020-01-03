import sys
import time

import numpy as np
import torch


class NNClassificationTrainer:

    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 device,
                 train_loader,
                 test_loader,
                 epochs: int,
                 ):

        self._epochs = epochs

        self._test_loader = test_loader
        self._train_loader = train_loader
        self._loss_function = loss
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._global_steps = 0
        self._acc_buffer = []
        self._loss_buffer = []
        self._metrics = []

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True):
        """ one iteration of forward-backward """

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)

        # forward pass
        accuracy, loss, out = self._forward_pass(x, y, train=train)

        # backward pass
        if train:
            self._backward_pass(loss)

        # free memory
        for tens in [out, y, x, loss]:
            tens.detach()

        return accuracy, loss.item()

    def _forward_pass(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      train: bool = True):
        """ implementation of a forward pass """

        if train:
            self._optimizer.zero_grad()

        out = self._model(x).squeeze()
        loss = self._loss_function(input=out, target=y)
        accuracy = self._get_accuracy(out, y)
        return accuracy, loss, out

    @staticmethod
    def _get_accuracy(output, y):
        predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
        correct = y.eq(predictions).sum().item()
        return correct / output.shape[0]

    def _backward_pass(self, loss):
        """ implementation of a backward pass """

        loss.backward()
        self._optimizer.step()

    def _epoch_iteration(self,
                         epoch: int):
        """ implementation of an epoch of training """

        print("\n")

        self._acc_buffer, self._loss_buffer = [], []

        for batch_num, batch in enumerate(self._train_loader):
            print(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            acc, loss = self._batch_iteration(*batch, self._model.training)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)

        self._log(epoch)

        print("\n")

    def _log(self, epoch: int):
        """ logs to terminal and tensorboard if the time is right"""

        # validate on test and train set
        train_acc, train_loss = np.mean(self._acc_buffer), np.mean(self._loss_buffer)
        test_acc, test_loss = self.validate()

        # log metrics
        self._add_metrics(test_acc, test_loss, train_acc, train_loss)

        # reset for next log
        self._acc_buffer, self._loss_buffer = [], []

        # print to terminal
        print("epoch", epoch)
        for key, value in self._metrics[-1].items():
            print(key, value)

    def validate(self):
        """ validates the model on test set """

        print("\n")

        # init test mode
        self._model.eval()
        cum_acc, cum_loss = [], []

        # loop through test-set and evaluate
        with torch.no_grad():
            for batch_num, batch in enumerate(self._test_loader):
                acc, loss = self._batch_iteration(*batch, train=False)
                cum_acc.append(acc)
                cum_loss.append(loss)
                print(f"\rEvaluating... {batch_num}/{len(self._test_loader)}", end='')
        print("\n")

        # put back into train mode
        self._model.train()

        return float(np.mean(cum_acc)), float(np.mean(cum_loss))

    def _add_metrics(self, test_acc, test_loss, train_acc, train_loss):
        """
        save metrics
        """

        metric_entry = {
            "acc/train": train_acc,
            "acc/overfitting": train_acc - test_acc,
            "loss/train": train_loss,
            "loss/test": test_loss,
            "loss/overfitting": test_loss - train_loss,
            "acc/test": test_acc
        }

        self._metrics.append(metric_entry)

    def train(self):
        """ main training function """

        self._model.train()

        try:

            print(f"Started training")

            # do training
            for epoch in range(self._epochs):
                print(f"\n\nEPOCH {epoch}\n")

                # do epoch
                self._epoch_iteration(epoch)

        except KeyboardInterrupt as e:
            print(f"Killed by user: {e} at {time.time()}")
        except Exception as e:
            print(f"Unexpected {e.__class__} error: {e} at {time.time()}")

        # flush prints
        sys.stdout.flush()

        return self._metrics, self._model
