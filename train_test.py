import torch
import data_manager
import models
from hyper_parameters import hparams


# Wrapper class to run PyTorch model
class Runner:
    def __init__(self, hparams):
        # __init__에 model 을 올림
        self.model = models.Baseline(hparams)
        # Loss function은 CrossEntropyLoss()를 호출
        self.criterion = torch.nn.CrossEntropyLoss()
        # Stocastic Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate)
        # Learning_rate도 저장
        self.learning_rate = hparams.learning_rate
        # 기본 디바이스는 cpu
        self.device = torch.device("cpu")

        # GPU
        if hparams.device:
            torch.cuda.set_device(hparams.device - 1)
            # model.cuda
            self.model.cuda(hparams.device - 1)
            # criterion.cuda
            self.criterion.cuda(hparams.device - 1)
            # device.cuda를 지정
            self.device = torch.device("cuda:" + str(hparams.device - 1))

    # Accuracy function works like loss function in PyTorch
    def accuracy(self, source, target):
        # 예측치와 실제 값을 비교하기 위해서
        source = source.max(1)[1].long().cpu()
        target = target.cpu()
        correct = (source == target).sum().item()

        return correct/float(source.size(0))

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode == 'train' else self.model.eval()

        epoch_loss = 0
        epoch_acc = 0
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            acc = self.accuracy(prediction, y)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()
            epoch_acc += prediction.size(0)*acc

        epoch_loss = epoch_loss/len(dataloader.dataset)
        epoch_acc = epoch_acc/len(dataloader.dataset)

        return epoch_loss, epoch_acc


def device_name(device):
    if device == 0:
        device_name = 'CPU'
    else:
        device_name = 'GPU:' + str(device - 1)

    return device_name


def main():
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams)

    print(f"Training on {device_name(hparams.device)}")
    for epoch in range(hparams.num_epochs):
        train_loss, train_acc = runner.run(train_loader, 'train')
        valid_loss, valid_acc = runner.run(valid_loader, 'eval')

        print(f"[Epoch {epoch + 1}/{hparams.num_epochs}] [Train Loss: {train_loss}] "
              f"[Train Acc: {train_acc}] [Valid Loss: {valid_loss}] [Valid Acc: {valid_acc}]")

    test_loss, test_acc = runner.run(test_loader, 'eval')
    print("Training Finished")
    print(f"Test Accuracy: {100 * test_acc}")


if __name__ == '__main__':
    main()
