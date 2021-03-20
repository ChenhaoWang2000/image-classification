import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

print('Loading data ...')

data_root = './timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

BATCH_SIZE = 8

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layerBN0 = nn.BatchNorm1d(429)
        self.layerBN1 = nn.BatchNorm1d(1024)
        self.layerBN2 = nn.BatchNorm1d(512)
        self.layerBN3 = nn.BatchNorm1d(128)
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.layer4 = nn.LSTM(429, 1024, 2)
        self.layer5 = nn.Linear(1024, 39)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(128, 39)

        self.act_fn = nn.Tanh()

    def forward(self, x):
        # x = self.layerBN0(x)
        x = self.layer1(x)
        x = self.layerBN1(x)
        # x = self.dropout(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.layerBN2(x)
        # x = self.dropout(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.layerBN3(x)
        # x = self.dropout(x)
        x = self.act_fn(x)
        x = self.out(x)
        # x = self.layer4(x)
        # s, b, h = x.size()
        # x = x.view(s * b, h)
        # x = self.layer5(x)

        return x


# check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# fix random seed for reproducibility
same_seeds(0)

# get device
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 20  # number of training epoch
learning_rate = 0.0001  # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def cal_loss(pred, target):
    """ Calculate loss """
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(param * param)
    # TODO: you may implement L2 regularization here
    return criterion(pred, target) + 0.01 * regularization_loss
    # return self.criterion(pred, target)


# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        # batch_loss = cal_loss(outputs, labels)
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (
                        val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
