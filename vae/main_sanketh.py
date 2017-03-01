from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="PyTorch MNIST Example - VAE ")
parser.add_argument('--batch-size',type=int, default=128, metavar='N'
                    help = "input batch size for training (default = 64)")
parser.add_argument('--epochs', type=int, default=10, metavar='N'
                    help = "number of epochs to train (default 2)")
parser.add_argument('--no-cuda',action='store_true',default=False
                    help = 'enables CUDA training')
parser.add_argument('--log-interval',type=int,default=10,metavar='N'
                    help = "how many minibatches to wait before logging the status")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',train=True, download=True,
                        transform=transforms.ToTensor()),
                        batch_size = args.batch_size, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',train=False,transform=transforms.ToTensor()),
        batch_size = args.batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.fc1 = nn.Linear(784,400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x,x)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add(-1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
