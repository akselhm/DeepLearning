import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# --------------- pivotial parameters -----------------

# dataset
dataset = "MNIST" #MNIST default
DSS_split = 0.7
D2_split1 = 0.7
D2_split2 = 0.2

# autoencoder
lr_ae = 0.001
loss_ae = "MSE"
opt_ae = ""
epochs_ae = 5

# classifier
lr_cl = 0.001
loss_cl = "MSE"
opt_cl = ""
epochs_cl = 5

# latent vector
lat_size = 5

# freeze decoder
freeze=False

#
ae_reconstructions = 10
tSNE_show=False

# ------------------------------------------------------


# --------------- load datasets -----------------------

# mnist
mnist_data = datasets.MNIST("./data", train=True, download=False, transform = transforms.Compose([transforms.ToTensor()]))

# fashion mnist


# cifar 10
cifar10_data = datasets.CIFAR10(root='data', train=True, download=False, transform=transforms.ToTensor())

data = mnist_data #default
#if dataset == "fmnist":
#    data = fashionmnist_data
if dataset == "cifar10":
    data = cifar10_data

# split
D1_size = int(DSS_split * len(dataset))
D2_size = len(dataset) - D1_size
D1, D2 = torch.utils.data.random_split(dataset, [D1_size, D2_size])

D2test_size = int(D2_split1 * len(D2))
D2validation_size = int(D2_split2 * len(D2))
D2test_size = len(D2) - D2test_size - D2validation_size
D2_test, D2_validation, D2_test = torch.utils.data.random_split(dataset, [D2test_size, D2validation_size, D2test_size])

inputsize = 28  #generate from dataset
classes = 10    #generate from dataset


# -----------------------------------------------------

# -------------- Autoencoder --------------------------

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, lat_size, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x #latent vector ?

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(lat_size, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, x):
        return self.decoder.forward(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

# -------------- Semi-supervised classifier -----------

class Chead(nn.Module):
    #head of the semi-supervised classifier
    def __init__(self):
        super(Chead, self).__init__()
        self.fc1 = nn.Linear(lat_size, 16)
        self.fc2 = nn.Linear(16, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class C1(nn.Module):
    # Semi-supervised learner consisting of encoder and classifier head (Chead)
    def __init__(self, encoder, chead):
        super(C1, self).__init__()
        self.encoder = encoder
        self.chead = chead
    
    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.chead.forward(x)
        return x
# -------------- Supervised classifier ----------------
# used as a benchmark

class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()
        self.fc1 = nn.Linear(inputsize**2, 64)
        self.fc2 = nn.Linear(64, 50)
        self.fc3 = nn.Linear(50, 64)
        self.fc4 = nn.Linear(64, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


# -----------------------------------------------------

# -------------------- run ----------------------------
"""
def reconstructions(index):
    plt.imshow(X[index].view(inputsize,inputsize))
    plt.show()
    # print recon as well
"""

encoder = Encoder()
decoder = Decoder()
autoencoder = Autoencoder(encoder, decoder)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Optimizer
if opt_ae == "Adam":
    opt_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae)


print(encoder)
print(decoder)
print(autoencoder)