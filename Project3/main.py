import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------- pivotial parameters ----------------------------------------------------

# dataset
dataset = "mnist" #MNIST default
DSS_split = 0.7
D2_split1 = 0.7
D2_split2 = 0.2

# autoencoder
lr_ae = 0.001
loss_ae = "MSE0"
opt_ae = ""
epochs_ae = 20

# classifier
lr_cl = 0.001
loss_cl = "MSE0"
opt_cl = ""
epochs_cl = 15

# latent vector
lat_size = 5

# freeze decoder
freeze=False

#
ae_reconstructions = 10
tSNE_show=False

# ----------------------------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------- load datasets ---------------------------

# mnist
mnist_data = datasets.MNIST("./data", train=True, download=False, transform = transforms.ToTensor())

# fashion-mnist
fashionmnist_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())


# cifar10
cifar10_data = datasets.CIFAR10(root='data', train=True, download=False, transform=transforms.ToTensor())

data = mnist_data #default
#if dataset == "fmnist":
#    data = fashionmnist_data
if dataset == "cifar10":
    data = cifar10_data
if dataset == "fmnist":
    data = fashionmnist_data

# ---------------------------------- split data ----------------------------------------------

# -- subset of dataset DSS --
DSS_size = int(0.1 * len(data))
rest_size = len(data) - DSS_size
DSS, rest = torch.utils.data.random_split(data, [DSS_size, rest_size])

# -- D1/D2 split --
D1_size = int(DSS_split * len(DSS))
D2_size = len(DSS) - D1_size
D1, D2 = torch.utils.data.random_split(DSS, [D1_size, D2_size])

D1_loader = torch.utils.data.DataLoader(D1, batch_size=10, num_workers=0)

# -- train/validate/test split --
D2train_size = int(D2_split1 * len(D2))
D2validation_size = int(D2_split2 * len(D2))
D2test_size = len(D2) - D2train_size - D2validation_size
D2_train, D2_validation, D2_test = torch.utils.data.random_split(D2, [D2train_size, D2validation_size, D2test_size])

D2_train_loader = torch.utils.data.DataLoader(D2_train, batch_size=10, num_workers=0)
D2_validation_loader = torch.utils.data.DataLoader(D2_validation, batch_size=10, num_workers=0)

# -- dimentions.. --
inputsize = len(data[0][0][0])  
inchannels = len(data[0][0])
classes = 10   

# --------------------------------------------------------------------------------------------
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# ------------------------------------------- Autoencoder ------------------------------------

# -- Encoder --
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(inchannels, 16, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, lat_size, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            #nn.Flatten()
        )

    def forward(self, x):
        return self.encode(x)
        """
        self.conv1 = nn.Conv2d(inchannels, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, lat_size, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        nn.Flatten()
    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        return x #latent vector ?
        """

# -- Decoder --
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            #View((-1, lat_size, int(inputsize/2), int(inputsize/2))),
            nn.ConvTranspose2d(lat_size, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, inchannels, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        #x.view((-1, lat_size, int(inputsize/2), int(inputsize/2)))
        return self.decode(x)
        """
        self.t_conv1 = nn.ConvTranspose2d(lat_size, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, inchannels, 2, stride=2)

    def forward(self, x):
        #to use sigmoid, use torch.sigmoid
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        
        return x
        """

# -- Autoencoder --
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        #print(x.shape)
        x = self.encode(x)
        #print(x.shape)
        x = self.decode(x)
        return x

# ---------------------------------- Semi-supervised classifier C1 ------------------------------

# -- head of the semi-supervised classifier --
class Chead(nn.Module):
    def __init__(self):
        super(Chead, self).__init__()
        self.fc1 = nn.Linear(lat_size, 16)
        self.fc2 = nn.Linear(16, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# -- Semi-supervised learner consisting of encoder and classifier head (Chead) --
class C1(nn.Module):
    def __init__(self, encoder, chead):
        super(C1, self).__init__()
        self.encoder = encoder
        self.chead = chead
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.chead(x)
        return x
# --------------------------------------- Supervised classifier C2 ---------------------------------
# -- used as a benchmark --

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
        return F.softmax(x, dim=0)


# ----------------------------------------------------------------------------

# ------------------------------- visualize ----------------------------------

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
"""
dataiter = iter(D1_loader)

images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(10):
    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    #ax.set_title(classes[labels[idx]])
plt.show()
"""
def reconstruct(autoencoder):  #only works for mnist currently
    dataiter = iter(D1_loader)
    images, labels = dataiter.next()
    reconstructions = autoencoder(images)
    images = images.numpy() # convert images to numpy for display
    reconstructions = reconstructions.detach().numpy()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        #ax.set_title(classes[labels[idx]])
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 20/2, idx+11, xticks=[], yticks=[])
        imshow(reconstructions[idx])
    plt.show()
    # print recon as well




# ------------------------------ make models ----------------------------------
encoder = Encoder()
decoder = Decoder()
autoencoder = Autoencoder(encoder, decoder)

head = Chead()
c1 = C1(encoder, head)

c2 = C2()

reconstruct(autoencoder)

# -- loss functions --

ae_loss = nn.BCELoss() #default

cl_loss = nn.NLLLoss() #default


# -- Optimizers --
opt_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae) #default

opt_cl = torch.optim.Adam(c2.parameters(), lr=lr_ae) #default

# ------------------------------ train autoencoder ---------------------------------

for epoch in range(epochs_ae):
    train_loss = 0
    for data in D1_loader:
        X, _ = data # will not use _ (the label)
        opt_ae.zero_grad() # eller autoencoder.zero_grad()?
        outputs = autoencoder(X)
        loss = ae_loss(outputs, X) #learn from images generated
        loss.backward()
        opt_ae.step()
        train_loss += loss.item()*X.size(0)
        train_loss = train_loss/len(D1_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

reconstruct(autoencoder)
exit()
# --------------------------------- train networks -------------------------------------

if freeze:      #not tested!
    # freeze parameter weigths in encoder
    for params in encoder.parameters():
        params.require_grad = False

# -- c2 --
"""
for epoch in range(epochs_cl):
    train_loss = 0
    val_loss = 0
    for data in D2_train_loader:
        X, y = data
        opt_cl.zero_grad() # eller c2.zero_grad()?
        outputs = c2(X.view(-1, inputsize*inputsize))
        #print(outputs)
        exit()
        loss = cl_loss(outputs, y)
        loss.backward()
        opt_cl.step()
        train_loss += loss.item()*X.size(0)
        train_loss = train_loss/len(D2_train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

    for data in D2_validation_loader:
        X, y = data
        #opt_cl.zero_grad() # eller c2.zero_grad()?
        outputs = c2(X.view(-1, inputsize*inputsize))
        loss = cl_loss(outputs, y)
        val_loss += loss.item()*X.size(0)
        val_loss = val_loss/len(D2_validation_loader)
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
"""
