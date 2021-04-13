import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# -------------------------------------- pivotial parameters ----------------------------------------------------

# dataset
dataset = "mnist" #MNIST default, other options are "fmnist", "cifar10" and "omniglot"
DSS_split = 0.7
D2_split1 = 0.7
D2_split2 = 0.2

# autoencoder
lr_ae = 0.001
loss_ae = "MSE0"    #BCE default,
optimizer_ae = ""   #Adam default,
epochs_ae = 15

# classifier
lr_cl = 0.001
loss_cl = "MSE0"    #Cross entropy default,
optimizer_cl = ""   #Adam default, other option "SGD"
epochs_cl = 20

# latent vector
lat_size = 50       # not implemented properly

# freeze decoder
freeze=True

#
ae_reconstructions = 15
tSNE_show=False

# ----------------------------------------------------------------------------------------------------------------
batchsize = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------- load datasets ---------------------------

# mnist
mnist_data = datasets.MNIST("./data", train=True, download=False, transform = transforms.ToTensor())

omniglot_data = datasets.Omniglot("./data", download=False, transform = transforms.ToTensor())

# fashion-mnist
fashionmnist_data = datasets.FashionMNIST(root='data', train=True, download=False, transform=transforms.ToTensor())
#UFC_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

VOC_data = datasets.VOCSegmentation(root='data', download=False, transform=transforms.ToTensor())

# emnist
#emnist_data = datasets.EMNIST(root='/data', download=True, transform=transforms.ToTensor())

# cifar10
cifar10_data = datasets.CIFAR10(root='data', train=True, download=False, transform=transforms.ToTensor())

data = mnist_data #default
#if dataset == "fmnist":
#    data = fashionmnist_data
if dataset == "cifar10":
    data = cifar10_data
elif dataset == "fmnist":
    data = fashionmnist_data
elif dataset == "voc":
    data = VOC_data
elif dataset == "omniglot":
    data = omniglot_data

# ---------------------------------- split data ----------------------------------------------

# -- subset of dataset DSS --
DSS_size = int(0.1 * len(data))
rest_size = len(data) - DSS_size
DSS, rest = torch.utils.data.random_split(data, [DSS_size, rest_size])

# -- D1/D2 split --
D1_size = int(DSS_split * len(DSS))
D2_size = len(DSS) - D1_size
D1, D2 = torch.utils.data.random_split(DSS, [D1_size, D2_size])

D1_loader = torch.utils.data.DataLoader(D1, batch_size=batchsize, num_workers=0)

# -- train/validate/test split --
D2train_size = int(D2_split1 * len(D2))
D2validation_size = int(D2_split2 * len(D2))
D2test_size = len(D2) - D2train_size - D2validation_size
D2_train, D2_validation, D2_test = torch.utils.data.random_split(D2, [D2train_size, D2validation_size, D2test_size])

D2_train_loader = torch.utils.data.DataLoader(D2_train, batch_size=batchsize, num_workers=0)
D2_validation_loader = torch.utils.data.DataLoader(D2_validation, batch_size=batchsize, num_workers=0)

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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=lat_size*batchsize):
        return input.view(input.size(0), size, 1, 1)

# -- Encoder --
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(inchannels, 16, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        #    Flatten()
        )
        #self.fc1 = nn.Linear(16*3*4+4, lat_size)
        #self.latent = nn.Linear(16*3*4+4, lat_size)

    def forward(self, x):
        x = self.encode(x)
        #print(x.shape)
        #x = self.latent(x)
        #print(x.shape)
        return x

        """
class Encoder(nn.Module):
        self.conv1 = nn.Conv2d(inchannels, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, lat_size, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.latent = nn.Flatten()
    
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
            #UnFlatten(),
            #View((-1, lat_size, int(inputsize/2), int(inputsize/2))),
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, inchannels, 2, stride=2),
            nn.Sigmoid()
        )
        #self.linear = nn.Linear(lat_size, 16*3*4+4)

    def forward(self, x):
        #x.view((-1, lat_size, int(inputsize/2), int(inputsize/2)))
        #print(x.shape)
        #x = self.linear(x)
        #print(x.shape)
        x = self.decode(x)
        return x
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
        self.fc1 = nn.Linear(int(inputsize*inputsize/4), 16)
        self.fc2 = nn.Linear(16, classes)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, 16, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            #Flatten()
        )
        self.fc1 = nn.Linear(int(inputsize*inputsize/4), lat_size)
        self.fc2 = nn.Linear(lat_size, 84)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)        

        return F.softmax(x, dim=0)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# ----------------------------------------------------------------------------


# ------------------------------- visualize ----------------------------------



# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def reconstruct(autoencoder, amount): 
    dataiter = iter(D1_loader)
    images, labels = dataiter.next()
    reconstructions = autoencoder(images)
    images = images.numpy() # convert images to numpy for display
    reconstructions = reconstructions.detach().numpy()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(amount):
        ax = fig.add_subplot(2, amount, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        #ax.set_title(classes[labels[idx]])
    for idx in np.arange(amount):
        ax = fig.add_subplot(2, amount, idx+amount+1, xticks=[], yticks=[])
        imshow(reconstructions[idx])
    plt.show()

def tSNEplot(features):
    tsne = TSNE(n_components=2).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    """
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = np.array(colors_per_class[label], dtype=np.float) / 255

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)
    """
    # build a legend using the labels we set previously
    #ax.legend(loc='best')

    # finally, show the plot
    plt.show()


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

#dataiter = iter(D1_loader)
#images, labels = dataiter.next()
#tSNEplot(images)

#exit()

# ------------------------------ make models ----------------------------------
encoder = Encoder()
decoder = Decoder()
autoencoder = Autoencoder(encoder, decoder)


head = Chead()
c1 = C1(encoder, head)
c2 = C2()

reconstruct(autoencoder, ae_reconstructions)    #try to reconstruct images without training

# -- loss functions --

ae_loss = nn.BCELoss() #default

cl_loss = nn.CrossEntropyLoss() #default


# -- Optimizers --
opt_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae) #default

opt_c1 = torch.optim.Adam(c1.parameters(), lr=lr_cl) #default
opt_c2 = torch.optim.Adam(c2.parameters(), lr=lr_cl) #default
if optimizer_cl == "SGD":
    opt_c1 = torch.optim.SGD(c1.parameters(), lr=lr_cl, momentum=0.9)
    opt_c2 = torch.optim.SGD(c2.parameters(), lr=lr_cl, momentum=0.9)

# ------------------------------ train autoencoder ---------------------------------
print("autoencoder training")
ae_loss_array = np.zeros(epochs_ae)

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
    ae_loss_array[epoch] = train_loss
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

reconstruct(autoencoder, ae_reconstructions)

plt.plot(ae_loss_array)
plt.show()


# --------------------------------- train networks -------------------------------------

# -- losses --
c1_trainacc = np.zeros(epochs_cl)
c2_trainacc = np.zeros(epochs_cl)
c1_valacc = np.zeros(epochs_cl)
c2_valacc = np.zeros(epochs_cl)

# -- freeze parameter weigths in encoder --
if freeze:      #not tested!
    for params in encoder.parameters():
        params.require_grad = False

# -- c1 --
print("c1 training")

for epoch in range(epochs_cl):
    correct = 0
    total = 0
    train_loss = 0
    # -- training --
    for data in D2_train_loader:
        X, y = data
        opt_c1.zero_grad() # eller opt.zero_grad()?
        outputs = c1(X)#.view(-1, inputsize*inputsize))
        #print(outputs)
        loss = cl_loss(outputs, y)
        loss.backward()
        opt_c1.step()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        train_loss += loss.item()*X.size(0)
    train_loss = train_loss/len(D2_train_loader)
    c1_trainacc[epoch] = 100 * correct / total
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    # -- validation --
    correct = 0
    total = 0
    for data in D2_validation_loader:
        X, y = data
        outputs = c1(X)#.view(-1, inputsize*inputsize))
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    c1_valacc[epoch] = 100 * correct / total

    


# -- c2 --
print("c2 training")
for epoch in range(epochs_cl):
    correct = 0
    total = 0
    train_loss = 0
    val_loss = 0
    for data in D2_train_loader:
        X, y = data
        opt_c2.zero_grad() # eller c2.zero_grad()?
        outputs = c2(X)#.view(-1, inputsize*inputsize))
        #print(outputs)
        loss = cl_loss(outputs, y)
        loss.backward()
        opt_c2.step()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        train_loss += loss.item()*X.size(0)
    train_loss = train_loss/len(D2_train_loader)
    c2_trainacc[epoch] = 100 * correct / total
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    # -- validation --
    correct = 0
    total = 0
    for data in D2_validation_loader:
        X, y = data
        outputs = c2(X)#.view(-1, inputsize*inputsize))
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    c2_valacc[epoch] = 100 * correct / total

# -- plot accuracy --
plt.plot(c1_trainacc, label="c1 train")
plt.plot(c1_valacc, label="c1 validate")
plt.plot(c2_trainacc, label="c2 train")
plt.plot(c2_valacc, label="c2 validate")
plt.legend()
plt.show()


# -- test on D1 -- 

#c1
correct = 0
total = 0
for data in D1_loader:
    X, y = data
    outputs = c1(X)#.view(-1, inputsize*inputsize))
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()
accuracy = 100 * correct / total

print("c1 accuracy on D1: ", accuracy)

#c2
correct = 0
total = 0
for data in D1_loader:
    X, y = data
    outputs = c2(X)#.view(-1, inputsize*inputsize))
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()
accuracy = 100 * correct / total

print("c2 accuracy on D1: ", accuracy)

