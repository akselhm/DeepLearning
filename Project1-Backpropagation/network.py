import numpy as np
import matplotlib.pyplot as plt

from layer import layer
from datagenerator import datagenerator

# ------------------------ CONFIGURATION ----------------------------------------------------------------------

n = 30              #pixels of image (n*n)
datasize = 500      #size of dataset
noise = 0.01        #amount of pixels influenced by noise; 0.01 = 1%
imshow = False      #set to True to show images on screen
centered = False    #set to True to center the images


#LAYERS (first is input, last is output) possible atcivation functions: 'sigmoid', 'tanh', 'ReLU', 'linear'
layer_spec = [[n*n, 'none'], [100, 'tanh'], [50, 'ReLU'], [10, 'sigmoid'], [4, 'sigmoid']]

loss_func = 'cross_entropy'
lr = 0.01

epochs = 300         #one epoch = the whole training set one time 

#--------------------------------------------------------------------------------------------------------

class network:

    def __init__(self, layers, images, target, lr, loss_func = 'cross_entropy', softmax_on= True):
        """
        Parameters  :______________________________________________
        hiddenarray : list of number of nodes in each layer in the hidden layer, example: [20, 15, 15]
        images      : array of the array representing each image
        target      : array of the labels for each image
        lr          : learning rate; determines how much the weigths are updated
        loss_func   : select loss function. 
        softmax_on  : initial value True. Change to False to turn of softmax layer
        """
        self.images = images
        self.target = target
        self.layers = layers
        self.lr     = lr
        self.softmax_on = softmax_on

        self.loss_list = []  #store the total loss


    def softmax(self, output):
        """
        parameters:__________________________________
        output: single array of the output for a given case
        ---------
        returns: single array of softmax applied on all outputs
        """
        return np.exp(output)/np.exp(output).sum()

    def d_softmax(self, z):
        """
        parameters:_______________________________________
        z: single array of the output for a given case (softmaxed)
        ---------
        returns: single array of softmax applied on all outputs
        """
        #return self.softmax(z)*(1 - self.softmax(z))
        return np.diag(z) - np.outer(z, z)

    # --- loss functions ---

    def cross_entropy(self, target, output):    #Usikker på om denne fungerer riktig
        """
        parameters:__________________________________
        target: single array of the target for a given case
        output: single array of the output for a given case
        ---------
        returns: single array of losses for ach class
        """
        # cross entropy loss funtion
        return - np.multiply(target, np.log(output)).sum()

    def MSE(self, target, output):
        """
        parameters:_______________________________
        target: single array of the target for a given case
        output: single array of the output for a given case
        ---------
        returns: single array of losses for each class
        """
        # mean squared error loss funtion
        return np.square(np.subtract(target,output)).mean() 

    # --- derivatives of loss functions ----

    def cross_entropy_deriv(self, targets, predictions):
        return np.where(predictions != 0, -targets/predictions, 0.0)

    def d_MSE(self, target, output):
        """
        parameters:___________________________
        target: single array of the target for a given case
        output: single array of the output for a given case
        ---------
        returns: single array of change in losses for each class with respect to ...
        """
        # not finished
        f = output # not correct
        return 2*(np.subtract(target,output)*f).mean()

    # ---- regularization -------------------
    #TODO: implement regulization to forpass and backpass

    def L1(self):
        sum = 0
        for l in self.layers:
            sum += ((l.weights)**2).sum()
        return 0.5*sum
    
    def L2(self):
        sum = 0
        for l in self.layers:
            sum += abs(l.weights).sum()
        return sum

    def d_L1(self, w):
        w[w < 0] = -1
        w[w == 0] = 0
        w[w > 0] = 1
        return w

    def d_L2(self, w):
        return w
    

    #  -- forward and backward pass --   

    def forward_pass(self, case):
        """
        parameters:
        case: index of the case (image) to pass. The image is represented as a single array
        returns: 
        out: the output of the forward pass given as a 1x4-array
        loss: single array of losses for each class
        """
        # iterate through the layers
        upstreamout =  self.images[case]
        for i in range(len(self.layers)):
            upstreamout = self.layers[i].forward_pass(upstreamout)
        out = upstreamout

        #sotmax
        if self.softmax_on:
            out = self.softmax(out)

        #loss function (make functionality for selection)
        loss = self.cross_entropy(self.target[case], out)
        #print(loss)
        self.loss_list.append(loss)
        return out, loss


    def backward_pass(self, loss, out, case):
        """
        parameters:
        out: the output of the forward pass given as a 1x4-array (typically softmaxed)
        loss: single array of losses for each class (1x4)
        case: index of the case (image) to pass. The image is represented as a single array
        returns: 
        """
        JLZ = loss # initialize in case the network do not have a softmax layer
        JLS = self.cross_entropy_deriv(self.target[case], out).reshape(len(self.target[case]), 1)
        

        if self.softmax_on:
            JSoft = self.d_softmax(out)
            
            JLZ = np.dot(JLS.T, JSoft).T
            
        
            #JSN = np.diag(out) - np.outer(out, out) #softmax jacobian
            #JLN = np.dot(JLS, JSN) #double check this (JLZ from lecture notes) 

        for i in range(len(self.layers)-1, 0, -1):  #iterate backwards through the layers
            JLZ = self.layers[i].backward_pass(JLZ, self.layers[i-1].nodes, self.lr)



    def plot_loss(self):
        #not complete 
        plt.figure(1)
        loss = self.loss_list
        plt.plot(loss[0:-1:210])
        plt.title('Loss')
        #plt.xlabel(r'image')
        plt.ylabel(r'loss')
        plt.show()


# ---------------------- test (and run)---------------------------------------

#generate images and convert to array

gen = datagenerator(n, datasize, noise, centered=centered, imshow=imshow)
images, target = gen.generate()
imgrid = gen.im2grid(images)
imarr = gen.grid2array(imgrid)

#TODO: split into train, validate, test


# create layers
layers = []

for i in range(1, len(layer_spec)):
    l = layer(layer_spec[i][0], layer_spec[i-1][0], act_func=layer_spec[i][1])
    layers.append(l)  

# create network
NN = network(layers, imarr, target, lr, loss_func=loss_func)
NN.layers[-1].weights

#print(net.forward_pass(0)) #first case through forward pass

#exit()
for e in range(epochs):
    for i in range(datasize):
        out, loss = NN.forward_pass(i)
        NN.backward_pass(loss, out, i)
        #print(loss)


NN.plot_loss()