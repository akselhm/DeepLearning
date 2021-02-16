import numpy as np

from layer import layer
from datagenerator import datagenerator


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

        self.loss_case = np.zeros(len(images))  #store the loss for each case
        self.lossiter = []  #store the total loss for each iteration (image)

        """
        # Prior way to initiate layers (DELETE?)

        self.inputlayer  = layer(len(images[0]), 0)
        self.outputlayer = layer(len(target[0]), hiddenarray[-1])
        self.hiddenlayers= []

        hiddenarray.insert(0, len(images[0]))   #weigths for the first hiddenlayer
        for i in range(len(self.hiddenlayers-1)):
            self.hiddenlayers.append(layer(hiddenarray[i+1], hiddenarray[i]))

        self.layers = self.hiddenlayers
        self.layers.insert(0, self.inputlayer)
        self.layers.append(self.outputlayer)
        """

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
        z: single array of the output for a given case
        ---------
        returns: single array of softmax applied on all outputs
        """
        return self.softmax(z)*(1 - self.softmax(z))

    # --- loss functions ---

    def cross_entropy(self, target, output):
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


    def forward_pass(self, case):
        """
        parameters:
        case: index of the case (image) to pass. The image is represented as a single array
        returns: 
        out: the output of the forward pass given as a 1x4-array
        loss: single array of losses for each class
        """
        """
        Objective (delete when done)
        1. Fetch a minibatch of training cases (or one case).
        2. Send each case through the network, from the input to the output layer. At each layer (L), multiply the
        outputs of the upstream layer by the weights and then add in the biases. Finally, apply the activation
        function to these sums to produce the outputs of L.
        3. Apply the softmax function to the values entering the output layer to produce the network’s outputs.
        Remember that softmax has no incoming weights.
        4. Compare the targets to the output values via the loss function.
        5. Cache any information (such as the outputs of each layer) needed for the backward stage
        """
        # iterate through the layers
        upstreamout =  self.images[case]
        for i in range(len(self.layers)):
            upstreamout = self.layers[i].forward_pass(upstreamout, case)
        out = upstreamout

        #sotmax
        if self.softmax_on:
            out = self.softmax(out)

        #loss function (make functionality for selection)
        loss = self.cross_entropy(self.target, out)
        print(loss)
        return out, loss


    def backward_pass(self, loss, out, case):
        """
        parameters:
        out: the output of the forward pass given as a 1x4-array
        loss: single array of losses for each class (1x4)
        case: index of the case (image) to pass. The image is represented as a single array
        returns: 
        """
        JLN = loss # if the network do not have a softmax layer

        #1. Compute the initial Jacobian (JLS) representing the derivative of the loss with respect to the network’s (typically softmaxed) outputs.
        if self.softmax_on:
            JLS = self.d_softmax(loss)
        
            #2. Pass JLS back through the Softmax layer, modifying it to JLN , which represents the derivative of the loss with respect to the outputs of the layer prior to the softmax, layer N.
            JSN = np.diag(out) - np.outer(out, out) #softmax jacobian
            JLN = np.dot(JLS, JSN) #duble check this

        #3. Pass JLN to layer N, which uses it to compute its delta Jacobian, δN .
        for i in range(len(self.layers)-1, 0, -1):
            JLN = self.layers[i].backward_pass(JLN, case, self.layers[i-1].nodes[case])
        #4. Use δN to compute: a) weight gradients JLW for the incoming weights to N, b) bias gradients JLB for the biases at layer N, and c) JLN−1 to be passed back to layer N-1.

        #5. Repeat steps 3 and 4 for each layer from N-1 to 1. Nothing needs to be passed back to the Layer 0, the input layer. 

        #6. After all cases of the minibatch have been passed backwards, and all weight and bias gradients havebeen computed and accumulated, modify the weights and biases.
"""
    def update_weigths(self):

"""

# ---------------------- test (or run)---------------------------------------

#generate images and convert to array
n = 12 #number of pixels in each direction
trainingsize = 3
gen = datagenerator(12, trainingsize, 0, centered=False)
images, target = gen.generate()
#print(images)
print(target)
imgrid = gen.im2grid(images)
imarr = gen.grid2array(imgrid)

# make layers
nodes1 = 20
nodes2 = 15

layer1 = layer(nodes1,n*n, trainingsize)
layer2 = layer(nodes2, nodes1, trainingsize)
outputlayer = layer(4, nodes2, trainingsize)

layerarray = [layer1, layer2, outputlayer]


net = network(layerarray, imarr, target, 0)

print(net.forward_pass(0)) #first case through forward pass