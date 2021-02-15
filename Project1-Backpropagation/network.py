import numpy as np

from layer import layer
from datagenerator import datagenerator


class network:

    def __init__(self, layers, images, target):
        """
        Parameters:
        inputs      :
        hiddenarray : list of number of nodes in each layer in the hidden layer, example: [20, 15, 15]
        images      : array of the array representing each image
        target      : array of the labels for each image
        """
        self.images = images
        self.target = target
        self.layers = layers

        """
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
        return np.exp(output)/np.exp(output).sum()


    def forward_pass(self, inputdata):
        # inputdata: an image represented as a single array
        """
        1. Fetch a minibatch of training cases.
        2. Send each case through the network, from the input to the output layer. At each layer (L), multiply the
        outputs of the upstream layer by the weights and then add in the biases. Finally, apply the activation
        function to these sums to produce the outputs of L.
        3. Apply the softmax function to the values entering the output layer to produce the network’s outputs.
        Remember that softmax has no incoming weights.
        4. Compare the targets to the output values via the loss function.
        5. Cache any information (such as the outputs of each layer) needed for the backward stage
        """
        upstreamout = inputdata #self.images[0]
        for i in range(len(self.layers)):
            upstreamout = self.layers[i].forward_pass(upstreamout)

        return self.softmax(upstreamout)

    def backward_pass(self, inputdata):
        return 0


# ---------------------- test (or run)---------------------------------------

#generate images and convert to array
n = 12 #number of pixels in each direction
gen = datagenerator(12, 2, 0, centered=False)
images, target = gen.generate()
#print(images)
print(target)
imgrid = gen.im2grid(images)
imarr = gen.grid2array(imgrid)

# make layers
nodes1 = 20
nodes2 = 15
layer1 = layer(nodes1,n*n)
layer2 = layer(nodes2, nodes1)
outputlayer = layer(4, nodes2)
layerarray = [layer1, layer2, outputlayer]
hiddenarray = [20, 20]

net = network(layerarray, imarr, target)

print(net.forward_pass(net.images))