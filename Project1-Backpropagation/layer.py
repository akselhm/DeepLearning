import numpy as np

class layer:

    def __init__(self, nodes, inputs):
        """
        parameters:
        nodes: number of nodes
        inputs: number of inputs
        """

        self.bias = 0.1     #select value (?)
        self.nodes = np.zeros(nodes)
        self.weights = np.ones((nodes, inputs))*0.5 #temporary way to initialize weights (to 0.5)


    def act_func(self, x):
        #sigmoid activation function
        return 1/(1 + np.exp(-x))


    def forward_pass(self, inputs):
        #inputs: outputs from the upstream layer (one or more array)
        """
        1. Fetch a minibatch of training cases. (done in network.forward_pass)
        2. Send each case through the network, from the input to the output layer. At each layer (L), multiply the
        outputs of the upstream layer by the weights and then add in the biases. Finally, apply the activation
        function to these sums to produce the outputs of L.
        3. Apply the softmax function to the values entering the output layer to produce the network’s outputs.
        Remember that softmax has no incoming weights.
        4. Compare the targets to the output values via the loss function.
        5. Cache any information (such as the outputs of each layer) needed for the backward stage
        """
        sums = np.matmul(self.weights,inputs) + self.bias
        output = self.act_func(sums)
        return output


# ----------------test -----------------------
"""
layer = layer(5, 15)

#print(layer.nodes)
#print(layer.weights)
inputs = [0,0.1,0.2,0.3,0.4,0.5,0.0,0.1,0.2,0.3,0.4,0.5,0.3,0.4,0.5]

print(layer.forward_pass(inputs))
"""