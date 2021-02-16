import numpy as np

class layer:

    def __init__(self, nodes, inputs, trainingsize, act_func= 'sigmoid'):
        """
        parameters:_________________________________
        nodes: number of nodes
        inputs: number of inputs
        trainingsize: size of the trainingdataset
        act_func: activation function for the layer. Valid inputs are sigmoid, ReLU, ...
        """

        self.bias = np.zeros(nodes)     
        self.nodes = np.zeros((trainingsize,nodes))    #need to store the values for each case in dataset? 
        self.weights = np.ones((nodes, inputs))*0.1 #temporary way to initialize weights (to 0.1), dim(nodes in layer, nodes in upstream layer)
        self.d_weights = np.zeros((nodes, inputs))

    #  -- activation functions --

    def sigmoid(self, x):
        #sigmoid activation function
        return 1/(1 + np.exp(-x))

    def ReLU(self, x):
        return max(0, x)

    #  -- derivatives of activation functions --

    def d_sigmoid(self, x):        
        """
        The derivative of the sigmoid function with respect to x
        parameters:___________________________
        target: single array of the target for a given case
        output: single array of the output for a given case
        ---------
        returns: single array of change in losses for each class with respect to x
        """
        return x*(1-x)

    def d_ReLU(self, x):
        if x > 0:
            return 1
        return 0

    #  -- forward and backward pass --

    def forward_pass(self, inputs, case):
        """
        parameters:__________________________________________
        inputs: outputs from the upstream layer (one or more array)
        case: index of the case (image) to pass. The image is represented as a single array
        returns: 
        out: the output of the forward pass given as a 1x4-array
        loss: single array of losses for each class
        """
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
        output = self.sigmoid(sums)     #dependent on activation function
        self.nodes[case] = output    
        #print(self.nodes)
        return output

    def backward_pass(self, JLN, case, upstream_nodes):
        """
        Parameters:_________________________________
        JLN: the jacobian for the downstream layer N
        case: index of the case (image) to pass. The image is represented as a single array
        upstream_nodes: the nodes of the upstream layer (to calculate JLW)
        returns: the jacobian JLM with respect to this layer M
        """
        #1. Compute the initial Jacobian (JLS) representing the derivative of the loss with respect to the network’s (typically softmaxed) outputs.
        #2. Pass JLS back through the Softmax layer, modifying it to JLN , which represents the derivative of the loss with respect to the outputs of the layer prior to the softmax, layer N.
        #3. Pass JLN to layer N, which uses it to compute its delta Jacobian, δN .

        JMSum = np.diag(self.d_sigmoid(self.nodes[case]))    #noted JZSum in lecture notes
        JNM = np.dot(JMSum, self.weights) #jacobian for this layer, noted JZY in lecture notes
        JLM = np.dot(JNM,JLN)
        #4. Use δN to compute: a) weight gradients JLW for the incoming weights to N, b) bias gradients JLB for the biases at layer N, and c) JLN−1 to be passed back to layer N-1.

        X_T = np.array([upstream_nodes]).T      #not sure if i need this
        Y_mat = np.array(upstream_nodes*len(JMSum)).reshape(len(JMSum), len(upstream_nodes)).T #create a matrix with upstream nodes (same node on the whole row)
        JMW = np.dot(Y_mat,JMSum)       #simplified version from lecture slide 2 p. 53
        JLW = 1

        JLB = 1
        #5. Repeat steps 3 and 4 for each layer from N-1 to 1. Nothing needs to be passed back to the Layer 0, the input layer. 
        return JLM
        #6. After all cases of the minibatch have been passed backwards, and all weight and bias gradients havebeen computed and accumulated, modify the weights and biases.



    #def update_weights(self):
        #TODO: iterate through the net and update weigths for each layer

# ----------------test -----------------------
"""
layer = layer(5, 15)

#print(layer.nodes)
#print(layer.weights)
inputs = [0,0.1,0.2,0.3,0.4,0.5,0.0,0.1,0.2,0.3,0.4,0.5,0.3,0.4,0.5]

print(layer.forward_pass(inputs))
"""