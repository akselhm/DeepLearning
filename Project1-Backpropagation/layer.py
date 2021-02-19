import numpy as np

class layer:

    def __init__(self, nodes, inputs, act_func= 'sigmoid'):
        """
        parameters:_________________________________
        nodes: number of nodes
        inputs: number of inputs (nodes in upstream layer)
        act_func: activation function for the layer. Valid inputs are sigmoid, ReLU, ...
        """

        self.bias = np.zeros(nodes)     
        self.nodes = np.zeros(nodes)    #need to store the values for each case in dataset? 
        self.act_func = act_func
        #self.weights = np.random.randn(nodes, inputs)

        #HE initialization
        self.weights = np.random.randn(nodes,inputs)*np.sqrt(2/inputs)

    #  -- activation functions -------------------

    def sigmoid(self, x):
        #sigmoid activation function
        """
        x       : an array with the same dimentions as nodes
        return  : an array with the same dimentions as nodes
        """
        return 1/(1 + np.exp(-x))

    def ReLU(self, x):
        """
        x       : an array with the same dimentions as nodes
        return  : an array with the same dimentions as nodes
        """
        return np.maximum(0, x)

    def tanh(self, x):
        """
        x       : an array with the same dimentions as nodes
        return  : an array with the same dimentions as nodes
        """
        return np.tanh(x)
    

    #  -- derivatives of activation functions ------------------

    def d_sigmoid(self, x):        
        """
        The derivative of the sigmoid function with respect to x
        x       : an array with the same dimentions as nodes
        return  : an array with the same dimentions as nodes
        """
        #return x*(1-x)
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def d_ReLU(self, x):
        """
        The derivative of the ReLU function with respect to x
        x       : an array with the same dimentions as nodes
        return  : an array with the same dimentions as nodes
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def d_tanh(self, x):
        """
        The derivative of the tan h function with respect to x
        x       : an array with the same dimentions as nodes
        return  : an array with the same dimentions as nodes
        """
        return 1/(np.cosh(x)**2)


    #  -- forward and backward pass --

    def forward_pass(self, inputs):
        """
        parameters:__________________________________________
        inputs: outputs from the upstream layer (one or more array)
        case: index of the case (image) to pass. The image is represented as a single array
        returns: 
        out: the output of the forward pass given as a 1x4-array
        loss: single array of losses for each class
        """
        sums = np.matmul(self.weights,inputs) + self.bias

        if self.act_func == 'ReLU':
            output = self.ReLU(sums)     #dependent on activation function (only sigmoid working now)
        elif self.act_func == 'tanh':
            output = self.tanh(sums)
        elif self.act_func == 'linear':
            output = sums
        else: # assume sigmoid
            output = self.sigmoid(sums) 

        self.nodes = output    
        #print(self.nodes)
        return output

    def backward_pass(self, JLY, upstream_nodes, lr):
        """
        Parameters:_________________________________
        JLY: the jacobian for the downstream layer Y (a little bad notation as Y is upstream in lecture notes)
        case: index of the case (image) to pass. The image is represented as a single array
        upstream_nodes: the nodes of the upstream layer (to calculate JLW)
        lr: learning rate; determines how much the weigths are updated
        --------------------
        returns: the jacobian JLM with respect to this layer M
        """
        if self.act_func == 'ReLU':
            JZSum = np.diag(self.d_ReLU(self.nodes))  
        elif self.act_func == 'tanh':
            JZSum = np.diag(self.d_tanh(self.nodes))
        elif self.act_func == 'linear':
            JZSum = np.diag(np.ones(len(self.nodes))) 
        else: #sigmoid
            JZSum = np.diag(self.d_sigmoid(self.nodes))  
        #print(JZSum)

        JZY = np.dot(JZSum, self.weights)
        JLZ = np.dot(JZY.T, JLY)


        JZW = np.outer(upstream_nodes, np.diag(JZSum)) #np.dot(Y_mat,JZSum)       #simplified version from lecture slide 2 p. 53 (y*z(1-z))
        #print(JZW.shape)
        #print(self.weights.shape)
        
        # tror noe av problemet ligger i JLM som er en vektor i første iterasjon (fra output layer), men er en matrise senere
        JLW = JZW * JLZ #np.dot(JMW,JLM)       # double check dimentions
        #print(JLW.shape)

        self.weights -= lr*JLW.T
        #print(self.weights)
    
        # bias
        
        JZB = np.dot(JZSum, self.bias).reshape(len(self.bias), 1) 
        #print(JLY.shape)
        #print(JZB.shape)
        #print(JLZ.shape)
        #JLB = JZB * JLZ #np.dot(JNB,JLN)
        JLB = JZB * JLY
        JLB = np.squeeze(np.asarray(JLB))
        #print(JLB.shape)
        
        self.bias += lr*JLB     # should it be += here?
        
        return JLZ


# ----------------test -----------------------
"""
layer = layer(5, 15)

#print(layer.nodes)
#print(layer.weights)
inputs = [0,0.1,0.2,0.3,0.4,0.5,0.0,0.1,0.2,0.3,0.4,0.5,0.3,0.4,0.5]

print(layer.forward_pass(inputs))
"""