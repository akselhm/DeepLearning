from layer import layer

class network:

    def __init__(self, inputs, hiddenlayers):
        self.inputs = inputs
        self.layers = []
        for i in range(hiddenlayers):
            layers[i] = layer(10)


    def forward_pass(self, inputdata):
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
        for layer in layers:
            layer.foreward_pass(inputs)


