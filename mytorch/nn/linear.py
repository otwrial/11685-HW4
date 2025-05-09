import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        self.input_shape = A.shape
        batch_size = int(np.prod(self.input_shape[:-1]))
        in_features = self.input_shape[-1]
        self.A_flat = A.reshape(batch_size, in_features)
        Z = np.dot(self.A_flat, self.W.T) + self.b
        out_features = self.W.shape[0]
        Z = Z.reshape(*self.input_shape[:-1], out_features)
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        batch_size = int(np.prod(self.input_shape[:-1]))
        out_features = self.W.shape[0]
        dZ_flat = dLdZ.reshape(batch_size, out_features)

        # Compute gradients (refer to the equations in the writeup)
        self.dLdA = np.dot(dZ_flat, self.W)
        

        self.dLdW = np.dot(dZ_flat.T, self.A_flat)
        self.dLdb = np.sum(dZ_flat, axis=0)
        self.dLdA = self.dLdA.reshape(self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA
