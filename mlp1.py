import numpy as np

dropout_rate = 0.0
dropout_factor = 1./(1.-dropout_rate)
dropout_mask = None


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """

    exp = np.exp(x-np.max(x))
    return exp/np.sum(exp)

def classifier_output(x, params):
    # YOUR CODE HERE.
    h, z = forward(x, params)
    probs = softmax(z)
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def forward(x, params):
  global dropout_mask

  U,b2, W, b1 = params
  h = np.tanh(x.dot(W)+b1)
  #h = np.maximum(x.dot(W) + b1, 0.) #RELU
  dropout_mask = np.where(np.random.rand(*h.shape) > dropout_rate, 1., 0.)
  h*=dropout_mask*dropout_factor
  z = h.dot(U) + b2
  return h,z

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W (size: in_dim x hid_dim)
    gb1: vector, gradients of b1 (size:hid_dim)
    gU = matrix, gradients of U (size: hid_dim x out_dim)
    gb2 = vector, gradients of b2 (size: out_dim)
    """

    x,y = np.array(x), np.array(y)
    U,b2, W, b1 = params
    h, z = forward(x, params)
    prediction = softmax(z)
    loss = -np.log(prediction[y])
    y_true = np.zeros(prediction.shape)
    y_true[y] = 1. 

    
    delta_output = prediction - y_true
    gU = np.outer(h, delta_output)

    gb2 = delta_output
    #rnd_matrix = np.random.ones(*U.shape)
    delta_hidden =  U.dot(delta_output) * (1-(h/dropout_factor)**2) * dropout_mask * dropout_factor
    #delta_hidden =  U.dot(delta_output) * (np.where(h>0., 1., 0))
    gW =  np.outer(x, delta_hidden) 
    gb1 = delta_hidden

    return loss, [gU, gb2, gW, gb1]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """

    """
    W = np.random.randn(in_dim, hid_dim)
    U = np.random.randn(hid_dim, out_dim)
    b2 = np.random.randn(out_dim)
    b1 = np.random.randn(hid_dim)
    """

    epsilon_U = np.sqrt(100)/(np.sqrt(hid_dim+out_dim))
    epsilon_W = np.sqrt(100)/(np.sqrt(in_dim+hid_dim))
    U = np.random.uniform(low=-epsilon_U, high=epsilon_U, size = (hid_dim, out_dim))
    b1 = np.zeros(hid_dim)
    b2 = np.zeros(out_dim)
    W = np.random.uniform(low=-epsilon_W, high=epsilon_W, size = (in_dim, hid_dim) ) 

    params = [U,b2, W, b1]
    return params


if __name__ == '__main__':

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    U,b2, W, b1 = create_classifier(3,5, 4)

    def _loss_and_W_grad(W):

        loss,grads = loss_and_gradients([1,2,3],0,[U, b2, W, b1])
        return loss,grads[2]

    def _loss_and_b1_grad(b1):

        loss,grads = loss_and_gradients([1,2,3],1,[U, b2, W, b1])
        return loss,grads[3]

    def _loss_and_U_grad(U):
        loss,grads = loss_and_gradients([1,2,3],0,[U, b2, W, b1])
        return loss,grads[0]

    def _loss_and_b2_grad(b2):

        loss,grads = loss_and_gradients([1,2,3],1,[U, b2, W, b1])
        return loss,grads[1]
    
    for _ in xrange(10):

        U = np.random.randn(U.shape[0],U.shape[1])
        b1 = np.random.randn(b1.shape[0])
        b2 = np.random.randn(b2.shape[0])
        W = np.random.randn(W.shape[0],W.shape[1])
 
        #set dropout_rate=0 before gradient test

        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_b2_grad, b2)


