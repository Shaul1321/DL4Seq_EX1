import numpy as np

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """

    exp = np.exp(x-np.max(x))
    return exp/np.sum(exp)

def classifier_output(x, params):
    H = forward(x, params)
    output_activation = H[-1]
    probs = softmax(output_activation)
    return probs

def forward(x, params):
  num_layers = len(params) / 2

  H = [x]
  h = x
  p = 0.01
  for i in xrange(num_layers):
       W, b = params[2*i], params[2*i+1]
       h = h.dot(W) + b
  
       if i < num_layers - 1: #if not output layer
         h = np.tanh(h)
         #h = np.maximum(h, 0.)
       H.append(h)

  assert len(H) == num_layers + 1 #+1 for input layer
  return H

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    grads = []
    H = forward(x, params)
    num_layers = len(H)
    prediction = softmax(H[-1])
    y_true = np.zeros(prediction.shape)
    y_true[y] = 1.
    loss = -np.log(prediction[y])

    delta = prediction - y_true

    for i in xrange(num_layers-1, 0, -1):

      prev_h = H[i-1]

      if i != num_layers - 1 : #not output layer      
        W_next =  params[2*i] 
        delta = W_next.dot(delta) * (1.- H[i]**2) 
        #delta = W_next.dot(delta) * (np.where(H[i]>0., 1., 0))
      gW = np.outer(prev_h, delta)
      gb = delta
      grads.append(gb)
      grads.append(gW)

    grads.reverse()

    for grad, param in zip(grads, params):
      assert grad.shape == param.shape

    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(1, len(dims)):
      prev_size, next_size = dims[i-1], dims[i]
      epsilon = np.sqrt(6)/(np.sqrt(prev_size+next_size))
      W_prev_next = np.random.uniform(low=-epsilon, high=epsilon, size = (prev_size, next_size))
      b = np.random.uniform(low=-0.1, high = 0.1, size = next_size)
      params.append(W_prev_next)
      params.append(b)
    return params

if __name__ == '__main__':
 params = create_classifier([3, 20, 30,  6])

 # Sanity checks. If these fail, your gradient calculation is definitely wrong.
 # If they pass, it is likely, but not certainly, correct.
 from grad_check import gradient_check


 def _loss_and__grad(param_index):
      loss,grads = loss_and_gradients(x,0,params)

      return loss,grads[param_index]

 def _loss_and_W1_grad(W1):
        params1 = np.array(params)
        params1[0] = W1

        loss,grads = loss_and_gradients(x,1,params1)
        return loss,grads[0]

 def _loss_and_W2_grad(W2):
        params1 = np.array(params)
        params1[2] = W2

        loss,grads = loss_and_gradients(x,2,params1)
        return loss,grads[2]

 def _loss_and_W3_grad(W3):
        params1 = np.array(params)
        params1[4] = W3

        loss,grads = loss_and_gradients(x,3,params1)
        return loss,grads[4]

 def _loss_and_b1_grad(b1):
        params1 = np.array(params)
        assert params1[1].shape == b1.shape
        params1[1] = b1
 
        loss,grads = loss_and_gradients(x,1,params1)
        return loss,grads[1]

 def _loss_and_b2_grad(b2):
        params1 = np.array(params)
        params1[3] = b2

        loss,grads = loss_and_gradients(x,1,params1)
        return loss,grads[3]

 def _loss_and_b3_grad(b3):
        params1 = np.array(params)
        params1[5] = b3

        loss,grads = loss_and_gradients(x,1,params1)
        return loss,grads[5]
 
 for i in range(10):
  x = np.random.randn(3)
  W1 = np.random.randn(*params[0].shape)
  W2 = np.random.randn(*params[2].shape)
  W3 = np.random.randn(*params[4].shape)
  b1 = np.random.randn(*params[1].shape)
  b2 = np.random.randn(*params[3].shape)
  b3 = np.random.randn(*params[5].shape)

  gradient_check(_loss_and_W1_grad ,W1)
  gradient_check(_loss_and_W2_grad ,W2)
  gradient_check(_loss_and_W3_grad ,W3)
  gradient_check(_loss_and_b1_grad ,b1)
  gradient_check(_loss_and_b2_grad ,b2)
  gradient_check(_loss_and_b3_grad ,b3)

