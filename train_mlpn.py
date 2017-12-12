import mlpn
import random
import utils
import numpy as np

momentum = 0.9

def feats_to_vec(features):
    feats_vec = np.zeros(len(vocab))
    indices = [F2I[f] if f in vocab else F2I['UNKNOWN'] for f in features]
    for feature_key in indices:
       feats_vec[feature_key]+=1.

    feats_vec/=len(features)
    return feats_vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        feats_vec = feats_to_vec(features)
        prediction = mlpn.predict(feats_vec, params)
        if prediction == L2I[label]:
           good+=1
        else: 
           bad+=1
    return good / (good + bad)

def test_classifier(test_data, test_raw, trained_params):
  f = open("test.pred", "w")
  for i, features in enumerate(test_data):
    x = feats_to_vec(features)
    category = mlpn.predict(x, trained_params)
    #print i, I2L[category], test_raw[i]
    f.write(I2L[category]+"\n")
  f.close()

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    weighted_grads = []
    for p in params:
      weighted_grads.append(np.zeros(p.shape))

    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
 
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = mlpn.loss_and_gradients(x,y,params)
            cum_loss += loss
            
            for j, grad in enumerate(grads):
               weighted_grads[j] = momentum * weighted_grads[j] + (1-momentum) * grad
               params[j] -= learning_rate * weighted_grads[j] 
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        if dev_accuracy >=0.89: 
          print dev_accuracy
          return params
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':

    train_data, dev_data, test_data = utils.TRAIN, utils.DEV, utils.TEST
    test_sentences = utils.test_sentences
    vocab = utils.vocab
    L2I, I2L, F2I = utils.L2I, utils.I2L,  utils.F2I
   
    in_dim, out_dim = len(vocab), len(L2I.keys())
    dims = [in_dim, 64, 64, out_dim]
   
    params = mlpn.create_classifier(dims)
    num_iterations = 75
    learning_rate =  5*1e-3
    
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    test_classifier(test_data, test_sentences, trained_params)
