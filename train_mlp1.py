import mlp1
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
    # Should return a numpy vector of features.
    return feats_vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        feats_vec = feats_to_vec(features)
        prediction = mlp1.predict(feats_vec, params)
        if prediction == L2I[label]:
           good+=1
        else: 
           bad+=1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    gU = np.zeros(params[0].shape)
    gb2 = np.zeros(params[1].shape)
    gW = np.zeros(params[2].shape)
    gb1 = np.zeros(params[3].shape)

    for I in xrange(num_iterations):

        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            gU = momentum * gU + (1.-momentum) * grads[0]
            gb2 = momentum * gb2 + (1.-momentum) * grads[1]
            gW = momentum * gW + (1.-momentum) * grads[2] 
            gb1 = momentum * gb1 + (1.-momentum) * grads[3]

            regularization = 0.01
            params[0]-=learning_rate * gU + regularization * gU
            params[1]-=learning_rate * gb2 
            params[2]-=learning_rate * gW + regularization * gW
            params[3]-=learning_rate * gb1 

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        if dev_accuracy > 0.89: 
           print dev_accuracy
           return params
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    train_data, dev_data = utils.TRAIN, utils.DEV
    vocab = utils.vocab
    L2I, F2I = utils.L2I, utils.F2I
   
    in_dim, hid_dim, out_dim = len(vocab), 32, len(L2I.keys())
    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    num_iterations = 60
    learning_rate = 0.04
    
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)


