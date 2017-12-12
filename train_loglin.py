import loglinear as ll
import random
import utils
import numpy as np


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
        prediction = ll.predict(feats_vec, params)
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
    #cumm_grad = [np.zeros(params[0].shape), np.zeros(params[1].shape)]
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            gW, gb = grads
            #cumm_grad[0]+=gW
            #cumm_grad[1]+=gb
            params[0]-=learning_rate*gW
            params[1]-=learning_rate*gb

        
        #params[0]-=learning_rate*cumm_grad[0]/len(train_data)
        #params[1]-=learning_rate*cumm_grad[1]/len(train_data)

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
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
   
    in_dim, out_dim = len(vocab), len(L2I.keys())
    params = ll.create_classifier(in_dim, out_dim)
    num_iterations = 150
    learning_rate = 0.4
    
    i = random.choice(range(100))
    feats_to_vec(train_data[i][1])
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

