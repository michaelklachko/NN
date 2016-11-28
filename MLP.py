import numpy as np
import cPickle
import time
    
def ReLU(z): return max(0.0, z)

#apply the function to each element in the input array
ReLU_vec = np.vectorize(ReLU)

def ReLU_prime(z):
    """derivative of ReLU function"""
    return 0.0 if z <= 0 else 1.0

ReLU_prime_vec = np.vectorize(ReLU_prime)   
      
def onehot(j):
    """take a number, and return a sparse vector of 10 elements"""   
    result = np.zeros((10,1))
    result[j] = 1.0   
    return result 
    
def initialize_weights(layers):
    n_in, n_hidden, n_out = layers
    
    W1 = np.random.randn(n_in, n_hidden)*np.sqrt(2.0/n_in)
    b1 = np.random.randn(1, n_hidden) 
    W2 = np.random.randn(n_hidden, n_out)*np.sqrt(2.0/n_hidden)
    b2 = np.random.randn(1, n_out)
    
    return W1, W2, b1, b2
    
def backprop(error, batch, outputs, W2):   
    z_hidden, output_hidden, z_out = outputs   
    #add bias update vectors for all images into one, and make it a column vector
    grad_b_out = np.sum(error, axis=0)#.reshape(1, -1)  #should be shape=(1,10)
    grad_w_out = np.dot(output_hidden.transpose(), error)  #(50, 200) x (200, 10)  = (50, 10)
               
    error = np.dot(error, W2.transpose()) * ReLU_prime_vec(z_hidden)  #(200, 50)
    
    grad_b_hidden = np.sum(error, axis=0)#.reshape(1,-1)   #should be shape=(1,50)
    grad_w_hidden = np.dot(batch.transpose(), error)   #(784, 50)
    
    return grad_w_hidden, grad_b_hidden, grad_w_out, grad_b_out

def feedforward(images, weights, biases):
    z_hidden = np.dot(images, weights[0]) + biases[0]   #(#imgs, 784) x (784, 50) = (#imgs, 50)
    output_hidden = ReLU_vec(z_hidden)
    z_out = np.dot(output_hidden, weights[1]) + biases[1]     #(#imgs, 50) x (50, 10) = (#imgs, 10)  
    
    return z_hidden, output_hidden, z_out   
            
def predict(images, weights, biases):
    """returns the position of the highest output for each image
    axis=0 means return a row of results (one per column)
    axis=1 means return a column of results (one per row)
    """
    return np.argmax(feedforward(images, weights, biases)[2], axis=1)  #shape=(#imgs)
    
def accuracy(images, labels, weights, biases):
    results = predict(images, weights, biases) 
    n_correct = np.sum((x == y) for (x, y) in zip(results, labels))
    
    return 100*n_correct/float(len(labels))

def train(layers, epochs, LR, batch_size):
    print "Network Size: {}\nLearning Rate: {:.2f}\nMinibatch size: {:d}\nNumber of epochs: {:d}".format(layers, LR, batch_size, epochs)
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_labels_onehot = np.asarray([onehot(lb) for lb in train_labels]).reshape(60000, 10)

    batches = [train_images[k:k+batch_size] for k in xrange(0, len(train_images), batch_size)]
    labels = [train_labels_onehot[k:k+batch_size] for k in xrange(0, len(train_labels), batch_size)]
              
    W1, W2, b1, b2 = initialize_weights(layers)
    
    start_time = time.time()
    best_accuracy = 0   
    
    print "\nAccuracy (training, testing), %:\n"
        
    for epoch in range(epochs):
        for batch, batch_labels in zip(batches, labels):   
            
            outputs = feedforward(batch, (W1, W2), (b1, b2))
            error = outputs[-1] - batch_labels          
            grad_w_hidden, grad_b_hidden, grad_w_out, grad_b_out = backprop(error, batch, outputs, W2)
                      
            W2 = W2 - (LR/batch_size)*grad_w_out
            b2 = b2 - (LR/batch_size)*grad_b_out

            W1 = W1 - (LR/batch_size)*grad_w_hidden
            b1 = b1 - (LR/batch_size)*grad_b_hidden

        test_accuracy = accuracy(test_images, test_labels, (W1, W2), (b1, b2)) 
        train_accuracy = accuracy(train_images, train_labels, (W1, W2), (b1, b2))
        
        print "Epoch {:d}: {:.2f}, {:.2f}".format(epoch, train_accuracy, test_accuracy)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

    print "\nBest Accuracy:    {:.2f}%\n".format(best_accuracy)                       
    print "--- Program ran for {:.1f} minutes ---\n".format((time.time() - start_time)/60.0) 

  
def load_mnist():
    """dataset is broken into training data (images (60k, 784) and labels (60k)), and test data (images (10k, 784) and labels (10k))"""
    f = open("C:\Users\Michael\Desktop\Research\Data\mnist\mnist_noval.pkl", 'rb')
    train_data, test_data = cPickle.load(f)
    f.close()
    
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    return train_images, train_labels, test_images, test_labels

print "\n**** This program will train a single layer neural net to classify MNIST images ****\n\n"

train(layers=[784, 50, 10], epochs=20, LR=0.02, batch_size=200)
