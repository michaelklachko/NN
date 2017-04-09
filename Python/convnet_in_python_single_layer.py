
import numpy as np
import cPickle
import time

def cost_mse(prediction, y):
        return 0.5*np.sum((prediction - y)**2)
        
def onehot(labels):
    """converts scalar labels into one hot encoded 10-vectors 
    (9 zeroes and a single 1 in the proper position"""
    l = np.zeros((len(labels), layers[-1]))
    for i, label in enumerate(labels):
        l[i,label] = 1.0    
    return l.transpose()  
    
def ReLU(z):
    return max(0.0, z)

ReLU_vec = np.vectorize(ReLU)

def ReLU_prime(z):
    """derivative of ReLU function"""
    return 0.0 if z <= 0 else 1.0

ReLU_prime_vec = np.vectorize(ReLU_prime)  

def grad_check(x, y, fs, weights, biases, grad_w, grad_b, minibatch_size, eps=0.00001):
    """dC/dw = [C(w+dw) - C(w)] / dw
    slightly change a single weight, and calculate how the cost changes    
    """   
    np.set_printoptions(threshold=1000, precision=2, linewidth=260, suppress=True)   
              
    def calc_grad(i, j, k):  
        #first calculate the original cost value     
        z0 = feedforward(x, weights, biases, fs)  #make sure there's no dropout 
        cost0 = cost_mse(z0, y)
    
        weights_mod = [np.array(w) for w in weights]  #create a copy of weights array
        #modify a single weight value (the weight from ith input pixel to ith neuron in the hidden layer, if j=0)
        weights_mod[i][j][k] += eps  
        
        z1 = feedforward(x, weights_mod, biases, fs)
        cost1 = cost_mse(z1, y)
        
        return (cost1 - cost0) / eps
    
    for j in range(1):
        for k in range(fs*fs):
            g = grad_w[0][j][k]
            c = calc_grad(0, j, k)
            diff = abs(g - c)/max(abs(g), abs(c), 0.000001)
            if diff > 0.001:
                print "g - c = {:.4f} - {:.4f} = {:.4f}   Failed!".format(g, c, diff) 
            
    
def feedforward(inpt, weights, biases, fs):
    #if input is a single image, put it in a minibatch of size=1
    if len(inpt.shape) != 4:
        inpt = inpt[np.newaxis, ...]
        
    z = conv_layer(inpt, weights[0], biases[0], fs)  #z.shape=(num_images, fms_out, img_conv, img_conv)
    z_pooled = pooling_layer(z, mode='test')        #z_pooled.shape=(num_images, fms_out, img_pool, img_pool)
    output = ReLU_vec(z_pooled)
    output = output.reshape(output.shape[0], -1).transpose()  #output.shape=(n_pool, num_images) 
    
    z = np.dot(weights[1], output) + biases[1]    #(n_hidden, n_pool)x(n_pool, num_images) = (n_hidden, num_images)
    a = ReLU_vec(z)
    
    z = np.dot(weights[2], a) + biases[2]      #(n_out, n_hidden)x(n_hidden, num_images) = (n_out, num_images)
    return z   
    
def predict(image, weights, biases, fs):
    """returns the position of the highest output"""
    return np.argmax(feedforward(image, weights, biases, fs), axis=0)   #argmax((n_out, num_images), axis=0) = (num_images,)

def accuracy((images, labels), weights, biases, fs):
    """counts number of correctly predicted labels"""   
    results = predict(images, weights, biases, fs)
    n_correct = np.sum(results == labels)
    return 100*n_correct/float(len(labels))

def conv_layer(minibatch, weights, biases, fs):
    
    num_img = minibatch.shape[0]
    img = minibatch.shape[-1] - fs + 1
    fm_out = weights.shape[0]
    
    z = np.zeros((num_img, fm_out, img, img), dtype=np.float64)
         
    for r in range(img):
        for c in range(img):
            patch = minibatch[:,:,r:r+fs,c:c+fs].reshape(num_img, -1)   #(num_images, fm_in, 5, 5)  >>> (num_img, 75)
            t = np.dot(patch, weights.transpose())                      # (num_images, 75) x (75, fm_out) 
            z[:,:,r,c] = t + biases.transpose()                         #(num_images, fm_out) + (1, fm_out)  - broadcasted to all images                   
    return z

                       
def pooling_layer(minibatch, stride=2, mode='train'):
    """stride is a scaling parameter, take (stride,stride) patch of inputs, and pick the maximum value."""
    
    num_images = minibatch.shape[0]
    fms = minibatch.shape[1]
    rows = minibatch.shape[-2]/stride
    columns = minibatch.shape[-1]/stride
    
    #initialize outputs of the pooling layer:
    z = np.zeros((num_images, fms, rows, columns), dtype=np.float64)
    
    mask = np.zeros(minibatch.shape)
    patch = minibatch[:,:, :stride, :stride]  #to initialize patch_mask_flat, so we don't have to do it inside the loop below
    patch_flat = patch.reshape(patch.shape[0], patch.shape[1], -1) #to initialize patch_mask_flat, so we don't have to do it inside the loop below
    patch_mask_flat = np.zeros(patch_flat.shape)
    
    #parallelize accross images and accross feature maps:
    for r in range(rows):
        for c in range(columns):
            patch = minibatch[:,:, stride*r : stride*(r+1), stride*c : stride*(c+1)]
            z[:,:,r,c] = np.max(patch, axis=(2,3)) #pool max values from patches into the pooling layer
            
            #if in training mode, construct the mask of max values:
            if mode == 'train':
                patch_flat = patch.reshape(patch.shape[0], patch.shape[1], -1)   #convert patch to vector, to make it easier to find the max value in the patch
                indexes_flat = np.argmax(patch_flat, axis=2)     #find the position of max value in the patch vector
                patch_mask_flat = np.zeros(patch_flat.shape)
                for i in range(patch_flat.shape[0]):             #construct the mask: set ones at the positions of max values 
                    for j in range(patch_flat.shape[1]):
                        patch_mask_flat[i,j,indexes_flat[i,j]] = 1
                patch_mask = patch_mask_flat.reshape(patch.shape) #shape the flattened mask to the patch shape
                mask[:,:, stride*r : stride*(r+1), stride*c : stride*(c+1)] = patch_mask   #construct the entire minibatch mask from patch masks     
              
    if mode == 'train':  
        mask = mask.astype(bool)  #make it a boolean array so it can be used as index to minibatch                    
        return z, mask 
        
    return z
        
        
def train(dataset, layers, fs, pool_stride, epochs, LR, minibatch_size, check_gradient=False):
    
    training_data, val_data, test_data = dataset
    #training_data, test_data = dataset
    test_images = test_data[0]
    test_labels = test_data[1]
    training_images = training_data[0]
    training_labels = training_data[1]
    
    test_images     =     test_images.reshape((-1, colors, img_in, img_in))
    training_images = training_images.reshape((-1, colors, img_in, img_in))
    
    training_data = zip(training_images, training_labels)
    test_data =     zip(test_images, test_labels)
     
    fms_in = colors
    fms_out, n_hidden, n_out = layers
    patch_size = fms_in*fs*fs #filter cube (1*5*5=25 for MNIST)
    img_conv = img_in - fs + 1  #image height after convolution (should be 28-5+1 = 24 for MNIST)
    img_pool = img_conv/pool_stride  #image height after pooling (should be 12 for MNIST)
    n_pool = fms_out*img_pool*img_pool #number of outputs from pooling layer (4*12*12=576)
    
    
    w_conv = np.random.randn(fms_out, patch_size)*np.sqrt(2.0/patch_size)  #each fm_out has its own receptive field weights (colors*fs*fs)
    b_conv = np.random.randn(fms_out,1)  #one bias per feature map
    w_hidden = np.random.randn(n_hidden, n_pool)*np.sqrt(2.0/n_pool)
    b_hidden = np.random.randn(n_hidden, 1) #+1 
    w_out = np.random.randn(n_out, n_hidden)*np.sqrt(2.0/n_hidden)
    b_out = np.random.randn(n_out, 1) 

    w_conv = w_conv.reshape(fms_out, patch_size)
    b_conv = b_conv.reshape(fms_out,1)
    
    w_hidden = w_hidden#.transpose()
    b_hidden = b_hidden.reshape(n_hidden, 1)
    
    #w_out = w_out.transpose()
    b_out = b_out.reshape(n_out, 1)
    
    #weights = [w_conv, w_hidden, w_out]
    #biases = [b_conv, b_hidden, b_out] 
    
    start_training = time.time() 
    
    for epoch in range(epochs):
        start_epoch = time.time()
             
        num_batches = len(training_labels) / minibatch_size
        
        for k in xrange(num_batches):

            x = training_images[k*minibatch_size : (k + 1)*minibatch_size]
            y = onehot(training_labels[k*minibatch_size : (k + 1)*minibatch_size])
                                                   
            grad_b_conv, grad_b_hidden, grad_b_out = [np.zeros(b.shape) for b in [b_conv, b_hidden, b_out]]
            grad_w_conv, grad_w_hidden, grad_w_out = [np.zeros(w.shape) for w in [w_conv, w_hidden, w_out]]
   
            z_conv = conv_layer(x, w_conv, b_conv, fs)  #z_conv.shape=(num_images, fms_out, img_conv, img_conv) 
            
            z_pooled, max_positions = pooling_layer(z_conv)              
            z_pooled = z_pooled.reshape(z_pooled.shape[0], -1).transpose()  #z_pooled.shape=(n_pool, num_images)           
            output_pooled = ReLU_vec(z_pooled)  
            
            z_hidden = np.dot(w_hidden, output_pooled) + b_hidden #(n_hidden, n_pooled)x(n_pooled, n_images) = (n_hidden, n_images)                    
            output_hidden = ReLU_vec(z_hidden)
            
            z_out = np.dot(w_out, output_hidden) + b_out
            
            error = z_out - y  
            
            #add bias update vectors for all images into one, and make it a column vector
            grad_b_out = np.sum(error, axis=1).reshape(-1,1)  #(-1,1) means automatically calculate the proper value for the first dimension 
            grad_w_out = np.dot(error, output_hidden.transpose())  #(10,1)x(1,4)=(10,4)

            error = np.dot(w_out.transpose(), error) * ReLU_prime_vec(z_hidden)  #(4,10)x(10,1)=(4,1)

            grad_b_hidden = np.sum(error, axis=1).reshape(-1,1)   #(4,1)
            grad_w_hidden = np.dot(error, output_pooled.transpose())  #(4,1)x(1,4)=(4,4)

            error_pool = np.dot(w_hidden.transpose(), error) * ReLU_prime_vec(z_pooled) #(4,4)x(4,1)=(4,1)
            error_pool = error_pool.transpose().reshape(minibatch_size, fms_out, img_pool, img_pool)  #to match z_conv shape
            error_conv = np.zeros(z_conv.shape)
            
            for r in range(img_conv/2):
                for c in range(img_conv/2):
                    patch_mask = max_positions[:,:, pool_stride*r : pool_stride*(r+1), pool_stride*c : pool_stride*(c+1)]
                    error_patch = error_pool[:,:,r,c].reshape(minibatch_size, fms_out, 1,1)
                    error_conv[:,:, pool_stride*r : pool_stride*(r+1), pool_stride*c : pool_stride*(c+1)] = error_patch * patch_mask
            
            for r in range(img_conv):
                for c in range(img_conv):                      
                    patch = x[:,:,r:r+fs,c:c+fs].reshape(minibatch_size, -1)  #shape=(num_images, fms_in, 5, 5) --> (num_imgs, patch_size)
                    grad_w_conv += np.dot(error_conv[:,:,r,c].transpose(), patch)   #(fms_out, num_imgs)x(num_imgs, patch_size) = (fms_out, patch size)
            
            grad_b_conv = np.sum(error_conv, axis=(0,2,3)).reshape(-1, 1)   #grad_b.shape=(fms_out,1)
            
            grad_b = [grad_b_conv, grad_b_hidden, grad_b_out] 
            grad_w = [grad_w_conv, grad_w_hidden, grad_w_out]
            
            
            if check_gradient:                  
                print "\n******************   Checking gradient   ***********************\n"
                grad_check(x, y, fs, [w_conv, w_hidden, w_out], [b_conv, b_hidden, b_out], grad_w, grad_b, minibatch_size)
                return
            
            w_conv, w_hidden, w_out = [w-(LR/minibatch_size)*dw for w, dw in zip([w_conv, w_hidden, w_out], grad_w)]                   
            b_conv, b_hidden, b_out = [b-(LR/minibatch_size)*db for b, db in zip([b_conv, b_hidden, b_out], grad_b)]   
        
        end_epoch = time.time()
        epoch_train_time = (end_epoch - start_epoch)/60.
        
        start_test_accuracy = time.time()
        best_accuracy = 0
        test_accuracy = accuracy((test_images, test_labels), [w_conv, w_hidden, w_out], [b_conv, b_hidden, b_out], fs)
        
        if test_accuracy < 12:
            print "\nBad initialization, let's try again...\n"
            return
        end_test_accuracy = time.time()
        test_time_test = (end_test_accuracy - start_test_accuracy)/60.
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            
        train_accuracy = 0
        test_time_train = 0
        if check_train_accuracy and epoch % 10 == 0 and epoch != 0:
            start_train_accuracy = time.time()     
            #train_accuracy = accuracy(training_data, [w_conv, w_hidden, w_out], [b_conv, b_hidden, b_out], fs)
            train_accuracy = accuracy((training_images, training_labels), [w_conv, w_hidden, w_out], [b_conv, b_hidden, b_out], fs)
            end_train_accuracy = time.time()
            test_time_train = (end_train_accuracy - start_train_accuracy)/60.
        
        if epoch % 30 == 0:
            LR = LR/10.
            
        print "Epoch {:d}: train: {:.2f}, test: {:.2f}".format(epoch, train_accuracy, test_accuracy)
        #epoch_time = epoch_train_time + test_time_test + test_time_train
        #print "\nEpoch took {:.2f} min: \ntraining: {:.2f} min \ntesting accuracy on test dataset: {:.2f} min \ntesting accuracy on train dataset: {:.2f} min\n".format(epoch_time, epoch_train_time, test_time_test, test_time_train)
    end_training = time.time()
    epoch_time = epoch_train_time + test_time_test + test_time_train
    total_time = (end_training - start_training)/60.
    print "\nEpoch took {:.2f} min: \ntraining: {:.2f} min \ntesting accuracy on test dataset: {:.2f} min \ntesting accuracy on train dataset: {:.2f} min\n".format(epoch_time, epoch_train_time, test_time_test, test_time_train)
    print "\nBest Test Accuracy: {:.2f}\n".format(best_accuracy)
    print "\nTotal training time: {:.2f} min\n\n".format(total_time)

LR = 0.01
epochs = 21   
minibatch_size = 16
layers = [8, 128, 10]   #fm, n_hidden, n_out    
fs=5
pool_stride=2
colors = 1
img_in = 28
check_train_accuracy = True
check_gradient = False

print "\n**** This program will train a simple conv. neural net to classify MNIST images ****\n\n"
print "Network Size: {}\nLearning Rate: {:.2f}\nMinibatch size: {:d}\n".format(layers, LR, minibatch_size)
print "\nTraining for {:d} epochs\n".format(epochs)

#input_path = 'C:\Users\Michael\Desktop\Research\Data\mnist\mnist_14_binary.pkl'
input_path = 'C:\Users\Michael\Desktop\Research\Data\mnist\mnist_binary.pkl'
f = open(input_path, 'rb')
dataset = cPickle.load(f)

#training takes 2-3 minutes per epoch on a modern 4-core CPU, depending on minibatch size, ~97% after 10 epochs, ~98% after 30 epochs (binary MNIST images)
#TODO: batchnorm, ADAM (or at least momentum), dropout, extra conv layer, better initialization, cross-entropy loss, modularity of layers, speedup 

train(dataset, layers, fs, pool_stride, epochs, LR, minibatch_size)

"""
#for minibatch_size in [16, 32, 64, 128, 256]:
for LR in [0.01]:#, 0.02, 0.03]:
    #print "\nminibatch size:", minibatch_size, "\n"
    print "\nLR:", LR, "\n"
    for i in range(3):
        train(dataset, layers, fs, pool_stride, epochs, LR, minibatch_size)
"""

