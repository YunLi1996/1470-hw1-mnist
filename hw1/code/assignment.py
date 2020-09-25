from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self):
        self.input_size = 28*28 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.5

        self.W = np.zeros((784,10))
        self.b = np.zeros(10)

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        W_all = np.concatenate((self.W, np.reshape(self.b, (1, 10))))
        inputs = np.append(inputs, np.ones((np.shape(inputs)[0], 1)), axis=-1)
        
        linear_layer = np.dot(inputs, W_all)
        probability_layer = np.exp(linear_layer)
        sum_e = np.sum(probability_layer,axis=-1)
        probability_layer = probability_layer / np.reshape(sum_e, (sum_e.size, 1))
        return probability_layer
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step). 
        NOTE: This function is not actually used for gradient descent 
        in this assignment, but is a sanity check to make sure model 
        is learning.
        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        losses = probabilities[range(probabilities.shape[0]), labels]
        losses = np.log(losses) * -1.0
        return (np.sum(losses) / losses.size)
    
    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. The learning 
        algorithm for updating weights and biases mentioned in 
        class works for one image, but because we are looking at 
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        y = np.zeros((labels.size, self.num_classes))
        y[np.arange(labels.size), labels] = 1.0
        
        delta_W = np.dot(np.transpose(inputs), (probabilities - y))
        delta_W = delta_W / labels.size
        delta_W = self.learning_rate * delta_W * -1.0
        
        delta_b = probabilities - y
        delta_b = np.mean(delta_b, axis=0)
        delta_b = self.learning_rate * delta_b * -1.0
        return (delta_W, delta_b)
    
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        max_prob = np.argmax(probabilities, axis=1)
        correct = np.sum(max_prob == labels)
        return 1.0 * correct / labels.size

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        self.W = self.W + gradW
        self.b = self.b + gradB
        return
    
def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    increment = model.batch_size
    batch_losses = []
    for i in range(int(train_labels.size / increment)):
        batch_inputs = train_inputs[i*increment:(i+1)*increment, ]
        batch_labels = train_labels[i*increment:(i+1)*increment]
        probabilities = model.call(batch_inputs)
        gradients = model.back_propagation(batch_inputs, probabilities, batch_labels)
        model.gradient_descent(gradients[0], gradients[1])
        
        batch_losses.append(model.loss(probabilities, batch_labels))
    visualize_loss(np.array(batch_losses))
    return

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment, 
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    probabilities = model.call(test_inputs)
    return model.accuracy(probabilities, test_labels)

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far, 
    and how do you know that?
    Optionally, add your answers to README!
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT
    
    :return: doesn't return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''

    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_data = get_data("data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz", 60000)
    train_inputs = train_data[0]
    train_labels = train_data[1]
    
    test_data = get_data("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz", 10000)
    test_inputs = test_data[0]
    test_labels = test_data[1]
    
    # TODO: Create Model
    m = Model()
    
    # TODO: Train model by calling train() ONCE on all data
    train(m, train_inputs, train_labels)
    
    # TODO: Test the accuracy by calling test() after running train()
    print(test(m, test_inputs, test_labels))

    # TODO: Visualize the data by using visualize_results()
    #visualize_results()
    
if __name__ == '__main__':
    main()
