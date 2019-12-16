# ICDSS-FashionMNIST

## Dataset
Fashion-MNIST is an image dataset compiled by Zalando Research to replace the MNIST dataset. It contains 70,000 28x28 grayscale images of clothing items from 10 labelled classes. 
The dataset is divided into a 60,000-image train set and 10,000-image test set. The data is provided in 2 .csv files contained in data.tar.gz file. [reader.py](https://github.com/sytan98/ICDSS-FashionMNIST/blob/master/reader.py) helps to open the tar.gz file and return the training images, training labels, testing images and testing labels.

### Visualization
I did dimensionality reduction using t-Distributed Stochastic Neighbour Embedding (t-SNE). In machine learning, dimensionality reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. 
 
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.  

### Data Preprocessing
After using the provided reader to import the data as NumPy arrays in Python, I transformed the NumPy arrays into a float32 array of shape (60000, 28 * 28) and included an extra dimension to the array as the dimension for image channels. 
I split the original training data into a training set of 48,000 images (80%) and a validation, of 12000 images (20%). 

For data augmentation, other than common data augmentation techniques such as horizontal flipping, shifting and zooming on the training samples, I also included a pre-processing function, which I learnt about during my internship. Random Erasing randomly selects a rectangle region in an image and erases its pixels with random 
values. Random Erasing is complementary to commonly used data augmentation techniques such as random cropping and flipping. It is very effective in combatting occlusion, but it also allows us to train our model to not be overly dependent on a certain feature in the image to classify it. Do refer to this [paper](https://arxiv.org/abs/1708.04896 ) for more information on this.

## Models
I experimented with a few models:
1. 4 layer CNN layer
2. ResNet
3. Transfer learning using InceptionV3

## Train

| Model         | Testing Accuracy | Model Weights |
| :-----------: |:----------------:|:-------------:|
| 4layer CNN    | 0.9135           |[train_model_4CNN.hdf5](https://drive.google.com/file/d/13tpQiTnfruziTi6pLwhVbL2FItDqTEhL/view?usp=sharing)|
| ResNet        | 0.9099           |[train_model_resnet.hdf5](https://drive.google.com/file/d/1LpeqzU0ZMjERWQIiWteLSYgMT2069EIo/view?usp=sharing)|


For the 4 layer CNN, I used a grid search to find the most optimal training parameters for the model. Unfortunately, keras wrapper for scikit-learn does not allow us to do a grid search with fit_generator. So we need to create a new class that defines a function "fit".

## Evaluation
Test accuracy is not sufficient as an indicator of whether the model is a good model. We need to analyse if the accuracy is balanced across all classes. 

I used a classification Report which shows precision, recall and f1 score of each class. 
Precision is true positive/(true positive and false positive). A high precision would mean classifier has a low false positive rate. Recall is true positive/(true positive and false negative. A high Recall would mean a low false negative rate.  F1 score is basically 2 x precision x recall/(precision + recall). It is the weighted average of Precision and Recall. T
 
I also displayed the confusion matrix, where the diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabelled by the classifier. The higher the diagonal values of the confusion matrix the better, indicating many correct predictions. 

For the 4 layer CNN model, I also tried to extracting feature maps at each layers to understand the intermediate outputs of the CNN model. In order to extract the feature maps we want to look at, we’ll create a Keras model that takes batches of images as input, and outputs the activations of all convolution and pooling layers. To do this, I used the Keras class Model functional api. 

## Requirements
Choose the latest versions of any of the dependencies below: 
* pandas 0.24.2
* numpy 1.16.2
* matplotlib 3.0.3
* sklearn 0.20.3
* keras 2.3.1
* tensorflow 2.0.0
