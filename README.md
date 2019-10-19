# Multiclass-Image-classifier-
 
 This repository will provide a multiple class image classifier code using tensorflow, keras and python

# Prerequisites

* Install python by downloading from https://www.python.org/downloads/ based on system requirements

* Instll tensorflow using command pip install tensorflow in command prompt

* Editor used : Anaconda, you can use any of the editor as per your interest.

# Installing libraries
  * Run following command in your command prompt for installing required libraries
    * os --> pip install os
    * numpy --> pip install numpy
    * pickle --> pip install pickle
    * matplotlib --> pip install matplotlib
    * keras --> pip install keras
    * layers --> pip install layers

# Preparing a Dataset

  This dataset contain total images of 2134 images, in which around 70% of images are used for training purpose, 22% of images
  used for validation data and remaining 8% is used for testing purpose.

  * In this example we have used 6 classes namely [cats, dogs, salman, messie,sharukh, syed]

  * The flow of dataset is as follows.
  
      * First creat a main directory (directory name)  say directory name  = Dataset
  
      * Under main directroy there are 3 sub directories named testset, validation , train

      ![dataset](https://user-images.githubusercontent.com/56253081/66737339-5249a880-ee89-11e9-86a1-08f92f6816b6.jpg)


      * Now under each subdirectories add class directories like cats, dogs, salman, messie, sharukh, syed
      * Below is the example of test directory same should be followed for validation and testset

       ![class](https://user-images.githubusercontent.com/56253081/66737343-54ac0280-ee89-11e9-8c62-1adfa772cf9f.png)

      * Now under each class add images.

     ![images](https://user-images.githubusercontent.com/56253081/66737350-5a094d00-ee89-11e9-9d83-17dc2019b290.jpg)
  
  
  # Steps to be followed after preparing dataset
  
  * Load the dataset using python by assigning each directory to train , validation and test directory
  * Import transfer learning model, in this case it is VGG16
  * Creat weights and features using VGG16
  * Create Convolutional Neural Network code
  * Compile and Fit the model
  * Save the trained model
  
  # Loss function used:
  
  ## Categorical Crossentropy
  
   * Its clear that we are using multiple classess so we are straight away diving into multiclass classification functions.
  
   * Reason : 
      * No limitation on how many classes our model can classify (we can use n number of classes to classify)
      * This function require that output layer is configured with n number of nodes, one for each class here we have 6 classes.
      * Soft max activation in order to predict the probability of each class

  ## VGG16 pre-trained model architecture
  
      ![VGG16_architecture](https://user-images.githubusercontent.com/56253081/67145614-74d02d00-f2a0-11e9-99d1-4d1ee64f3f36.png)
 
  # Testing the model
  
  * Open trained model using load_model function
  * test the model for the images dedicated for testing in the testing folder
  * Evaluate the accuracy
  
  # Output of the testing model
  
    * 8% of our test set which is not trained is inducted into prediction.
    * output will be in the form of array, in which indexes define the classess.
    * Example say a image is predicted as cats the array will be [1,0,0,0,0,0], and if it is messi [0,0,1,0,0,0]
    * Please read indexes accordingly [cats, dogs, messi, salman, sharukh, syed]
    * Here the accuracy has been rounded to a integer instead of float.
    
  
# Accuracy Details:
  * Training:
      * accuracy : 83.7 %
      * loss : 0.475
  * Validation :
      * accuracy : 81 %
      * loss : 0.61
  * Testing:
      * testing_accuracy : 86%
      * lable_accuracy : 68%




