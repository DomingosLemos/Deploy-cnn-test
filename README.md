# Identifying Meals – DLNN Group Project
## Abstract
The purpose of the project is to classify pictures of food according to 101 given food categories. The methodological approach is based on a Convolutional Neural Network model that analyses a dataset of pictures of gastronomic dishes divided into 101 categories. The best results were achieved using the inception model with an accuracy of 80% on the test datasets.
## Introduction
A technology that is becoming more and more present in social media is the image recognition. It can be used to identify people, products, brands and places in pictures with an insignificant margin of error. This project aims to recreate this technology, in a lower scale
The methodological approach designed for the project consists in 5 blocks:

![alt text](https://github.com/DomingosLemos/Deploy-cnn-test/blob/main/static/Fluxo.png)

*	The step 1 of the project consists in preparing data to be train the model. In this step it will be detailed the source of the dataset, the import procedure, and dataset partition.
*	The 2nd stage is the model development. Two types of models will be developed: one baseline model designed from zero and one pre-trained model identified in Kaggle.
*	In the 3rd stage, the most accurate model from each type will be selected.
*	The final step is to develop the deployment of the final model.


## 1. Data Preparation

### 1.1 Dataset
The dataset used is Food Images (Food 101) available on Kaggle1. The dataset includes 101 categories of food, each category with 1000 pictures of food, totaling 101000 food images. 
### 1.2 Import Procedure
To optimize the access to the dataset and ease the collaborative work on the notebook, Google Colab was chosen to develop the model. The dataset was imported to google drive and then imported to the Colab notebook, using drive from the google.colab library.
### 1.3 Data partition
For later analysis, each of the 100 categories of the dataset was broken in 3: train (70%), validation (20%) and test (10%). The library splitfolders was used for this purpose.

## 2.	Model Development

For this project, two types of models were developed. The first type is a baseline model made from scratch. For the baseline model, several parameters and layers were tested with the aim of increasing the accuracy of the achieving an acceptable model. For the pre-trained model, several existing models available in Kaggle were analyzed and one was selected.
## 2.1	Baseline model
For the baseline model the chosen approach consists in designing an initial simple model and successively add layers and configurations until reaching the highest accuracy possible. 
The initial model was developed using 4 layers. Sequential library from tensorflow was used. The input layer consists in 2 layers: one 2D convolutional layer with a relu activation to help producing a tensor of outputs and a pooling layer in order to reduce computation complexity. It was decided to use MaxPooling instead of AveragePooling. In what regards image detection, MaxPooling2D identify high color changes used to highlight more important traits of the images such as lines, curves and parabolas that characterize the image. 
The hidden layer is composed by 2048 output units and relu activation and the output layer has 101 units (the number of categories) and activation softmax to ensure that the probability of all results sum 1. 

![alt text](https://github.com/DomingosLemos/Deploy-cnn-test/blob/main/static/figure1.png)

The model was tuned using Adam optimizer, crossentropy loss and early stop as callback. After 14 epochs, the model resulted in an accuracy of 7,97% in the validation dataset. Using this model as a base, several changes were applied resulting in several different models. Some of them can be observed in table 1.
Since there is not a specific criteria or formula to identify the optimal number of layers, additional layers were introduced until the accuracy of the models did not increased. Firstly, additional Convolutional layers were introduced one by one, followed by dense layers. It was observed that adding more dense layers did not improve the accuracy of the model. 
In addition, neural networks tend to overfit a training dataset very quickly. By using dropout, that randomly drop out nodes during training, it is possible to reduce this effect. Thus, the next step to develop the baseline model was the introduction of dropout. Several dropout percentages were tested, achieving the highest accuracy with 30%. The Kernel Constraint limits the weight vectors eliminating the possibility of blowing up. This constraint was added, however the results remained quite similar.
Additional settings were changed and tested concerning the optimizer, such as changing the learning rate. Furthermore, the optimizer was changed from Adam to RMSProp, which provided better results. Several data augmentation techniques were tested, all of them providing not promising results.

| Model | Tuning | Accuracy |
| ----- | ------ | -------- |
| 2     | + 2 extra input layers: Conv2D and MaxPoolingD | 11.56%  |
| 3     | + 2 extra input layers Conv2D and MaxPoolingD | 17.34%  |
| 4	    | + 2 extra input layers Conv2D and MaxPoolingD	| 0.94% |
| 5	    | + 1 Extra Hidden Layer (Dense 4096)	| 1.09% |
| 6	    | + 1 Extra Hidden Layer (Dense 4096)	| 0.78% |
| 7	    | Add dropout rate (0.2) and remove added hidden layers | 22.97% |
| 8	    | Increase dropout rate (0.3) | 23.59% |
| 9	    | Combine Kernel Constraint with Dropout Rate (0.3) | 21.88% |
| 10	  | Include Learning Rate on Optimizer | 20.94% |
| 11	  | Change optimizer to RMSprop | 23.13% |
| 12	  | Increase Batch Size and Steps per epoch | 26.25% |
| 13	  | Change padding to 'same' | 25.02% |
| 14	  | Include Data Augmentation techniques | 1.61% |
| 15	  | Additional Data Augmentation techniques | 16.32% |

Model 12 has the highest accuracy level, 26,25%

## 2.2	Pre-trained model
To complement the followed approach, 3 pre-trained models were also tested: VGG16, Inception_V3 and Resnet50.
Since the dataset is small, only includes 700 images for each category, Image Augmentation was used to increase the size of the train dataset (validation set remained equal). Since there is no need to train all the layers, they were made non_trainable.

### 2.2.1 Inception V3 Model
Smaller steps were tried per season, so as not to show all the images of the training source to the model for each epoc, but the results observed were slightly worse, so it was chosen to present all the images in each one and used only 30 epocs because of the time needed for processing, since it burst with larger values.
The size used for the shape of the images was 299x299x3, which complies with almost the entire park, but also because it is the standard size of this pre-trained model.
Model that approximates a sparse CNN with a normal dense construction. Since only a small number of neurons are effective the number of convolutional filters was kept small. After running, the accuracy of this model was 79.33%, the highest achieved on this project. 
 
### 2.2.2 Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16 Model)
As training parameters, it was chosen to split the batch size into 20 and apply a shape to the 224x224x3 images, since the overwhelming majority of the dataset has larger images, trying to make the most of the most information and not jeopardizing the GPU memory.
In this model it was used include_top equal to False to not make a full connect on the last layer. The "imagenet" font already prepared for image classification was used for the weights
Layers from a previously trained model were frozen them, so as to avoid destroying any of the information they contain during future training rounds.
A fully connected layer with 512 hidden units and ReLU activation, a dropout rate of 0.5 and a final sigmoid layer for classification. An accuracy of 28,60% was achieved, which was worse than the Inception model. 

### 2.2.3 ResNet50
In this model it was used the same technique used in Inception to make all images known in each epoc, as they have slightly better results.
The Residual Network is used for extremely deep neural networks models with dozens of layers. This model was also applied, and the final accuracy was 56,31%. 

### 2.2.4 Autoencoder
In every stage of a CNN bits of information are rejected to make classification possible. This makes it difficult to pre-train a deep CNN with an autoencoder simply because the loss of information means it can't be reconstructed the input.

## 3.	Model Selection
When observing the models previously developed either the baseline model and its subsequent modifications and the pre-trained models it is possible to  highlight the most accurate from each category. The 12th variant of the baseline model offers the highest accuracy, 26,25% on the other hand, the best pre-trained model is the inception model with an accuracy of 79,33%. Having almost 3 times the accuracy of the baseline model, the inception model was selected for the rest of the project.

## 4.	Deployment
The operating system used for the entire deployment was Windows 10.
A small python app was made in Visual Studio Code, based on the example given in the practical classes.
The Flask library is the one that allowed the creation of a web app with python. 
The application structure is in the following image

![alt text](https://github.com/DomingosLemos/Deploy-cnn-test/blob/main/static/struct_code.png)

Project structure:
*	app.py is the main application code block
*	templates/home.html is responsible for the page layout
*	static is the folder that contains the background image as well as the images loaded, on web, for predict classification
*	notebooks is the folder where we have ower code used in google colab to reach the ideal model

Used tools/Software:
*	python 3.7.9
*	Visual Studio Code
*	Git/Github Desktop
*	Guithub

Guaranteed care:
*	Reshape of images equal to the model
*	Resize of matrix values equal of training

For deployment it was tried to use Heroku, but given the storage limitation per deployment of 500MB, we opted to look for alternatives. Within several attempts, such as cloud foundary, netlify, versel, the conclusion was to create a VM on Google Cloud.
At the end the result is in http://34.122.81.128/

Imagem de entrada:
![alt text](https://github.com/DomingosLemos/Deploy-cnn-test/blob/main/static/GUI_init.PNG)

Imagem de classificação de imagem:
![alt text](https://github.com/DomingosLemos/Deploy-cnn-test/blob/main/static/GUI_predict.PNG)

## Proposed extras that were not used

Autoencoder: In every stage of a CNN bits of information are rejected to make classification possible. This makes it difficult to pre-train a deep CNN with an autoencoder simply because the loss of information means it can't be reconstructed the input. For this reason, autoencoder was not considered for this project.
Ensemble: Since the final model has a much higher accuracy than the others trained, it was decided not to create an ensemble model. The best chance would be to reproduce this best model with slight changes in the hyperparameters to obtain a model with a close accuracy. However, due to the time it would take to train the model, it was opted not to pursue this option.

## Next Steps

Having the model defined and deployed, there are some aspects that can have further improvement. These aspects can be either in terms of modelling and in terms of scope of the model. In terms of modelling, there is room for improvement to increase the accuracy rate. With more time and processing capacity models with more layers and different features can be testes. In terms of scope, the model can be modified to identify the images by meal type (breakfast, lunch, dinner, etc), to identify by the type of cuisine and to indicate whether the food is vegetarian or not.

## References

*Dataset*
1. https://www.kaggle.com/kmader/food41

*Baseline Model*
2. https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu#:~:text=In%20general%2C%20batch%20size%20of,best%20to%20start%20experimenting%20with
3. https://ruder.io/optimizing-gradient-descent/
4. https://keras.io/api/optimizers/rmsprop/
5. https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
6. https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
7. https://stackoverflow.com/questions/43237124/what-is-the-role-of-flatten-in-keras
8. https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
9. https://stackoverflow.com/questions/49922252/choosing-number-of-steps-per-epoch#:~:text=Traditionally%2C%20the%20steps%20per%20epoch,by%202%20or%203%20etc
10. https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu#:~:text=In%20general%2C%20batch%20size%20of,best%20to%20start%20experimenting%20with

*Pre-trained models*
11. https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
12. https://www.kaggle.com/theimgclist/multiclass-food-classification-using-tensorflow
13. https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/

*Deployment*
14. https://www.youtube.com/watch?v=vGphzPLemZE
15. https://tecadmin.net/install-python-3-7-on-centos-8/
16. https://dashboard.ngrok.com/get-started/setup
17. https://medium.com/bhavaniravi/build-your-1st-python-web-app-with-flask-b039d11f101c
18. https://medium.com/techkylabs/getting-started-with-python-flask-framework-part-1-a4931ce0ea13
19. https://github.com/MafSV20180073/deploying-cnn-model
20. https://apoie.me/cursoemvideo
21. https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/


