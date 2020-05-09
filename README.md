# covid-xray-prediction
Introduction
- 
The goal of this project is to use machine learning to diagnose patients with COVID-19, viral pneumonia, and backterial pneumonia from images of chest X-rays. Early in the epidemic, physicials were actually diagnosing cases of coronavirus using X-Ray and CT images. Since COVID X-Rays are frequently confused with ordinary pneumonia, this becomes a difficult classification problem. 

Preprocessing
-
I first started with exploratory data analysis to understand the data that was provided. This dataset is a multiclass classification problem, with distributions as follows: 

![Distribution of Classes](/images/classes.png)

Because of the large discrepancy between the Covid images and the rest of images, it may be beneficial to augment the Covid data to increase the samples. Analyzing the dataset, I found that most of the images are different sizes. This would require resizing the data to a specific size to specify the features. I used an image size of `250x250`, but increasing this value would most likely support results. Below are a few of the samples from each class, with each row containing a different class:
![Before Preprocessing](/images/before_pre.png)
To the untrained eye, these images are hard to distinguish from each other. I attempted to use a method of [bone suppression](https://arxiv.org/pdf/1811.02628.pdf), which has historically performed well; however, the Generative Adverserial Network used to train the bone suppression model was trained using X-Rays that looked very different from the data provided. However, it may be useful to look into in the future.

![Bone Suppression Example](/images/bone_sup.png)

Furthermore, some are taken from different angles and have black borders. I used a few variations of color encoding and cropping, to emphasize the non-black aspects and of the image and set the close to black pixels to be completely black. This was able to help the accuracy of the model. 

![After Preprocessing](/images/before_pre.png)

After testing the model, I decided that no data augmentation was necessary to achieve good accuracy on the problem. In addition, data augmentation may overfit our data, so it was not used. Other preprocessing steps include creating normalizing the pixel values to be between $0$ and $1$, and converting the greyscale images to RGB. Furthermore, the categorical labels were converted into one hot encoded numerical labels for the model to train and test on. 

The Model
-
I chose to use Convolutional Neural Networks, and partially transfered from VGG16. VGG16 is pretrained on ImageNet, and will help speed up the model as well as understand general "ideas". From there, I added layers to further fine tune the model.  This is because first few layers capture general details like color blobs, patches, edges, etc. Instead of randomly initialized weights for these layers, it would be much better if you fine tune them. I attempted to use [Depthwise Convolution](https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution) for various depths, as well as batch size = 10. This method also performed well in similar classification tasks. Below are the amount of parameters of our model. 

`Total params: 119,597,380
Trainable params: 104,873,476
Non-trainable params: 14,723,904. `

Analysis
-
Below is the training sequence using early stopping on the validation loss to ensure we don't overfit.

![Training the Model](/images/epoch.png)

Early stopping wasn't employed because it had a patience of 4; however, we use the model trained at epoch $7$ to perform our prediction. This can be better visualized by viewing the accuracy and loss. 

![Model Accuracy](/images/model_acc.png)
![Model Loss](/images/model_loss.png)

In a clinical setting, this accuracy means that we should be wary of the results of employed by this model. Because neural networks are generally black boxes, the interpretability of this model suffers, this is the tradeoff for good performance with relatively simple code. However, in a clinical setting, this model may be useful in helping clinicians validate diagnoses, and may serve as a base for more complex models. There may be potential in understanding the features that are important using saliency maps, but I was unable to explore this method. 


Conclusion
-
Throughout the project, I encountered a few bugs and issues. Importing VGG16 weights and adding them as an initial weighting parameter was fairly tricky. There were a few lines of code that the model required before flattening, and it was causing issues for a while. A bug that I had made was iterating through the test set, however indexing my training set; this led to an abysmal score on Kaggle, and I considered scraping the code until I found the error. Other than that, this was a fairly simple implementation of the model without much rigorous testing. 
