# CMIL-Assessment

## Running the project

This project was built within Anaconda Environment.
The version of Python used is 3.11.5. The modules necessary are os, pandas, numpy, tensorflow, sklearn, matplotlib

The model is available for download via Google Drive @ https://drive.google.com/file/d/1dJecfWYHqSX5kjef3b7353aK1HeYw7BD/view?usp=sharing

Once run ensure that the data is formatted in the structure as follows:
```
(Project Folder)
-(Public)
--(globally_sclerotic_glomeruli)
--(non_globally_sclerotic_glomeruli)
--public.csv
-Evaluation.py
-CMIL_model.py
-glomeruli_model.h5
```

Run the evaluation.py file to run the project. An evaluation.csv file will be created if the program compiles successfully


## Background

This project was done to detect whether sclerotic glomeruli were global (GSG) or non global (N-GSG). The data was provided by the CMIL Lab in the format of two folders and a csv file. One folder contained the GSG images and the other contained the NGSG images. The csv file had the file name of the image along with its ground truth value.

## Approach

I wanted to approach this binary classification task using a CNN. A CNN has been proven to be good at image classification tasks and it is a highly customizable and intuitive model.

The first step was to format the data. I set up the above folder structure and defined the proper paths for each directory. While doing this I took a look at the images and did some research on sclerosis. I also looked into glomeruli and their function within the body. When doing research I noticed that we have many more images in the N-GSG folder than in the GSG folder. This meant we had a skew in data within our dataset. This changed my approach to account for this skew when setting up the network.

The next step was pre-processing. Pre-processing the images to make them appropriate for the model meant defining the labels as well as standardizing each image. Currently, the images were all in various sizes. In order to standardize them I chose 244 by 244 as the image size. To load the files, I read each one separately, and made an array with the images, and another with the labels. The labels were extracted depending on the folder they were in. After all of this, I also scaled down each pixel value from (0,255) to [0,1]. This normalizes the images, making it easier to process them. 

Next, I split the dataset into training and testing. This was done using the scikit tools "train_test_split" command. I chose 80/20 as my split as a good starting point that gave me enough data. Before starting the training process, I also calculated the class weights of the data. This simply deals with the aformentioned skew in images we had. By assigning more weight to the GSG images it ensures that the model is able to learn at similar rates for both classes. The class weights, if curious were: [0.60958179, 2.78140097]. 

Now we can start training and building the model. The model I chose was a CNN. I have a standard template I used before for image classification tasks. This is a standard 10 layer sequential CNN. The model details can be examined more in depth within the code, as the separate layers are defined. The motivation for using this standard model was just to see how complex the task is and if it was necessary to optimize the various components. Using a standard relu function for the activation layers, and a 50% drop out before combining the image, I was able to get decent training values. I ran 30 epochs with a calidation split of 8/20 and shuffling the data each epoch. The graph is shown below.

![d3676116-ca08-4516-a363-901dab2bc91f](https://github.com/i-mohammed2/CMIL-Assessment/assets/106894101/9efbf033-3edf-4c8e-b6a9-e574f32e1146)

![a05441ac-733d-4bd0-be24-8357b44afe43](https://github.com/i-mohammed2/CMIL-Assessment/assets/106894101/5db793cd-cd49-49a4-8a65-7bffd6a89f1d)

The validation data is also shown above. We can see that we were able to achieve a high training accuracy of about 99% by the end of training. The validation accuracy reached near 90%. This tells us that our model isn't over fitting as the lines aren't identical. In addition, the accuracy is fairly high indicating that the model is able to differentiate between N-GSG and GSG images. The last step is evaluating our model on the test data. Now with the loss, we see that the validation loss was slowly inncreasing. This is an indicator that my model is infact overfitting. However the switch from decreasing to increasing tells me that this is likely because of the small dataset available. The model has probably memorized the data set rather than actually pinpointing differences in GSG and NGSG. This hapened around the 5 epoch mark. A future goal would be to add data augmentation into the pipeline. This would hopefully complicate the dataset a bit more to help decrease this chance of overfitting. The mdoel could probably be modified as well to fix this issue.

I colleected some metrics to evaluate this model after running it on the test set. The test accuracy was a 90.97%.

```
     p    n
tp [[902  24]
tn [ 80 146]]
              precision    recall  f1-score   support

           0       0.92      0.97      0.95       926
           1       0.86      0.65      0.74       226

    accuracy                           0.91      1152
   macro avg       0.89      0.81      0.84      1152
weighted avg       0.91      0.91      0.90      1152
```

We can see here that the F1 score is fairly high. This indicates that our model is doing pretty well. We see lower values for GSG predictions, which is expected due the skew in the data set. Overall, we had a high number of images that were correctly identified. Our model does seem to have an understanding of differentiating between N-GSG and GSG. Another graph we can look at is the ROC curve, shown below.

![cbaaa364-7293-405b-9000-b72fd88ddf11](https://github.com/i-mohammed2/CMIL-Assessment/assets/106894101/ac8f6330-5ae1-42d5-801f-221db17b5897)

The area under the curve is 0.81 which is better than a 50/50. We can see that the model still has a lot of room for improvement, but it can get there through optimization and increasing the data set size.

![92e40fdb-9644-49c6-8e6e-2a1f75d8a7ae](https://github.com/i-mohammed2/CMIL-Assessment/assets/106894101/b6dca763-f31e-47fb-b154-5e239db42105)

Here I've attached an image showcasing some of the predictions, as well as their ground truth value.

This was the entire process taken when completing this task. More details can be given if necessary, and there are some bugs needed to be fixed. I wasn't able to conserve the initial file names, which may serve as an issue later on, however, this should be easily fixed with some more work.
