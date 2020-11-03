# License-Plate-Recognition

Licence plate recognition is used in car licence plate identification system where the system tries to identify the character on the number plate.

The project consists of set of images (Normal and HDR quality) and annotation file (dataset.csv)

In order to read car plates, first detected characters using connected contours then padding and reshaping to have the same input shape and finally use a convolutional neural network for prediction.

## Folder Structure:
*	Image_Dataset – All images of normal and HDR quality, dataset is randomly divided in 80:20 ratio.
      * Train – 80% images(519) of normal and HDR quality.
      * Test – 20% images(133) of normal and HDR quality.
*	Dataset.csv – CSV file containing annotations of all the images.

## Training Steps:
1.	The dataset is divided into train and test. [Code](https://github.com/AlankritaSrivastava/License_Plate_Recognition/blob/master/resize.py)
2.	Each license plate image is resized to a single width and height ratio, by maintaining the aspect ratio.

    ![image](https://user-images.githubusercontent.com/73849427/97909844-d873ae00-1d6e-11eb-9b57-2d0f34710247.png)
 
3.	Input image is a noisy image. Filtered image with a 5x5 gaussian kernel to remove the noise, then applied Otsu thresholding. Thresholding and Gaussian kernel is applied on the resized images for preprocessing to remove the noise so as to make the image easier to analyze.
 
    ![image](https://user-images.githubusercontent.com/73849427/97909909-f6411300-1d6e-11eb-9567-6aacf140b060.png)

4.	All the characters are extracted using contours method and the cropped characters are saved in external folder.
  
    ![image](https://user-images.githubusercontent.com/73849427/97909944-05c05c00-1d6f-11eb-87dd-5eba746c54b2.png)

5.	Using the annotation file (dataset.csv), cropped images for the specific number plate “crop_h1/I00001.png” are labelled with the corresponding annotation “6B94558”.
      
    ![image](https://user-images.githubusercontent.com/73849427/97910007-1c66b300-1d6f-11eb-9da3-7e48ac9d7bfd.png) ![image](https://user-images.githubusercontent.com/73849427/97910024-21c3fd80-1d6f-11eb-9b8c-e110b87de139.png) ![image](https://user-images.githubusercontent.com/73849427/97910031-25578480-1d6f-11eb-8b8b-3d4861513359.png) ![image](https://user-images.githubusercontent.com/73849427/97910042-27b9de80-1d6f-11eb-94bb-7613305d570f.png) ![image](https://user-images.githubusercontent.com/73849427/97910050-2b4d6580-1d6f-11eb-88c6-002c36c33825.png) ![image](https://user-images.githubusercontent.com/73849427/97910058-2dafbf80-1d6f-11eb-8961-4ebfedb301c1.png) ![image](https://user-images.githubusercontent.com/73849427/97910072-33a5a080-1d6f-11eb-85cd-a35b76752213.png)           

6.	Individual characters are saved in the specific folder structure.

    ![image](https://user-images.githubusercontent.com/73849427/97910133-4f10ab80-1d6f-11eb-9a5f-5784b856494c.png)


7.	Using Tensorflow framework and preloaded dataset MNIST (which only has digits) has been extended with the cropped characters and it’s folder labels.

   
8.	The aggregated dataset (MNIST + Provided Dataset) has been trained using following architecture:

    ![image001](https://user-images.githubusercontent.com/73849427/97909163-c80f0380-1d6d-11eb-80ab-0d92d67d833d.png)
 

9.	The loss is 0.0197 and the accuracy 99% for 10 epochs having steps of 62107.

    ![image002](https://user-images.githubusercontent.com/73849427/97909689-a4988880-1d6e-11eb-9820-7b55632cbe59.png)




## Prediction Steps:
1.	Each license plate image is resized to a single width and height ratio, by maintaining the aspect ratio.

    ![image](https://user-images.githubusercontent.com/73849427/97910279-8aab7580-1d6f-11eb-957a-4baffd1403e1.png)

2.	Input image is a noisy image. Filtered image with a 5x5 gaussian kernel to remove the noise, then applied Otsu thresholding. Thresholding and Gaussian kernel is applied on the resized images for pre-processing to remove the noise so as to make the image easier to analyze.

    ![image](https://user-images.githubusercontent.com/73849427/97910286-8d0dcf80-1d6f-11eb-811b-d00de4d69c22.png)  

3.	All the characters are extracted using contours method and the cropped character are predicted using the trained model. The above steps are iterated for all the test images.

    ![image](https://user-images.githubusercontent.com/73849427/97910296-8f702980-1d6f-11eb-996a-55efb0355f02.png)

4.	The results are aggregated in the dataframe which is saved in excel format. [Open Result](https://github.com/AlankritaSrivastava/License_Plate_Recognition/blob/master/Results.csv)

## Accuracy Metrics:
The actual and predicted labels are compared and out of 133 test images, 66 images were correctly predicted (all the characters in the given sequence).
## Accuracy = (66/133)*100 ~ 50%
