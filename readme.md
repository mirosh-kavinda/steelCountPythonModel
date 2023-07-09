The code and the algorithm are for non-commercial use only.

Paper:" Automated Steel Bar Counting and Center Localization with Convolutional Neural Networks"

Author: Zhun Fan, Jiewei Lu, Benzhang Qiu, Tao Jiang, Kang An

Date: December 16, 2018

Version: 1.0

Copyright(c) 2018, Benzhang Qiu

----------------------------------------------------------

Notes:

(1)Operating environment: Linux 16.04, tensorflow 1.8.0

(2)Dependent package: scikit-image 0.14.0

		      tensorflow-gpu 1.8.0

		     	

		      numpy 1.14.5

---------------------------------------------------------

The architecture of folder:

	-image                      the folder of test image

	-model                      the model of CNN

	-steel_bar_detection.py     the source code of steel bar detection


This Python script appears to be a computer vision algorithm for steel bar detection in images. It uses a convolutional neural network (CNN) for candidate center detection and performs clustering to group the candidate centers. Finally, it shows the results by visualizing the candidate centers, bounding boxes around the clustered centers, and the final clustered centers.

Here's a breakdown of the main components and steps in the script:

The script imports necessary libraries, sets the TensorFlow log level to suppress warnings, and defines some helper functions.

The detection function takes an original image and a stride value as inputs. It uses a pre-trained CNN model to detect candidate centers of steel bars in the image. The image is divided into patches, and each patch is fed through the CNN to obtain a classification result. The function returns a matrix of candidate centers.

The clustering function takes the matrix of candidate centers and a distance threshold as inputs. It performs clustering by calculating the Euclidean distance between each pair of candidate centers. Centers within the distance threshold are grouped together. The function returns a list of clusters, where each cluster contains the indices of candidate centers belonging to that cluster.

The center_clustering function takes the matrix of candidate centers and the list of clusters as inputs. It calculates the mean coordinates for each cluster to obtain the final centers of steel bars. The function returns a matrix of final center coordinates.

There are three "show" functions that visualize the results using OpenCV. They display the original image with candidate centers marked in red, the original image with bounding boxes around clustered centers marked in green, and the original image with final clustered centers marked in red.

The main part of the script loads an input image, sets the stride and distance threshold parameters, and calls the detection, clustering, and center_clustering functions to obtain the final clustered centers. Then, it calls the "show" functions to display the results.

Please note that this script relies on some external files such as the pre-trained CNN model and the input image. Make sure to have the necessary files in the correct locations and modify the script accordingly to run it successfully.