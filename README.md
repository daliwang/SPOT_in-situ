# SPOT_in-situ

This repository contains code and data associated with following publication. 

Title: Autonomous Nondestructive Evaluation of Resistance SPOT Welded Joints
Authors: Jian Zhou, Dali Wang, , Jian Chen, Zhili Feng, Blair Clarson, Amberlee Baselhuhn

Code files:
1.	Train_3DCNN.py: 
Code is designed for training 3D CNN model based on ground truth files and assembled IR welding video datasets. Ground truths are ‘GrdTruhX.PNG’ files, and each one is a ground truth corresponding to one particular IR welding video, which are located in ‘Inputs/Label’. Assembled IR welding video datasets are named as ‘Input_X_(ExaNor).npy’, and each one is assembled from one particular IR video, which are located in the same file of the code. The 3D CNN model is trained by using GPU. The output of the code is a trained 3D CNN architecture saved as model3D.ckpt in the current directory.

2.	Test_3DCNN.py:
Code is developed for testing a trained 3D CNN model based on ground truth files and assembled testing IR welding video datasets. Ground truths are named as ‘GrdTruh.png’, which are located in the directory ‘Ground_Truth’. Then select the ground truth directory based on particular IR welding video dataset, for example, the directory ‘Ground_Truth/Target/E065/000015/GrdTrh_relocated’. F1 scores and nugget diameter prediction results are the output to evaluate the 3D CNN model performance.

3.	Model3DTest.py:
A 3D CNN architecture with customization corresponding to the one described in the paper.

4.	CNN3D_archt2.py:
Another 3D CNN architecture with customization.

5.	Assumbly_3DCNN_input_2.py:
Code for IR welding video frames sampling and assembling as described in the paper to create datasets for training and testing 3D CNN model.

6.	Load_inputs.py:
Code for reading selected frames from particular IR welding videos and conduct normalization.

7.	train.py:
Code for training 2D CNN model. Output is saved as model.ckpt.

8.	model.py:
The designed 2D CNN architecture with customization as described in the paper.

9.	load_data.py:
Code for reading gourd truth files and normalized IR welding video datasets.

10.	test.py:
Code for testing a trained 2D CNN model with F1 scores are used for evaluation metric.

Dataset files:
1.	‘Inputs’ folder:
It includes folders that contain ground truth files (GrdTruhX.png) for specific IR welding video datasets used for training 3D CNN models in this research. The ‘data’ folder inside contains assembled datasets for 3D CNN models based on considered IR welding video datasets.

2.	‘Ground_Truth’ folder:
It includes folders that contain ground truth files (GrdTruh.png) for all available IR welding video datasets and ground truth files (GrdT_X.png) for selected IR video frames.

3.	‘2DCNN’ model:
The trained 2D CNN models based on particular IR welding video datasets.

4.	‘3DCNN’ model:
The trained 3D CNN models based on all considered IR welding video datasets.
