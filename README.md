# DeepFaceRecognizer
Face Recognition using Deep Convolutional Nets

Convolutional neural networks are the current state-of-art architecture for image classification. 
They’re used in practice today in facial recognition, self-driving cars, and detecting an object.
A CNN typically consists of 3 types of layers:
  1.	Convolution Layer
  2.	Pooling Layer
  3.	Fully Connected Layer
For more clear concepts you can further read 
https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/

<b>Step 1: Install Dependencies</b>
    -	First Install Python (I use python 3.4)
    -	Then install scipy, dlib, tensorflow, tflearn, scikit-image
Using pip install –user scipy dlib tensorflow tflearn scikit-image

<b>Step 2: Approach</b>
We will use deep neural network to produce face encodings using the following steps. 
When we pass two different images of the same person the network should return closer output (numbers) 
for both images and when we pass images of two different people, it should return two different outputs. 
This means, the NN should be trained to automatically identify features of faces and calculate the encodings. 
In this example we will not build our neural network to train every image, rather we will use trained model through dlib.
dlib gives us face encodings, when we pass in the image of someone’s face and compare encodings of faces from different 
images which will tell us if someone’s face matches with anyone we have.

<b>Setp 3: Setup your project</b>
    Main_Folder
      |- images
      |- testimages
      |- dlib_face_recognition_resnet_model_v1.dat
      |- shape_predictor_68_face_landmarks.dat
      |- reconizer.py
      |- find_match.py
Here 
  -	put jpg images of the known faces to ‘images’ folder (a single instance of each face with the name of the person.
  E.g. Kiros.jpg, Gereziher.jpg, Haftom.jpg, Tesfu.jpg).
  -	Then put a multiple instance of jpg images you want to recognize inside ‘testimages’ folder
  -	Then download and extract the pre-trained model of dlib libraries and put inside the main folder. 
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    
<b>Step 4: Run find_match.py and you will get output that looks like</b>
      testimages/image_0092.jpg==> 			Kiros
      Please change image: testimages/ image_0036.jpg - it has 0 faces; it can only have one
      testimages/image_0006.jpg ==> 		Gereziher
      testimages/image_0006.jpg==> 			Haftom
      testimages/image_0116.jpg==> 			Tesfu
      testimages/image_0093.jpg==> 			Tesfu
      testimages/image_0003.jpg==> 			Haftom
(Deep neural network for face recognition using our own CNN Model…. Coming soon).
