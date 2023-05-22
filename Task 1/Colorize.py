import numpy as np
import cv2

prototxt_path='Models/colorization_deploy_v2.prototxt'
model_path = 'Models/colorization_release_v2.caffemodel'
kernel_path = 'Models/pts_in_hull.npy'
image_path = 'girl.jpg'

#neural network loading
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

#convolute the kernel into size 1 x 1
points = points.transpose().reshape(2, 313, 1, 1)

net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# LAB -> L = Lightness A* and B* are color values

#reading the black and white image
bw_image = cv2.imread(image_path)

#This will help get a value between 0 and 1
normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

#model is trained to take images from 224x224 so resize the image to the value needed
resized = cv2.resize(lab, (224,224))

#Now split the channel to the resized image
L = cv2.split(resized)[0]
L -= 50

#Now to get the color channels take L as an input and transpose the input
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))

#Resizing the image to the original size
ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
#now we get the same lightness level back
L = cv2.split(lab)[0]

#Combining lightness with colors
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis = 2)

#Converting the image back from LAB to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

#Now we scale the image we normalized back to normal
colorized = (255.0 * colorized).astype("uint8")

#To display the image
cv2.imshow("BW Image", bw_image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()