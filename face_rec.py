import cv2, os
import numpy as np
from PIL import Image


cascadePath = "/Users/saracamnasio/Dropbox/Research/code/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
	images = []
	labels = []
	for image_path in image_paths:
		image_pil = Image.open(image_path).convert('L')
		image = np.array(image_pil, 'uint8')
		nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
		faces = faceCascade.detectMultiScale(image)
		for (x, y, w, h) in faces:
			images.append(image[y: y + h, x: x + w])
			labels.append(nbr)
			cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
			cv2.waitKey(50)
	return images, labels

path = '/Users/saracamnasio/Pictures/yalefaces'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()
recognizer.train(images, np.array(labels))

image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
for image_path in image_paths:
	predict_image_pil = Image.open(image_path).convert('L')
	predict_image = np.array(predict_image_pil, 'uint8')
	faces = faceCascade.detectMultiScale(predict_image)
	for (x, y, w, h) in faces:
		nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
		nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
		if nbr_actual == nbr_predicted:
			print "Individual {} matched with {}. Correctly Recognized with confidence {}".format(nbr_actual, nbr_predicted, conf)
		else:
			print "{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted)
		cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
		cv2.waitKey(1000)
cv2.destroyAllWindows()

#importing from FACEBOOK API:
# import json
# import requests
# token = 'EAACEdEose0cBAOEFNnbjZCgxxki7rLRyEBeQvz5DjUUoHqZC3yYLEWOz7JKAIfx1SYj48UymuECPjfEwo68dnPevBbOo6nwCbeWR9bk10iMeMEh2f1RCQbTXhAXuwrSQN3GFZAbJyaFc3QpAzvl3ok0tXyRfVX2XylDSjwH0RcYBmodzUtohnQAvT9Y1KBdYuSVsdlbVAZDZD'
# api_url = "https://graph.facebook.com/v2.1/"
# params = {'access_token' : token}

# #https://graph.facebook.com/[uid]/albums?access_token=[AUTH_TOKEN]
# call = "me/friends?fields=picture.height(500).width(500),name,location,albums{id,name,type}"
# response = requests.get(api_url + call, params=params)
# r = (json.loads(response.content))




