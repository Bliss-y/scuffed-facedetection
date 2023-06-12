import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Union
import math
import cv2
import numpy as np

model_path = './blaze_face_short_range.tflite'


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
	base_options=BaseOptions(model_asset_path='./blaze_face_short_range.tflite'),
	running_mode=VisionRunningMode.IMAGE)

detector = vision.FaceDetector.create_from_options(options)

# annotated_image = visualize(image_copy, face_detector_result)

def _normalized_to_pixel_coordinates(
	normalized_x: float, normalized_y: float, image_width: int,
	image_height: int) -> Union[None, Tuple[int, int]]:

	def is_valid_normalized_value(value: float) -> bool:
		return (value > 0 or math.isclose(0, value)) and (value < 1 or
													  math.isclose(1, value))

	if not (is_valid_normalized_value(normalized_x) and
		  is_valid_normalized_value(normalized_y)):
	# TODO: Draw coordinates even if it's outside of the image bounds.
		return None
	x_px = min(math.floor(normalized_x * image_width), image_width - 1)
	y_px = min(math.floor(normalized_y * image_height), image_height - 1)
	return x_px, y_px

from datetime import datetime
def visualize( model, image,detection_result) -> np.ndarray:
	annotated_image = image.copy()
	height, width,_ = image.shape

	for detection in detection_result.detections:
	# Draw bounding_box
		bbox = detection.bounding_box
		cv2.imwrite("./right.jpg",annotated_image[bbox.origin_y:bbox.origin_y+bbox.height, bbox.origin_x:bbox.origin_x+bbox.width])
		res = model.predict('./right.jpg')
		cv2.rectangle(image,(bbox.origin_x,bbox.origin_y), (bbox.origin_x+ bbox.width, bbox.origin_y + bbox.height), (1,255,0),3)
		if(res != 'Nishedh'):
			name = "./mistakes/mistakes" + str(datetime.now().minute) +str(datetime.now().second)+".jpg"
			cv2.imwrite(name ,annotated_image[bbox.origin_y:bbox.origin_y+bbox.height, bbox.origin_x:bbox.origin_x+bbox.width])
		print(res);
from model import model
m = model.load('./oopmodel/', 'model')


import cv2

def cap():
	stream = cv2.VideoCapture(0)
	mistakes = 0
	frame_num = 0
	while(True):
		(pic, frame) = stream.read()
		if not pic:
			break
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
		res = detector.detect(mp_image)
		visualize(m, frame, res)
		cv2.imshow('image', frame)
		key = cv2.waitKey(200) & 0xFF
		if key == ord('q'): break
		

	stream.release()
	cv2.waitKey(100)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	print(f'{mistakes=}, {frame_num=}, accuracy={(frame_num -mistakes)/frame_num}')

cap()

