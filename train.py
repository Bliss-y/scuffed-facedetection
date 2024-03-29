from model import model
m = model.createDataSet('./images/train', './images/test')
m.train(50)
m.saveModel('./oopmodel/', 'model')

# ####################################################
# import json
# dir  ='./oopmodel/'; name = 'model2';
# f = open(dir+name+'/model.json', 'r')
# ResultMap = json.load(f)
# ResultMap['3']
# c = {}
# it = 0
# for i in ResultMap:
# 	c[it] = ResultMap[str(it)];
# 	it+=1
# c
# f.close()
# ####################################################

# import os

# testfolder = './images/wild/'
# total =0
# correct = 'panas_nudes'
# mistake = 0
# for file in os.listdir(testfolder):
# 	total +=1
# 	res = m.predict(testfolder+file)
# 	if res != correct:
# 		print(testfolder + file)
# 		print(res)
# 		print('-'*3)
# 		mistake+=1
# print(f'accuracy {(total -mistake)*100/total}')



# ################################################################

# '''
# WEB CAM
# '''

# import cv2

# def cap():
# 	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
# 	screen_width = 1280
# 	screen_height = 720
# 	stream = cv2.VideoCapture(0)
# 	mistakes = 0
# 	frame_num = 0
# 	while(True):
# 		(pic, frame) = stream.read()
# 		if not pic:
# 			break
# 		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)
# 		face = 0
# 		for (x, y, w, h) in faces:
# 			face+=1
# 			color = (0,255,255)
# 			stroke = 5;
# 			name = "C:\\general\\ai2\\temp\\tmp.jpg"
# 			cv2.imwrite(name, frame[y:y+h, x:x+w])
# 			cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
# 			result = m.predict('./temp/tmp.jpg')
# 			cv2.putText(frame, result, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2, color=0)
# 			print(result)
# 			print()
# 			if(result != 'Nishedh'):
# 				mistakes +=1
# 				# cv2.imwrite('C:\\general\\ai2\\Nishedh\\nishedh_new_%d.jpg'%frame_num, frame[y:y+h, x:x+w])

# 			frame_num+=1
# 		cv2.imshow('image', frame)
# 		key = cv2.waitKey(1) & 0xFF
# 		if key == ord('q'): break
		

# 	stream.release()
# 	cv2.waitKey(100)
# 	cv2.destroyAllWindows()
# 	cv2.waitKey(1)
# 	print(f'{mistakes=}, {frame_num=}, accuracy={(frame_num -mistakes)/frame_num}')

# cap()




