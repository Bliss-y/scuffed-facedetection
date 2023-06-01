import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

screen_width = 1280
screen_height = 720

stream = cv2.VideoCapture(0)

frame_num = 0
while(True):
    (pic, frame) = stream.read()
    if not pic:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)
    face = 0
    for (x, y, w, h) in faces:
        face+=1
        color = (0,255,255)
        stroke = 5;
        name = f"C:\\general\\ai2\\frames\\frame_{frame_num}_%d.jpg"%face
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    cv2.imshow('image', frame)
    key = cv2.waitKey(500) & 0xFF
    frame_num+=1
    if key == ord('q'): break

stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
500