import cv2, os
from main import  detect_faces

path = 'trainningData.yml'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(path)

dataset = ["", "Rajita Ghosal", "Vishal Sinha", "Ashutosh Agarwal", "Aneesh Dixit", "Kshitiz Khatri", "Nihar Chitnis"]

def predict(img) :
	face, rect = detect_faces(img)

	if face is not None :
		for i in range(0, len(face)) :
			#label = face_recognizer.predict(face[i])
			label, conf = face_recognizer.predict(face[i])
			label_text = dataset[label]

			(x, y, w, h) = rect[i]
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
			cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	return img

#video_capture = cv2.VideoCapture(1)
video_capture = cv2.VideoCapture(0)

while True :
	ret, frame = video_capture.read()

	frame = predict(frame)
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q') :
		break
# img = cv2.imread('Dataset/1/1.jpg')
# img = predict(img)
# cv2.imshow('Image', img)
# cv2.waitKey(1000)

video_capture.release()
cv2.destroyAllWindows()