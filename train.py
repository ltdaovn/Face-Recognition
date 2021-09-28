import cv2, os, numpy



def detect_faces(img) :
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	faces = faceCasc.detectMultiScale(gray, 1.3, 5)
	graylist = []
	faceslist = []

	if len(faces) == 0 :
		return None, None

	for i in range(0, len(faces)) :
		(x, y, w, h) = faces[i]
		graylist.append(gray[y:y+w, x:x+h])
		faceslist.append(faces[i])

	return graylist, faceslist

def detect_face(img) :
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	faces = faceCasc.detectMultiScale(gray, 1.3, 5)
	graylist = []
	faceslist = []

	if len(faces) == 0 :
		return None, None

	(x, y, w, h) = faces[0]
	return gray[y:y+w, x:x+h], faces[0]

def data() :
	dirs = os.listdir("Dataset_lfw")


	faces = []
	labels = []

	for i in dirs :
		set = "Dataset/" + i

		label = int(i)

		for j in os.listdir(set):
			path = set + "/" + j
			img = cv2.imread(path)
			face, rect = detect_face(img)

			if face is not None :
				faces.append(face)
				labels.append(label)

	cv2.destroyAllWindows()
	cv2.waitKey(1)
	cv2.destroyAllWindows()

	return faces, labels

faces, labels = data()

#face_recognizer = cv2.face.createLBPHFaceRecognizer()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, numpy.array(labels))
face_recognizer.save('trainningData.yml')

