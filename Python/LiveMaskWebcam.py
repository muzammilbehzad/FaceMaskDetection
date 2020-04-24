# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%                                               %%%%%
# %%%%%          BISMILLAH HIRRAHMA NIRRAHEEM         %%%%%
# %%%%%                                               %%%%%
# %%%%%         Programmed By: Muzammil Behzad        %%%%%
# %%%%% Center for Machine Vision and Signal Analysis %%%%%
# %%%%%              University of Oulu               %%%%%
# %%%%%                 Oulu, Finland                 %%%%%
# %%%%%                                               %%%%%
# %%%%%        Email: muzammil.behzad@oulu.fi         %%%%%
# %%%%%                                               %%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# import stuff
import cv2
import keras
import numpy as np

# restore the pre-trained model for mask prediction
mask_model = keras.models.load_model('quarantine_face_mask.h5')

# initialize camera capture
cap = cv2.VideoCapture(0)

# create the haar cascade for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# capture frame-by-frame
	ret, frame = cap.read()

	# some operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in camera's current image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.5,
		minNeighbors=5,
		minSize=(30, 30)
	)

	# display the information accordingly
	print("Found {0} face(s)!".format(len(faces)))

	# draw a rectangle around the faces
	for (x, y, w, h) in faces: # run for all the found faces
		frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 1)
		current_image = np.array(frame)
		
		# predict if the current face has a mask
		model_prediction = mask_model.predict(current_image.reshape(-1,160,160,3)) == mask_model.predict(current_image.reshape(-1,160,160,3)).max()
		my_result = np.sum(mask_model.predict(current_image.reshape(-1,160,160,3)), axis = 0)
		if my_result[0] >= my_result[1]:
			display_string = 'With Mask'
		else:
			display_string = 'Without Mask'
		cv2.putText(frame, display_string, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 1)

	# display the resulting frame
	cv2.imshow('Webcam Live Face Mask Detection | Muzammil Behzad', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
