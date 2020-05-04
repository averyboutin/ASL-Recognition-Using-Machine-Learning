from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

# load model
model = load_model('./model/trained_model.h5')
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # press escape to quit
        print("Exiting...")
        break
    elif k % 256 == 32:
        # press space to take picture
        img_name = "frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} saved!".format(img_name))

        # resize screenshot
        src = cv2.imread("frame_{}.png".format(img_counter))
        dsize = (100, 100)
        resizedImage = cv2.resize(src, dsize)
        resizedImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

        # predicting image
        x = image.img_to_array(resizedImage)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict_classes(images)
        blah = pred.tolist()
        print(blah)
        pred_label = labels[blah[0]]
        print(pred_label)
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
