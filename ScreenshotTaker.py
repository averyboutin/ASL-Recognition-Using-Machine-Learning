import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

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
        # runs screenshot through model


        img_counter += 1

cam.release()
cv2.destroyAllWindows()
