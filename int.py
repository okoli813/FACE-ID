import cv2

face_harr_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

image = cv2.imread("potrait.jpg")
if image is None:
    print("Error: Image 'potrait.jpg' not found or could not be loaded.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray", gray)

faces = face_harr_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

cv2.imshow("Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


