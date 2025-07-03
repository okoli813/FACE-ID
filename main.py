import cv2

face_hear_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread("my pic.jpg")
if image is None:
    raise FileNotFoundError("Image file 'my pic.jpg' not found.")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(500)

faces = face_hear_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

cv2.imshow("faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
