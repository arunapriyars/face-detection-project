import cv2
import os

# Load Haar Cascade from OpenCV (built-in path)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Safety check
if face_cascade.empty():
    print("Error loading Haar Cascade file")
    exit()

# Folder path
image_folder = 'images'

# Loop through all images
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read {image_name}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show image
    cv2.imshow(image_name, img)
    print(f"{image_name}: {len(faces)} face(s) detected")

    cv2.waitKey(0)

cv2.destroyAllWindows()