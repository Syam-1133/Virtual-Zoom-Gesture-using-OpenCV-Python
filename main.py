import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
startDist = None
zoom_factor = 1.0  # Initialize zoom_factor to 1.0 (no zoom)
cx, cy = 640, 360  # Default values for cx, cy (center of the screen)

# Ensure the image is loaded correctly
img1 = cv2.imread('cute.webp')
if img1 is None:
    print("Error: Unable to load image 'cute.webp'. Check the file path.")
    exit()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if len(hands) == 2:
        # Get the landmark lists for both hands
        lmList1 = hands[0]['lmList']  # Landmark list of the first hand
        lmList2 = hands[1]['lmList']  # Landmark list of the second hand

        print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1]))
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            print('Zoom Gesture')

            # Extract the x, y coordinates of the tip of the index finger for both hands
            x1, y1 = lmList1[8][0], lmList1[8][1]
            x2, y2 = lmList2[8][0], lmList2[8][1]

            if startDist is None:
                # Get the distance between the tip of the index finger of both hands
                length, info, img = detector.findDistance((x1, y1), (x2, y2), img)
                startDist = length

            length, info, img = detector.findDistance((x1, y1), (x2, y2), img)
            scale = length / startDist  # Calculate the scale factor based on hand distance
            zoom_factor *= scale  # Adjust the zoom factor
            startDist = length  # Update the start distance for the next frame
            cx, cy = info[4:]  # Update cx, cy from the info variable
            print(zoom_factor)

    else:
        startDist = None

    h1, w1, _ = img1.shape
    newH, newW = int(h1 * zoom_factor), int(w1 * zoom_factor)  # Calculate new dimensions based on zoom_factor

    # Ensure the new dimensions are within the image bounds
    newH = max(10, min(newH, img.shape[0]))  # Avoid going out of bounds vertically
    newW = max(10, min(newW, img.shape[1]))  # Avoid going out of bounds horizontally

    # Resize img1 to the correct size based on zoom_factor and bounds
    img1_resized = cv2.resize(img1, (newW, newH))

    # Ensure that the placement of the image fits within the bounds of img
    x_start = max(0, cx - newW // 2)
    y_start = max(0, cy - newH // 2)
    x_end = min(img.shape[1], cx + newW // 2)
    y_end = min(img.shape[0], cy + newH // 2)

    # Replace the region in img with the resized img1
    img[y_start:y_end, x_start:x_end] = img1_resized[:(y_end - y_start), :(x_end - x_start)]

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

