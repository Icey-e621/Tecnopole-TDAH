import os
import cv2
from PIL import Image, ImageDraw, ImageFont

# Create a directory to save the extracted frames
if not os.path.exists('not-adhd'):
    os.makedirs('not-adhd')

# Open the video file
video = cv2.VideoCapture('Hmaps/Con_adhd/heatmap2.avi')

# Initialize frame counter
count = 0

# Process each frame in the video
while True:
    # Read the next frame
    ret, frame = video.read()

    # If the frame was not read, exit the loop
    if not ret:
        break

    Finaljpg = cv2.resize(frame,(250,250),interpolation=Image.LANCZOS)

    # Save the frame to a file
    cv2.imwrite(f"Datos/Con_adhd/frame_{count}-2.jpg", Finaljpg)

    # Increment the frame counter
    count += 1

# Release the video file and exit
video.release()
cv2.destroyAllWindows()