import os
import cv2

# Create a directory to save the extracted frames
if not os.path.exists('not-adhd'):
    os.makedirs('not-adhd')

# Open the video file
video = cv2.VideoCapture('not-adhd.mp4')

# Initialize frame counter
count = 0

# Process each frame in the video
while True:
    # Read the next frame
    ret, frame = video.read()

    # If the frame was not read, exit the loop
    if not ret:
        break

    # Save the frame to a file
    cv2.imwrite(f'not-adhd/frame_{count}.jpg', frame)

    # Increment the frame counter
    count += 1

# Release the video file and exit
video.release()
cv2.destroyAllWindows()