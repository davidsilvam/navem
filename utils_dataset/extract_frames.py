import cv2
import os

video_name = "2020_06_29-19_33_39"
video_directory = "./../raw_datasets_videos"
images_directory = "./../raw_datasets_images"

if not os.path.exists(os.path.join(images_directory, video_name)):
    os.makedirs(os.path.join(images_directory, video_name))
    print("Path", os.path.join(images_directory, video_name), "created.")
else:
    print("Directory already exist.")

vidcap = cv2.VideoCapture(os.path.join(video_directory, video_name, "video.mp4"))
success, image = vidcap.read()
count = 0
while success:
    image_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(images_directory, video_name, "%d.jpg" % count), image_rotate_90_clockwise)#save frame as JPEG file
    success, image = vidcap.read()
    print ('Save frame: ', count)
    count += 1
print("Finish")
