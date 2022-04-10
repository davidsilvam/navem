import cv2
import os

video_name = "PilotGuru-V2"
video_directory = "./../../raw_datasets_videos"
images_directory = "./../../raw_datasets_images"

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
    #print(image_rotate_90_clockwise.shape[0], image_rotate_90_clockwise.shape[1])
    h = image_rotate_90_clockwise.shape[0]
    w = image_rotate_90_clockwise.shape[1]
    x = 0
    y = 216
    crop_img = image_rotate_90_clockwise[y:y + h, x:x + w]
    #cv2.imshow("cropped", crop_img)
    #cv2.imshow("cropped", image_rotate_90_clockwise)
    #cv2.waitKey(0)

    #os.system('pause')

    cv2.imwrite(os.path.join(images_directory, video_name, "%d.jpg" % count), crop_img)#save frame as JPEG file
    success, image = vidcap.read()
    print ('Save frame: ', count)
    count += 1
print("Finish")
