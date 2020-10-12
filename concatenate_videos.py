import cv2
import os

# this two lines are for loading the videos.
# in this case the video are named as: cut1.mp4, cut2.mp4, ..., cut15.mp4
# videofiles = [n for n in os.listdir('.') if n[0]=='c' and n[-4:]=='.mp4']
# videofiles = sorted(videofiles, key=lambda item: int( item.partition('.')[0][3:]))

video_name = '2020_06_25-16_49_23'
path = os.path.join('./../')

videofiles = [n for n in os.listdir(path) if n[-4:] == '.avi']

print(videofiles)

videofiles = sorted(videofiles, key=lambda item: int(item.partition('.')[0][-4:]))

for i in range(len(videofiles)):
    videofiles[i] = os.path.join(path, videofiles[i])
print(videofiles)

# os.system('pause')

video_index = 0
cap = cv2.VideoCapture(videofiles[0])

# video resolution: 1624x1234 px
# out = cv2.VideoWriter("video.avi",
#                       cv2.cv.CV_FOURCC('F','M','P', '4'),
#                       15, (1624, 1234), 1)

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('cutout.mp4', fourcc, 20, (640, 480))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join('./../', video_name + '.avi'), fourcc, 30.0, (2160, 3840))

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        print ("end of video " + str(video_index) + " .. next one now")
        video_index += 1
        if video_index >= len(videofiles):
            break
        cap = cv2.VideoCapture(videofiles[ video_index ])
        ret, frame = cap.read()
    #cv2.imshow('frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print ("end.")
