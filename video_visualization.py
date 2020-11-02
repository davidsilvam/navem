import cv2
import os
import imutils
import pandas as pd


video_dir = os.path.join("./../raw_datasets_videos", "2020_06_25-14_14_59", "video.mp4")
cap = cv2.VideoCapture(video_dir)

video_name = "2020_06_25-14_14_59"
exp_accx = "exp_035"
exp_psi = "exp_040"

name_accx = os.path.join('./../experiments', exp_accx, video_name + ".txt")
name_psi = os.path.join('./../experiments', exp_psi, video_name + ".txt")

df_accx = pd.read_csv(name_accx, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
df_psi = pd.read_csv(name_accx, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])

cap.set(3, 640)
cap.set(4, 480)

index_df = 0

f = 0
size = 680
a = -50
while (True):
    try:
        #print(df.iloc[index_df]['pred'])
        # Capture frames in the video
        ret, img = cap.read()
        img = imutils.resize(img, width=size)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # if(a < 0):
        #     cv2.line(img=frame, pt1=(200 + a, 10), pt2=(200, 10), color=(0, 0, 255), thickness=5, lineType=10)
        #     b = 1
        # else:
        #     cv2.line(img=frame, pt1=(200, 10), pt2=(200 + a, 10), color=(0, 0, 255), thickness=5, lineType=10)
        #     b = 1
        #
        # a+=1
        index_df += 1

        #print(100*df.iloc[index_df]['pred'])
        r_psi = df_accx.iloc[index_df]['pred']
        # print(int(r))

        cv2.line(img=img, pt1=(int(size / 2) , 10), pt2=(int(size / 2) + 100, 10), color=(0, 255, 255), thickness=5,
                 lineType=cv2.LINE_4)
        if (r_psi > 0.46):
            cv2.line(img=img, pt1=(500 - int(round(100 * r_psi)), 10), pt2=(10, 10), color=(0, 0, 255), thickness=5,
                     lineType=cv2.LINE_4)
        else:
            cv2.line(img=img, pt1=(10, 10), pt2=(10, 10), color=(0, 0, 255), thickness=5,
                     lineType=cv2.LINE_4)

        # cv2.line(img=frame, pt1=(10, 10), pt2=(int(round(100*r)), 10), color=(0, 0, 255), thickness=5, lineType=10)

        if(a == 50):
            a = -50

        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX
        # a+=1
        # if(a == 15):
        #     a = 0

        #Use putText() method for
        #inserting text on video
        cv2.putText(img,
                    str(r_psi),
                    (50, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)

        # Display the resulting frameq
        #if (index_df > 735):
        cv2.imshow('video', img)
        print(index_df)
        # creating 'q' as the quit
        # button for the video
        f += 1
    except:
        print("over")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()
