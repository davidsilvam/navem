import cv2
import numpy as np
import glob
import os
import pandas as pd

image_name = "2020_06_25-14_14_59"
exp_accx = "exp_035"
exp_accy = "exp_040"
exp_psi = "exp_042"

start_frame = 0
end_frame = 10

image_dir = os.path.join("./../datasets", image_name, "*.jpg")
video_dir = os.path.join('./../project_' + str(start_frame) + '_' + str(end_frame) + '.avi')

name_accx = os.path.join('./../experiments', exp_accx, image_name + ".txt")
df_accx = pd.read_csv(name_accx, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])

img_array = []
# \param c maximun frames
c = 0

index_df = 0
# begin space in show text
a = 15

pos_text_info_h = 850
pos_text_info_w = 50
pos_text_size = 4
for filename in glob.glob(image_dir):
    # print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    font = cv2.FONT_HERSHEY_SIMPLEX

    s = len(filename.split('\\'))
    # print(int(filename.split('\\')[s - 1].split('.')[0]))

    current = int(filename.split('\\')[s - 1].split('.')[0])
    print("%.2f%% of images was processed. Current %d" % (100 * current / df_accx.shape[0], current))

    if(current > start_frame):
        r = df_accx.iloc[index_df]['pred']
        cv2.line(img=img, pt1=(0, 200), pt2=(int(round(width * r)), 200), color=(0, 0, 255), thickness=100,
                 lineType=cv2.LINE_4)

        percent = width * r
        if percent > 0 and percent <= width / 5:
            cv2.rectangle(img=img, pt1=(a, 550), pt2=(round(width / 5) - a, 650), color=(255, 0, 0), thickness=-1)
            cv2.putText(img, "Pare", (50, pos_text_info_h), font, pos_text_size, (0, 255, 255), 6, cv2.LINE_4)
        elif percent > width / 5 and percent <= width * 2 / 5:
            cv2.rectangle(img=img, pt1=(a, 550), pt2=(round(width / 5) - a, 650), color=(255, 0, 0), thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5), 550), pt2=(round(width * 2 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.putText(img, "Velocidade muito lenta", (50, pos_text_info_h), font, pos_text_size, (0, 255, 255), 6, cv2.LINE_4)
        elif percent > width * 2 / 5 and percent <= width * 3 / 5:
            cv2.rectangle(img=img, pt1=(a, 550), pt2=(round(width / 5) - a, 650), color=(255, 0, 0), thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5), 550), pt2=(round(width * 2 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5) * 2, 550), pt2=(round(width * 3 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.putText(img, "Velocidade lenta", (50, pos_text_info_h), font, pos_text_size, (0, 255, 255), 6, cv2.LINE_4)
        elif percent > width * 3 / 5 and percent <= width * 4 / 5:
            cv2.rectangle(img=img, pt1=(a, 550), pt2=(round(width / 5) - a, 650), color=(255, 0, 0), thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5), 550), pt2=(round(width * 2 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5) * 2, 550), pt2=(round(width * 3 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5) * 3, 550), pt2=(round(width * 4 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.putText(img, "Mantenha velocidade", (50, pos_text_info_h), font, pos_text_size, (0, 255, 255), 6, cv2.LINE_4)
        else:
            cv2.rectangle(img=img, pt1=(a, 550), pt2=(round(width / 5) - a, 650), color=(255, 0, 0), thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5), 550), pt2=(round(width * 2 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5) * 2, 550), pt2=(round(width * 3 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5) * 3, 550), pt2=(round(width * 4 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.rectangle(img=img, pt1=(a + round(width / 5) * 4, 550), pt2=(round(width * 5 / 5) - a, 650), color=(255, 0, 0),
                          thickness=-1)
            cv2.putText(img, "Acelere", (50, pos_text_info_h), font, pos_text_size, (0, 255, 255), 6, cv2.LINE_4)
        # Put text value predicted
        cv2.putText(img, str(r), (50, 450), font, 7, (0, 255, 255), 6, cv2.LINE_4)

        img_array.append(img)

        # c += 1
        # if c > 50:
        #   break
        if(current >= end_frame):
            break

    index_df += 1

out = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    print("%.2f%% was maked on video" % (100 * i / len(img_array)))
    out.write(img_array[i])
print("Finish")
out.release()
