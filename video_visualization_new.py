import cv2
import numpy as np
import glob
import os
import pandas as pd

image_name = "2020_06_25-14_14_59"
exp_accx = "exp_035"
exp_psi = "exp_040"
exp_accy = "exp_042"

start_frame = 3000
end_frame = 3500

show_accx = True
show_accy = True
show_psi = True

image_dir = os.path.join("./../datasets", image_name, "*.jpg")
video_dir = os.path.join('./../project_' + str(start_frame) + '_' + str(end_frame) + '.avi')

name_accx = os.path.join('./../experiments', exp_accx, image_name + ".txt")
name_psi = os.path.join('./../experiments', exp_psi, image_name + ".txt")
name_accy = os.path.join('./../experiments', exp_accy, image_name + ".txt")

df_accx = pd.read_csv(name_accx, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
df_accy = pd.read_csv(name_accy, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
df_psi = pd.read_csv(name_psi, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])

img_array = []
# \param c maximun frames
c = 0

index_df = 0
# begin space in show text
a = 15

pos_text_info_h = 850
pos_text_info_w = 50
pos_text_size = 3

d_ini_accy = 550
d_fim_accy = 650

d_ini_psi = 1050
d_fim_psi = 1150

def transformRange(value, r1, r2):
  scale = (max(r2) - min(r2)) / (max(r1) - min(r1))
  return (value - min(r1)) * scale

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
        # # continue line
        r_accx = df_accx.iloc[index_df]['pred']
        # cv2.line(img=img, pt1=(0, 200), pt2=(int(round(width * r_accx)), 200), color=(0, 0, 255), thickness=100,
        #          lineType=cv2.LINE_4)

        percent = width * r_accx

# Conditions from accx
        if(show_accx):
            if percent > 0 and percent <= width / 5:
                cv2.rectangle(img=img, pt1=(30, height - a), pt2=(130, height - round(height / 5)), color=(255, 0, 0),
                              thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - a), pt2=(width - 30, height - round(height / 5)),
                              color=(255, 0, 0),
                              thickness=-1)

                cv2.putText(img, "Pare", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0),
                            thickness=9, lineType=4, fontScale=4)
                cv2.putText(img, "Pare", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                            thickness=8, lineType=2, fontScale=4)
            elif percent > width / 5 and percent <= width * 2 / 5:
                cv2.rectangle(img=img, pt1=(30, height - a), pt2=(130, height - round(height / 5)), color=(255, 0, 0),
                              thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - a), pt2=(width - 30, height - round(height / 5)),
                              color=(255, 0, 0),
                              thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) - a), pt2=(130, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) - a),
                              pt2=(width - 30, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)

                cv2.putText(img, "Velocidade muito lenta", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0),
                            thickness=9, lineType=4, fontScale=4)
                cv2.putText(img, "Velocidade muito lenta", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                            thickness=8, lineType=2, fontScale=4)
            elif percent > width * 2 / 5 and percent <= width * 3 / 5:
                cv2.rectangle(img=img, pt1=(30, height - a), pt2=(130, height - round(height / 5)), color=(255, 0, 0),
                              thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - a), pt2=(width - 30, height - round(height / 5)),
                              color=(255, 0, 0),
                              thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) - a), pt2=(130, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) - a),
                              pt2=(width - 30, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) * 2 - a),
                              pt2=(130, height - round(height / 5) * 3),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) * 2 - a),
                              pt2=(width - 30, height - round(height / 5) * 3),
                              color=(255, 0, 0), thickness=-1)

                cv2.putText(img, "Velocidade lenta", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0),
                            thickness=9, lineType=4, fontScale=4)
                cv2.putText(img, "Velocidade lenta", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                            thickness=8, lineType=2, fontScale=4)
            elif percent > width * 3 / 5 and percent <= width * 4 / 5:
                cv2.rectangle(img=img, pt1=(30, height - a), pt2=(130, height - round(height / 5)), color=(255, 0, 0),
                              thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - a), pt2=(width - 30, height - round(height / 5)),
                              color=(255, 0, 0),
                              thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) - a), pt2=(130, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) - a),
                              pt2=(width - 30, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) * 2 - a),
                              pt2=(130, height - round(height / 5) * 3),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) * 2 - a),
                              pt2=(width - 30, height - round(height / 5) * 3),
                              color=(255, 0, 0), thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) * 3 - a),
                              pt2=(130, height - round(height / 5) * 4),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) * 3 - a),
                              pt2=(width - 30, height - round(height / 5) * 4),
                              color=(255, 0, 0), thickness=-1)

                cv2.putText(img, "Mantenha velocidade", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0),
                            thickness=9, lineType=4, fontScale=4)
                cv2.putText(img, "Mantenha velocidade", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                            thickness=8, lineType=2, fontScale=4)
            else:
                cv2.rectangle(img=img, pt1=(30, height - a), pt2=(130, height - round(height / 5)), color=(255, 0, 0),
                              thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - a), pt2=(width - 30, height - round(height / 5)),
                              color=(255, 0, 0),
                              thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) - a), pt2=(130, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) - a),
                              pt2=(width - 30, height - round(height / 5) * 2),
                              color=(255, 0, 0), thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) * 2 - a),
                              pt2=(130, height - round(height / 5) * 3),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) * 2 - a),
                              pt2=(width - 30, height - round(height / 5) * 3),
                              color=(255, 0, 0), thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) * 3 - a),
                              pt2=(130, height - round(height / 5) * 4),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) * 3 - a),
                              pt2=(width - 30, height - round(height / 5) * 4),
                              color=(255, 0, 0), thickness=-1)

                cv2.rectangle(img=img, pt1=(30, height - round(height / 5) * 4 - a),
                              pt2=(130, height - round(height / 5) * 5),
                              color=(255, 0, 0), thickness=-1)
                cv2.rectangle(img=img, pt1=(width - 130, height - round(height / 5) * 4 - a),
                              pt2=(width - 30, height - round(height / 5) * 5),
                              color=(255, 0, 0), thickness=-1)

                cv2.putText(img, "Acelere", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0),
                            thickness=9, lineType=4, fontScale=4)
                cv2.putText(img, "Acelere", org=(150, 290), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0),
                            thickness=8, lineType=2, fontScale=4)
            # Put text value predicted
            cv2.putText(img, str(r_accx), org=(150, 160), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        thickness=9, lineType=4, fontScale=7)
            cv2.putText(img, str(r_accx), org=(150, 160), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0),
                        thickness=8, lineType=2, fontScale=7)

# Conditions from accy
        if(show_accy):
            r_accy = df_accy.iloc[index_df]['pred']
            percent_accy = width * r_accy

            if(r_accy >= 0.88):
                cv2.line(img=img, pt1=(int(round((width/2))), d_ini_accy + 350), pt2=(int(round((width/2) + int(round((width/2) * transformRange(r_accy, [0.88, 1], [0, 1]))))), d_ini_accy + 350), color=(0, 255, 0), thickness=100,
                        lineType=cv2.LINE_4)
            else:
                cv2.line(img=img, pt1=(int(round(width/2) - int(round((width/2) * (1 - transformRange(r_accy, [0, 0.88], [0, 1]))))), d_ini_accy + 350), pt2=(int(round(width/2)), d_ini_accy + 350), color=(0, 255, 0), thickness=100,
                        lineType=cv2.LINE_4)

            if r_accy > 0 and r_accy <= 0.80:
                cv2.rectangle(img=img, pt1=(a, d_ini_accy), pt2=(round(width / 5) - a, d_fim_accy), color=(0, 255, 0), thickness=-1)
            elif r_accy > 0.80 and r_accy <= 0.85:
                cv2.rectangle(img=img, pt1=(a + round(width / 5), d_ini_accy), pt2=(round(width * 2 / 5) - a, d_fim_accy), color=(0, 255, 0),
                              thickness=-1)
            elif r_accy > 0.85 and r_accy <= 0.90:
                cv2.rectangle(img=img, pt1=(a + round(width / 5) * 2, d_ini_accy), pt2=(round(width * 3 / 5) - a, d_fim_accy), color=(0, 255, 0),
                              thickness=-1)
            elif r_accy > 0.90 and r_accy <= 0.95:
                cv2.rectangle(img=img, pt1=(a + round(width / 5) * 3, d_ini_accy), pt2=(round(width * 4 / 5) - a, d_fim_accy), color=(0, 255, 0),
                              thickness=-1)
            else:
                cv2.rectangle(img=img, pt1=(a + round(width / 5) * 4, d_ini_accy), pt2=(round(width * 5 / 5) - a, d_fim_accy), color=(0, 255, 0),
                              thickness=-1)

            # Put text value predicted
            cv2.putText(img, str(r_accy), org=(150, d_fim_accy + 160), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        thickness=9, lineType=4, fontScale=7)
            cv2.putText(img, str(r_accy), org=(150, d_fim_accy + 160), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
                        thickness=8, lineType=2, fontScale=7)

# Conditions from psi
        if(show_psi):
            r_psi = df_psi.iloc[index_df]['pred']
            percent_psi = width * r_psi

            if(r_psi >= 0.46):
                # cv2.line(img=img, pt1=(int(round((width/2))), d_ini_psi + 350), pt2=(int(round((width/2) + int(round((width/2) * transformRange(r_psi, [0.46, 1], [0, 1]))))), d_ini_psi + 350), color=(0, 0, 255), thickness=100,
                #         lineType=cv2.LINE_4)
                cv2.line(img=img, pt1=(int(round(width/2) - int(round((width/2) * transformRange(r_psi, [0.46, 1], [0, 1])))), d_ini_psi + 350), pt2=(int(round(width/2)), d_ini_psi + 350), color=(0, 0, 255), thickness=100,
                        lineType=cv2.LINE_4)

            else:
                # cv2.line(img=img, pt1=(int(round(width/2) - int(round((width/2) * (1 - transformRange(r_psi, [0, 0.45], [0, 1]))))), d_ini_psi + 350), pt2=(int(round(width/2)), d_ini_psi + 350), color=(0, 0, 255), thickness=100,
                #         lineType=cv2.LINE_4)
                cv2.line(img=img, pt1=(int(round((width/2))), d_ini_psi + 350), pt2=(int(round((width/2) + int(round((width/2) * (1 - transformRange(r_psi, [0, 0.45], [0, 1])))))), d_ini_psi + 350), color=(0, 0, 255), thickness=100,
                        lineType=cv2.LINE_4)

            if r_psi > 0 and r_psi <= 0.34:#r_psi > 0 and r_psi <= 0.83
                # cv2.rectangle(img=img, pt1=(a, d_ini_psi), pt2=(round(width / 5) - a, d_fim_psi), color=(0, 0, 255), thickness=-1)
                cv2.rectangle(img=img, pt1=(a + round(width / 5) * 4, d_ini_psi), pt2=(round(width * 5 / 5) - a, d_fim_psi),
                              color=(0, 0, 255),
                              thickness=-1)
            elif r_psi > 0.34 and r_psi <= 0.42:
                # cv2.rectangle(img=img, pt1=(a + round(width / 5), d_ini_psi), pt2=(round(width * 2 / 5) - a, d_fim_psi),
                #               color=(0, 0, 255),
                #               thickness=-1)
                cv2.rectangle(img=img, pt1=(a + round(width / 5) * 3, d_ini_psi), pt2=(round(width * 4 / 5) - a, d_fim_psi),
                              color=(0, 0, 255),
                              thickness=-1)
            elif r_psi > 0.42 and r_psi <= 0.5:
                cv2.rectangle(img=img, pt1=(a + round(width / 5) * 2, d_ini_psi), pt2=(round(width * 3 / 5) - a, d_fim_psi),
                              color=(0, 0, 255),
                              thickness=-1)
            elif r_psi > 0.5 and r_psi <= 0.58:
                # cv2.rectangle(img=img, pt1=(a + round(width / 5) * 3, d_ini_psi), pt2=(round(width * 4 / 5) - a, d_fim_psi),
                #               color=(0, 0, 255),
                #               thickness=-1)
                cv2.rectangle(img=img, pt1=(a + round(width / 5), d_ini_psi), pt2=(round(width * 2 / 5) - a, d_fim_psi),
                              color=(0, 0, 255),
                              thickness=-1)
            else:
                # cv2.rectangle(img=img, pt1=(a + round(width / 5) * 4, d_ini_psi), pt2=(round(width * 5 / 5) - a, d_fim_psi),
                #               color=(0, 0, 255),
                #               thickness=-1)
                cv2.rectangle(img=img, pt1=(a, d_ini_psi), pt2=(round(width / 5) - a, d_fim_psi), color=(0, 0, 255), thickness=-1)


            # Put text value predicted
            cv2.putText(img, str(r_psi), org=(150, d_fim_psi + 160), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        thickness=9, lineType=4, fontScale=7)
            cv2.putText(img, str(r_psi), org=(150, d_fim_psi + 160), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255),
                        thickness=8, lineType=2, fontScale=7)

        img_array.append(img)

        if(current >= end_frame):
            break

    index_df += 1

out = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    print("%.2f%% was maked on video" % (100 * i / len(img_array)))
    out.write(img_array[i])
print("Finish")
out.release()
