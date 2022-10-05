import numpy as np
import cv2, math, dlib, random
from itertools import product

# cap = cv2.VideoCapture('../video/chaplin.mp4')         #讀入影片
# cap = cv2.VideoCapture('../video/Alec_Baldwin.mp4')         #讀入影片
cap = cv2.VideoCapture('../cat.mp4')  # 讀入影片

FPS = cap.get(cv2.CAP_PROP_FPS)  # 每秒幾偵放映 Frame Per Second
F_Count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # frame count
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 取得畫面尺寸
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

seg_count = F_Count / 4  # seg=10 segmentation
print(f'FPS : {FPS:.2f} f/s,\tF_Count : {F_Count},\tw : {w},\th :　{h}')

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 編碼
out = cv2.VideoWriter('./hw.mp4', fourcc, FPS, (w, h))
pos = (10, 30);
font = 0;
color = (0, 255, 255)

r = math.ceil((seg_count) / 90)
sift = cv2.xfeatures2d.SIFT_create()  # create object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    C_Count = cap.get(cv2.CAP_PROP_POS_FRAMES)

    ##### 1. Flip
    if C_Count < seg_count * 1:
        cv2.putText(frame, f'1. Original', pos, font, 1, color, 2)

    #### 2. Rotation
    elif C_Count < seg_count * 2:
        ang = (C_Count % seg_count) / seg_count * 360 * r
        M1 = cv2.getRotationMatrix2D((w / 2, h / 2), ang,
                                     (C_Count % seg_count) / (seg_count))  # 表示旋轉的中心點,表示旋轉的角度,圖像縮放因子
        frame = cv2.warpAffine(frame, M1, (w, h))
        cv2.putText(frame, f'2. Rotation: {r} & Scale : {ang:.0f} degs.', pos, font, 1, color, 2)

    ##### 3. Sobel Canny Laplacian
    elif C_Count < seg_count * 3:
        sobelx = cv2.Sobel(frame[:, :w // 3], cv2.CV_64F, 1, 0, ksize=-1)
        sobely = cv2.Sobel(frame[:, :w // 3], cv2.CV_64F, 0, 1, ksize=-1)
        sobelx = cv2.convertScaleAbs(sobelx)  # 轉回 uint8
        sobely = cv2.convertScaleAbs(sobely)
        frame1 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        frame2 = cv2.cvtColor(cv2.Canny(frame[:, w // 3:w // 3 * 2], 64, 192), cv2.COLOR_GRAY2BGR)

        lapl = cv2.Laplacian(frame[:, w // 3 * 2:w], cv2.CV_64F)
        frame3 = cv2.convertScaleAbs(lapl)

        frame = np.hstack([frame1, frame2, frame3])
        cv2.putText(frame, f'3. Sobel{"Canny":>24}{"Laplacian":>30}', pos, font, 1, color, 2)

    #### 4. Sift
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps = sift.detect(frame, None)
        frame = cv2.drawKeypoints(frame, kps, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # image : image output
        cv2.putText(frame, f'10. Sift : {len(kps)} kps', pos, font, 1, color, 2)

    cv2.putText(frame, f'{C_Count:.0f}/{F_Count:.0f} frames, FPS : {FPS:.0f}', (10, 70), font, 1, color, 2)
    cv2.imshow('frame', frame)
    out.write(frame)  # 寫入影格

    if cv2.waitKey(1) == 27:  # c==27 (ascii code) key escape
        break

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)