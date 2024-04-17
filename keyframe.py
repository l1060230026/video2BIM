import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm

if __name__=="__main__":
    video_path = "IMG_1419.MP4"
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if os.path.exists('database.db'):
        os.remove('database.db')
    if os.path.exists('photo'):
        shutil.rmtree('photo')
    if os.path.exists('match'):
        shutil.rmtree('match')
    os.makedirs('photo')
    os.makedirs('match')
    name = 1
    old_name = 1
    step = 30
    min = 150
    max = 200
    sift = cv2.SIFT_create()
    count = 0
    while success:
        # cv2.imwrite(os.path.join('match', str(name) + '.jpg'), frame)
        # name += step
        # cap.set(cv2.CAP_PROP_POS_FRAMES, name)
        # success, frame = cap.read()
        if name >= total_frames:
            break

        if name == 1:
            cv2.imwrite(os.path.join('match', str(name) + '.jpg'), frame)
            kp1, des1 = sift.detectAndCompute(frame, None)
            name += step
            continue
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, name)
            success, frame = cap.read()
            kp2, des2 = sift.detectAndCompute(frame, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            pts1 = []
            pts2 = []
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.8 * n.distance:
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)

            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

            try:
                pts1 = pts1[mask.ravel() == 1]
            except:
                print('a')
            pts2 = pts2[mask.ravel() == 1]

            if len(pts1) > max:
                while len(pts1) > max:
                    if name - old_name > 120:
                        break
                    name += 10
                    cap.set(cv2.CAP_PROP_POS_FRAMES, name)
                    success, frame = cap.read()
                    kp2, des2 = sift.detectAndCompute(frame, None)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    pts1 = []
                    pts2 = []
                    for i, (m, n) in enumerate(matches):
                        if m.distance < 0.8 * n.distance:
                            pts2.append(kp2[m.trainIdx].pt)
                            pts1.append(kp1[m.queryIdx].pt)

                    pts1 = np.int32(pts1)
                    pts2 = np.int32(pts2)
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

                    pts1 = pts1[mask.ravel() == 1]
                    pts2 = pts2[mask.ravel() == 1]
                if len(pts1) < min:
                    name -= 10
                    cap.set(cv2.CAP_PROP_POS_FRAMES, name)
                    success, frame = cap.read()
                    cv2.imwrite(os.path.join('match', str(name) + '.jpg'), frame)
                    kp1, des1 = kp2, des2
                else:
                    cv2.imwrite(os.path.join('match', str(name) + '.jpg'), frame)
                    kp1, des1 = kp2, des2
                    old_name = name
                name += step
                continue
            if len(pts1) < min:
                temp = [len(pts1)]
                for i in range(1, 6):
                    temp_name = name - 5 * i
                    cap.set(cv2.CAP_PROP_POS_FRAMES, temp_name)
                    success, frame = cap.read()
                    kp2, des2 = sift.detectAndCompute(frame, None)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    pts1 = []
                    pts2 = []
                    for i, (m, n) in enumerate(matches):
                        if m.distance < 0.8 * n.distance:
                            pts2.append(kp2[m.trainIdx].pt)
                            pts1.append(kp1[m.queryIdx].pt)

                    pts1 = np.int32(pts1)
                    pts2 = np.int32(pts2)
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

                    pts1 = pts1[mask.ravel() == 1]
                    pts2 = pts2[mask.ravel() == 1]
                    if len(pts2) < min:
                        temp.append(len(pts2))
                    else:
                        break
                name -= np.argmax(np.array(temp)) * 5
                cap.set(cv2.CAP_PROP_POS_FRAMES, name)
                success, frame = cap.read()
                cv2.imwrite(os.path.join('match', str(name) + '.jpg'), frame)
                kp1, des1 = kp2, des2
                name += step
                continue
            cv2.imwrite(os.path.join('match', str(name) + '.jpg'), frame)
            kp1, des1 = kp2, des2
            old_name = name
            name += step
