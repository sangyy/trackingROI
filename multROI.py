import cv2
import imutils
import numpy as np

# 这八种工具包括：

# BOOSTING Tracker：和Haar cascades（AdaBoost）背后所用的机器学习算法相同，但是距其诞生已有十多年了。这一追踪器速度较慢，并且表现不好，但是作为元老还是有必要提及的。（最低支持OpenCV 3.0.0）

# MIL Tracker：比上一个追踪器更精确，但是失败率比较高。（最低支持OpenCV 3.0.0）

# KCF Tracker：比BOOSTING和MIL都快，但是在有遮挡的情况下表现不佳。（最低支持OpenCV 3.1.0）

# CSRT Tracker：比KCF稍精确，但速度不如后者。（最低支持OpenCV 3.4.2）

# MedianFlow Tracker：在报错方面表现得很好，但是对于快速跳动或快速移动的物体，模型会失效。（最低支持OpenCV 3.0.0）

# TLD Tracker：我不确定是不是OpenCV和TLD有什么不兼容的问题，但是TLD的误报非常多，所以不推荐。（最低支持OpenCV 3.0.0）

# MOSSE Tracker：速度真心快，但是不如CSRT和KCF的准确率那么高，如果追求速度选它准没错。（最低支持OpenCV 3.4.1）

# GOTURN Tracker：这是OpenCV中唯一一深度学习为基础的目标检测器。它需要额外的模型才能运行，本文不详细讲解。（最低支持OpenCV 3.2.0）

# 我个人的建议：

# 如果追求高准确度，又能忍受慢一些的速度，那么就用CSRT

# 如果对准确度的要求不苛刻，想追求速度，那么就选KCF

# 纯粹想节省时间就用MOSSE

TrDict = {'csrt': cv2.TrackerCSRT_create,
         'kcf' : cv2.TrackerKCF_create,
         'boosting' : cv2.TrackerBoosting_create,
         'mil': cv2.TrackerMIL_create,
         'tld': cv2.TrackerTLD_create,
         'medianflow': cv2.TrackerMedianFlow_create,
         'mosse':cv2.TrackerMOSSE_create}

# tracker = TrDict['csrt']()
#tracker = cv2.TrackerCSRT_create()
trackers = cv2.MultiTracker_create()

v = cv2.VideoCapture(r'mot.mp4') # video
# v = cv2.VideoCapture(0)

ret, frame = v.read()
frame = imutils.resize(frame,width=800)
k = 2
for i in range(k):
    cv2.imshow('Frame',frame)
    bbi = cv2.selectROI('Frame',frame)
    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i,frame,bbi)

# frameNumber = 2
# baseDir = r'TrackingResults'

while True:
    ret, frame = v.read()
    if not ret:
        break
    frame = imutils.resize(frame,width=800)
    (success,boxes) = trackers.update(frame)
    # np.savetxt(baseDir + '/frame_'+str(frameNumber)+'.txt',boxes,fmt='%f')
    # frameNumber+=1
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()
