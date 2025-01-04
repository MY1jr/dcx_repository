# -*- coding: utf-8 -*-
import  cv2
import numpy as np


img_path = 'G:/NN_py/circle_detection/circle_group_2_2.png'

#载入并显示图片
img=cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit()

cv2.imshow('1',img)
# #降噪（模糊处理用来减少瑕疵点）
# img = cv2.blur(img, (5,5))
# cv2.imshow('2',img)
#灰度化
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('3',gray_img)
 
# #param1的具体实现，用于边缘检测   
# img = cv2.Canny(img, 50, 150)  
# cv2.imshow('4', img) 
 
 
#霍夫变换圆检测
circles = cv2.HoughCircles(
    gray_img,  # 输入图像
    cv2.HOUGH_GRADIENT,  # 检测方法
    dp=1,  # 图像分辨率与累加器分辨率的比值
    minDist=100,  # 增加最小距离，避免重复检测
    param1=100,   # 增加边缘检测的阈值
    param2=30,    # 增加检测圆的阈值
    minRadius=50, # 增加最小半径
    maxRadius=200  # 增加最大半径
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    #根据检测到圆的信息，画出每一个圆
    for (x, y, r) in circles[0]:
        #画圆心
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        #画圆轮廓
        cv2.circle(img, (x, y), r, (0, 0, 255), 2)

#显示新图像
img = cv2.resize(img, (480, 480))
cv2.imshow('5',img)
 
#按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()