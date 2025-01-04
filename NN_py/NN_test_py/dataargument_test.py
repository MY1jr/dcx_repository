import cv2
import albumentations as A
from circleaugment import CircleAugmentation

def seq_augmentation(img, p_aug):
    if 0 < p_aug <= 0.05:
        transform = A.Compose([
            A.RandomResizedCrop(height=64, width=64, scale=(0.8, 1.0))
        ])
    elif 0.05 < p_aug <= 0.35:
        transform = A.Compose([
            A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.05)
        ])
    elif 0.35 < p_aug < 0.4:
        transform = A.Compose([
            A.MotionBlur(blur_limit=(3, 3))
        ])
    elif 0.4 <= p_aug <= 0.5:
        transform = CircleAugmentation()
        return transform(img)
    else:
        return img  # 直接返回原图,不转换

    img = transform(image=img) # albumentation库的要求
    return img['image']

img = cv2.imread('./dcx_mini/600/320/15.png')
cv2.imshow("Original Image", img)

augmented = seq_augmentation(img,p_aug=0.15)
cv2.imshow("Augmented Image", augmented)

# 等待按键事件，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()