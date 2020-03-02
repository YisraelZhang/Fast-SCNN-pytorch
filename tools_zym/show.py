import cv2
import numpy as np
import os

def show_cam_on_image(img, mask):
    print(mask[0][0])
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))

def show_cam_on_image_save(mask, num, laterals):
    root = os.path.join('./result', 'test', str(laterals))
    # root = os.path.join('/home/sensetime/Desktop/result/crop_in_fpn_1.0/person', '400_1_4', str(laterals))
    # root = os.path.join('/home/sensetime/Desktop/result/crop_in_fpn_1.0/person', '200_1_4', str(laterals))
    # root = os.path.join('/home/sensetime/Desktop/result/crop_in_fpn_1.0/person', '100_1_4', str(laterals))
    if not os.path.exists(root):
        os.makedirs(root)
    min = mask.min()
    max = mask.max()
    mask.add_(-min).div_(max - min + 1e-5)
    mask = mask.detach().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cv2.imwrite(os.path.join(root,str(num)+'.jpg'), np.uint8(heatmap))
    cv2.imshow(str(num) + ".jpg", np.uint8(heatmap))
    cv2.waitKey(0)

def show_cam_on_image(mask, num, laterals):
    min = mask.min()
    max = mask.max()
    mask.add_(-min).div_(max - min + 1e-5)
    mask = mask.detach().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cv2.imshow("cam.jpg", np.uint8(heatmap))
    cv2.waitKey(0)

def show_mask(mask, laterals=0):
    N, C, H, W = mask.shape
    for i in range(N):
        mask_tmp = mask[i]
        for num, j in enumerate(mask_tmp):
            show_cam_on_image(j, num, laterals)