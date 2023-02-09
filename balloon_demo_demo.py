from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np 
# 指定模型的配置文件和 checkpoint 文件路径
config_file = '/root/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco_balloon.py'
checkpoint_file = '/root/mmdetection/work_dirs/balloon/latest.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# # 在一个新的窗口中将结果可视化
# model.show_result(img, result)
# # 或者将可视化结果保存为图片
# model.show_result(img, result, out_file='result.jpg')

# 测试视频并展示结果
video = mmcv.VideoReader('/root/mmdetection/data/test_video.mp4')
frame_id = 0
for frame in video:
    # bgr转灰度
    r = frame[:,:,-1]
    b = frame[:,:,0]
    g = frame[:,:,1]
    gray = (r**1.8*0.2446+g**1.8*0.6720+b**1.8*0.0833)**(1/1.8) # 转灰度
    gray =gray.astype(np.uint8) # 转uint8
    gray = np.expand_dims(gray, axis=2) # 升维
    gray = np.concatenate((gray,gray,gray),axis=2) # 拼接灰度三通道 

    # 推理
    result = inference_detector(model, frame)
    masks = result[-1][0] # [mask]

    #mask取并
    mask_sum = np.zeros(frame.shape[0:2],dtype=bool)
    for mask in masks:
        mask_sum = mask+mask_sum # 布尔或运算
    gray[mask_sum] = frame[mask_sum] # 色彩拼接
    # print(mask_sum)
    # print(gray)
    # cv2.imshow('gray',gray)
    # print('frame',frame.shape)
    cv2.imwrite('/root/mmdetection/data/colorframes/{}.jpg'.format("%06d"%frame_id), gray)
    frame_id += 1
    print('frame {} has been transformed'.format(frame_id))

