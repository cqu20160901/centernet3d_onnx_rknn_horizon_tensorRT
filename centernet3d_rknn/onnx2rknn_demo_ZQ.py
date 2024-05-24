import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './Monodle_centerNet3D.onnx'
RKNN_MODEL = './Monodle_centerNet3D.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True



CLASSES = ['Pedestrian', 'Car', 'Cyclist']

class_num = len(CLASSES)
input_h = 384
input_w = 1280

object_thresh = 0.6

output_h = 96
output_w = 320
downsample_ratio = 4
num_heading_bin = 12


class ScoreXY:
    def __init__(self, score, c, h, w):
        self.score = score
        self.c = c
        self.h = h
        self.w = w


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def nms(heatmap, heatmapmax):
    keep_heatmap = []
    for b in range(1):
        for c in range(class_num):
            for h in range(output_h):
                for w in range(output_w):
                    if heatmapmax[c * output_h * output_w + h * output_w + w] == heatmap[c * output_h * output_w + h * output_w + w] and heatmap[c * output_h * output_w + h * output_w + w] > object_thresh:
                        temp = ScoreXY(heatmap[c * output_h * output_w + h * output_w + w], c, h, w)
                        keep_heatmap.append(temp)
    return keep_heatmap


def sigmoid(x):
    return 1 / (1 + exp(-x))


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


class Calibration(object):
    def __init__(self, calib_file):
        with open(calib_file) as f:
            lines = f.readlines()
        obj = lines[2].strip().split(' ')[1:]
        self.P2 = np.array(obj, dtype=np.float32).reshape(3, 4)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def img_to_rect(self, u, v, depth_rect):
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = [x, y, depth_rect]
        return pts_rect

    def alpha2ry(self, alpha, u):
        ry = alpha + np.arctan2(u - self.cu, self.fu)
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry


def postprocess(outputs, calibs):
    heatmap = outputs[0]
    offset_2d = outputs[1]
    size_2d = outputs[2]
    heatmapmax = outputs[7]

    depths = outputs[3]
    offset_3d = outputs[4]
    size_3d = outputs[5]
    heading = outputs[6]

    keep_heatmap = nms(heatmap, heatmapmax)
    top_heatmap = sorted(keep_heatmap, key=lambda t: t.score, reverse=True)

    boxes2d = []
    output3d = []

    for i in range(len(top_heatmap)):
        if i > 50:
            break
        classId = top_heatmap[i].c
        score = top_heatmap[i].score
        w = top_heatmap[i].w
        h = top_heatmap[i].h

        # 解码 2d 框
        bx = (w + offset_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        by = (h + offset_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bw = (size_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bh = (size_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio

        xmin = (bx - bw / 2) / input_w
        ymin = (by - bh / 2) / input_h
        xmax = (bx + bw / 2) / input_w
        ymax = (by + bh / 2) / input_h

        keep_flag = 0
        for j in range(len(boxes2d)):
            xmin1 = boxes2d[j].xmin
            ymin1 = boxes2d[j].ymin
            xmax1 = boxes2d[j].xmax
            ymax1 = boxes2d[j].ymax
            if IOU(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1) > 0.45:
                keep_flag += 1
                break

        if keep_flag == 0:
            bbox = DetectBox(classId, score, xmin, ymin, xmax, ymax)
            boxes2d.append(bbox)

            # 解码 3DBox
            dimensions = []
            headings = []

            depth = depths[0 * output_h * output_w + h * output_w + w]
            sigma = depths[1 * output_h * output_w + h * output_w + w]

            depth = 1. / (sigmoid(depth) + 1e-6) - 1.
            sigma = np.exp(-sigma)

            x3d = (w + offset_3d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
            y3d = (h + offset_3d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio

            dimensions.append(size_3d[0 * output_h * output_w + h * output_w + w])
            dimensions.append(size_3d[1 * output_h * output_w + h * output_w + w])
            dimensions.append(size_3d[2 * output_h * output_w + h * output_w + w])

            for k in range(24):
                headings.append(heading[k * output_h * output_w + h * output_w + w])

            locations = calibs.img_to_rect(x3d, y3d, depth)
            locations[1] += dimensions[0] / 2
            alpha = get_heading_angle(headings)
            ry = calibs.alpha2ry(alpha, x3d)

            # 结果输出
            output3d.append(score)
            output3d.append(classId)
            output3d.append(alpha)

            # 理论上这里的 xmin, ymin,xmax, ymax 用 3D 结果计算(这里直接用的 2d 解码的框)
            output3d.append(xmin)
            output3d.append(ymin)
            output3d.append(xmax)
            output3d.append(ymax)

            output3d.append(dimensions[0])
            output3d.append(dimensions[1])
            output3d.append(dimensions[2])

            output3d.append(locations[0])
            output3d.append(locations[1])
            output3d.append(locations[2])

            output3d.append(ry)

    return boxes2d, output3d


def class2angle(cls, residual, to_label_format=False):
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)


class Object3d(object):
    def __init__(self, data):
        # extract label, truncation, occlusion
        self.score = data[0]  # score
        self.type = data[1]  # 'Car', 'Pedestrian', ...
        self.alpha = data[2]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[3]  # left
        self.ymin = data[4]  # top
        self.xmax = data[5]  # right
        self.ymax = data[6]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[7]  # box height
        self.w = data[8]  # box width
        self.l = data[9]  # box length (in meters)
        self.t = (data[10], data[11], data[12])  # location (x,y,z) in camera coord.
        self.ry = data[13]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]


def project_to_image(pts_3d, P):
    '''
    将相机坐标系下的3D边界框的角点, 投影到图像平面上, 得到它们在图像上的2D坐标
    输入: pts_3d是一个nx3的矩阵, 包含了待投影的3D坐标点(每行一个点), P是相机的投影矩阵, 通常是一个3x4的矩阵。
    输出: 返回一个nx2的矩阵, 包含了投影到图像平面上的2D坐标点。
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)  => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)   => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]  # 获取3D点的数量
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))  # 将每个3D点的坐标扩展为齐次坐标形式（4D），通过在每个点的末尾添加1，创建了一个nx4的矩阵。
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # 将扩展的3D坐标点矩阵与投影矩阵P相乘，得到一个nx3的矩阵，其中每一行包含了3D点在图像平面上的投影坐标。每个点的坐标表示为[x, y, z]。
    pts_2d[:, 0] /= pts_2d[:, 2]  # 将投影坐标中的x坐标除以z坐标，从而获得2D图像上的x坐标。
    pts_2d[:, 1] /= pts_2d[:, 2]  # 将投影坐标中的y坐标除以z坐标，从而获得2D图像上的y坐标。
    return pts_2d[:, 0:2]  # 返回一个nx2的矩阵,其中包含了每个3D点在2D图像上的坐标。


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d(obj, P):
    '''
    计算对象的3D边界框在图像平面上的投影
    输入: obj代表一个物体标签信息,  P代表相机的投影矩阵-内参。
    输出: 返回两个值, corners_3d表示3D边界框在 相机坐标系 的8个角点的坐标-3D坐标。
                                     corners_2d表示3D边界框在 图像上 的8个角点的坐标-2D坐标。
    '''
    # 计算一个绕Y轴旋转的旋转矩阵R，用于将3D坐标从世界坐标系转换到相机坐标系。obj.ry是对象的偏航角
    R = roty(obj.ry)

    # 物体实际的长、宽、高
    l = obj.l
    w = obj.w
    h = obj.h

    # 存储了3D边界框的8个角点相对于对象中心的坐标。这些坐标定义了3D边界框的形状。
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # 1、将3D边界框的角点坐标从对象坐标系转换到相机坐标系。它使用了旋转矩阵R
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # 3D边界框的坐标进行平移
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

    # 2、检查对象是否在相机前方，因为只有在相机前方的对象才会被绘制。
    # 如果对象的Z坐标（深度）小于0.1，就意味着对象在相机后方，那么corners_2d将被设置为None，函数将返回None。
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # 3、将相机坐标系下的3D边界框的角点，投影到图像平面上，得到它们在图像上的2D坐标。
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)



def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.37]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs


if __name__ == '__main__':
    print('This is main ...')

    cal_path = './test.txt'
    image_path = './test.png'
    origin_image = cv2.imread(image_path)
    image_h, image_w = origin_image.shape[:2]
    
    
    resize_image = cv2.resize(origin_image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    
    img = np.expand_dims(resize_image, 0)

    rknn_output = export_rknn_inference(img)

    outputs = []
    for i in range(len(rknn_output)):
        outputs.append(rknn_output[i].reshape(-1))

    calibs = Calibration(cal_path)
    boxes2d, results3d = postprocess(outputs, calibs)

    print('detect num is:', len(boxes2d))

    # 画 3D 框
    for l in range(0, len(results3d), 14):
        obj3d = []
        for q in range(14):
            obj3d.append(results3d[l + q])
        obj3ds = Object3d(obj3d)
        box3d, box3d_pts_3d = compute_box_3d(obj3ds, calibs.P2)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(origin_image, (int(box3d[i, 0]), int(box3d[i, 1])), (int(box3d[j, 0]), int(box3d[j, 1])), (0, 0, 255), 2)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(origin_image, (int(box3d[i, 0]), int(box3d[i, 1])), (int(box3d[j, 0]), int(box3d[j, 1])), (0, 0, 255), 2)

            i, j = k, k + 4
            cv2.line(origin_image, (int(box3d[i, 0]), int(box3d[i, 1])), (int(box3d[j, 0]), int(box3d[j, 1])), (0, 0, 255), 2)

    for i in range(len(boxes2d)):
        classid = boxes2d[i].classId
        score = boxes2d[i].score
        xmin = int(boxes2d[i].xmin * image_w)
        ymin = int(boxes2d[i].ymin * image_h)
        xmax = int(boxes2d[i].xmax * image_w)
        ymax = int(boxes2d[i].ymax * image_h)

        cv2.rectangle(origin_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = '%s:%.2f' % (CLASSES[classid], score)
        cv2.putText(origin_image, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_onnx_result.jpg', origin_image)

