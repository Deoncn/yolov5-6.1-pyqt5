# 图形库QT5 （必备）
# -*- coding: utf-8 -*-
# @Modified by: Ren
# @ProjectName:6.1-pyqt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
# 检测界面 （必备）
from ui.detect_ui import Ui_MainWindow  # 导入 UI目录下的 detect_ui
import detect

# 检测功能所需要的工具包 （必备）
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

# 附加库
import numpy as np
import time
import random
import matplotlib.pyplot as plt

# YOLO

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)


# 创建主函数类
class UI_Logic_Window(QtWidgets.QMainWindow):
    # 导航栏——退出
    def nav_exit(self):
        # print("退出")
        sys.exit(0)

    # 基本
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)

        self.timer_video = QtCore.QTimer()  # 创建定时器
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_slots()  # 绑定控件

        self.cap = cv2.VideoCapture()
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.output_folder = 'output/'
        self.vid_writer = None

        # 权重初始文件名
        self.openfile_name_model = None
        self.img_name = None

    # 控件绑定相关操作
    def init_slots(self):
        self.ui.action.triggered.connect(self.nav_exit)
        self.ui.pushButton_7.clicked.connect(self.load_model)
        # self.ui.pushButton_3.clicked.connect(self.open_vid)
        self.ui.pushButton_5.clicked.connect(self.open_img)
        # self.ui.pushButton_8.clicked.connect(self.open_cam)
        self.ui.pushButton_6.clicked.connect(self.detect)
        self.ui.pushButton.clicked.connect(self.stop_detect)
        # self.ui.pushButton_9.clicked.connect(self.save_ss)
        # self.timer_video.timeout.connect(self.show_video_frame)

    # 载入模型
    def load_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self, '选择权重模型文件(以.pt结尾)', 'weights/')

        # 若没有选择权重文件，弹出文件选择失败。
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开权重失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        # 若正常选择权重文件，则在命令行窗口打印地址字符串。
        else:
            if self.openfile_name_model:
                QtWidgets.QMessageBox.warning(self, u"ok", u"载入完成！", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            print(self.openfile_name_model)
            print('加载权重模型weights文件地址为：' + str(self.openfile_name_model))
            # self.model_init(self,  **self.openfile_name_model )

    # 模型初始化
    def model_init(self):
        print('hello')

    # 载入图片
    def open_img(self):
        try:
            self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "data/images",  "*.jpg ; *.png;All Files(*)")
        except OSError as reason:
            print(str(reason))
        else:
            if not self.img_name:
                QtWidgets.QMessageBox.warning(self, u"⚠警告", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # label_11
                Qimg = QtGui.QPixmap(self.img_name).scaled(self.ui.label_11.width(), self.ui.label_11.height())
                self.ui.label_11.setPixmap(Qimg)
                self.ui.textBrowser.setText(self.img_name)

    # 载入视频
    def open_vid(self):
        print("视频选择...")
        self.vid_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "data/videos", "*.mp4;*.mkv;")
        flag = self.cap.open(self.vid_name)
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            vid = QtGui.QPixmap(self.vid_name).scaled(self.ui.label_11.width(), self.ui.label_11.height())
            self.ui.label_11.setPixmap(vid)
            print('视频地址为：' + str(self.vid_name))

    # 摄像头开启
    def open_cam(self):
        print("打开摄像头")

    # 闭包
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        self.opt = parser.parse_args()
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        # print_args(FILE.stem, opt)
        return self.opt

    # 检测
    def detect(self):
        print("开始检测按钮")
        opt = self.parse_opt()
        opt.weights = self.openfile_name_model
        if self.img_name:
            print("图像检测开始")
            self.save_dir = increment_path(Path('runs\detect') / 'exp', exist_ok=False)
            print(self.save_dir)
            opt.source = self.img_name

            detect.run(**vars(opt))  # 保存检测结果
            print("检测完成")
            self.test()
        else:
            print("图片为空")

    # 检测结束
    def test(self):
        self.img_over = "E:/K01/Documents/pythonProject1/yolov5-6.1/runs/detect/exp/bus.jpg"
        print(self.img_over)
        print(self.img_name)

        # self.save_dir = increment_path(Path('runs/detect') / 'exp', exist_ok=False)
        # print(self.save_dir)
        # tfile_path = save_dir + '/' + m_file
        # Qtheimg = tfile_path
        # theimg = QtGui.QPixmap(Qtheimg).scaled(self.ui.label_5.width(), self.ui.label_5.height())
        # self.ui.label_5.setPixmap(theimg)

        m_file = os.path.basename(self.img_name)  # 文件名
        print(m_file)

        tfile_path = 'E:/K01/Documents/pythonProject1/yolov5-6.1/' + str(self.save_dir) + '/' + m_file
        print(tfile_path)

        # simg = cv2.imread(tfile_path)
        # cv2.imshow('img', simg)

        Qimg = QtGui.QPixmap(tfile_path).scaled(self.ui.label_11.width(), self.ui.label_11.height())
        self.ui.label_5.setPixmap(Qimg)

        print('设置了结果画面')

    # 结束检测
    def stop_detect(self):
        print("stopped")

        # 图像路径状态
        self.img_name = None

        self.ui.label_11.clear()
        self.ui.label_5.clear()
        self.ui.pushButton_5.setDisabled(False)
        # self.ui.pushButton_3.setDisabled(False)
        # self.ui.pushButton_8.setDisabled(False)

    # 保存路径
    def save_ss(self):
        print("the savepath")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())
