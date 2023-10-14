import configparser
import ctypes
import threading
import torch
import winsound
from numpy.random import randint
from win32capture import capture,User32
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from ctypes import windll
from sendinput import leftdown,leftup,moveR
from math import ceil

conf = configparser.ConfigParser()
path = './conf.txt'
conf.read(path,encoding='utf-8')
#Q开黑交流群：814694645（删除会导致失效）
xfov = float(conf.get('config', 'x'))
yfov = float(conf.get('config', 'y'))
w = int(conf.get('config', 'gamex'))
h = int(conf.get('config', 'gamey'))
sec = int(conf.get('config', 'sec'))
#好了
# print(w)
# print(h)
windll.winmm.timeBeginPeriod(1)
stop = windll.kernel32.Sleep

GetKeyState = User32.GetKeyState
GetKeyState.argtypes = (ctypes.c_int,)


'------------------------------------------------------------------------------------'
#分辨率设置   Q开黑交流群：814694645（删除会导致失效）

# w = 1920    #游戏的宽
# h = 1080      #游戏的高
img_w = 320    #截图的宽
img_h = 320   #截图的高  32的倍数


####################



offset = img_h /2
baim = True
autofire = False


def lock():
    global autofire

    aims_copy = aims

    if baim:
        aims_copy = [x for x in aims_copy if x[0] in [0, 1]]
        if 0 in [x[0] for x in aims_copy]:  # 有body
            aims_copy = [x for x in aims_copy if x[0] in [0]]
    else:
        aims_copy = [x for x in aims_copy if x[0] in [1]]

    autofire = False
    if len(aims_copy):
        dist_list = []

        for _, x_c, y_c, _ in aims_copy:
            dist = (x_c - offset) ** 2 + (y_c - offset) ** 2
            dist_list.append(dist)

        tag, x_center, y_center, height = aims_copy[dist_list.index(min(dist_list))]

        move_x = ceil((x_center - offset) * xfov)
        body_y = ceil((y_center - offset - height / 7) * yfov)
        head_y = ceil((y_center - offset) * yfov)

        if GetKeyState(0X02) > 2:#此处设置左键右键1是左键2是右键

            if tag in [0]:

                moveR(move_x, body_y)

            elif tag in [1]:

                autofire = True
                moveR(move_x, head_y)



n = 0
def baimset():
    global baim, n
    while True:
        if GetKeyState(123) >1 and GetKeyState(123) != n:     #设置切换打头按键 默认为 F12

            baim = not baim
            n = GetKeyState(123)           
            if not baim:
                winsound.PlaySound('./neverlose.wav', flags=1)
        stop(50)

t =threading.Thread(target=baimset)
t.start()







conf_thres = 0.65   #置信度
iou_thres = 0.55

device = select_device()
model = DetectMultiBackend('cf.pt', device=device)

g = capture(w, h, img_w, img_h,"CrossFire")      #句柄截图构造

aims = []


while True:

    img0 = g.cap()
    img = img0.transpose((2, 0, 1))

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.
    img = img[None]

    pred = model(img, False, False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

    aims.clear()
    if not len(pred):
        continue

    for det in pred:

        for *xyxy, _, cls in det:
            line = [cls,
                    ceil((xyxy[0] + xyxy[2]) / 2),
                    ceil((xyxy[1] + xyxy[3]) / 2),
                    ceil(xyxy[3] - xyxy[1])]

            aims.append(line)

    lock()
    stop(sec)
