import multiprocessing
import mss
import numpy
import cv2
import sys
import random
import ctypes
import collections
from ultralytics import YOLO

from utils.mouse import MouseControls
from utils.wind_mouse import wind_mouse
from utils.grabber import Grabber
from utils.FPS import FPS
from utils.WinHelper import WinHelper




model = YOLO('../nn/models_nn/rust_nn_v6_s.pt')

aiming_percentages = 0.7
colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(4)]
# game_window_rect = WinHelper.GetWindowRect("Rust", (8, 30, 16, 39)) 

def CAPSLOCK_STATE():
    hllDll = ctypes.WinDLL("User32.dll")
    VK_CAPITAL = 0x14
    return hllDll.GetKeyState(VK_CAPITAL)


def grab_process(q):
    grabber = Grabber()
    while True:
        img = grabber.get_image({
            "left": 0,
            "top": 0,
            "width": 1920,
            "height": 1080}) 

        q.put_nowait(img)
        q.join()


def draw_boxes(img, xyxycscs):
    for i in xyxycscs:
      x1, y1, x2, y2, class_name, score, color = i
      img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
      img = cv2.putText(img, '{} {:.4f}'.format(class_name, score), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

    return img


def get_xyxycscs(res):
    box = res[0].boxes
    names = res[0].names
    xyxycscs = []
    for i in range(len(box.xyxy)):
        x1, y1, x2, y2 = int(box.xyxy[i][0]), int(box.xyxy[i][1]), int(box.xyxy[i][2]), int(box.xyxy[i][3])
        cls = names[int(box.cls[i])]
        score = box.conf[i].item()
        color = colors[int(box.cls[i])]
        xyxycscs.append([x1, y1, x2, y2, cls, score, color])

    return xyxycscs


def aiming(mouse, xyxycscs):
    for i in xyxycscs:
        if not i[5] > aiming_percentages:
            continue
        if not i[4] in ["plh", "pl"]:
            continue
        x, y = mouse.get_position()
        wind_mouse(x, y, i[0] + (i[2] - i[0]) / 2, i[1] + (i[3] - i[1]) / 2, move_mouse=mouse.move_relative) #  , 
        break
    

def cv2_process(q):
    fps = FPS()
    font = cv2.FONT_HERSHEY_SIMPLEX

    mouse = MouseControls()
    
    while True:
        if not q.empty():
            img = q.get_nowait()
            q.task_done()
            
            res = model.predict(source=img, verbose=True)
            
            xyxycscs = get_xyxycscs(res)
            
            if CAPSLOCK_STATE():
                aiming(mouse, xyxycscs)
                
            img = draw_boxes(img, xyxycscs)
            
            cv2.putText(img, f"{fps():.2f}", (5, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            img = cv2.resize(img, (850, 500))
            cv2.imshow("Playerid", img)
            if cv2.waitKey(1) == ord("l"):
                sys.exit(0)
            


if __name__ == "__main__":
    q = multiprocessing.JoinableQueue()

    p1 = multiprocessing.Process(target=grab_process, args=(q,))
    p2 = multiprocessing.Process(target=cv2_process, args=(q,))

    p1.start()
    p2.start()
