import pyautogui
import time
import os
from ultralytics import YOLO
import base64
import json
from PIL import Image
import keyboard


model = YOLO('../nn/models_nn/rust_nn_v5.pt')

mon = (100, 201, 927, 522)
# screenshot = pyautogui.screenshot(region=mon)
# screenshot.save(f"aaa_test.jpg")
# raise
btn_screenshot = 'y'
SCORE = 0.7
num_screenshots = 3000
num_screenshots_in_minute = 60
first_name = 3000 # 3000
time_between_screenshots = 60 / num_screenshots_in_minute
print("timer")

time.sleep(10)


def get_xyxyncs(res):
    box = res[0].boxes
    xyxyncs = []
    for i in range(len(res[0].boxes.xyxyn)):
        score = box.conf[i].item()
        if not score > 0.70:
            continue
        x1, y1, x2, y2 = box.xyxyn[i][0], box.xyxyn[i][1], box.xyxyn[i][2], box.xyxyn[i][3]
        cls = int(box.cls[i])
        xyxyncs.append([x1, y1, x2, y2, cls])

    return xyxyncs


def get_xyxycs(res):
    box = res[0].boxes
    names = res[0].names
    xyxycs = []
    for i in range(len(res[0].boxes.xyxy)):
        score = box.conf[i].item()
        if not score > SCORE:
            continue
        x1, y1, x2, y2 = box.xyxy[i][0].item(), box.xyxy[i][1].item(), box.xyxy[i][2].item(), box.xyxy[i][3].item()
        cls = names[int(box.cls[i])]
        xyxycs.append([x1, y1, x2, y2, cls])

    return xyxycs
    

def create_label(name, xyxync):
    x1, y1, x2, y2, cls = xyxync
    with open(f"datasets/rust/dataset_nn/labels/train/{name}.txt", "w") as f:
        f.write(f"{cls} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}")


def create_json(name, xyxycs):
    im = Image.open(f"datasets/rust/dataset_nn/{name}.jpg")
    width, height = im.size
    
    with open(f"datasets/rust/dataset_nn/{name}.jpg", "rb") as f:
        img_data = str(base64.b64encode(f.read()))[2:-1]

    data = {"version": "5.2.1",
            "flags": {},
            "shapes": [{"label": i[4],
                        "points": [
                            [i[0], i[1]],
                            [i[2], i[3]]
                            ],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {}
                        } for i in xyxycs],
            "imagePath": f"{name}.jpg",
            "imageData": img_data,
            "imageHeight": height,
            "imageWidth": width
            }
    with open(f"datasets/rust/dataset_nn/{name}.json", "w") as f:
        f.write(json.dumps(data, indent=2))
    

def screen():
    global screenshots, first_name
    
    screenshot = pyautogui.screenshot(region=mon)
    
    res = model.predict(source=screenshot, verbose=False)
    xyxycs = get_xyxycs(res)

    screenshot.save(f"datasets/rust/dataset_nn/{first_name}.jpg")
    create_json(first_name, xyxycs)
    
    screenshots += 1
    first_name += 1
    
    print(screenshots)
    
print("start")
screenshots = 0

while True:
    if screenshots >= num_screenshots:
        break
    
    screenshot = pyautogui.screenshot(region=mon)
    
    res = model.predict(source=screenshot, verbose=False)
    
    xyxycs = get_xyxycs(res)
    if not xyxycs and not keyboard.is_pressed(btn_screenshot):
        continue

    screenshot.save(f"datasets/rust/dataset_nn/{first_name}.jpg")
    create_json(first_name, xyxycs)
    
    screenshots += 1
    first_name += 1
    
    print(screenshots)
    time.sleep(time_between_screenshots)

print("finish")
