import os
from os import walk
from ultralytics import YOLO
import base64
import json
from PIL import Image
import keyboard


model = YOLO('../nn/models_nn/rust_nn_v6_s.pt')
path = 'datasets/rust/dataset_nn/augment/output'

SCORE = 0.7


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


def create_json(name, xyxycs, exp=".jpg"):
    im = Image.open(f"{name}{exp}")
    width, height = im.size
    
    with open(f"{name}{exp}", "rb") as f:
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
    with open(f"{name}.json", "w") as f:
        f.write(json.dumps(data, indent=2))
    

files_img = [i for i in next(walk(path), (None, None, []))[2] if ".jpg" in i]

for i in files_img:
    res = model.predict(source=f"{path}/{i}", verbose=False)
    
    xyxycs = get_xyxycs(res)
    create_json(f"{path}/{i.rsplit('.', 1)[0]}", xyxycs)
