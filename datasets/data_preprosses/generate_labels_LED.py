# Adpatation of generate_labels file for LED dataset

import numpy as np
import os
import re
import cv2
import json
import pickle
from tqdm import tqdm

# Ensure numbers are always zero-padded (e.g., 01, 02, ..., 196)
def zero_pad(number):
    return f"{int(number):02d}"  # Converts to integer and re-formats with 2 digits

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_phase_dict(json_file):
    phase_dict = {}
    for entry in json_file:
        # Extract the relevant part from the image_name
        image_path = entry["image_name"]
        formatted_key = "/".join(image_path.split("/")[-2:])  # Extract "video_xx/XXXXX.jpg"
        
        # Store in dictionary
        phase_dict[formatted_key] = entry["phases"]

    return phase_dict

def main():
    ROOT_DIR = "/home/ohosie/Surgformer/data/LED/"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'frames'))
    # [ORIGINAL IMPLEMENTATION]
    # VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])

    # TODO: Nico addded sorting in other way
    # Natural sorting (extract numeric part and sort)
    VIDEO_NAMES = sorted(VIDEO_NAMES, key=lambda x: int(re.search(r'\d+', x).group()))

    TRAIN_NUMBERS = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
        102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 
        126, 127, 128, 129, 130, 131, 132, 136, 137, 138, 139, 140, 141, 142, 146, 147, 148, 149, 150, 151, 152, 
        156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182
    ]

    VAL_NUMBERS = [
        15, 16, 17, 18, 19, 20, 21, 
        62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
        118, 119, 120, 121, 122, 123, 124, 125, 
        133, 134, 135, 143, 144, 145, 153, 154, 155, 
        183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196
    ]
    
    # TRAIN_NUMBERS = [zero_pad(x) for x in TRAIN_INDEX]
    # VAL_NUMBERS = [zero_pad(x) for x in TEST_INDEX]
    TEST_NUMBERS = []

    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    train_pkl = dict()
    val_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    id2phase = {
        0: "Preparation",
        1: "Calot Triangle Dissection",
        2: "Clipping Cutting",
        3: "Gallbladder Dissection",
        4: "Gallbladder Packaging",
        5: "Cleaning Coagulation",
        6: "Gallbladder Retraction",
        7: "Dividing Ligament and Peritoneum",
        8: "Dividing Uterine Vessels and Ligament",
        9: "Transecting the Vagina",
        10: "Specimen Removal",
        11: "Suturing",
        12: "Washing",
        13: "Trocar Placement",
        14: "General Preparation and Orientation in the Abdomen",
        15: "Dissection of Lymph Nodes and Blood Vessels en Bloc",
        16: "Retroperitoneal Preparation towards Lower Pancreatic Border",
        17: "Retroperitoneal Preparation of Duodenum and Pancreatic Head",
        18: "Mobilization of Sigmoid Colon and Descending Colon",
        19: "Mobilization of Splenic Flexure",
        20: "Mobilization of Transverse Colon",
        21: "Mobilization of Ascending Colon",
        22: "Dissection and Resection of the Rectum",
        23: "Extra-abdominal Preparation of Anastomosis",
        24: "Intra-abdominal Preparation of Anastomosis",
        25: "Creation of Stoma",
        26: "Finalization of Operation",
        27: "Exception (Unique or Unexpected Phases)"
    }

    train_json = load_json("/home/naparicioc/ENDOVIS/GraSP/TAPIS/data/Led/annotations/train.json")['annotations']
    test_json = load_json("/home/naparicioc/ENDOVIS/GraSP/TAPIS/data/Led/annotations/test.json")['annotations']

    for video_id in VIDEO_NAMES:
        print(f"Video: {video_id}")

        vid_id = int(video_id.split("_")[-1])
        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
            ann = create_phase_dict(train_json)
        elif vid_id in VAL_NUMBERS:
            unique_id = unique_id_val
            ann = create_phase_dict(test_json)
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        video_path = os.path.join(ROOT_DIR, "frames", video_id)
        frames_list = os.listdir(video_path)

        frame_infos = list()
        for frame_id in tqdm(sorted(frames_list)):
            info = dict()
            index_video = int(frame_id.split(".")[0])
            info['unique_id'] = unique_id
            info['frame_id'] = index_video
            info['original_frame_id'] = index_video
            info['video_id'] = video_id
            info['tool_gt'] = None
            info['frames'] = len(frames_list)
            phase = ann[video_id+"/"+frame_id]
            info['phase_gt'] = phase
            info['phase_name'] = id2phase[int(phase)]
            info['fps'] = 1
            frame_infos.append(info)
            unique_id += 1
        
        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frames_list)
            unique_id_train = unique_id
        elif vid_id in VAL_NUMBERS:
            val_pkl[video_id] = frame_infos
            VAL_FRAME_NUMBERS += len(frames_list)
            unique_id_val = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += len(frames_list)
            unique_id_test = unique_id
    
    
    breakpoint()
    train_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    val_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
        pickle.dump(val_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)


    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('VAL Frams', VAL_FRAME_NUMBERS, unique_id_val)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    main()