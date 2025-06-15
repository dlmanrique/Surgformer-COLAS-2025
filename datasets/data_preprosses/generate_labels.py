#Code file that generates the pkl files for the selected dataset

import numpy as np
import os
import re
import cv2
import json
import pickle
from tqdm import tqdm
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Create annotations pkl file')
parser.add_argument('--dataset', default='Cholec80', type=str)
args = parser.parse_args()



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def process_split_data(train_dataframes, test_dataframes, valid_dataframes=None):
    """
    This function process the information from the .json file and returns the neccesary information for the .pkl file
    """
    
    train_images, train_annots, phase_info = train_dataframes
    test_images, test_annots = test_dataframes

    
    # Change the original dict, number as key and names as values
    id_to_phase = {v: k for k, v in phase_info.items()}

    general_id_train = 0
    all_train_info = {}

    
    # Iterate over each video for training split
    for video_id in tqdm(train_images['video_name'].unique()):
        
        # Filter the information based on video_id
        video_train_images = train_images[train_images['video_name'] == video_id].reset_index(drop=True)
        video_train_annots = train_annots[train_annots['video_name'] == video_id].reset_index(drop=True)

        assert len(video_train_annots) == len(video_train_images)

        video_frames_info = []
        # Iterate dataframe rows and create dict for each frame
        for i in range(len(video_train_images)):

            frame_info = {'unique_id': general_id_train,
                          'frame_id': video_train_images['frame_num'][i],
                          'original_frame_id': video_train_images['frame_num'][i],
                          'video_id': video_train_images['video_name'][i],
                          'tool_gt': None,
                          'frames': len(video_train_images),
                          'phase_gt': video_train_annots['phases'][i],
                          'phase_name': id_to_phase[video_train_annots['phases'][i]],
                          'fps': 1
                          }

            video_frames_info.append(frame_info)
            general_id_train += 1
        
        all_train_info[video_id] = video_frames_info


    general_id_test = 0
    all_test_info = {}

    # Iterate over each video for training split
    for video_id in tqdm(test_images['video_name'].unique()):
        
        # Filter the information based on video_id
        video_test_images = test_images[test_images['video_name'] == video_id].reset_index(drop=True)
        video_test_annots = test_annots[test_annots['video_name'] == video_id].reset_index(drop=True)

        assert len(video_test_annots) == len(video_test_images)

        video_frames_info = []
        # Iterate dataframe rows and create dict for each frame
        for i in range(len(video_test_images)):

            frame_info = {'unique_id': general_id_test,
                          'frame_id': video_test_images['frame_num'][i],
                          'original_frame_id': video_test_images['frame_num'][i],
                          'video_id': video_test_images['video_name'][i],
                          'tool_gt': None,
                          'frames': len(video_test_images),
                          'phase_gt': video_test_annots['phases'][i],
                          'phase_name': id_to_phase[video_test_annots['phases'][i]],
                          'fps': 1
                          }

            video_frames_info.append(frame_info)
            general_id_test += 1
        
        all_test_info[video_id] = video_frames_info


    all_valid_info = None

    if valid_dataframes:
        valid_images, valid_annots = valid_dataframes

        general_id_valid = 0
        all_valid_info = {}
        
        # Iterate over each video for training split
        for video_id in tqdm(valid_images['video_name'].unique()):
            
            # Filter the information based on video_id
            video_valid_images = valid_images[valid_images['video_name'] == video_id].reset_index(drop=True)
            video_valid_annots = valid_annots[valid_annots['video_name'] == video_id].reset_index(drop=True)

            assert len(video_valid_annots) == len(video_valid_images)

            video_frames_info = []
            # Iterate dataframe rows and create dict for each frame
            for i in range(len(video_valid_images)):

                frame_info = {'unique_id': general_id_valid,
                            'frame_id': video_valid_images['frame_num'][i],
                            'original_frame_id': video_valid_images['frame_num'][i],
                            'video_id': video_valid_images['video_name'][i],
                            'tool_gt': None,
                            'frames': len(video_valid_images),
                            'phase_gt': video_valid_annots['phases'][i],
                            'phase_name': id_to_phase[video_valid_annots['phases'][i]],
                            'fps': 1
                            }

                video_frames_info.append(frame_info)
                general_id_valid += 1
            
            all_valid_info[video_id] = video_frames_info


    return all_train_info, all_test_info, all_valid_info
    



def create_pickle_file(args):
    """
    This function creates all the .pkl files for a desired dataset
    """
    # Load all json files (all corresponding splits) 
    train_data = load_json(f'DATASETS/PHASES/annotations/Original_Datasets_Splits_Annotations/json_files/long_term_{args.dataset}_train.json')

    # Exception for Autolaparo
    try:
        valid_data = load_json(f'DATASETS/PHASES/annotations/Original_Datasets_Splits_Annotations/json_files/long_term_{args.dataset}_valid.json')
    except:
        valid_data = None

    test_data = load_json(f'DATASETS/PHASES/annotations/Original_Datasets_Splits_Annotations/json_files/long_term_{args.dataset}_test.json')

    # Create DataFrames for all the json_data
    train_images_df, train_annots_df, phases_info = pd.DataFrame(train_data['images']),  pd.DataFrame(train_data['annotations']), train_data['phases_categories']
    test_images_df, test_annots_df = pd.DataFrame(test_data['images']),  pd.DataFrame(test_data['annotations'])

    if valid_data:
        valid_images_df, valid_annots_df = pd.DataFrame(valid_data['images']),  pd.DataFrame(valid_data['annotations'])

    
    # Call the function that process the dataframes and return the pickle information
    if valid_data:
        all_train_info, all_test_info, all_valid_info = process_split_data([train_images_df, train_annots_df, phases_info],
                                                                            [test_images_df, test_annots_df],
                                                                            [valid_images_df, valid_annots_df])
    else:
        all_train_info, all_test_info, all_valid_info = process_split_data([train_images_df, train_annots_df, phases_info],
                                                                            [test_images_df, test_annots_df])

    os.makedirs(f'data/{args.dataset}', exist_ok=True)
    with open(os.path.join(f'data/{args.dataset}', '1fps_train.pickle'), 'wb') as file:
        pickle.dump(all_train_info, file)

    with open(os.path.join(f'data/{args.dataset}', '1fps_test.pickle'), 'wb') as file:
        pickle.dump(all_test_info, file)

    if all_valid_info:
        with open(os.path.join(f'data/{args.dataset}', '1fps_valid.pickle'), 'wb') as file:
            pickle.dump(all_valid_info, file)

if __name__ == "__main__":
    create_pickle_file(args)