import pandas as pd
import ast
import os
import json

dataset_name = 'M2CAI'
preds_paths = [
    'results/M2CAI/surgformer_HTA_M2CAI_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/0.txt',
    'results/M2CAI/surgformer_HTA_M2CAI_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/1.txt',
    'results/M2CAI/surgformer_HTA_M2CAI_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/2.txt'
]


#Open each file and join all info in a dataframe
datos = []
for path in preds_paths:
    with open(path, 'r') as f:
        for linea in f:
            partes = linea.strip().split(' ', 3)  # separamos en: index, video, frame, resto
            idx = int(partes[0])
            video_id = partes[1]
            frame_id = int(partes[2])
            
            # separar features y clase
            features_str, clase_str = partes[3].rsplit(']', 1)
            features = ast.literal_eval(features_str + ']')  # convierte string a lista de floats
            clase = int(clase_str.strip())
            
            datos.append([idx, video_id, frame_id, features, clase])


df_file = pd.DataFrame(datos, columns=['idx', 'video_id', 'frame_id', 'probs', 'class'])
df_file = df_file.sort_values(by='idx').reset_index(drop=True)

# Save all info in json file, first recover the original path for all frames and add it as a new column of the dataframe
df_file['frame_path'] = df_file.apply(
    lambda row: f"{row['video_id']}/{str(row['frame_id']).zfill(5)}.png", axis=1
)

dict_preds = dict(zip(df_file['frame_path'], df_file['probs']))

os.makedirs(f'Surgformer_preds/{dataset_name}')
with open(f'Surgformer_preds/{dataset_name}/preds_phases.json', "w", encoding="utf-8") as f:
        json.dump(dict_preds, f, ensure_ascii=False, indent=4)