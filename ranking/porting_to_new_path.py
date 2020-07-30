import pandas as pd
import os, glob, sys
csv_files = glob.glob('/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/datasets/*/*/*.csv')
new_prefix = /media/data_cifs/jacob/Fossil_Project/data/raw_data/Leaves_updated/
for f in csv_files: 
    data = pd.read_csv(f)
    new_name = f[:-3]+'old.csv'    
    if os.path.exists(new_name)==False:
        data.to_csv(new_name)
    if 'label' in data.columns:
        files = data['file_name']
        labels = data['label']
    elif 'x' in data.columns: 
        files = data['x']
        labels = data['y']
    else:
        print(data.columns)
        print('No column matched ')
    new_file_names =[]
    for name in files:

