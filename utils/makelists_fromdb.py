import os 
import pandas as pd 

def get_fossil_leaves_list(df):
    fossils,leaves=[],[]
    for i in range(len(df)):
        row = df.iloc[i]
        if 'Fossils' in row['file_name']:
            fossils.append([row['file_name'],row['label']])
        else: 
            leaves.append([row['file_name'],row['label']])
    return fossils,leaves 
local_datasets_dir = '/users/irodri15/scratch/Fossils/Experiments/softmax_triplet/datasets/leaves_fossils'
train_csv_file = 'lists/train_leaves_fossils_list.csv'
val_csv_file = 'lists/val_leaves_fossils_list.csv'
test_csv_file = 'lists/test_leaves_fossils_list.csv'

print(local_datasets_dir)
print(os.path.join(local_datasets_dir,train_csv_file))

train = pd.read_csv(os.path.join(local_datasets_dir,train_csv_file))
test = pd.read_csv(os.path.join(local_datasets_dir,test_csv_file))
val = pd.read_csv(os.path.join(local_datasets_dir,val_csv_file))

train_fossils , train_leaves = get_fossil_leaves_list(train)
test_fossils, test_leaves = get_fossil_leaves_list(test)
val_fossils, val_leaves = get_fossil_leaves_list(val)

train_fossils_df = pd.DataFrame(train_fossils+val_fossils,columns=['file_name','label'])
test_fossils_df = pd.DataFrame(test_fossils,columns=['file_name','label'])


train_leaves_df = pd.DataFrame(train_leaves+val_leaves,columns=['file_name','label'])
test_leaves_df = pd.DataFrame(test_leaves,columns=['file_name','label'])
import pdb;pdb.set_trace()
output_dir = '/users/irodri15/scratch/Fossils/Experiments/softmax_triplet/datasets/gan_fossils_leaves'
os.makedirs(output_dir,exist_ok=True)

train_fossils_df.to_csv(os.path.join(output_dir,'train_gan_fossils.csv'))
test_fossils_df.to_csv(os.path.join(output_dir,'test_gan_fossils.csv'))

train_leaves_df.to_csv(os.path.join(output_dir,'train_gan_leaves.csv'))
test_leaves_df.to_csv(os.path.join(output_dir,'test_gan_leaves.csv'))

