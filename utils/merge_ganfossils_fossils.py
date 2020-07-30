import pandas as pd 
import os 
import glob

dir = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/datasets/leaves_fossils//lists/'

file = os.path.join(dir,'train_leaves_fossils_list.csv')


inputdf = pd.read_csv(file)
lookup = {}

for i in range(len(inputdf)):
    key = inputdf.iloc[i]['file_name'].split('/')[-2]
    lookup[key]=inputdf.iloc[i]['label']
print(inputdf['label'].unique())
print(lookup)

images_to_add = glob.glob('/media/data_cifs/irodri15/Leafs/Experiments/cyclegan/checkpoints/cyclegan_unconditioned/samples_testing/Leaves2Fossils/*-*.jpg')

datas= [[p,lookup[p.split('/')[-1].split('-')[0]]]  for p in images_to_add if p.split('/')[-1].split('-')[0] in lookup]

newdf = pd.DataFrame(datas,columns=['file_name','label'])

totaldf = pd.concat([inputdf,newdf])
print(totaldf)
totaldf.to_csv('/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/datasets/leaves_fossils//lists/gan_train_leaves_fossils_list.csv')

