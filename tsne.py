import pandas as pd 
import numpy as np 
import argparse
import numpy as np 
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw, ImageFont,ImageOps
import matplotlib as mlp
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from lapjv import lapjv
import os
from tensorflow.python.keras.preprocessing import image
COLORS =[(184, 134, 11),(0, 100, 0),(189, 183, 107),(85, 107, 47),(255, 140, 0),(153, 50, 204),
        (233, 150, 122),(143, 188, 143),(72, 61, 139),(0, 206, 209),(148, 0, 211),(255, 20, 147),
            (0, 191, 255),(30, 144, 255),(252, 230, 201),(0, 201, 87),(178, 34, 34),(139, 26, 26),
                (255, 250, 205),(238, 233, 191),(104, 131, 139),(240, 128, 128),(122, 139, 139)]
def load_img(paths,out_dim,out_res):
    pred_img = paths[:np.square(out_dim)]
    img_collection = []
    for idx, img in enumerate(pred_img):
        img_collection.append(image.load_img(img, target_size=(out_res, out_res)))
        
    if (np.square(out_dim) > len(img_collection)):
        raise ValueError("Cannot fit {} images in {}x{} grid".format(len(img_collection), out_dim, out_dim))
    return img_collection

def generate_tsne(activations,perplexity=100,tsne_iter=5000):
    tsne = TSNE(perplexity=perplexity, n_components=2, init='random', n_iter=tsne_iter)
    X_2d = tsne.fit_transform(activations)
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d

def save_tsne_grid(img_collection,labels, X_2d, out_res, out_dim,out_dir,out_name='tsne_visualization.png',quality=75,subsampling=2):
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    border = 10
    out = np.ones((out_dim*(out_res+2*border), out_dim*(out_res+2*border), 3))
    print(out.shape)
    fnt = ImageFont.load_default()
    fnt.size = 40
    for pos, img,l in zip(grid_jv, img_collection,labels):
        h_range = int(np.floor(pos[0]* (out_dim - 1) * (out_res+2*border)))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * (out_res+2*border)))
        #img =image.img_to_array(img)
        img_with_border = ImageOps.expand(img,border=border,fill=COLORS[l%len(COLORS)])
        d = ImageDraw.Draw(img_with_border)
        d.text((10,10),str(l), font=fnt,fill=COLORS[l%len(COLORS)])
        out[h_range:h_range + (out_res+2*border), w_range:w_range + (out_res+2*border)]  = img_with_border
    print(out.shape)
    im = image.array_to_img(out)
    im.save(os.path.join(out_dir,out_name), quality=quality,subsampling=subsampling)

import random
def make_tsne(dataset,activations_path,out_dim,out_res,out_dir,out_name,tsne_iter=2000,quality_saving=75,subsampling_saving=2):
    data = pd.read_csv(dataset)
    print(data.head(2))
    print(data.shape)
    try: 
        paths = list(data['x'])
        labels = list(data['y'])
    except:
        paths = list(data['file_name'])
        labels = list(data['label'])
    activations = np.load(activations_path)[:np.square(out_dim),:]
    c = list(zip(paths, labels,activations))
    random.shuffle(c)
    paths, labels,activations= zip(*c)
    print('loading images')
    img_collection = load_img(paths,out_dim,out_res)
    print('generating tsne ')
    
    X_2d = generate_tsne(activations,tsne_iter)
    print("Generating image grid.")
    save_tsne_grid(img_collection,labels[:np.square(out_dim)], X_2d, out_res, out_dim,out_dir,out_name,quality=quality_saving,subsampling=subsampling_saving)

if __name__ == '__main__':
    import os 
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int,default=42, help="number of small images in a row/column in output image")
    parser.add_argument('-d', '--dir', type=str, help="source directory for images")
    parser.add_argument('-r', '--res', type=int, default=150, help="width/height of output square image")
    parser.add_argument('-n', '--name', type=str, default='tsne_grid.jpg', help='name of output image file')
    parser.add_argument('-p', '--path', type=str, default='./', help="destination directory for output image")
    parser.add_argument('-x', '--per', type=int, default=50, help="tsne perplexity")
    parser.add_argument('-i', '--iter', type=int, default=5000, help="number of iterations in tsne algorithm") # leaves_fossils_resnet50_leaves_lr0.01_B40_caf10_iter10K_lambda1_trn_mode_hard_anchor__debug_anchor_resize_fixed_299_0
    parser.add_argument('-f','--features_path',type=str,default='/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/checkpoints/leaves_fossils_resnet50_leaves_lr0.01_B40_caf10_iter10K_lambda1_trn_mode_hard_anchor__debug_anchor_resize_fixed_299_0/features/',help='features saved path') #validation_pnas_resnet50_leaves_pretrained_lr0.01_B45_caf10_iter15K_lambda1_trn_mode_hard__validation_5050_pretrained_log_0
    parser.add_argument('-data','--dataset',type=str,default ='/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/datasets/validation/test_data.csv',help='Used dataset for features')
    args = parser.parse_args()
    out_res = args.res
    out_name = args.name
    out_dim = args.size
    to_plot = np.square(out_dim)
    perplexity = args.per
    tsne_iter = args.iter
    out_name = 'tsne_visualization_%05d_%02d.png'%(tsne_iter,out_dim)
    
    features_path = args.features_path #'/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/checkpoints/validation_pnas_resnet50_leaves_pretrained_lr0.01_B45_caf10_iter15K_lambda1_trn_mode_hard__validation_5050_pretrained_log_0/features/'
    dataset = args.dataset #'/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/datasets/validation/test_data.csv'
    pooling_path = os.path.join(features_path,'pooling.npy') 
    
    make_tsne(dataset,pooling_path,out_dim,out_res,features_path,out_name)