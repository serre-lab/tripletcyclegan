import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
import pandas as pd
import data
import module
import neptune 
import os 
import tensorflow_addons as tfa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, recall_score,precision_score, f1_score
from tsne import save_tsne_grid
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


#from tensorflow import ConfigProto
#from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
tf.config.experimental.set_lms_enabled(True)
neptune.set_project('Serre-Lab/paleo-ai')


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--outdir', default='/users/irodri15/data/irodri15/Fossils/Experiments/cyclegan/checkpoints/')
py.arg('--train_datasetA', default='/users/irodri15/data/irodri15/Fossils/Experiments/datasets/gan_fossils_leaves_v1/fossils_train_oscar_processed.csv')
py.arg('--train_datasetB', default='/users/irodri15/data/irodri15/Fossils/Experiments/datasets/gan_fossils_leaves_v1/leaves_train_oscar_processed.csv')
py.arg('--test_datasetA', default='/users/irodri15/data/irodri15/Fossils/Experiments/datasets/gan_fossils_leaves_v1/fossils_test_oscar_processed.csv')
py.arg('--test_datasetB', default='/users/irodri15/data/irodri15/Fossils/Experiments/datasets/gan_fossils_leaves_v1/leaves_test_oscar_processed.csv')
py.arg('--experiment_name')
py.arg('--kernels_num', type=int, default=64)
py.arg('--load_size', type=int, default=600)  # load image to this size
py.arg('--crop_size', type=int, default=600)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--batch_size_triplet', type=int, default=15)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=50)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=1.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--triplet_loss_weight', type=float, default=1.5)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--grayscale',type=bool,default= False)
py.arg('--triplet_margin', type = float, default= 1.0)
py.arg('--evaluate_every', type = int, default= 500)
args = py.args()

params = vars(args)
neptune.create_experiment(name=args.experiment_name,params=params)
neptune.append_tag('cycleGAN')
# output_dir
output_dir = os.path.join(args.outdir,args.experiment_name)#py.join('output', args.outdir)
print(output_dir)
os.makedirs(output_dir,exist_ok=True)

# save settings
print(output_dir)
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================
A = pd.read_csv(args.train_datasetA)
B = pd.read_csv(args.train_datasetB)
print(B)
A_img_paths = list(A['file_name']) #py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
A_labels = list(A['label'])
print(type(A_img_paths[0]))
B_img_paths = list(B['file_name'])#py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
B_labels = list(B['label'])



A_test = pd.read_csv(args.test_datasetA)
B_test = pd.read_csv(args.test_datasetB)
A_img_paths_test = list(A_test['file_name'])#py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
A_test_labels = list(A_test['label'])
B_img_paths_test = list(B_test['file_name'])#py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
B_test_labels = list(B_test['label'])
with tf.device('/device:GPU:0'):
    A_B_dataset_test, _ = data.make_zip_dataset3(A_img_paths_test,A_test_labels, B_img_paths_test,B_test_labels, args.batch_size, args.load_size, args.crop_size, training=False, grayscale=args.grayscale, repeat=True)
    A_B_dataset, len_dataset = data.make_zip_dataset3(A_img_paths,A_labels, B_img_paths,B_labels, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False,shuffle=False,grayscale=args.grayscale)
    A2B_pool = data.ItemPool(args.pool_size)
    B2A_pool = data.ItemPool(args.pool_size)

with tf.device('/device:GPU:1'):
        
    A_B_dataset_triplet, len_dataset_triplet = data.make_zip_dataset_triplet2(A_img_paths,A_labels, B_img_paths,B_labels, args.batch_size_triplet, args.load_size, args.crop_size,Triplet_K=3, training=True, repeat=False,shuffle=True,grayscale=args.grayscale)

    A_B_dataset_test_triplet,len_dataset_test_triplet = data.make_zip_dataset3(A_img_paths_test,A_test_labels, B_img_paths_test,B_test_labels, args.batch_size_triplet, args.load_size, args.crop_size, training=False, grayscale=args.grayscale, repeat=False,shuffle =False)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================
with tf.device('/device:GPU:0'):
    G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3),dim=args.kernels_num)
    G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3),dim=args.kernels_num)
    cycle_loss_fn = tf.losses.MeanAbsoluteError()
    identity_loss_fn = tf.losses.MeanAbsoluteError()
    G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
    G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)

with tf.device('/device:GPU:2'):  
    D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3),dim=args.kernels_num)
    D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3),dim=args.kernels_num)
    d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
    D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
    D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)
    
with tf.device('/device:GPU:1'):
    T_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset_triplet, args.epoch_decay * len_dataset_triplet)
    T = module.Resnet50embeddings(input_shape=(args.crop_size, args.crop_size, 3),embedding_size=256)
    T_optimizer = keras.optimizers.Adam(learning_rate=T_lr_scheduler, beta_1=args.beta_1)

#triplet_loss_fn = tf.losses.MeanAbsoluteError()
#classification_loss_fn = tf.losses.BCE

#gpus = tf.config.list_physical_devices('GPU')

#tf.debugging.set_log_device_placement(True)
#gpus = tf.config.experimental.list_logical_devices('GPU')
# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B,A_triplet,B_triplet):
    with tf.GradientTape() as t:

        with tf.device('/device:GPU:0'):
            A2B = G_A2B(A, training=True)
            B2A = G_B2A(B, training=True)
            A2B2A = G_B2A(A2B, training=True)
            B2A2B = G_A2B(B2A, training=True)
            A2A = G_B2A(A, training=True)
            B2B = G_A2B(B, training=True)
            
            A2B_d_logits = D_B(A2B, training=True)
            B2A_d_logits = D_A(B2A, training=True)

            A2B_g_loss = g_loss_fn(A2B_d_logits)
            B2A_g_loss = g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
            B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
            A2A_id_loss = identity_loss_fn(A, A2A)
            B2B_id_loss = identity_loss_fn(B, B2B)
        
        with tf.device('/device:GPU:1'):
            
            A2B_triplet = tf.TensorArray(tf.float32,size=0,dynamic_size =True)
            i=0
            for A_t in A_triplet[0]:
               A2B_triplet = A2B_triplet.write(i,G_A2B(tf.expand_dims(A_t,axis=0),training=False))
               i+=1
            B2A_triplet = tf.TensorArray(tf.float32,size=0,dynamic_size =True)
            j=0
            for B_t in B_triplet[0]:
                B2A_triplet = B2A_triplet.write(j,G_B2A(tf.expand_dims(B_t,axis=0),training=False))
                j+=1
                
            TA = T(A_triplet[0])
            TB = T(B_triplet[0])
            TA2B = T(A2B_triplet.concat())
            TB2A = T(B2A_triplet.concat())
            triplet_a_loss = tfa.losses.triplet_hard_loss(A_triplet[1],TA,margin = args.triplet_margin)
            triplet_b_loss = tfa.losses.triplet_hard_loss(B_triplet[1],TB,margin = args.triplet_margin)
            triplet_a2b_loss = tfa.losses.triplet_hard_loss(A_triplet[1],TA2B,margin=args.triplet_margin)
            triplet_b2a_loss = tfa.losses.triplet_hard_loss(B_triplet[1],TB2A,margin=args.triplet_margin)

        
        
        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight + (triplet_a_loss + triplet_b_loss+triplet_a2b_loss+triplet_b2a_loss )*0.25* args.triplet_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss,
                      'TA_loss':triplet_a_loss,
                      'TB_loss':triplet_b_loss,
                      'TA2B_loss':triplet_a2b_loss,
                      'TB2A_loss':triplet_b2a_loss,
                      'G_loss':G_loss }


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        with tf.device('/device:GPU:2'):
            A_d_logits = D_A(A, training=True)
            B2A_d_logits = D_A(B2A, training=True)
            B_d_logits = D_B(B, training=True)
            A2B_d_logits = D_B(A2B, training=True)

            A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
            D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
            D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

            D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}

@tf.function
def train_T(A_triplet,B_triplet):
    with tf.GradientTape() as t:
        with tf.device('/device:GPU:1'):
            TA = T(A_triplet[0],training=True)
            TB = T(B_triplet[0],training=True)
            A2B_triplet = G_A2B(A_triplet[0])
            B2A_triplet = G_B2A(B_triplet[0])
            TA2B = T(A2B_triplet)
            TB2A = T(B2A_triplet)
            triplet_a_loss = tfa.losses.triplet_hard_loss(A_triplet[1],TA,margin=args.triplet_margin)
            triplet_b_loss = tfa.losses.triplet_hard_loss(B_triplet[1],TB,margin=args.triplet_margin)
            triplet_a2b_loss = tfa.losses.triplet_hard_loss(A_triplet[1],TA2B,margin = args.triplet_margin)
            triplet_b2a_loss = tfa.losses.triplet_hard_loss(B_triplet[1],TB2A,margin = args.triplet_margin)

            T_loss = triplet_a_loss + triplet_b_loss +triplet_a2b_loss + triplet_b2a_loss
    T_grad = t.gradient(T_loss,T.trainable_variables)
    T_optimizer.apply_gradients(zip(T_grad,T.trainable_variables))
        
    return {'T_A_loss': triplet_a_loss,
            'T_B_loss': triplet_b_loss,
            'T_A2B_loss': triplet_a2b_loss,
            'T_B2A_loss': triplet_b2a_loss,
            'T_loss': T_loss
            }


def train_step(A, B,A_triplet,B_triplet):

    
    A2B, B2A, G_loss_dict = train_G(A, B,A_triplet,B_triplet)
    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU
    D_loss_dict = train_D(A, B, A2B, B2A)
    T_loss_dict = train_T(A_triplet,B_triplet)


    return G_loss_dict, D_loss_dict,T_loss_dict


@tf.function
def sample(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B

@tf.function
def test_triplet(A_triplet,B_triplet):
    TA = T(A_triplet,training=False)
    TB = T(B_triplet,training=False)
    A2B_triplet = G_A2B(A_triplet,training=False)
    B2A_triplet = G_B2A(B_triplet,training=False)
    TA2B = T(A2B_triplet,training=False)
    TB2A = T(B2A_triplet,training=False)

    return TA,TB,TA2B,TB2A

def clasifyKnn( X,y, K=1):
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(X, y)
    pred = neigh.predict(X)
    return pred

def metrics_triplet(y_true,y_pred,title):
    print(title)
    print(classification_report(y_true,y_pred))
    return recall_score(y_true,y_pred,average='micro'),precision_score(y_true,y_pred,average='micro'),f1_score(y_true,y_pred,average='micro')
# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
train_triplet_iter = iter(A_B_dataset_triplet)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        #for A_triplet, B_triplet in tqdm.tqdm(A_B_dataset_triplet, desc='Inner Epoch Loop', total=len_dataset):
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            A_triplet, B_triplet = next(train_triplet_iter)

            G_loss_dict, D_loss_dict,T_loss_dict = train_step(A[0], B[0],A_triplet,B_triplet)

            ## Logging to neptune
            for k in  G_loss_dict : neptune.log_metric(k,G_loss_dict[k])
            for k in  D_loss_dict : neptune.log_metric(k,D_loss_dict[k])
            for k in  T_loss_dict : neptune.log_metric(k,T_loss_dict[k])
            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary(T_loss_dict, step=T_optimizer.iterations, name='T_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')
            # sample
            if G_optimizer.iterations.numpy() %  args.evaluate_every == 0:
                A, B = next(test_iter)
            
                A2B, B2A, A2B2A, B2A2B = sample(A[0], B[0])
                img = im.immerge(np.concatenate([A[0], A2B, A2B2A, B[0], B2A, B2A2B], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
                neptune.log_image( 'iter-%09d'%G_optimizer.iterations.numpy(), py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
                count = 0
                for A_t, B_t in tqdm.tqdm(A_B_dataset_test_triplet, desc='Testing Triplet', total=len_dataset_test_triplet): 
                
                    TA,TB,TA2B,TB2A = test_triplet(A_t[0],B_t[0])
                    if count==0:
                        embeddingsA,embeddingsB,embeddingsA2B,embeddingsB2A = TA,TB,TA2B,TB2A
                        labelsA,labelsB = A_t[1],B_t[1]
                        imagesA = A_t[0]
                        imagesB = B_t[0]
                    else:
                        embeddingsA = np.concatenate([embeddingsA, TA]) 
                        embeddingsB = np.concatenate([embeddingsB, TB])
                        embeddingsA2B = np.concatenate([embeddingsA2B, TA2B])
                        embeddingsB2A = np.concatenate([embeddingsB2A, TB2A]) 
                        labelsA = np.concatenate([labelsA,A_t[1]])
                        labelsB = np.concatenate([labelsB,B_t[1]])
                        imagesA = np.concatenate([imagesA,A_t[0]])
                        imagesB = np.concatenate([imagesB,B_t[0]])
                    count += 1
                print('batches %05d total %05d'%(count,count*args.batch_size_triplet))
                out_dim =42
                imagesAB = np.concatenate([imagesA[:np.square(out_dim)//2],imagesB[:np.square(out_dim)]//2])[:np.square(out_dim)]
                embeddingsAB = np.concatenate([embeddingsA[:np.square(out_dim)//2],embeddingsB[:np.square(out_dim)//2]])[:np.square(out_dim)]
                labelsAB = np.concatenate([labelsA[:np.square(out_dim)//2],labelsB[:np.square(out_dim)//2]])[:np.square(out_dim)]
            
            
                tsne_name = 'tsne_visualization_%05d.png'%G_optimizer.iterations.numpy()
                #import pdb;pdb.set_trace()
                save_tsne_grid(imagesAB,labelsAB,embeddingsAB,args.load_size,out_dim,output_dir,out_name=tsne_name,border=10)
                neptune.log_image('tsne',os.path.join(output_dir,tsne_name))
                pred_A = clasifyKnn(embeddingsA,labelsA,K=3) 
                pred_B = clasifyKnn(embeddingsB,labelsB,K=3) 
                pred_A2B = clasifyKnn(embeddingsA2B,labelsA,K=3)
                pred_B2A = clasifyKnn(embeddingsB2A,labelsB,K=3)
                recallA, precisionA,f1A = metrics_triplet(labelsA,pred_A,title='A') 
                recallB, precisionB,f1B = metrics_triplet(labelsB,pred_B,title='B')
                recallA2B, precisionA2B,f1A2B = metrics_triplet(labelsA,pred_A2B,title='A2B')
                recallB2A, precisionB2A,f1B2A = metrics_triplet(labelsB,pred_B2A,title='B2A')
                neptune.log_metric('f1_A',f1A)
                neptune.log_metric('f1_B',f1B)
                neptune.log_metric('f1_A2B',f1A2B)
                neptune.log_metric('f1_B2A',f1B2A)
                #neptune.log_metric('f1_top1_')
                #neptune.log_metric('f1_top1_')
                #neptune.log_metric('f1_top1_')
        # sve checkpoint
        checkpoint.save(ep)
