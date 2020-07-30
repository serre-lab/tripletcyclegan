import os
import getpass
import argparse
import constants

# import constants as const

TRAIN_MODE_CHOICES = (
    'vanilla', # Softmax loss only
    'semi_hard', # Softmax  + semi-hard triplet loss
    'hard', # Softmax  + hard triplet loss
    'hard_awtl', # Softmax  + Adaptive weight triplet loss
    'cntr',
    'hard_anchor',
    'hard_anchor_fossils' # Softmax  + center loss
    # 'mgnt', # Softmax  + Magnet loss (not supported in this code base)
)

DB_CHOICES = (
    'flowers',
    'aircrafts',
    'pnas',
    'leaves',
    'fossil',
    'leaves_61',
    'leaves_pnas',
    'leaves_fossils',
    'gan_leaves_fossils',
    'all_gan_leaves_fossils',
    'gan_leaves_fossils_leaves',
    'leaves_fossils_out',
    'validation_fossils',
    'validation_leaves',
    'validation_pnas',
    'leaves_out',
    'pretrain'
    # 'dogs',
    # 'birds',
    # 'cars',
)

IMG_AUG_STYLE = (
    'batch',
    'img',
)

def float_or_string(arg):
    """Tries to convert the string to float, otherwise returns the string."""
    try:
        return float(arg)
    except (ValueError, TypeError):
        return arg

class BaseConfig:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--emb_dim', type=int, default=256,
                                 help='Embedding dimension')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch Size')
        self.parser.add_argument('--gpu', type=str, default='3',
                                 help='which gpu')
        self.parser.add_argument('--checkpoint_dir', type=str, default=None,
                                 help='where to save experiment log and model')
        self.parser.add_argument('--db_name', type=str, default='flowers',choices=DB_CHOICES,
                                 help='Database name')
        self.parser.add_argument('--net', type=str, default='resnet50',
                                 help='Which networks? resnet50, inc4,densenet161')
        self.parser.add_argument('--tuple_loader_queue_size', type=int, default=10,
                                 help='')
        self.parser.add_argument('--train_mode', type=str, default='vanilla',choices=TRAIN_MODE_CHOICES,
                                 help='')

        self.parser.add_argument('--aug_style', type=str, default='batch', choices=IMG_AUG_STYLE,
                                 help='Group augment images per batch or augment images individually')
        self.parser.add_argument('--aug_rotate', type=int, default=120,
                                 help='Group augment images per batch or augment images individually')

        self.parser.add_argument('--triplet_loss_lambda', type=float, default=1,
                                 help='')
        self.parser.add_argument('--softmax_loss_lambda', type=float, default=1,
                                 help='')
        self.parser.add_argument('--caffe_iter_size', type=int, default=1,
                                 help='')
        self.parser.add_argument('--logging_threshold', type=int, default=500,
                                 help='')
        self.parser.add_argument('--test_interval', type=int, default=10,
                                 help='')
        self.parser.add_argument('--train_iters', type=int, default=40000,
                                 help='')

        self.parser.add_argument('--margin', type=float_or_string, default=0.2,
                                 help='')

        self.parser.add_argument('--Triplet_K', type=int, default=4,
                                 help='')
        self.parser.add_argument('--checkpoint_suffix', type=str, default='base_config',
                                 help='')

        self.parser.add_argument('--checkpoint_filename', type=str, default='model.ckpt',
                                 help='')

        self.parser.add_argument('--learning_rate', type=float, default=0.01,
                                 help='')
        self.parser.add_argument('--end_learning_rate', type=float, default=0,
                                 help='')

        self.parser.add_argument('--log_filename', type=str, default='logger',
                                 help='')

        self.parser.add_argument('--frame_size', type=int, default=constants.frame_width,
                                 help='')
        self.parser.add_argument('--username',type=str,default=None)
        self.parser.add_argument('--label_out',type=int,default=None)
        self.parser.add_argument('--training_mode_debug',type=bool,default=False)
        self.parser.add_argument('--tsne_visualization',type=bool,default=False)
        self.parser.add_argument('--tsne_log_every',type=int, default=500)
        self.parser.add_argument('--tsne_iter',type=int,default=5000)
        self.parser.add_argument('--tsne_out_dim',type=int,default=42)
        self.parser.add_argument('--tsne_out_res',type=int,default=150)

    def _load_user_setup(self,username=None):

        
        if username==None:
            username = getpass.getuser()
        else:
            username = username 
        print(username)
        if username == 'irodri15':  ## VC
            local_datasets_dir = '/media/data_cifs2/irodri15/Leafs/Experiments/softmax_triplet/datasets/'
            pretrained_weights_dir = '/media/data_cifs2/irodri15/Leafs/Experiments/softmax_triplet/pretrained/'
            training_models_dir = '/media/data_cifs2/irodri15/Leafs/Experiments/softmax_triplet/checkpoints/'
            caffe_iter_size = 1
            logging_threshold = 100
            batch_size = 32
        elif username == 'irodri15p1':  ## VC
            local_datasets_dir = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/datasets/'
            pretrained_weights_dir = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/pretrained/'
            training_models_dir = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/checkpoints/'
            caffe_iter_size = 1
            logging_threshold = 100
            batch_size = 32
        elif username == 'irodri15_validation':  ## VC
            local_datasets_dir = '/media/data_cifs/jacob/Fossil_Project/replication_data/'
            pretrained_weights_dir = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/pretrained/'
            training_models_dir = '/media/data_cifs/irodri15/Leafs/Experiments/softmax_triplet/checkpoints/'
            caffe_iter_size = 1
            logging_threshold = 100
            batch_size = 32
        elif username == 'irodri15_oscar':
            local_datasets_dir = '/users/irodri15/scratch/Fossils/Experiments/softmax_triplet/datasets/'
            pretrained_weights_dir = '/users/irodri15/scratch/Fossils/Experiments/softmax_triplet/pretrained/'
            training_models_dir = '/users/irodri15/scratch/Fossils/Experiments/softmax_triplet/checkpoints/'
            caffe_iter_size = 1
            logging_threshold = 100
            batch_size = 32

        else:
            raise NotImplementedError('Invalid username {}. Please set the configuration of this username/machine inside config/base_config.py'.format(username))


        return local_datasets_dir,pretrained_weights_dir,training_models_dir,logging_threshold,batch_size,caffe_iter_size

    def parse(self,args):
        cfg = self.parser.parse_args(args)
        local_datasets_dir, pretrained_weights_dir, training_models_dir, logging_threshold, batch_size, caffe_iter_size = self._load_user_setup(cfg.username)
        cfg.num_classes, cfg.db_path, cfg.db_tuple_loader, cfg.train_csv_file, cfg.val_csv_file, cfg.test_csv_file    = self.db_configuration(cfg.db_name,local_datasets_dir,cfg.label_out)
        cfg.network_name, cfg.imagenet__weights_filepath, cfg.preprocess_func, cfg.preprocessing_module = self._load_net_configuration(cfg.net,pretrained_weights_dir)

        if cfg.checkpoint_dir is None:
            checkpoint_dir = [cfg.db_name, cfg.net, 'lr' + str(cfg.learning_rate), 'B' + str(cfg.batch_size),
                              'caf' + str(cfg.caffe_iter_size), 'iter' + str(cfg.train_iters // 1000) + 'K',
                              'lambda' + str(cfg.triplet_loss_lambda), 'trn_mode_' + str(cfg.train_mode),
                              cfg.checkpoint_suffix]
            checkpoint_dir = '_'.join(checkpoint_dir)
            cfg.checkpoint_dir = os.path.join(training_models_dir, checkpoint_dir)
        else:
            cfg.checkpoint_dir = os.path.join(training_models_dir,cfg.checkpoint_dir)


        cfg.test_interval = cfg.test_interval * cfg.logging_threshold

        return cfg

    def _load_net_configuration(self,model,pretrained_weights_dir):
        if model == 'resnet50':
            network_name = 'nets.resnet_v2.ResNet50'
            imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v2_50/resnet_v2_50.ckpt'
            preprocess_func = 'inception_v1'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        elif model == 'resnet50_leaves':
            network_name = 'nets.resnet_v2.ResNet50'
            imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v2_50/resnet_v2_50.ckpt'
            preprocess_func = 'inception_leaves'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        elif model == 'resnet50_leaves_pretrained':
            network_name = 'nets.resnet_v2.ResNet50'
            imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v2_50_leaves/model.ckptbest' # model.ckptbest
            preprocess_func = 'inception_leaves'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        elif model == 'resnet50_leaves_color':
            network_name = 'nets.resnet_v2.ResNet50'
            imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v2_50/resnet_v2_50.ckpt'
            preprocess_func = 'inception_leaves_color'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        elif model == 'resnet50_v1':
            network_name = 'nets.resnet_v1.ResNet50'
            imagenet__weights_filepath = pretrained_weights_dir + 'resnet_v1_50/resnet_v1_50.ckpt'
            preprocess_func = 'vgg'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        elif model == 'densenet161':
            network_name = 'nets.densenet161.DenseNet161'
            imagenet__weights_filepath = pretrained_weights_dir + 'tf-densenet161/tf-densenet161.ckpt'
            preprocess_func = 'densenet'
            preprocessing_module = 'data_sampling.augmentation.densenet_preprocessing'
        elif model == 'inc4':
            network_name = 'nets.inception_v4.InceptionV4'
            imagenet__weights_filepath = pretrained_weights_dir + 'inception_v4/inception_v4.ckpt'
            preprocess_func = 'inception_v1'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        elif model == 'inc3':
            network_name = 'nets.inception_v3.InceptionV3'
            imagenet__weights_filepath = pretrained_weights_dir + 'inception_v3.ckpt'
            preprocess_func = 'inception_v1'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        elif model == 'mobile':
            network_name = 'nets.mobilenet_v1.MobileV1'
            imagenet__weights_filepath = pretrained_weights_dir + 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
            preprocess_func = 'inception_v1'
            preprocessing_module = 'data_sampling.augmentation.inception_preprocessing'
        else:
            raise NotImplementedError('network name not found')

        return network_name,imagenet__weights_filepath,preprocess_func,preprocessing_module

    def db_configuration(self, dataset_name, datasets_dir,label_out=None):

        if dataset_name == 'flowers':
            num_classes = 102
            db_path = datasets_dir + 'flower102'
            db_tuple_loader = 'data_sampling.flower_tuple_loader.FLower102TupleLower'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_all_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'cars':
            num_classes = 196
            db_path = datasets_dir + 'stanford_cars'
            db_tuple_loader = 'data_sampling.cars_tuple_loader.CarsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_all_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'aircrafts':
            num_classes = 100
            db_path = datasets_dir + 'aircrafts'
            db_tuple_loader = 'data_sampling.aircrafts_tuple_loader.AircraftsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_all_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'dogs':
            num_classes = 120
            db_path = datasets_dir + 'Stanford_dogs'
            db_tuple_loader = 'data_sampling.dogs_tuple_loader.DogsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'birds':
            num_classes = 555
            db_path = datasets_dir + 'nabirds'
            db_tuple_loader = 'data_sampling.birds_tuple_loader.BirdsTupleLoader'
            train_csv_file = '/lists/train_all_sub_list.csv'
            val_csv_file = '/lists/val_sub_list.csv'
            test_csv_file = '/lists/test_all_sub_list.csv'
        elif dataset_name == 'pnas':
            num_classes = 19
            db_path = datasets_dir +'pnas'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_pnas_list.csv'
            val_csv_file = '/lists/val_pnas_list.csv'
            test_csv_file = '/lists/test_pnas_list.csv'
        elif dataset_name == 'leaves':
            num_classes = 190
            db_path = datasets_dir +'leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_leaves_list.csv'
            val_csv_file = '/lists/val_leaves_list.csv'
            test_csv_file = '/lists/test_leaves_list.csv'
        elif dataset_name =='pretrain':
            num_classes = 190
            db_path = datasets_dir +'leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/pretrain.csv'
            val_csv_file = '/lists/pretrain.csv'
            test_csv_file = '/lists/pretrain.csv'
        elif dataset_name == 'leaves_61':
            num_classes = 61
            db_path = datasets_dir +'leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_leaves_list_61.csv'
            val_csv_file = '/lists/val_leaves_list_61.csv'
            test_csv_file = '/lists/test_leaves_list_61.csv'

        elif dataset_name == 'leaves_pnas':
            num_classes =19
            db_path = datasets_dir +'leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_leaves_list_pnas.csv'
            val_csv_file = '/lists/val_leaves_list_pnas.csv'
            test_csv_file = '/lists/test_leaves_list_pnas.csv'
        elif dataset_name == 'leaves_pnas':
            num_classes =19
            db_path = datasets_dir +'leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_leaves_list_pnas.csv'
            val_csv_file = '/lists/val_leaves_list_pnas.csv'
            test_csv_file = '/lists/test_leaves_list_pnas.csv'

        elif dataset_name == 'leaves_fossils':
            num_classes = 18
            db_path = datasets_dir +'leaves_fossils'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_leaves_fossils_list.csv'
            val_csv_file = '/lists/val_leaves_fossils_list.csv'
            test_csv_file = '/lists/test_leaves_fossils_list.csv'

        elif dataset_name == 'gan_leaves_fossils':
            num_classes = 18
            db_path = datasets_dir +'gan_fossils_leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/gan_train_leaves_fossils_list.csv'
            val_csv_file = '/lists/val_leaves_fossils_list.csv'
            test_csv_file = '/lists/test_leaves_fossils_list.csv'

        elif dataset_name == 'gan_leaves_fossils_leaves':
            num_classes = 18
            db_path = datasets_dir +'gan_fossils_leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/gan_train_leaves_fossils_list_leaves.csv'
            val_csv_file = '/lists/val_leaves_fossils_list.csv'
            test_csv_file = '/lists/test_leaves_fossils_list.csv'

        elif dataset_name == 'all_gan_leaves_fossils':
            num_classes = 18
            db_path = datasets_dir +'gan_fossils_leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/all_gan_train_leaves_fossils_list.csv'
            val_csv_file = '/lists/val_leaves_fossils_list.csv'
            test_csv_file = '/lists/test_leaves_fossils_list.csv'

        
        elif dataset_name == 'leaves_fossils_out':
            l = label_out
            num_classes = 18
            db_path = datasets_dir +'leaves_fossils'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_leaves_fossils_list_%02d.csv'%l
            val_csv_file = '/lists/test_leaves_fossils_list_%02d.csv'%l
            test_csv_file = '/lists/test_leaves_fossils_list_%02d.csv'%l
            
        elif dataset_name == 'leaves_out':
            l = label_out
            num_classes = 18
            db_path = datasets_dir +'leaves_fossils'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_leaves_out_list_000_%02d.csv'%l
            val_csv_file = '/lists/test_leaves_out_list_000_%02d.csv'%l
            test_csv_file = '/lists/test_leaves_out_list_000_%02d.csv'%l
        elif dataset_name == 'fossil':
            num_classes = 20
            db_path = datasets_dir +'fossil'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/lists/train_fossil_list.csv'
            val_csv_file = '/lists/val_fossil_list.csv'
            test_csv_file = '/lists/test_fossil_list.csv'

        elif dataset_name == 'validation_fossils':
            num_classes = 20
            db_path = datasets_dir +'single-domain_experiments/Fossil'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/Fossil/train_data.csv'
            val_csv_file = '/Fossil/val_data.csv'
            test_csv_file = '/Fossil/test_data.csv'

        elif dataset_name == 'validation_leaves':
            num_classes = 190
            db_path = datasets_dir +'single-domain_experiments/Leaves'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/Leaves/train_data.csv'
            val_csv_file = '/Leaves/val_data.csv'
            test_csv_file = '/Leaves/test_data.csv'

        elif dataset_name == 'validation_pnas':
            num_classes = 19
            db_path = datasets_dir +'single-domain_experiments/PNAS'
            db_tuple_loader = 'data_sampling.pnas_tuple_loader.PnasTupleLoader'
            train_csv_file = '/PNAS/train_data.csv'
            val_csv_file = '/PNAS/test_data.csv'
            test_csv_file = '/PNAS/test_data.csv'
        else:
            raise NotImplementedError('dataset_name not found')

        return num_classes,db_path,db_tuple_loader,train_csv_file,val_csv_file,test_csv_file

def parser_triplet(args):
     return  [
            '--db_name',str(args['db_name']),
            '--gpu', str(args['gpu']),
            '--net', str(args['net']),
            '--train_mode',str(args['train_mode']),
            '--margin', str(args['margin']),
            '--batch_size',str(args['batch_size']),
            '--caffe_iter_size', str(args['caffe_iter_size']),
            '--logging_threshold', str(args['logging_threshold']),
            '--train_iters', str(args['train_iters']),
            '--test_interval',str(args['test_interval']),
            '--learning_rate', str(args['learning_rate']),
            '--Triplet_K',str(args['Triplet_K']),
            '--triplet_loss_lambda',str(args['triplet_loss_lambda']),
            '--softmax_loss_lambda',str(args['softmax_loss_lambda']),
            '--tuple_loader_queue_size',str(args['tuple_loader_queue_size']),
            #'--training_mode_debug','True'']),
            '--aug_style', str(args['aug_style']),
            '--aug_rotate',str(args['aug_rotate']),
            '--tsne_visualization',str(args['tsne_visualization']),
            '--tsne_log_every',str(args['tsne_log_every']),
            '--username',str(args['username']),
            '--tsne_iter',str(args['tsne_iter']),
            '--frame_size',str(args['frame_size']),
            '--checkpoint_suffix', str(args['checkpoint_suffix']),
        ]

if __name__ == '__main__':
    args = [
        '--db_name','flowers'
    ]
    cfg = BaseConfig().parse(args)
    print(cfg.num_classes,cfg.train_csv_file)
    if hasattr(cfg,'abc'):
        print(cfg.abc)
    else:
        print('Something is wrong')

