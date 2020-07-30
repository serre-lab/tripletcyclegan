import numpy as np
import tensorflow as tf
import tf2lib as tl


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,grayscale=False, shuffle=False, repeat=1):
    if training:
       
        @tf.function
        def _map_fn(img):  # preprocessing
            #toss = np.random.uniform(0,1)
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, grayscale=True,repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1
    A_dataset = make_dataset(A_img_paths,batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths,batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)


def make_dataset2(img_paths, labels, batch_size, load_size, crop_size, training, drop_remainder=True,grayscale=False, shuffle=False, repeat=1):
    if training:
       
        @tf.function
        def _map_fn(img,label):  # preprocessing
            
            #toss = np.random.uniform(0,1)
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return [img,label]
    else:
        @tf.function
        def _map_fn(img,label):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return [img,label]

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       labels=labels,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset2(A_img_paths,A_labels, B_img_paths,B_labels, batch_size, load_size, crop_size, training, shuffle=True, grayscale=True,repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1
    A_dataset = make_dataset2(A_img_paths,A_labels, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=A_repeat)
    B_dataset = make_dataset2(B_img_paths,B_labels, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset

def make_dataset_triplet(img_paths, labels, batch_size, load_size, crop_size, training, num_classes=18,drop_remainder=True,grayscale=False, shuffle=False, repeat=1):

    if training:
       
        @tf.function
        def _map_fn(img,label):  # preprocessing
            
            #toss = np.random.uniform(0,1)
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img, tf.one_hot(label, num_classes,dtype=tf.int32)
    else:
        @tf.function
        def _map_fn(img,label):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img, tf.one_hot(label, num_classes,dtype=tf.int32)
    
    return tl.disk_image_batch_dataset_triplet(img_paths,
                                       batch_size,
                                       crop_size,
                                       labels=labels,
                                       drop_remainder=drop_remainder,
                                       shuffle=shuffle,
                                       repeat=repeat)

def make_zip_dataset_triplet(A_img_paths,A_labels, B_img_paths,B_labels, batch_size, load_size, crop_size, training, shuffle=True, grayscale=True,repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1
    A_dataset = make_dataset_triplet(A_img_paths,A_labels, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=A_repeat)
    B_dataset = make_dataset_triplet(B_img_paths,B_labels, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset,len_dataset