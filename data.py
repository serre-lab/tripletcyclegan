import numpy as np
import tensorflow as tf
import tf2lib as tl

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.compat.v1.to_float(height)
  width = tf.compat.v1.to_float(width)
  smallest_side = tf.compat.v1.to_float(smallest_side)

  scale = tf.compat.v1.cond(tf.compat.v1.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.compat.v1.to_int32(height * scale)
  new_width = tf.compat.v1.to_int32(width * scale)
  return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.compat.v1.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.compat.v1.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image



def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,grayscale=False, shuffle=False, repeat=1):
    if training:
       
        @tf.function
        def _map_fn(img):  # preprocessing
            #toss = np.random.uniform(0,1)
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.random_flip_left_right(img)
            
            img = tf.image.resize_with_pad(img, load_size, load_size, antialias = True)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize_with_pad(img,crop_size, crop_size, antialias = True)  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
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
            maxside = tf.math.maximum(tf.shape(img)[0],tf.shape(img)[1])
            while tf.math.square(tf.shape(img)[0]-tf.shape(img)[1])>100:
                padx = tf.math.minimum(maxside - tf.shape(img)[0],tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1]))
                pady = tf.math.minimum(maxside - tf.shape(img)[1],tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1]))
                paddings = [[padx/2,padx/2],[pady/2,pady/2],[0, 0]]
                img = tf.pad(img,paddings,'SYMMETRIC')#tf.image.resize_with_pad(img, load_size, load_size, antialias = True)
            img = tf.image.resize(img, [load_size*+10,load_size+10],preserve_aspect_ratio=True)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return [img,label]
    else:
        @tf.function
        def _map_fn(img,label):  # preprocessing
            img =_aspect_preserving_resize(img,load_size)# tf.image.resize(img, [load_size,load_size])
            #padx = load_size - tf.shape(img)[0]
            #pady = load_size -tf.shape(img)[1]
            #paddings = [[padx/2,padx/2],[pady/2,pady/2],[0, 0]]
            #img = tf.pad(img,paddings,'SYMMETRIC')
            #img = tf.image.resize_with_pad(img,crop_size, crop_size, antialias = True)  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
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

def make_dataset_triplet(img_paths, labels, batch_size, load_size, crop_size, training,Triplet_K=4, num_classes=18,drop_remainder=True,grayscale=False, shuffle=False, repeat=1):

    if training:
       
        @tf.function
        def _map_fn(img,label):  # preprocessing
            
            #toss = np.random.uniform(0,1)
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
            #img = tf.image.random_flip_left_right(img)
            maxside = tf.math.maximum(tf.shape(img)[0],tf.shape(img)[1])
            while tf.math.square(tf.shape(img)[0]-tf.shape(img)[1])>100:
                padx = tf.math.minimum(maxside - tf.shape(img)[0],tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1]))
                pady = tf.math.minimum(maxside - tf.shape(img)[1],tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1]))
                paddings = [[padx/2,padx/2],[pady/2,pady/2],[0, 0]]
                img = tf.pad(img,paddings,'SYMMETRIC')#tf.image.resize_with_pad(img, load_size, load_size, antialias = True)
            img = tf.image.resize(img, [load_size*+10,load_size+10],preserve_aspect_ratio=True)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img, tf.one_hot(label, num_classes,dtype=tf.int32)
    else:
        @tf.function
        def _map_fn(img,label):  # preprocessing
            
            img = _aspect_preserving_resize(img,load_size-60) #tf.image.resize(img, crop_size, crop_size, antialias = True)  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            #img = tf.image.resize(img, [load_size*+10,load_size+10],preserve_aspect_ratio=True)
            #padx = load_size - tf.shape(img)[0]
            #pady = load_size -tf.shape(img)[1]
            #paddings = [[padx/2,padx/2],[pady/2,pady/2],[0, 0]]
            #img = tf.pad(img,paddings,'SYMMETRIC')#tf.image.resize_with_pad(img, load_size, load_size, antialias = True)
            tf.print(tf.shape(img))
            if grayscale:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img, tf.one_hot(label, num_classes,dtype=tf.int32)
    
    return tl.disk_image_batch_dataset_triplet(img_paths,
                                       batch_size,
                                       crop_size,
                                       labels=labels,
                                       Triplet_K=Triplet_K,
                                       drop_remainder=drop_remainder,
                                       shuffle=shuffle,
                                       repeat=repeat)

def make_zip_dataset_triplet(A_img_paths,A_labels, B_img_paths,B_labels, batch_size, load_size, crop_size, training,Triplet_K=4, shuffle=True, grayscale=True,repeat=False):
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
    A_dataset = make_dataset_triplet(A_img_paths,A_labels, batch_size, load_size, crop_size, training,Triplet_K=Triplet_K, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=A_repeat)
    B_dataset = make_dataset_triplet(B_img_paths,B_labels, batch_size, load_size, crop_size, training,Triplet_K=Triplet_K, drop_remainder=True, shuffle=shuffle, grayscale=grayscale, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset,len_dataset