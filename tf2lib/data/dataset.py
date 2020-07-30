import multiprocessing

import tensorflow as tf
import numpy as np
from nets import img_augment
def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset


def disk_image_batch_dataset(img_paths,
                             batch_size,
                             labels=None,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """Batch dataset of disk image for PNG and JPEG.

    Parameters
    ----------
    img_paths : 1d-tensor/ndarray/list of str
    labels : nested structure of tensors/ndarrays/lists

    """
    #import pdb;pdb.set_trace()
    if labels is None:
        memory_data = img_paths
    else:
        memory_data = (img_paths, labels)

    def parse_fn(path, *label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)  # fix channels to 3
        return (img,) + label

    if map_fn:  # fuse `map_fn` and `parse_fn`
        def map_fn_(*args):
            return map_fn(*parse_fn(*args))
    else:
        map_fn_ = parse_fn

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat)

    return dataset






def disk_image_batch_dataset_triplet(img_paths,
                             batch_size,
                             frame_size,
                             labels=None,
                             drop_remainder=True,
                             Triplet_K=4,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):

        def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
            """ Given a PID, select K FIDs of that specific PID. """
            
            possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))
        
            # The following simply uses a subset of K of the possible FIDs
            # if more than, or exactly K are available. Otherwise, we first
            # create a padded list of indices which contain a multiple of the
            # original FID count such that all of them will be sampled equally likely.
            count = tf.shape(possible_fids)[0]
            padded_count = tf.cast(tf.math.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
            full_range = tf.math.mod(tf.range(padded_count), count)

            # Sampling is always performed by shuffling and taking the first k.
            shuffled = tf.random.shuffle(full_range)
            selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

            return selected_fids, tf.fill([batch_k], pid)

        
        train_imgs = np.array(img_paths)
        train_lbls = np.array(labels,dtype=np.int64)
        unique_pids = np.unique(labels)
        class_per_batch = batch_size / Triplet_K
        if len(unique_pids) < class_per_batch:
            unique_pids = np.tile(unique_pids, int(np.ceil(class_per_batch / len(unique_pids))))
        print(train_imgs)
        dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
        dataset = dataset.shuffle(len(unique_pids))
        num_classes = batch_size // Triplet_K
        dataset = dataset.take((len(unique_pids) // num_classes) * num_classes)
        dataset = dataset.repeat()  #None ## Such sampling is always used during training

        # For every PID, get K images.
        dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
            pid, all_fids=img_paths, all_pids=train_lbls, batch_k=Triplet_K))

        def _parse_function(filename, label,num_classes=18):
            #tf.print(filename)
            image_string = tf.io.read_file(filename)
            #image_string = tf.Print(image_string,[filename,label],'img name ')
            image_decoded = tf.image.decode_image(image_string,channels=3)
            shape =tf.shape(image_decoded)  
            return image_decoded, label#tf.one_hot(label, num_classes,dtype=tf.int64)
       
        dataset = dataset.apply(tf.data.experimental.unbatch() )
        dataset = dataset.map(_parse_function,num_parallel_calls=n_map_threads)



        is_training = True ## Such sampling is always used during training
        if is_training:
            dataset = dataset.map(lambda im, lbl: (
            img_augment.preprocess_for_train(im, frame_size,frame_size,
                                                 preprocess_func='inception_leaves_color'), lbl))
            dataset = dataset.batch(batch_size)
               
        
        dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)
        #dataset = dataset.prefetch(1)

        return dataset 