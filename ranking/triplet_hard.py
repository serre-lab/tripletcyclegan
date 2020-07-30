import numbers
import tensorflow as tf
from ranking.triplet_semi import pairwise_distance

# def batch_hard(dists, pids, margin):
def batch_hard(labels, embeddings, margin=1.0):
    dists = pairwise_distance(embeddings, squared=True)
    with tf.name_scope("batch_hard"):
        same_identity_mask = tf.equal(tf.expand_dims(labels, axis=1),
                                      tf.expand_dims(labels, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(labels)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists * tf.cast(positive_mask, tf.float32), axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_max(tf.boolean_mask(x[0], x[1])),
                                     (dists, negative_mask), tf.float32)
        # Another way of achieving the same, though more hacky:
        # closest_negative = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative
        if isinstance(margin, numbers.Real):
            diff = tf.maximum(diff + margin, 0.0)
            print('hard Margin Utilized')
        elif margin == 'soft':
            diff = tf.nn.softplus(diff)
            print('Soft-margin Utilized')
        elif margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(margin))

    return diff

def batch_hard_fossils(labels,styles, embeddings, margin=1.0,style =-1):
   
    
    dists = pairwise_distance(embeddings, squared=True)
    with tf.name_scope("batch_hard_fossils"):
        
        style_bool = tf.cond(tf.equal(style,-1),lambda:tf.greater_equal(styles,0),lambda:tf.equal(styles,style))
        style_mask = tf.boolean_mask(styles,style_bool)
        
        same_identity_mask = tf.equal(tf.expand_dims(labels, axis=1),tf.expand_dims(labels, axis=0))
        negative_mask = tf.logical_and(tf.logical_not(same_identity_mask),style_bool)
        positive_mask = tf.logical_and(tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(labels)[0], dtype=tf.bool)),style_bool)

        furthest_positive = tf.reduce_max(dists * tf.cast(positive_mask, tf.float32), axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_max(tf.boolean_mask(x[0], x[1])),
                                     (dists, negative_mask), tf.float32)
        # Another way of achieving the same, though more hacky:
        # closest_negative = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative
        if isinstance(margin, numbers.Real):
            diff = tf.maximum(diff + margin, 0.0)
            print('hard Margin Utilized')
        elif margin == 'soft':
            diff = tf.nn.softplus(diff)
            print('Soft-margin Utilized')
        elif margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(margin))

    return diff
