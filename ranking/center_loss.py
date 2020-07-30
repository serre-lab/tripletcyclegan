import tensorflow as tf

def get_center_loss(features, labels, alpha, num_classes):


    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(), trainable=False)

    labels = tf.reshape(labels, [-1])


    centers_batch = tf.gather(centers, labels)

    loss = tf.nn.l2_loss(features - centers_batch)


    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels,diff,name='centers_update_op')

    return loss, centers, centers_update_op,appear_times,diff

# triplet loss
def center_loss(centeriods,anchor):
    pos_dist1 = tf.reduce_sum(tf.square(tf.subtract(centeriods, anchor)), 1)
    return pos_dist1

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def triplet_hard_fossils(labels,styles, embeddings, margin=1.0,style =-1):
   
    
    dists = pairwise_distance(embeddings, squared=True)
    with tf.name_scope("batch_hard_fossils"):
        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(dists, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(dists, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        style_bool = tf.cond(tf.equal(style,-1),lambda:tf.greater_equal(styles,0),lambda:tf.equal(styles,style))
        style_mask = tf.boolean_mask(styles,style_bool)

        mask = _get_triplet_mask(labels)
        combined_mask = tf.logical_and(mask,style_mask)

        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
  
        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets
        
        


    

    

    

    