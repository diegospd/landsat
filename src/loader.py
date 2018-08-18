import random
import sys
import rasterio
import numpy as np
import tensorflow as tf

# Turn this off to check local and global accuracy
do_train       = True

do_overfit     = True
do_eval        = True

epochs = 1000


# Don't mess with these! 
# Changing them redefines the graph
rebuild_labels = False
submap_length = 64
batch_size = 32
dropout_rate = 0.4
learning_rate = 1e-4

# Obsolete
# amount of random tests 
# validation_examples = 100
# Proportion of random points to take for training
# Total amount of observations: 33,183,691
# NOTE: Using 180 of each for now
training_ratio = 1e-5
# Make sure to use at least this amount of
# datapoints for each class during training.
# NOTE: Using 180 of each for now
min_class_size = 100


def build_layers(features, mode):
    input_layer = tf.cast(features, tf.float32)
    print('input layer', input_layer.shape)
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=16, kernel_size=[11,11], padding='SAME', activation=tf.nn.relu)
    print('conv1', conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print('pool1', pool1.shape)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[5,5], padding='SAME', activation=tf.nn.relu)
    print('conv2', conv2.shape)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    print('pool2', pool2.shape)

    conv3 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
    print('conv3', conv3.shape)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)
    print('pool3', pool3.shape)


    flat = tf.layers.flatten(pool3)
    print('flat', flat.shape)

    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
    print('dense', dense.shape)

    logits = tf.layers.dense(inputs=dense, units=32)
    print('logits', logits.shape)
    return logits


def my_model_fn(features, labels, mode, params):
    labels = tf.cast(labels, tf.int32)
    print('entropy labels', labels.shape)
    # labels = tf.reshape(labels, [-1])
    # print('entropy labels', labels.shape)

    # Softmax output of the neural network.
    logits = build_layers(features, mode)
    y_pred = tf.nn.softmax(logits=logits)
    # logits = tf.reshape(logits, [None] + logits.shape)
    print('entropy logits', logits.shape)
    print('y_pred', y_pred.shape)
    
    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)
    print('y_pred_cls', y_pred_cls.shape)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_cls)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.
        
        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = \
        {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
    return spec




#-------------------------------------------------------------------------------------------
#      Labels preprocessing
#-------------------------------------------------------------------------------------------
def preprocess_labels():
    """ Loads labeled data from disk.
    Writes training and validation datasets to disk.
    Returns training dataset as a tensor of tuples (x,y,value)
    where x,y are the pixel coordinates in the image and value the label"""
    train = rasterio.open('../data/training.tif').read()
    train = train.reshape(train.shape[1:])
    half = submap_length // 2
    train = np.pad(train, ((half, half), (half, half)), 'constant')
    train = flatten_training(train) # [(x,y,nonzero)]
    labels, test = smart_shuffle(train)
    print('Using', len(labels), ' training points...\n\n')
    np.save('../train_labels.npy', labels)
    np.save('../validation_labels.npy', test)
    return labels    

def smart_shuffle(labels):
    """ makes sure to take the same proportion 
    of observations for each class"""
    
    def taker(ls):
        return ls[:180], ls[-10:]
        # n = int(len(ls) * training_ratio)
        # if n <= min_class_size:
        #     print("WARNING! Taking too few samples. Increase training_ratio")
        #     print("Original:", len(ls), "Took:", n)
        #     if ls:
        #         print("Class value", ls[0][2])
        #     # sys.exit(1)
        #     print("WARNING! I'll use the first",min_class_size, "items")
        #     n = min_class_size

        # return ls[:n], ls[n:]

    train = []
    test = []
    buckets = segregate(labels, lambda x: x[-1])
    for b in buckets:
        random.shuffle(b)
        left, right = taker(b)
        train += left
        test += right
    # Only keep 10 unseen samples from each class as test set
    return np.array(train), np.array(test[:10])


def flatten_training(train):
    x,y = train.shape
    flat = []
    for i in range(x):
        for j in range(y):
            if train[i][j]:
                flat.append((i,j,train[i,j]))
    return flat

def segregate(xs, key=lambda x: x):
    """
    Returns a list of lists, all duplicate elements will be in the
    same inner list. Also accepts a sort key function.
    Warning: Original list is sorted.
    """
    xs = sorted(xs, key=key)
    res = []
    if not xs:
        return res
    a = xs[0]
    sub = [a]
    for x in xs[1:]:
        if key(a) != key(x):
            res.append(sub)
            sub = [x]
            a = x
        else:
            sub.append(x)
    
    res.append(sub)
    return res


#-------------------------------------------------------------------------------------------
#      Input functions
#-------------------------------------------------------------------------------------------
def get_submap(map, x, y):
    half = submap_length // 2
    return map[x - half : x + half, y - half : y + half, : ]


def input_fn_train(map, labels, shuffle=False):
    x = np.array([get_submap(map, d[0], d[1]) for d in labels])
    # x = x.reshape((len(labels), submap_length, submap_length, map.shape[2]))
    if len (x.shape) < 2:
        print("\n\n\n WARNING! Please rebuild the labels for submap_length", submap_length)
        sys.exit(1)
    print('....', x.shape)
    y = np.array([ d[2] - 1 for d in labels])
    # y = np.array([ to_one_hot(d[2]) for d in labels])

    print('input x', x.shape)
    print('INPUT Y', y.shape)
    return tf.estimator.inputs.numpy_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=shuffle,
        num_threads=1
        )

def input_fn_validation(map):
    labels = np.load('../validation_labels.npy')
    buckets = segregate(labels, lambda x: x[-1])
    labels = []
    for b in buckets:
        labels += b[:10]
    return input_fn_train(map, labels)



  
def load_data():
    """ Loads channels and squashes them into a tensor (x,y,values)"""
    b2 = rasterio.open('../data/b2.tif').read()
    b3 = rasterio.open('../data/b3.tif').read()
    b4 = rasterio.open('../data/b4.tif').read()
    b5 = rasterio.open('../data/b5.tif').read()
    map = np.concatenate((b2,b3,b4,b5))
    bands, width, height = map.shape
    map = map.reshape(width, height, bands)
    half = submap_length // 2
    map = np.pad(map, ((half, half), (half, half), (0,0)), 'constant')
    return map
    

#-------------------------------------------------------------------------------------------
#      Main
#-------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    if rebuild_labels:
        print("processing labels... (Memory intensive, 1 core)")
        labels = preprocess_labels()
    else:
        print('load labels...')
        labels = np.load('../train_labels.npy')
    print('labels.shape', labels.shape)
    

    print('load data...')
    map = load_data()
    print('map.shape', map.shape)

    print('\n\n\n')

    tf.logging.set_verbosity(tf.logging.INFO)
    params = {'learning_rate': learning_rate}
    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir='../checkpoints/',
        params=params
        )


    if do_train:
        classifier.train(input_fn_train(map, labels))

    if do_overfit:
        overfit = classifier.evaluate(
            input_fn=input_fn_train(map, labels, shuffle=False),
            steps=100
            )
        print('\n\n\n',overfit)
        print("Overfit accuracy: {0:.2%}\n\n".format(overfit["accuracy"]))

    if do_eval:
        result = classifier.evaluate(input_fn=input_fn_validation(map))
        print('\n\n\n', result)
        print("\n\n\nClassification accuracy: {0:.2%}\n\n".format(result["accuracy"]))







