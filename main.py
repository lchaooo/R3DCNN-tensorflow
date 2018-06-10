"""Main driver to run model."""
import numpy as np
import tensorflow as tf
# used to add values to summary
from tensorflow.core.framework import summary_pb2
import time
import sys
import os
import argparse
import pickle
import subprocess
# from video_recorder import VideoRecorder
from input_data import read_clip_and_label
sys.path.insert(0, './preprocessors')
sys.path.insert(0, './networks')

NUM_CLASSES=26

# def randomize(dataset, labels):
#     """Randomize dataset and labels with the same permutation."""
#     print("Randomize...")

#     permutation = np.random.permutation(labels.shape[0])

#     shuffled_dataset = dataset[permutation]
#     shuffled_labels = labels[permutation]
#     return shuffled_dataset, shuffled_labels


def sparse_tensor_feed(labels):
    """Generate a tuple for feeding it to sparse tensor with dimensions (batch, time).

    Args:
        labels: a list size batch_size of ints representing the labels
    """
    batch_size = len(labels)
    NO_GESTURE_LABEL = 0
    indices = []
    values = []
    max_len = 0

    for batch_i, batch in enumerate(labels):
        length = 0
        # check if only single label or multiple
        if isinstance(batch, int):
            length = 1
            batch = [batch]
        else:
            length = len(batch)

	steps = 1
	max_len = 1
	for step in range(steps):
	    indices.append([batch_i, step])
	    values.append(batch[step]) 
        #steps = 2 * length + 1
        #if steps > max_len:
        #    max_len = steps
        #for step in range(steps):
        #    indices.append([batch_i, step])
        #    if(step % 2 == 1):
        #        idx = (step - 1)/2
        #        values.append(batch[int(idx)])
        #    else:
        #        values.append(NO_GESTURE_LABEL)

    # size 2 * P + 1
    return (np.asarray(indices, dtype=np.int64), np.asarray(values),
            np.asarray([batch_size, max_len], dtype=np.int64))

    #return indices, values, shape

def calculate_label(probs, threshold):
    """Assign a single label to a clip.
    Args:
        probs: The probabilites for the [batch] clips (right now time major)
        threshold: The threshold that is necessary to overcome to assign a label
                   Otherwise the no gesture label is assigned
    Return:
        The assigned label
    """
    # print(probs.shape)
    probs = np.sum(probs, axis=0)
    # print(probs.shape)
    # set no gesture and to zero in each batch
    probs[:, -1] = 0
    # max probability for each batch
    labels = np.argmax(probs, axis=1)
    # print(probs)
    #for idx, label in enumerate(labels):
    #    if probs[idx, label] < threshold:
    #        labels[idx] = 0

    # print(labels)
    return labels

def sparse2arr(sparse):
    """Convert sparse tensor value to array.

    Args:
        sparse: tf.SparseTensorValue
    Returns
        array: np.array representing the sparse tensor
    """
    indices = sparse.indices
    values = sparse.values
    shape = sparse.dense_shape

    array = np.zeros(shape)

    # indx is the counter
    for indx, index in enumerate(indices):
        # is this always two dimensional?
        # index dimension depends on dimension of shape
        # this is always two dimensional:
        # batch x seq_len
        array[index[0], index[1]] = int(values[indx])

    return array

# TODO: train, pretrain, test, and livetest can be put in one function

def train(sess, model, train_list, num_labels, threshold):
    """Train R3DCNN model for one epoch on train_X with train_Y from end to end.
    Args:
        sess: The current session
        model: The model that is trained on
        train_X: The input that is used for training
        train_y: The targets that are true for the train_X
    Returns:
        Those values that are necessary to evaluate the training
        And the summary created by the model
    """
    batch_size = model.options.batch_size
    hidden_cells = model.options.hidden_cells
    train_size = len(train_list)
    num_frames = model.options.num_frames

    steps = int(train_size / batch_size)
    # TODO: some data will not be used
    train_cost = 0
    train_ler = 0
    train_labels = []
    train_preds = []
    train_preds2 = []
    # initial state that is overwritten in each step
    # reset for every epoch
    temp_state = np.zeros([1, 2, batch_size, hidden_cells])
    for step in range(steps):
        start_time = time.time()
        # Generate a minibatch.
	    offset = step * batch_size
        batch_data, batch_y = read_clip_and_label(train_list[offset: offset + batch_size], num_frames_per_clip=num_frames, height=112, width=112, crop_center = False, shuffle = True)
	    #print('Step %d: ' % (step))
        #print(batch_y)
        old_shape = list(batch_data.shape[2:])
        shape = [-1] + old_shape
        batch_data = batch_data.reshape(shape)
        batch_labels = sparse_tensor_feed(batch_y)
        # input is batch major
        # keep state over one epoch (to improve forget gate)
        batch_cost, batch_probs, batch_ler, batch_decoded,  _, temp_state, summary = sess.run(
            [model.loss, model.norm_score, model.ler, model.decoded, model.train_op,
             model.final_state, model.summary],
            feed_dict={
                model.inputs: batch_data,
                model.targets: batch_labels,
                model.state: temp_state
            }
        )
	    #print(sparse2arr(decoded[0]))
        train_labels.extend(batch_y)
        train_batch_preds = calculate_label(batch_probs, threshold)
        train_preds.extend(train_batch_preds)
	    #train_pred2 = sparse2arr(decoded[0])
	    #for i in range(batch_size):
        #    train_preds2.extend(train_pred2[i])

        train_cost += batch_cost * batch_size
        train_ler += batch_ler * batch_size
        #print('Step %d: %.3f sec' % (step, duration))

    acc = accuracy(train_labels, train_preds)
    confusion_matrix(train_labels, train_preds, num_labels)
    print("loss: " + "{:.5f}".format(train_cost))
    acc = create_summary('accuracy', acc)
    return summary, acc

def pretrain(sess, model, train_list, num_labels, threshold):
    """Train C3d model for one epoch on train_X with train_Y.

    Args:
        sess: The current session
        model: The model that is trained on
        train_X: The input that is used for training
        train_y: The targets that are true for the train_X

    Returns:
        Those values that are necessary to evaluate the training
        And the summary created by the model
    """
    batch_size = model.options.batch_size
    train_size = len(train_list)
    num_frames = model.options.num_frames

    steps = int(train_size / batch_size)
    # TODO: some data will not be used
    train_cost = 0
    train_labels = []
    train_preds = []
    start_idx = 0
    # initial state that is overwritten in each step
    # reset for every epoch
    for step in range(steps):
        # start_time = time.time()
        # Generate a minibatch.
        offset = step * batch_size
        batch_data, batch_y = read_clip_and_label(train_list[offset: offset + batch_size], num_frames_per_clip=num_frames, height=112, width=112, crop_center = False, shuffle = True)

        # shape = [batch_data.shape[0]] + [-1] + list(batch_data.shape[3:])
        # batch_data = batch_data.reshape(shape)
        # resample to num_frames
        a=np.random.randint(low = 16, high = 65)
        a=np.arange(a, a+8, 1)
	    b=np.random.randint(low = 0, high = batch_size)
        firstclip = np.random.choice([True, False])
        no_gesture_flag = np.random.choice([False, False, True])
        batch_data = batch_data.reshape([batch_size, -1] + list(batch_data.shape[3:]))
        if firstclip:
            no_gesture_clip=batch_data[b, [0,1,2,3,4,5,6,7],:]
        else:
            no_gesture_clip=batch_data[b,[72, 73, 74, 75, 76, 77, 78, 79], :]
        batch_data = batch_data[:,a, :]
        if no_gesture_flag:
            batch_data[b,:] = no_gesture_clip
            batch_y[b] = NUM_CLASSES-1
        # batch_y = np.repeat(batch_y, 3)
        # input is batch major
        batch_cost, batch_probs, _, summary = sess.run(
            [model.loss, model.norm_score, model.train_op,
             model.summary],
            feed_dict={
                model.inputs: batch_data,
                model.targets: batch_y
            }
        )
        train_labels.extend(batch_y)
        train_preds.extend(np.argmax(batch_probs, axis=1))
        #print('Step %d :' % (step))
        #print(batch_probs)
        #print(batch_y)
        #print(np.argmax(batch_probs, axis=1))
        # duration = time.time() - start_time
        train_cost += batch_cost * batch_size
        # print('Step %d: %.3f sec' % (step, duration))

    acc = accuracy(train_labels, train_preds)
    #print("acc: " + "{:.5f}".format(acc))
    confusion_matrix(train_labels, train_preds, num_labels)
    print("loss: " + "{:.5f}".format(train_cost))
    acc = create_summary('accuracy', acc)
    return summary, acc


def test(sess, model, test_list, num_labels, threshold, debug=False):
    """Test the model without training."""
    # to change dropout rate a place holder is neccessary
    batch_size = model.options.batch_size
    hidden_cells = model.options.hidden_cells
    num_frames = model.options.num_frames

    iterations = len(test_list) // batch_size
    test_cost = 0
    test_ler = 0
    test_labels = []
    test_preds = []
    test_preds2 = []
    temp_state = np.zeros([1, 2, batch_size, hidden_cells])
    for index in range(iterations):
        # test_images = test_X[index*batch_size:(index+1)*batch_size, :]
        offset = index * batch_size
        test_images, batch_y = read_clip_and_label(test_list[offset: offset + batch_size], num_frames_per_clip=num_frames, height=112, width=112, crop_center = True)
        # remove time dimension
        old_shape = list(test_images.shape[2:])
        shape = [-1] + old_shape
        batch_X = test_images.reshape(shape)
        batch_y_sparse = sparse_tensor_feed(batch_y)
	    batch_cost, batch_probs, batch_ler, decoded, temp_state, summary = sess.run(
            [model.loss, model.norm_score, model.ler, model.decoded,
             model.final_state, model.summary],
            feed_dict={
                model.inputs: batch_X,
                model.targets: batch_y_sparse,
                model.state: temp_state
            }
        )
	    #print('Step %d: ' % (index))
        #print(batch_y)
	    #print(batch_probs)
	
        #print(sparse2arr(decoded[0]))
        test_labels.extend(batch_y)
        test_batch_preds = calculate_label(batch_probs, threshold)
        test_preds.extend(test_batch_preds)
        #test_pred2 = sparse2arr(decoded[0])
        #for i in range(batch_size):
        #    test_preds2.extend(test_pred2[i])
        test_cost += batch_cost * batch_size
        test_ler += batch_ler * batch_size

    # only last prediction

    # print("last decoding:")
    # print(sparse2arr(decoded[0]))
    acc = accuracy(test_labels, test_preds)
    #acc = accuracy(test_labels, test_preds2)
    confusion_matrix(test_labels, test_preds, num_labels-1)
    print("acc: " + "{:.5f}".format(acc))
    print("loss: " + "{:.5f}".format(test_cost))
    acc = create_summary('accuracy', acc)
    return summary, acc


# def dump_predictions(sess, model, test_X, files, debug=False):
#     """Test the model without training similar to live."""
#     # to change dropout rate a place holder is neccessary
#     batch_size = 1
#     hidden_cells = model.options.hidden_cells

#     iterations = test_y.shape[0] // batch_size
#     temp_state = np.zeros([1, 2, batch_size, hidden_cells])
#     print(test_X.shape)

#     for index in range(iterations):
#         video = test_X[index]
#         # batch_y_sparse = sparse_tensor_feed(batch_y)
#         # state is always 0
#         video_probs = []
#         for clip in video:
#             # add single dimension to match input tensor
#             clip = clip[np.newaxis, :]
#             clip_probs, temp_state = sess.run(
#                 [model.norm_score, model.state],
#                 feed_dict={
#                     model.inputs: clip,
#                     model.state: temp_state
#                 })
#             # print(clip_probs)
#             video_probs.extend(clip_probs)
#         path = os.path.join(files[index][0], model.name)
#         if(not os.path.exists(path)):
#             os.mkdir(path)

#         video_probs = np.array(video_probs)
#         # remove clip dimension (always one)
#         video_probs.shape = (video_probs.shape[0], video_probs.shape[-1])
#         print(video_probs.shape)
#         with open(os.path.join(path, files[index][1] + ".npy"), "wb+") as handle:
#             np.save(handle, video_probs)


def create_summary(name, value):
    """Create a summary for a value."""
    value = summary_pb2.Summary.Value(tag=name, simple_value=value)
    summary = summary_pb2.Summary(value=[value])
    return summary


def accuracy(labels, preds):
    """Calculate the accuracy between predictions."""
    correct = 0
    for idx, label in enumerate(labels):
        pred = preds[idx]
        if isinstance(label, int):
            if label == pred:
                correct += 1
	    else:
		print(idx+1)
        else:
            if pred in label:
                correct += 1
	    else:
		print(idx+1)
    print(correct)
    return float(correct)/len(labels)


def confusion_matrix(labels, preds, num_labels):
    """Make confusion matrix."""
    matrix = np.zeros(shape=(num_labels, num_labels), dtype=np.int16)
    for idx, label in enumerate(labels):
        pred = preds[idx]
        matrix[label][pred] += 1
    print(matrix)


def create_session():
    """Initialize all variables and create a tensorflow session."""
    init = tf.global_variables_initializer()
    # Create a saver for writing training checkpoints.
    # Create a session for running Ops on the Graph.
    session = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    )
    session.run(init)
    return session


def serialize(obj, location, name):
    """Save serialized object in location."""
    try:
        string = str(obj)
        file_name = os.path.join(location, name + '.txt')
        with open(file_name, 'w+') as f:
            f.write(string)
    except Exception as e:
        print(e)
        print("Failed to convert to string. Only serialize as byte object.")

    file_name = os.path.join(location, name + '.pickle')
    with open(file_name, 'wb+') as f:
        pickle.dump(obj, f)

# def get_preprocessor(config, args):
#     if args.task == "pretrain":
#         raise NotImplemented
#     # elif args.task == "csv_train":
#     #     from r3dcnn_csv_preprocessor import R3DCNNCSVPreprocessor
#     #     pre = R3DCNNCSVPreprocessor(config, args)
#     #     return pre
#     else:
#         from r3dcnn_preprocessor import R3DCNNPreprocessor
#         pre = R3DCNNPreprocessor(config, args)
#         return pre

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('task',
                        help="Tasks that are to be perfomed: train, test, live, pretrain")
    parser.add_argument('-e', '--epochs', default=150, type=int,
                        help='''The number of complete iterations (epochs) through the complete dataset ''')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        help='''The batch size, number of training data that is calculated simultaneously. This depends on the number of memory availbale.''')
    #parser.add_argument('-l', '--learning_rate', default=1e-5, type=float,
    #                    help='''The learning rate for the optimizer''')
    parser.add_argument('--num_frames', default=8, type=int,
                        help='''The number of frames per clip''')
    parser.add_argument('--frames', default=80, type=int,
                        help='''The number of frames per clip''')
    #parser.add_argument('-hc', '--hidden_cells', default=256, type=int,
    #                    help='''The number of hidden cells in the LSTM RNN''')
    parser.add_argument('-se', '--save', default=5, type=int, 
                        help='''save models after some epoch''')
    parser.add_argument('-pt', '--prob_threshold', default=0.3, type=float,
                        help='''The probabilistic treshold which has to be overcome to assign a label''')
    parser.add_argument('-dt', '--depth_threshold', default=1.1, type=float,
                        help='''The treshold of 255 which has to be overcome to be evaluated (still saved)''')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='''The number of available gpus''')
    parser.add_argument('-d', '--dropout', default=0.5, type=float,
                        help='''Dropout rate for Convolution layers as well as RNN''')
    parser.add_argument('-m', '--model',
                        help='''Path to the model to restore. If this is set, it will be tried to be restored''')
    # parser.add_argument('-c', '--config', default='datasets/realsense/realsense_task.json',
    #                     help='''The path to a config file''')
    parser.add_argument('-c3d', '--c3d_model', default='./conv3d_deepnetA_sport1m_iter_1900000_TF.model',
                        help='''Path to a pretrained C3D model''')
    parser.add_argument('--log', default=True,
                        help='''Flag if log is written''')
    parser.add_argument('--debug', action='store_true',
                        help='''Flag to use debug mode, which switches off saving of the model and uses less samples''')
    parser.add_argument('-a', '--annotate', action='store_true',
                        help='''Flag to annotate recorded videos with predictions''')
    parser.add_argument('-pd', '--predata', action='store_true',
                        help='''Flag to use predata for live-training''')
    # parser.add_argument('--test_subject', default="",
    #                     help='''A comma seperated list of test subjects''')
    # parser.add_argument('--csv_folders',  nargs='*',
                        # help='''A list of csv files''')
    args = parser.parse_args()
    # if args.test_subject == "":
    #     test_subject = []
    # else:
    #     test_subject = args.test_subject.split(',')

    # config = args.config
    with tf.Graph().as_default():
        if args.task == "train":
            from r3dcnn import R3DCNN
            # pre = get_preprocessor(config, args)

            model = R3DCNN(args)

            # TODO.lc load data
            # pre.load_train_data(test_subject)
            # pre.load_val_data(test_subject)

            # train_X = pre.train_X
            # train_y = pre.train_y
            # val_X = pre.val_X
            # val_y = pre.val_y
            train_lines = open('train.list', 'r')
            train_list = list(train_lines)
            np.random.shuffle(train_list)

            test_lines = open('test.list', 'r')
            test_list = list(test_lines)
            np.random.shuffle(test_list)

            sess = create_session()

            if(args.model):
                model.restore(args.model, sess)
		print('loading r3dcnn model')
            else:
                print('loading c3d model')
                # otherwise just resort parts of this model
                model.restore_3DN(args.c3d_model, sess)
            # save all variables for now
            # TODO save only necessary
            saver = tf.train.Saver()

            model_save_dir = os.path.join('models', 'R3DCNN')
            if args.debug:
                model_save_dir = os.path.join(model_save_dir, 'debug')

            dir_name = os.path.join(model_save_dir, time.strftime("%y-%m-%d_%H%M"))
            save_model = os.path.join(dir_name, 'model')

            if args.log:
                train_writer = tf.summary.FileWriter(dir_name + '/train',
                                                     sess.graph)
                val_writer = tf.summary.FileWriter(dir_name + '/val',
                                                   sess.graph)
                serialize(args, dir_name, 'args')

                # pre.config.dump(os.path.join(dir_name, "config.json"))

                # with open(dir_name + '/dataset.txt', 'a') as the_file:
                #     the_file.write("Train shape: " + str(train_X.shape))
                #     the_file.write("Validation shape: " + str(val_X.shape))

            for epoch in range(1, args.epochs+1):
		        np.random.shuffle(train_list)

                print("Epoch %i:" % (epoch))
                summary, acc = train(sess, model, train_list, NUM_CLASSES,
                                     args.prob_threshold)

                if args.log:
                    train_writer.add_summary(summary, epoch)
                    train_writer.add_summary(acc, epoch)
                # train_X, train_y = randomize(train_X, train_y)
                np.random.shuffle(train_list)
                if not args.debug and epoch % args.save == 0:
                    # print(save_model)
                    saver.save(sess, save_model, global_step=epoch)

                #if (epoch % 1) == 0:
                #    print('testing')
                #    val_summary, val_acc = test(sess, test_model, test_list,
                #                                NUM_CLASSES, args.prob_threshold, False)
                    #if args.log:
                    #    val_writer.add_summary(val_summary, epoch)
                    #    val_writer.add_summary(val_acc, epoch)

        elif args.task == "test":
            from r3dcnn_test import R3DCNNTEST
            test_lines = open('test.list', 'r')
            test_list = list(test_lines)
	        print(len(test_list))
            #np.random.shuffle(test_list)
            # pre = get_preprocessor(config, args)

            # pre.load_val_data(test_subject)
            # ensure dropout is set to 1, there should no be other possibility?
	        args.dropout = 1.0
            model = R3DCNNTEST(args)
            sess = create_session()
            # TODO
            # test_X = pre.val_X
            # test_y = pre.val_y
            model.restore(args.model, sess)
            #test(sess, model, test_list,
            #     pre.num_labels, args.prob_threshold, args.debug
            test(sess, model, test_list,
                               NUM_CLASSES, args.prob_threshold, args.debug)
        # elif args.task == "csv_dump":
        #     from r3dcnn_csv_preprocessor import R3DCNNCSVPreprocessor
        #     from r3dcnn import R3DCNN
        #     pre = R3DCNNCSVPreprocessor(config, args)
        #     pre.load_train_data(extend=False)
        #     print(pre.files)
        #     # ensure dropout is set to 1, there should no be other possibility?
        #     args.dropout = 1.0
        #     pre.num_clips = 1
        #     args.batch_size = 1
        #     model = R3DCNN(pre, args)
        #     sess = create_session()
        #     # TODO
        #     test_X = pre.train_X
        #     test_y = pre.train_y

        #     model.restore(args.model, sess)
        #     dump_predictions(sess, model, test_X, pre.files, args.debug)

        # elif args.task == "csv_train":
        #     """Train on data collected in the wild."""
        #     from r3dcnn import R3DCNN

        #     pre = get_preprocessor(config, args)

        #     model = R3DCNN(pre, args)
        #     sess = create_session()
        #     if args.model:
        #         model.restore(args.model, sess)
        #     pre.load_train_data(test_subject, extend=False)
        #     train_X = pre.train_X
        #     train_y = pre.train_y
        #     saver = tf.train.Saver(max_to_keep=20)

        #     model_save_dir = os.path.join('models', "live")
        #     if args.debug:
        #         model_save_dir = os.path.join(model_save_dir, 'debug')

        #     dir_name = os.path.join(model_save_dir, time.strftime("%y-%m-%d_%H%M"))
        #     if args.model:
        #         model_name = os.path.split(args.model)[-1]
        #         print(model_name)
        #     else:
        #         model_name = 'model'
        #     save_model = os.path.join(dir_name, model_name)

        #     if args.log:
        #         train_writer = tf.summary.FileWriter(dir_name + '/train',
        #                                              sess.graph)
        #         val_writer = tf.summary.FileWriter(dir_name + '/val',
        #                                            sess.graph)
        #         serialize(args, dir_name, 'args')
        #     for epoch in range(1, args.epochs + 1):
        #         print("Epoch: " + str(epoch))
        #         train_X, train_y = randomize(train_X, train_y)
        #         summary, acc = train(sess, model, train_X, train_y,
        #                              pre.num_labels, args.prob_threshold)
        #         if args.log:
        #             train_writer.add_summary(summary, epoch)
        #             train_writer.add_summary(acc, epoch)
        #         if epoch % 5 == 0:
        #             saver.save(sess, save_model, global_step=epoch)

        # elif args.task == "live_test":
        #     """Originally test with similar approach.
        #     TODO change to actual live test.
        #     """
        #     from r3dcnn import R3DCNN

        #     pre = get_preprocessor(config, args)

        #     pre.load_val_data(test_subject)
        #     # ensure dropout is set to 1, there should no be other possibility?
        #     args.dropout = 1.0
        #     pre.num_clips = 1
        #     args.batch_size = 1
        #     model = R3DCNN(pre, args)
        #     sess = create_session()
        #     # TODO
        #     test_X = pre.val_X
        #     test_y = pre.val_y
        #     #model.restore(args.model, sess)
        #     #livetest(sess, model, test_X, test_y,
        #     #         pre.num_labels, args.prob_threshold, args.debug)


        elif args.task == "pretrain":
            """Pretrain C3D network."""
            from c3d import C3D

            # pre = get_preprocessor(config, args)

            # pre.load_pretrain_data(test_subject)
            model = C3D(args)
            # train_X = pre.pretrain_X
            # train_y = pre.pretrain_y
            train_lines = open('train.list', 'r')
            train_list = list(train_lines)
            np.random.shuffle(train_list)

            test_lines = open('test.list', 'r')
            test_list = list(test_lines)
            np.random.shuffle(test_list)

            sess = create_session()
            #if(args.c3d_model):
            model.restore(args.c3d_model, sess)
            #    print('restore c3d model')
            # save all variables for now
            # TODO save only necessary
            saver = tf.train.Saver()

            model_save_dir = os.path.join('models', 'C3D')
            if args.debug:
                model_save_dir = os.path.join(model_save_dir, 'debug')

            dir_name = os.path.join(model_save_dir, time.strftime("%y-%m-%d_%H%M"))
            save_model = os.path.join(dir_name, 'model')

            if args.log:
                train_writer = tf.summary.FileWriter(dir_name + '/train',
                                                     sess.graph)
                val_writer = tf.summary.FileWriter(dir_name + '/val',
                                                   sess.graph)
                serialize(args, dir_name, 'args')
                # pre.config.dump(os.path.join(dir_name, "config.json"))
            print('start pretrain') 
            for epoch in range(1, args.epochs+1):

                print("Epoch %i:" % (epoch))
                summary, acc = pretrain(sess, model, train_list, NUM_CLASSES,
                                        args.prob_threshold)

                if args.log:
                    train_writer.add_summary(summary, epoch)
                    train_writer.add_summary(acc, epoch)
                # train_X, train_y = randomize(train_X, train_y)
                np.random.shuffle(train_list)
                if not args.debug and epoch % args.save == 0:
                    # print(save_model)
                    saver.save(sess, save_model, global_step=epoch)
        else:
            print("Undefined task: %s" % args.task)
            parser.print_help()
