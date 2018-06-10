import tensorflow as tf
import numpy as np
import r3dcnn_model
from model import Model

# TODO
MOVING_AVERAGE_DECAY = 0.9999
NUM_LABELS = 26

class R3DCNNTEST(Model):
    def __init__(self, options):
        """Initialize model.

        Args:
            pre: The preprocessor that carries information how data looks like
            options: More general informaion how the model should be built

        """
        self._options = options
        # Get the sets of images and labels for training, validation, and

        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        # feed the input as one stream to the c3d network
        # the c3d network does not care about dimension
        images_placeholder, labels_placeholder = self.placeholder_inputs(
            options
        )
        # http://stackoverflow.com/questions/39112622/how-do-i-set-tensorflow-rnn-state-when-state-is-tuple-true
        state_placeholder = tf.placeholder(tf.float32, [1, 2, options.batch_size, options.hidden_cells])
        l = tf.unstack(state_placeholder, axis=0)
        rnn_tuple_state = tuple(
                 [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                  for idx in range(1)]
        )

        self._inputs = images_placeholder
        self._targets = labels_placeholder
        self._state = state_placeholder

        logits = []
        tower_grads1 = []
	    tower_grads2 = []

	    #opt1 = tf.train.AdamOptimizer(1e-4)
        #opt2 = tf.train.AdamOptimizer(3e-4)	

        with tf.variable_scope('var_name') as var_scope:
            for gpu_index in range(0, options.gpus):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('%s_%d' % ('gestabase', gpu_index)) as scope:
                        weights = {
                            'wc1': self._variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0000),
                            'wc2': self._variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0000),
                            'wc3a': self._variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0000),
                            'wc3b': self._variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0000),
                            'wc4a': self._variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0000),
                            'wc4b': self._variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0000),
                            'wc5a': self._variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0000),
                            'wc5b': self._variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0000),
                            # changed
                            'wd1': self._variable_with_weight_decay('wd1', [8192, 4096], 0.0001),
                            'wd2': self._variable_with_weight_decay('wd2', [4096, 4096], 0.0002),
                            'out': self._variable_with_weight_decay('wout', [options.hidden_cells, NUM_LABELS], 0.0005)
                        }
                        biases = {
                            'bc1': self._variable_with_weight_decay('bc1', [64], 0.000),
                            'bc3a': self._variable_with_weight_decay('bc3a', [256], 0.000),
                            'bc2': self._variable_with_weight_decay('bc2', [128], 0.000),
                            'bc3b': self._variable_with_weight_decay('bc3b', [256], 0.000),
                            'bc4a': self._variable_with_weight_decay('bc4a', [512], 0.000),
                            'bc4b': self._variable_with_weight_decay('bc4b', [512], 0.000),
                            'bc5a': self._variable_with_weight_decay('bc5a', [512], 0.000),
                            'bc5b': self._variable_with_weight_decay('bc5b', [512], 0.000),
                            'bd1': self._variable_with_weight_decay('bd1', [4096], 0.000),
                            'bd2': self._variable_with_weight_decay('bd2', [4096], 0.000),
                            'out': self._variable_with_weight_decay('bout', [NUM_LABELS], 0.000),
                        }
		            varlist2 = [ weights['out'],biases['out'] ]
                    varlist1 = list( set(weights.values() + biases.values()) - set(varlist2))
                    # TODO multiple gpus
                    self._weights = weights
                    self._biases = biases
                    logit, self._final_state = r3dcnn_model.inference_r3dcnn(
                        images_placeholder[
                            gpu_index * options.batch_size * 10:(gpu_index + 1) * 10 * options.batch_size, :, :, :, :],
                        options.dropout,
                        options.batch_size,
                        10,
                        options.hidden_cells,
                        rnn_tuple_state,
                        weights,
                        biases
                    )

                    # keep time major and reshape to one long sequence
                    logit = tf.reshape(logit, [10, options.batch_size, NUM_LABELS])

                    """ sequence length, the number of gestures in this sequence
                    the ctc loss function tries to align the gestures in such
                    manner that it fits the labels_placeholder
                    The X input size must not match the y input size
                    In this case, training data always assume:
                    TODO:
                    no gesture, gesture, no gesture
                    or no gesture, gesture
                    or gesture, no gesture

                    this means we look at the one complete training clip and
                    demand those three gestures as an output
                    sequence len is primarily used to indicate the function
                    about the dimensions of the sparse tensor. Because in case
                    of batch processing, it might happen that one is padded
                    """
                    seq_len = np.ones(options.batch_size) * 2

                    loss = self.tower_loss(scope, logit, labels_placeholder, seq_len)
		            #grads1 = opt1.compute_gradients(loss, varlist1)
                    #grads2 = opt2.compute_gradients(loss, varlist2)
		            #tower_grads1.append(grads1)
                    #tower_grads2.append(grads2)
                    logits.append(logit)
                    tf.get_variable_scope().reuse_variables()
        logits = tf.concat(logits, 0)

        # NOTE: this has to be fixed when using multiple gpus?
        self._norm_score = tf.nn.softmax(logits)

        #grads1 = self.average_gradients(tower_grads1)
        #grads2 = self.average_gradients(tower_grads2)
        #apply_gradient_op1 = opt1.apply_gradients(grads1, global_step = global_step)
        #apply_gradient_op2 = opt2.apply_gradients(grads2, global_step=global_step)
        #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        #variables_averages_op = variable_averages.apply(tf.trainable_variables())
	    #train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
        #null_op = tf.no_op()
        self._decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)

        # greedy decode only returns one possibility. get it through 0
        self._ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), labels_placeholder))

        self._summary = tf.summary.merge_all()
        #self._train_op = train_op
        self._loss = loss


    def tower_loss(self, name_scope, logit, labels, seq_len):
        """Calculate loss over multiple gpus."""
        loss = tf.nn.ctc_loss(labels, logit, seq_len)
        # tf.summary.scalar('loss', tower_loss(scope, loss))
        cost = tf.reduce_mean(loss)
        weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
        tf.summary.scalar(name_scope + 'weight decay loss', weight_decay_loss)
        tf.add_to_collection('losses', cost)
        losses = tf.get_collection('losses', name_scope)
        # Calculate the total loss for the current tower.
        total_loss = cost+ weight_decay_loss 
        tf.summary.scalar(name_scope + 'total loss', total_loss)

        return total_loss

    def placeholder_inputs(self, options):
        """Generate placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        code and will be fed from the downloaded data in the .run() loop, below.

        Args:
          batch_size: The batch size will be baked into both placeholders.

        Returns:
          images_placeholder: Images placeholder.
          labels_placeholder: Labels placeholder.
        """
        # Note that the shapes of the placeholders match the shapes of the full
        # image and label tensors, except the first dimension is now batch_size
        # rather than the full size of the train or test data sets.
        images_placeholder = tf.placeholder(tf.float32,
                                            shape=(options.batch_size * 10,
                                                   options.num_frames,
                                                   112,
                                                   112,
                                                   3))
        # print(shape.get_shape())
        labels_placeholder = tf.sparse_placeholder(tf.int32)
        # print(labels_placeholder.get_shape())
        return images_placeholder, labels_placeholder

