"""C3D model by https://github.com/hx173149/C3D-tensorflow."""
import tensorflow as tf
import c3d_model
from model import Model

MOVING_AVERAGE_DECAY = 0.9999
NUM_CLASSES = 26


class C3D(Model):
    """C3D model class."""

    def __init__(self, options):
        self._options = options
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        images_placeholder, labels_placeholder = self.placeholder_inputs(
            options
        )
        self._inputs = images_placeholder
        self._targets = labels_placeholder

        tower_grads1 = []
        tower_grads2 = []
        logits = []
        learning_rate = tf.train.exponential_decay(3e-4, global_step, 1600, 0.3) 
        opt1 = tf.train.AdamOptimizer(3e-5)
        opt2 = tf.train.AdamOptimizer(1e-4)
        with tf.variable_scope('var_name') as var_scope:
            for gpu_index in range(0, options.gpus):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('%s_%d' % ('gestabase', gpu_index)) as scope:
                        weights = {
                            'wc1': self._variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                            'wc2': self._variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                            'wc3a': self._variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                            'wc3b': self._variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                            'wc4a': self._variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                            'wc4b': self._variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                            'wc5a': self._variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                            'wc5b': self._variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                            # changed
                            'wd1': self._variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                            'wd2': self._variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                            'out': self._variable_with_weight_decay('wout', [4096, NUM_CLASSES], 0.0005)
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
                            'out': self._variable_with_weight_decay('bout', [NUM_CLASSES], 0.000),
                        }
                    varlist2 = [ weights['out'],biases['out'] ]
                    varlist1 = list( set(weights.values() + biases.values()) - set(varlist2)) 
                    self._weights = weights
                    self._biases = biases
                    logit = c3d_model.inference_c3d(
                        images_placeholder[
                            gpu_index * options.batch_size * 1: 1 * (gpu_index + 1) * options.batch_size, :, :, :, :],
                        options.dropout,
                        options.batch_size * 1,
                        weights,
                        biases
                    )
                    loss = self.tower_loss(
                        scope,
                        logit,
                        labels_placeholder
                    )
                    grads1 = opt1.compute_gradients(loss, varlist1)
                    grads2 = opt2.compute_gradients(loss, varlist2)
                    tower_grads1.append(grads1)
                    tower_grads2.append(grads2)
                    logits.append(logit)
                    tf.get_variable_scope().reuse_variables()
        logits = tf.concat(logits, 0)

        self._norm_score = tf.nn.softmax(logits)
        grads1 = self.average_gradients(tower_grads1)
        grads2 = self.average_gradients(tower_grads2)
        apply_gradient_op1 = opt1.apply_gradients(grads1)
        apply_gradient_op2 = opt2.apply_gradients(
            grads2, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        train_op = tf.group(apply_gradient_op1,
                            apply_gradient_op2, variables_averages_op)
        null_op = tf.no_op()
        self._summary = tf.summary.merge_all()
        self._train_op = train_op
        self._loss = loss

    def tower_loss(self, name_scope, logit, labels):
        """Calculate loss over multiple gpus."""
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logit)
        )
        tf.summary.scalar(
            name_scope + 'cross entropy',
            cross_entropy_mean
        )
        # weight_decay_loss = tf.get_collection('losses')
	# tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

	# total_loss = cross_entropy_mean + weight_decay_loss 
	# tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
	# return total_loss

	    weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
        tf.summary.scalar(name_scope + 'weight decay loss', tf.reduce_mean(weight_decay_loss))
        # tf.add_to_collection('losses', cross_entropy_mean)
        # losses = tf.get_collection('losses', name_scope)
	    total_loss = cross_entropy_mean + weight_decay_loss
	    tf.summary.scalar(name_scope + 'total loss', tf.reduce_mean(total_loss))
	    return total_loss
        # Calculate the total loss for the current tower.
        # total_loss = tf.add_n(losses, name='total_loss')
        # tf.summary.scalar(name_scope + 'total loss', total_loss)

        # Compute the moving average of all individual losses and the total loss.
        # loss_averages = tf.train.ExponentialMovingAverage(0.99, name='loss')
        # loss_averages_op = loss_averages.apply(losses + [total_loss])
        # with tf.control_dependencies([loss_averages_op]):
        #     total_loss = tf.identity(total_loss)
        # return total_loss

    def restore(self, location, sess):
        #varlist2 = [self._weights['out'], self._biases['out']]
        #varlist1 = list(set(self._weights.values() + self._biases.values()) - set(varlist2))
        #saver = tf.train.Saver(varlist1)
        #saver.restore(sess, location) 
        variables = list()
        for key, value in self.biases.items():
        #    if key == 'out':
        #        print("Skipped out layer")
        #        continue
            variables.append(value)
        for key, value in self.weights.items():
        #    if key == 'out':
        #        print("Skipped out layer")
        #        continue
            variables.append(value)
        # self.restore(location, sess, variables)
        saver = tf.train.Saver(variables)
        saver.restore(sess, location) 

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
        images_placeholder = tf.placeholder(tf.float32, shape=(options.batch_size * 1,
                                                               options.num_frames,
                                                               112,
                                                               112,
                                                               3))
        labels_placeholder = tf.placeholder(tf.int64, shape=(options.batch_size * 1))
        return images_placeholder, labels_placeholder
