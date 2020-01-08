"""Provide classes to perform private training and private prediction with
logistic regression"""
import tensorflow as tf
import tf_encrypted as tfe


class LogisticRegression:
  """Contains methods to build and train logistic regression."""
  def __init__(self, num_features):
    self.w = tfe.define_private_variable(
        tf.random_uniform([num_features, 1], -0.01, 0.01))
    self.w_masked = tfe.mask(self.w)
    self.b = tfe.define_private_variable(tf.zeros([1]))
    self.b_masked = tfe.mask(self.b)

  @property
  def weights(self):
    return self.w, self.b

  def forward(self, x):
    with tf.name_scope("forward"):
      out = tfe.matmul(x, self.w_masked) + self.b_masked
      y = tfe.sigmoid(out)
      return y

  def backward(self, x, dy, learning_rate=0.01):
    batch_size = x.shape.as_list()[0]
    with tf.name_scope("backward"):
      dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
      db = tfe.reduce_sum(dy, axis=0) / batch_size
      assign_ops = [
          tfe.assign(self.w, self.w - dw * learning_rate),
          tfe.assign(self.b, self.b - db * learning_rate),
      ]
      return assign_ops

  def loss_grad(self, y, y_hat):
    with tf.name_scope("loss-grad"):
      dy = y_hat - y
      return dy

  def fit_batch(self, x, y):
    with tf.name_scope("fit-batch"):
      y_hat = self.forward(x)
      dy = self.loss_grad(y, y_hat)
      fit_batch_op = self.backward(x, dy)
      return fit_batch_op

  def fit(self, sess, x, y, num_batches):
    fit_batch_op = self.fit_batch(x, y)
    for batch in range(num_batches):
      print("Batch {0: >4d}".format(batch))
      sess.run(fit_batch_op, tag='fit-batch')

  def evaluate(self, sess, x, y, data_owner):
    """Return the accuracy"""
    def print_accuracy(y_hat, y) -> tf.Operation:
      with tf.name_scope("print-accuracy"):
        correct_prediction = tf.equal(tf.round(y_hat), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print_op = tf.print("Accuracy on {}:".format(data_owner.player_name),
                            accuracy)
        return print_op

    with tf.name_scope("evaluate"):
      y_hat = self.forward(x)
      print_accuracy_op = tfe.define_output(data_owner.player_name,
                                            [y_hat, y],
                                            print_accuracy)

    sess.run(print_accuracy_op, tag='evaluate')


class DataOwner:
  """Contains code meant to be executed by a data owner Player."""
  def __init__(self, player_name, local_data_file, data_schema,
           header=False, index=False, field_delim=',', na_values=['nan'], batch_size=128):
    self.player_name = player_name
    self.local_data_file = local_data_file
    self.data_schema = data_schema
    self.batch_size = batch_size
    self.header = header
    self.index = index
    self.na_values = na_values
    self.field_delim = field_delim

  def provide_data(self):

    def decode(line):
      fields = tf.string_split([line], self.field_delim).values
      if self.index: # Skip index
        fields = fields[1:]
      fields = tf.regex_replace(fields, '|'.join(self.na_values), 'nan')
      fields = tf.string_to_number(fields, tf.float32)
      return fields

    def fill_na(fields, fill_values):
      fields = tf.where(tf.is_nan(fields), fill_values, fields)
      return fields

    dataset = tf.data.TextLineDataset(self.local_data_file)
    if self.header: # Skip header
      dataset = dataset.skip(1)
    dataset = dataset\
        .map(decode)\
        .map(lambda x: fill_na(x, self.data_schema.field_defaults))\
        .repeat()\
        .batch(self.batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    batch = tf.reshape(batch, [self.batch_size, self.data_schema.field_num])
    return batch

class DataSchema:
  def __init__(self, field_types, field_defaults):
    self.field_types = field_types
    self.field_defaults = field_defaults

  @property
  def field_num(self):
    return len(self.field_types)

class ModelOwner:
  """Contains code meant to be executed by a model owner Player."""
  def __init__(self, player_name):
    self.player_name = player_name

  @tfe.local_computation
  def receive_weights(self, *weights):
    return tf.print("Weights on {}:".format(self.player_name), weights)


class PredictionClient:
  """Contains methods meant to be executed by a prediction client."""
  def __init__(self, player_name, num_features):
    self.player_name = player_name
    self.num_features = num_features

  @tfe.local_computation
  def provide_input(self):
    return tf.random.uniform(
        minval=-.5,
        maxval=.5,
        dtype=tf.float32,
        shape=[1, self.num_features])

  @tfe.local_computation
  def receive_output(self, result):
    return tf.print("Result on {}:".format(self.player_name), result)

