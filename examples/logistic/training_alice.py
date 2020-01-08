"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import tensorflow as tf
from common import DataOwner, ModelOwner, LogisticRegression, DataSchema

def main(server):
  
  num_rows = 7000
  num_features = 32
  num_epoch = 5
  batch_size = 100
  num_batches = (num_rows // batch_size ) * num_epoch

  #who shall receive the output
  model_owner = ModelOwner('alice')
  
  data_schema0 = DataSchema([tf.float64]*16, [0.0]*16) 
  data_schema1 = DataSchema([tf.int64]+[tf.float64]*16, [0]+[0.0]*16) 
  data_owner_0 = DataOwner('alice',
         'aliceTrainFile.csv',
         data_schema0,
         batch_size = batch_size)
  data_owner_1 = DataOwner('bob',
         'bobTrainFileWithLabel.csv',
         data_schema1,
         batch_size = batch_size)

  tfe.set_protocol(tfe.protocol.Pond(
      tfe.get_config().get_player(data_owner_0.player_name),
      tfe.get_config().get_player(data_owner_1.player_name)
  ))

  x_train_0 = tfe.define_private_input(
            data_owner_0.player_name,
            data_owner_0.provide_data
              )
  x_train_1 = tfe.define_private_input(
            data_owner_1.player_name,
            data_owner_1.provide_data
              )
  y_train = tfe.gather(x_train_1, 0, axis=1)
  y_train = tfe.reshape(y_train, [batch_size, 1])

  #Remove bob's first column (which is label)
  x_train_1 = tfe.strided_slice(x_train_1, [0,1], [x_train_1.shape[0],x_train_1.shape[1]], [1,1]) 
  
  x_train = tfe.concat([x_train_0, x_train_1], axis=1)

  model = LogisticRegression(num_features)
  reveal_weights_op = model_owner.receive_weights(model.weights)

  with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(),
             tag='init')

    model.fit(sess, x_train, y_train, num_batches)
    # TODO(Morten)
    # each evaluation results in nodes for a forward pass being added to the graph;
    # maybe there's some way to avoid this, even if it means only if the shapes match
   # model.evaluate(sess, x_train, y_train, data_owner_0)
   # model.evaluate(sess, x_train, y_train, data_owner_1)

    sess.run(reveal_weights_op, tag='reveal')
  
  
def start_master(cluster_config_file=None):
  print("Starting alice...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  tfe.set_protocol(tfe.protocol.Pond())
  players = remote_config.players
  server0 = remote_config.server(players[0].name)
  main(server0)


if __name__ == "__main__":
  start_master("config.json")
