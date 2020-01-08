"""Private training on combined data from several data owners"""
import tf_encrypted as tfe

def start_slave(cluster_config_file):
  print("Starting bob...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  tfe.set_protocol(tfe.protocol.Pond())
  players = remote_config.players
  bob = remote_config.server(players[1].name)
  print("server_name = ", players[1].name)
  bob.join()

if __name__ == "__main__":
  start_slave("config.json")
