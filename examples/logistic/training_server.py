"""Private training on combined data from several data owners"""
import tf_encrypted as tfe

def start_slave(cluster_config_file):
  print("Starting crypto producer...")
  remote_config = tfe.RemoteConfig.load(cluster_config_file)
  tfe.set_config(remote_config)
  tfe.set_protocol(tfe.protocol.Pond())
  players = remote_config.players
  server = remote_config.server(players[2].name)
  print("server_name = ", players[2].name)
  server.join()

if __name__ == "__main__":
  start_slave("config.json")
