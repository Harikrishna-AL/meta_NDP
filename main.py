from meta_guided_ndp import meta_ndp
import networkx as nx
import torch
import yaml, uuid
import matplotlib.pyplot as plt
from meta_guided_ndp.utils import get_feat


def main(config):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configure the config file")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to your config file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f, Loader=yaml.FullLoader)

    config["id"] = uuid.uuid1()
    print("Model ID: ", config["id"])
    path = "saved_models/" + str(config["id"]) + ".pth"
    config["output_path"] = path

    print("Config: ", config)

    main(config=config)
