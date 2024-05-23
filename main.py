from meta_guided_ndp import meta_ndp, utils, train
import networkx as nx
import torch
import yaml, uuid
import matplotlib.pyplot as plt
from meta_guided_ndp.utils import get_feat
import numpy as np
import pathlib

def main(config):
    pathlib.Path(config["_path"]).mkdir(parents=True, exist_ok=False)

    config["seed"] = np.random.randint(10**7) if config["seed"] is None else config["seed"]
    utils.seed_python_numpy_torch_cuda(config["seed"])
    print("Seed: ", config["seed"])
    print("Config: ", config)

    solution_best, solution_centroid, early_stopping_executed, logger_df = train.train(config = config)
    if not early_stopping_executed:
        print(f"\n Saving models and config file  - Run ID {config['id']}")
        print(f"\n Final model has {config['num_trainable_params']} parameters")
        if config["save_model"]:
            np.save(path + "/" + "solution_centroid", solution_centroid)
            np.save(path + "/" + "solution_best", solution_best)
            if logger_df is not None:
                logger_df.to_csv(path + "/" + "logger.csv")

        with open(config["_path"] + "/" + "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    else:
        print("Early stopping executed")


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
