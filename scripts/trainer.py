import hydra
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from methods.train.net import Net
from methods.train.gan import GAN
from methods.train.kd import KD
from methods.train.ws import WS
import datetime

import mlflow
#from mlflow.models import infer_signature




@hydra.main(config_path="../configs/", config_name="train.yaml")
def main(cfg):

    # mlflow log params
    # mlflow.log_params(dict(cfg))
    
      
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(cfg.train.common.seed)
    np.random.seed(cfg.train.common.seed)

    trainer = None
    if cfg.train.common.method == "NET":
        trainer = Net
    elif cfg.train.common.method == "GAN":
        trainer = GAN
    elif cfg.train.common.method == "KD":
        trainer = KD
    elif cfg.train.common.method == "WS":
        trainer = WS

    #mlflow start
    if cfg.train.common.use_mlflow:
        # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        uri = f"http://{cfg.train.mlflow.host}:{cfg.train.mlflow.port}"
        mlflow.set_tracking_uri(uri=uri)
        mlflow.set_experiment(cfg.train.mlflow.experiment_name)
        now = datetime.datetime.now()
        mlflow.set_tag('mlflow.runName',now.strftime('%Y-%m-%d-%H-%M-%S'))
        mlflow.end_run()
        with mlflow.start_run():
            
            # log_params
            cfg_dict = dict(cfg)

            cfg_dict["dataset"] = cfg_dict["train"]["dataset"]
            cfg_dict["common"] = cfg_dict["train"]["common"]
            cfg_dict["ddp"] = cfg_dict["train"]["ddp"]
            cfg_dict["loss"] = cfg_dict["train"]["loss"]["lists"]
            cfg_dict["optim_type"] = cfg_dict["train"]["optim"]["type"]
            cfg_dict["mlflow"] = cfg_dict["train"]["mlflow"]
            for key,val in cfg_dict["train"]["optim"][cfg_dict["optim_type"]].items():
                cfg_dict["optim_"+key] = val
            

            cfg_dict["scheduler_type"] = cfg_dict["train"]["scheduler"]["type"]
            for key,val in cfg_dict["train"]["scheduler"][cfg_dict["scheduler_type"]].items():
                cfg_dict["scheduler_"+key] = val
            cfg_dict["metrics"] = cfg_dict["train"]["metrics"]["types"]


            del cfg_dict["train"]
            mlflow.log_params(cfg_dict)

            if torch.cuda.device_count() > 1:
                print("Train with multiple GPUs")
                cfg.train.ddp.distributed = True
                gpus = torch.cuda.device_count()
                cfg.train.ddp.world_size = gpus * cfg.train.ddp.nodes

                mp.spawn(
                    trainer,
                    nprocs=gpus,
                    args=(cfg,),
                )
                dist.destroy_process_group()
            else:
                trainer(0, cfg)
        
        # mlflow.end_run()

    else:
        if torch.cuda.device_count() > 1:
            print("Train with multiple GPUs")
            cfg.train.ddp.distributed = True
            gpus = torch.cuda.device_count()
            cfg.train.ddp.world_size = gpus * cfg.train.ddp.nodes

            mp.spawn(
                trainer,
                nprocs=gpus,
                args=(cfg,),
            )
            dist.destroy_process_group()
        else:
            trainer(0, cfg)


if __name__ == "__main__":
    main()
    # mlflow.end_run()
    # with mlflow.start_run():
    #     main()
