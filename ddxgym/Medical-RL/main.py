import ray
#import ray.rllib.agents.impala as IMPALA
#import ray.rllib.algorithms.impala as IMPALA
from argparse import Namespace
from typing import Dict
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.impala import Impala, ImpalaConfig
from tqdm import tqdm
from ray import air, tune
from arguments import parseArguments
from model import IMPALATransformer,IMPALATransformerDiseasePrediction, IMPALAFruitfly, IMPALATransformerLM, IMPALATransformerLMDiseasePrediction, IMPALAFullyConnected, IMPALALSTM
from utils import actionCallbacks
from ray.rllib.examples.policy.random_policy import RandomPolicy
from random_agent import RandomAgentConfig, RandomAgent

def build_config(args: Namespace) -> Dict:
    if args.model_name_or_path == "random":
        config = RandomAgentConfig().to_dict()
    else:
        config = ImpalaConfig().to_dict()
    #config['_use_trajectory_view_api'] = False
    #config                                     = IMPALA.DEFAULT_CONFIG.copy()
    config["framework"]                        = "torch"
    config["num_gpus"]                         = args.num_gpu
    config["num_multi_gpu_tower_stacks"]       = 1
    #config["min_time_s_per_iteration"]         = 1
    #config["num_multi_gpu_tower_stacks"]       = 0
    config["num_workers"]                      = args.num_workers
    config["num_envs_per_worker"]              = args.inference_batch_size
    config["num_cpus_per_worker"]              = args.inference_batch_size + 1 #if args.num_gpu_per_worker > 0 else args.num_workers * 24
    config["num_gpus_per_worker"]              = args.num_gpu_per_worker
    config["rollout_fragment_length"]          = args.rollout_fragment_length
    config["train_batch_size"]                 = args.batch_size
    config["replay_proportion"]                = args.replay_proportion
    config["replay_buffer_num_slots"]          = 128
    #config["recreate_failed_workers"]          = True
    #config["batch_mode"]                       = "complete_episodes"
    config["env"]                              = "gym_medical:doctorsim-v0"
    config["disable_env_checking"]             = True
    
    config["log_level"]                        = args.log_level
    config["env_config"] = {}
    config["env_config"]["observation_length"] = args.sequence_length
    config["env_config"]["tokenizer"]          = args.model_name_or_path
    if args.fruitfly:
        config["env_config"]["tokenizer"] = args.fruitfly_tokenizer

    config["env_config"]["is_csv"]        = args.is_csv
    config["env_config"]["data_path"]     = args.data_path
    config["env_config"]["max_diseases"]  = args.max_diseases
    config["env_config"]["fruitfly"]      = args.fruitfly
    
#    config["remote_worker_envs"]               = True
    #config["remote_env_batch_wait_ms"]         = 20 
    config["lr"]                               = args.learning_rate
    config['learner_queue_size'] = 1
    config["lr_schedule"]=[[0,args.learning_rate],[12_000_000, args.learning_rate/10]] 
    #config["exploration_config"] = {'type':'EpsilonGreedy',
    #                               'warmup_timesteps':10000,
    #                               'epsilon_timesteps':5e5
    #                               }
    config['callbacks'] = actionCallbacks
    model_config                        = MODEL_DEFAULTS.copy()
    
    model_config["custom_model"]        = "IMPALATransformer" 
    if args.predict_disease:
        model_config["custom_model"]        = "IMPALATransformerDiseasePrediction" 
    if args.transformer_mlm:
        model_config["custom_model"]        = "IMPALATransformerLM" 
    if args.transformer_mlm and args.predict_disease:
        model_config["custom_model"]        = "IMPALATransformerLMDiseasePrediction" 
    if args.sequence_lstm:
        model_config["custom_model"]        = "IMPALALSTM" 
    model_config["custom_model_config"] = {
        "model_name_or_path": args.model_name_or_path,
    #   "num_diseases": args.max_diseases
    }
    
    if args.fruitfly:
       model_config["custom_model"]        = "IMPALAFruitfly"
       model_config["custom_model_config"] = {
           "K": 400,
           "k": 50,
           "tokenizer_name":args.fruitfly_tokenizer,
           "model_path":args.fruitfly_model_path,
           "fs_path":args.fruitfly_model_fs
       }
       
    if args.model_name_or_path == "lstm":
        config["env"]   = "gym_medical:doctorsimmultibinary-v0"
        model_config["custom_model"] = "IMPALAFullyConnected"
        model_config["use_lstm"] = True
    config["model"] = model_config

    #just use FF


    return config


if __name__ == "__main__":

    args   = parseArguments()
    config = build_config(args)
    
    if args.model_name_or_path == "random":
        ray.init(ignore_reinit_error=True, local_mode=True)
        config["num_gpus"] = 0
        config["num_gpus_per_worker"] = 0
        config["framework"] = None
        config["custom_model"] = None
        config["custom_model_config"] = None
        random_config = RandomAgentConfig.from_dict(config)
        algo = random_config.build()
        for _ in range(1_000_000):
            res = algo.train()
            print(f"mean_reward:{res['episode_reward_mean']}")
            print(f"mean_len:{res['episode_len_mean']}")
        ray.shutdown()
        exit()

    ray.init(num_cpus=args.num_cpus or None)
    ModelCatalog.register_custom_model("IMPALATransformer", IMPALATransformer)
    ModelCatalog.register_custom_model("IMPALATransformerLM",IMPALATransformerLM)
    ModelCatalog.register_custom_model("IMPALATransformerDiseasePrediction", IMPALATransformerDiseasePrediction)
    ModelCatalog.register_custom_model("IMPALATransformerLMDiseasePrediction", IMPALATransformerLMDiseasePrediction)
    ModelCatalog.register_custom_model("IMPALAFruitfly", IMPALAFruitfly)
    ModelCatalog.register_custom_model("IMPALAFullyConnected", IMPALAFullyConnected)
    ModelCatalog.register_custom_model("IMPALALSTM", IMPALALSTM)
    #trainer = IMPALA.ImpalaTrainer(
    #    config = config,
    #    env    = "gym_medical:doctorsim-v0",
    #)

    stop = {
        "timesteps_total": 80_000_000
    }




    #print(IMPALA.ImpalaTrainer.default_resource_request(config=config)._bundles)
    tuner = tune.Tuner("IMPALA",
                       param_space=config,
                       run_config=air.RunConfig(stop=stop,
                                                verbose=3,
                                                local_dir=args.log_dir,
                                                name=f"{args.model_name_or_path.replace('/','')}-{args.max_diseases}",
                                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=args.checkpoint_frequency)
                                                )
                       )
    tuner.fit()
    ray.shutdown()

