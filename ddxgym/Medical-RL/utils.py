from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
import numpy as np

class actionCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["action"] = []
        episode.hist_data["actions"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        episode.user_data["action"].append(episode._agent_to_last_action['agent0'])    

    def on_episode_end(
         self,
         *,
         worker: RolloutWorker,
         base_env: BaseEnv,
         policies: Dict[str, Policy],
         episode: Episode,
         env_index: int,
         **kwargs
     ):
         # Check if there are multiple episodes in a batch, i.e.
         # "batch_mode": "truncate_episodes".
         if worker.policy_config["batch_mode"] == "truncate_episodes":
             # Make sure this episode is really done.
             assert episode.batch_builder.policy_collectors["default_policy"].batches[
                 -1
             ]["dones"][-1], (
                 "ERROR: `on_episode_end()` should only be called "
                 "after episode is done!"
             )
         
         episode.hist_data["actions"] = episode.user_data["action"]

