import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import (DEFAULT_CONFIG, execution_plan,
                                      get_policy_class, validate_config,
                                      warn_about_bad_reward_scales)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import (ConcatBatches, ParallelRollouts,
                                             SelectExperiences,
                                             StandardizeFields)
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker

# from utils.simulation import simulate_soccer
# from adapenvs.markov_soccer.markov_soccer import MarkovSoccer

class MyCallbacksField(DefaultCallbacks):
    def __init__(self):
        super(MyCallbacksField, self).__init__()
        self.avg_targets = [0]

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
        lifetimes = np.zeros(len(train_batch['infos']))
        for i, agent_info_dict in enumerate(train_batch['infos']):
            lifetimes[i] = agent_info_dict.get('lifetime', np.nan)
        result['avg_agent_lifetime'] = np.mean(lifetimes[~np.isnan(lifetimes)])

        for i, agent_info_dict in enumerate(train_batch['infos']):
            if "targets_found" in agent_info_dict:
                self.avg_targets.append(agent_info_dict["targets_found"])
                if len(self.avg_targets) > 20:
                    self.avg_targets = self.avg_targets[1:]
        result['avg_targets_found'] = sum(self.avg_targets) / len(self.avg_targets)
        result['max_targets_found'] = max(self.avg_targets)


class MyCallbacksSoccer(DefaultCallbacks):
    def __init__(self):
        super(MyCallbacksSoccer, self).__init__()
        self.winner = [0]
        self.draw = [0]
        self.soccer_config = {"max_env_actions": 5, "context_size": 3, "use_context": 3, "discrete_context": False}

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
        for i, agent_info_dict in enumerate(train_batch['infos']):
            if "Draw" in agent_info_dict:
                if agent_info_dict['Draw'] == 0:
                    self.winner.append(agent_info_dict['A']*-1 + agent_info_dict['B']*1)
                self.draw.append(agent_info_dict['Draw'])
        if len(self.winner) > 20:
            self.winner = self.winner[len(self.winner)-10:]
        if len(self.draw) > 20:
            self.draw = self.draw[len(self.draw)-10:]
        result['winner'] = sum(self.winner) / len(self.winner)
        result['draw'] = sum(self.draw) / len(self.draw)


class MyCallbacksFarm(DefaultCallbacks):
    def __init__(self):
        super(MyCallbacksFarm, self).__init__()
        self.last_life = 0
        self.last_avt = 0
        self.last_avc = 0
        self.last_attacktropy = 0

    def on_episode_start(self, *, worker: RolloutWorker, base_env,
                         policies,
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"
        # print("episode {} (env-idx={}) started.".format(
            # episode.episode_id, env_index))
        episode.user_data["avt"] = []
        episode.user_data['avc'] = []
        episode.user_data['c_t_attacktropy'] = []

    def on_episode_step(self, *, worker, base_env,
                        episode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        for agent_id, _ in episode.agent_rewards:
            info = episode.last_info_for(agent_id)
            if info != None and 'avt' in info:
                episode.user_data['avt'].append(info['avt'])
            if info != None and 'avc' in info:
                episode.user_data['avc'].append(info['avc'])
            if info != None and 'c_t_attacktropy' in info:
                episode.user_data['c_t_attacktropy'].append(info['c_t_attacktropy'])

    def on_episode_end(self, *, worker: "RolloutWorker", base_env, policies, episode, env_index, **kwargs) -> None:
        # return super().on_episode_end(worker, base_env, policies, episode, env_index=env_index, **kwargs)
        assert episode.batch_builder.policy_collectors[
            "policy_1"].buffers["dones"][-1], \
            "ERROR: `on_episode_end()` should only be called " \
            "after episode is done!"
        episode.custom_metrics['avt'] = np.mean(episode.user_data['avt'])
        episode.custom_metrics['avc'] = np.mean(episode.user_data['avc'])
        episode.custom_metrics['c_t_attacktropy'] = np.mean(episode.user_data['c_t_attacktropy'])

def execution_plan_maker(standardize_adv = False):
    def execution_plan(workers: WorkerSet,
                    config: TrainerConfigDict) -> LocalIterator[dict]:
        """Execution plan of the PPO algorithm. Defines the distributed dataflow.

        Args:
            workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
                of the Trainer.
            config (TrainerConfigDict): The trainer's configuration dict.

        Returns:
            LocalIterator[dict]: The Policy class to use with PPOTrainer.
                If None, use `default_policy` provided in build_trainer().
        """
        rollouts = ParallelRollouts(workers, mode="bulk_sync")

        # Collect batches for the trainable policies.
        rollouts = rollouts.for_each(
            SelectExperiences(workers.trainable_policies()))
        # Concatenate the SampleBatches into one.
        rollouts = rollouts.combine(
            ConcatBatches(
                min_batch_size=config["train_batch_size"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            ))
        # Standardize advantages.
        if standardize_adv:
            rollouts = rollouts.for_each(StandardizeFields(["advantages"]))

        # Perform one training step on the combined + standardized batch.
        train_op = rollouts.for_each(
            TrainOneStep(
                workers,
                num_sgd_iter=config["num_sgd_iter"],
                sgd_minibatch_size=config["sgd_minibatch_size"]))

        # Update KL after each round of training. --> This seems to have bugs when I use it with these trainers, just skip for now.
        # train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))

        # Warn about bad reward scales and return training metrics.
        return StandardMetricsReporting(train_op, workers, config) \
            .for_each(lambda result: warn_about_bad_reward_scales(config, result))
    return execution_plan
