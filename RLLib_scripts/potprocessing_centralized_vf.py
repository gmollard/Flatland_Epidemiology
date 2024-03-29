from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Postprocessing(object):
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"


@DeveloperAPI
def compute_advantages(rollout,
                       last_r,
                       gamma=0.9,
                       lambda_=1.0,
                       use_gae=True,
                       use_centralized_vf=False):
    """Given a rollout, compute its value targets and the advantage.
    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_centralized_vf (bool): whether we should compute the
            advantages using a centralized value function
    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    if use_gae:
        assert SampleBatch.VF_PREDS in rollout, "Values not found!"
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
            traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        advantages = discount(delta_t, gamma * lambda_)

        if use_centralized_vf:
            assert "central_vf_preds" in rollout, "Central values not found!"
            central_vpred_t = np.concatenate(
                [rollout[SampleBatch.VF_PREDS],
                 np.array([last_r])])
            central_delta_t = (
                traj[SampleBatch.REWARDS] + gamma * central_vpred_t[1:] -
                central_vpred_t[:-1])
            central_advantages = discount(central_delta_t, gamma * lambda_)
            traj[Postprocessing.ADVANTAGES] = central_advantages
            traj["central_value_targets"] = (
                central_advantages + traj["central_vf_preds"]).copy().astype(
                    np.float32)
        else:
            traj[Postprocessing.ADVANTAGES] = advantages
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.VALUE_TARGETS] = (
            advantages + traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array([last_r])])
        traj[Postprocessing.ADVANTAGES] = discount(rewards_plus_v, gamma)[:-1]
        # TODO(ekl): support using a critic without GAE
        traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
            traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)