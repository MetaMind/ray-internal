import numpy as np
import scipy.signal
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Postprocessing:
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"


@DeveloperAPI
def compute_advantages(rollout,
                       last_r,
                       gamma=0.9,
                       lambda_=1.0,
                       use_gae=True,
                       use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"
    
    if use_gae:
        if len(rollout[SampleBatch.VF_PREDS].shape) == len(np.array(last_r).shape)+1:
            last_r = [last_r]
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array(last_r)])  # SS** -> changed [last_r] to last_r
        delta_t = (
            traj[SampleBatch.REWARDS].reshape(-1, 1) + gamma * vpred_t[1:] - vpred_t[:-1]) # SS** -> added reshape
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
            traj[Postprocessing.ADVANTAGES] +
            traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array(last_r)])  # SS** -> changed [last_r] to last_r
        discounted_returns = discount(rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)

        if use_critic:
            traj[Postprocessing.
                 ADVANTAGES] = discounted_returns - rollout[SampleBatch.
                                                            VF_PREDS]
            traj[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            traj[Postprocessing.ADVANTAGES] = discounted_returns
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    
    return SampleBatch(traj)


# SS** Use stack to compute advantages in a vectorized manner
@DeveloperAPI
def compute_advantages_vectorized(agent_batches,
                                  last_rs,
                                  gamma=0.9,
                                  lambda_=1.0,
                                  use_gae=True,
                                  use_critic=True):
    """
    Given a set of rollouts, compute their value targets and the advantage.

    Args:
        agent_batches (Dict(policy, SampleBatch)): a set of agent trajectories
        last_rs (float): Value estimations for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        Dict of SampleBatches (SampleBatch): Objects with experience from rollout and
            processed rewards.
    """
    traj = {}
    agent_ids = ['a']
    rollout = agent_batches['a'][1]
    trajsize = len(rollout[SampleBatch.ACTIONS])
    rollout_keys = rollout.keys()

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"
    
    for key in rollout_keys:
        traj[key] = rollout[key]
        
    if use_gae:
        if len(rollout[SampleBatch.VF_PREDS].shape) == len(np.array(last_rs).shape) + 1:
            last_rs = [last_rs]
            
        vpred_t = np.concatenate((traj[SampleBatch.VF_PREDS],
                                  np.array(last_rs)))

        delta_t = (
                traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
                traj[Postprocessing.ADVANTAGES] +
                traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate((traj[SampleBatch.REWARDS],
                                         np.array(list(last_rs)).reshape(1, -1)))
        
        discounted_returns = discount(rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)
        
        if use_critic:
            traj[Postprocessing.
                ADVANTAGES] = discounted_returns - traj[SampleBatch.VF_PREDS]
            traj[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            traj[Postprocessing.ADVANTAGES] = discounted_returns
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])
    
    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)
    
    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
        
    # # Merge the batch and n_agents axes
    # for key in traj:
    #     print("before", key, traj[key].shape)
    #     # SS**
    #     if len(traj[key].shape) == 2:
    #         print("type F")
    #         prod = np.product(traj[key].shape[:2])
    #         traj[key] = traj[key].reshape(prod, )
    #     elif len(traj[key].shape) > 2:
    #         print("type G")
    #         prod = np.product(traj[key].shape[:-1])
    #         traj[key] = traj[key].reshape(prod, -1)
    #     else:
    #         print('type H')
    #         traj[key] = np.array(traj[key])
    #     print("after", key, traj[key].shape)

    return {agent_ids[0]: SampleBatch(traj)}
    
    # return {agent_id: SampleBatch(sample_batches[agent_id]) for agent_id in agent_ids}
