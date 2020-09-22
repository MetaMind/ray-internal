import logging

import ray
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.framework import try_import_tf, get_variable
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.sgd import standardized

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)

PLANNER_ADVANTAGES = "planner_advantages"


class WPPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 agent_ids,
                 value_targets,
                 advantages,
                 planner_advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 cur_lambda_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            planner_advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation for the planner.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for action prob output
                from the previous (before update) Model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from the previous (before update) Model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current WPPO KL
                coefficient.
            cur_lambda_coeff (Variable): Variable holding the current WPPO lambda.
            valid_mask (Optional[tf.Tensor]): An optional bool mask of valid
                input elements (for max-len padded sequences (RNNs)).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:

            def reduce_mean_valid(t):
                return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        else:

            def reduce_mean_valid(t):
                return tf.reduce_mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))

        planner_surrogate_loss = tf.minimum(
            planner_advantages * logp_ratio,
            planner_advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))

        lambda_param = tf.gather(cur_lambda_coeff, agent_ids)
        both_surrogate_loss = (planner_surrogate_loss - lambda_param * surrogate_loss) / (1 + lambda_param)

        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        self.mean_adv_policy_loss = reduce_mean_valid(planner_surrogate_loss)
        self.mean_both_policy_loss = reduce_mean_valid(both_surrogate_loss)


        if use_gae:
            vf_loss1 = tf.math.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.math.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                both_surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(both_surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


def wppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])

    policy.loss_obj = WPPOLoss(
        dist_class,
        model,
        train_batch["agent_id"],
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[Postprocessing.PLANNER_ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        policy.lambda_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lambda_coeff": tf.cast(policy.lambda_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "me_policy_loss": policy.loss_obj.mean_policy_loss,
        "adv_policy_loss": policy.loss_obj.mean_adv_policy_loss,
        "joint_policy_loss": policy.loss_obj.mean_both_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


def vf_preds_fetches(policy):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
    }


def postprocess_wppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch["state_out_{}".format(i)][-1])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    planner_batch = other_agent_batches["p"]
    assert "agent_id" in batch
    batch[PLANNER_ADVANTAGES] = standardized(planner_batch[Postprocessing.ADVANTAGES])
    return batch


def clip_gradients(policy, optimizer, loss):
    variables = policy.model.trainable_variables()
    if policy.config["grad_clip"] is not None:
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads,
                                                 policy.config["grad_clip"])
        clipped_grads = list(zip(policy.grads, variables))
        return clipped_grads
    else:
        return optimizer.compute_gradients(loss, variables)


class LambdaMixin:
    def __init__(self, config):
        # Lagrange multipliers
        self.lambda_coeff_val = config["lambda_coeff"]
        self.lambda_coeff = get_variable(
            [float(self.lambda_coeff_val)] * config["env_config"]["env_config_dict"]["n_agents"],
            tf_name="lambda_coeff", trainable=False)

    def update_lambda(self, new_lambda):
        self.lambda_coeff_val = new_lambda
        self.lambda_coeff.load(self.lambda_coeff_val, session=self.get_session())
        return self.lambda_coeff_val


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient
        self.kl_coeff_val = config["kl_coeff"]
        self.kl_target = config["kl_target"]
        self.kl_coeff = get_variable(
            float(self.kl_coeff_val), tf_name="kl_coeff", trainable=False)

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        self.kl_coeff.load(self.kl_coeff_val, session=self.get_session())
        return self.kl_coeff_val


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                        [prev_reward]),
                    "is_training": tf.convert_to_tensor([False]),
                }, [tf.convert_to_tensor([s]) for s in state],
                                          tf.convert_to_tensor([1]))
                return self.model.value_function()[0]

        else:

            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._value = value


def setup_config(policy, obs_space, action_space, config):
    # auto set the model option for layer sharing
    config["model"]["vf_share_layers"] = config["vf_share_layers"]


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    LambdaMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


WPPOTFPolicy = build_tf_policy(
    name="WPPOTFPolicy",
    get_default_config=lambda: ray.rllib.agents.wppo.wppo.DEFAULT_CONFIG,
    loss_fn=wppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_fetches,
    postprocess_fn=postprocess_wppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin,
        LambdaMixin,
    ])
