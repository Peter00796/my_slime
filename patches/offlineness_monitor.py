"""Monkey patch for policy_loss_function to monitor offlineness metrics.

This module provides a monkey patch that wraps the original policy_loss_function
to compute and report offlineness metrics (ESS, Mismatch KL) without modifying
the original source code.
"""

from argparse import Namespace
from collections.abc import Callable

import torch

from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.utils.types import RolloutBatch

# Store reference to original function
_original_policy_loss_function = None


def compute_offlineness_metrics(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Computes ESS and Mismatch KL for monitoring data staleness."""
    with torch.no_grad():
        # Flatten and filter by mask
        # Note: input log_probs/masks are flattened/concatenated before this function is called
        mask = loss_masks.bool().view(-1)
        valid_log_probs = log_probs.view(-1)[mask]
        valid_old_log_probs = old_log_probs.view(-1)[mask]
        
        if valid_log_probs.numel() == 0:
            return {}

        # Calculate Ratio
        log_ratio = valid_log_probs - valid_old_log_probs
        ratio = torch.exp(log_ratio)

        # 1. ESS Calculation: (Sum w)^2 / Sum (w^2)
        numerator = ratio.sum() ** 2
        denominator = (ratio ** 2).sum() + 1e-8
        ess = numerator / denominator
        ess_ratio = ess / ratio.numel()

        # 2. Mismatch KL (k3 estimator): 0.5 * (r-1)^2
        mismatch_kl = 0.5 * ((ratio - 1) ** 2).mean()

        # 3. Clip Fraction (Standard PPO bounds [0.8, 1.2])
        clip_frac = ((ratio < 0.8) | (ratio > 1.2)).float().mean()

    return {
        "offlineness/ess_ratio": ess_ratio,
        "offlineness/mismatch_kl": mismatch_kl,
        "offlineness/clip_frac_02": clip_frac,
        "offlineness/max_ratio": ratio.max(),
    }


def patched_policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapped policy_loss_function that adds offlineness metrics.
    
    This function calls the original policy_loss_function and then computes
    offlineness metrics (ESS, Mismatch KL) and adds them to the reported_loss dict.
    """
    # Call the original function
    loss, reported_loss = _original_policy_loss_function(args, batch, logits, sum_of_sample_mean)
    
    # Compute offlineness metrics
    # We need to get the concatenated log_probs and old_log_probs
    # Extract old_log_probs from batch
    old_log_probs_list = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    
    # Recompute current log_probs from logits
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    
    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=max_seq_lens,
    )
    
    log_probs_list = log_probs_and_entropy["log_probs"]
    
    # Concatenate both
    log_probs = torch.cat(log_probs_list, dim=0)
    old_log_probs = torch.cat(old_log_probs_list, dim=0)
    
    # Concatenate loss_masks
    loss_masks = torch.cat(batch["loss_masks"], dim=0)
    
    # Compute offlineness metrics
    offlineness_metrics = compute_offlineness_metrics(log_probs, old_log_probs, loss_masks)
    
    # Aggregate metrics using sum_of_sample_mean and add to reported_loss
    for key, value in offlineness_metrics.items():
        if value.numel() > 0:  # Only add if tensor is not empty
            reported_loss[key] = sum_of_sample_mean(value).clone().detach()
    
    return loss, reported_loss


def apply_patch() -> None:
    """Apply the monkey patch by replacing the original policy_loss_function."""
    global _original_policy_loss_function
    
    from slime.backends.megatron_utils import loss as loss_module
    
    # Store original function if not already stored
    if _original_policy_loss_function is None:
        _original_policy_loss_function = loss_module.policy_loss_function
    
    # Replace with patched version
    loss_module.policy_loss_function = patched_policy_loss_function


def remove_patch() -> None:
    """Remove the monkey patch and restore the original policy_loss_function."""
    global _original_policy_loss_function
    
    if _original_policy_loss_function is not None:
        from slime.backends.megatron_utils import loss as loss_module
        loss_module.policy_loss_function = _original_policy_loss_function
        _original_policy_loss_function = None

