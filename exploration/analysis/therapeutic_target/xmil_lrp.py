"""Retained xMIL-LRP propagation for late-fusion ABMIL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from downstream_tasks.models.pdl import PDropout


EPS = 1e-8


def stabilize(values: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    signs = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
    return values + eps * signs


def lrp_step(
    rule_output: torch.Tensor,
    relevance: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    activations: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    scaled = rule_output * (relevance / stabilize(rule_output)).detach()
    gradients = torch.autograd.grad(scaled.sum(), inputs, retain_graph=True)
    return tuple(
        activation * torch.nan_to_num(gradient)
        for activation, gradient in zip(activations, gradients)
    )


def projector_forward(projector: nn.Module | None, bag: torch.Tensor):
    if projector is None:
        return bag, []
    current = bag
    steps = []
    for layer in projector:
        if isinstance(layer, PDropout):
            continue
        if isinstance(layer, nn.ReLU):
            current = layer(current)
            continue
        activation = current
        inputs = current.detach().clone().requires_grad_(True)
        current = layer(inputs)
        steps.append((activation, inputs, F.linear(inputs, layer.weight, None)))
    return current, steps


def layernorm_rule(norm: nn.LayerNorm, inputs: torch.Tensor) -> torch.Tensor:
    mean = inputs.mean(dim=-1, keepdim=True)
    std = inputs.std(dim=-1, keepdim=True).detach()
    propagated = (inputs - mean) / stabilize(std, norm.eps)
    if norm.elementwise_affine:
        propagated = propagated * norm.weight + norm.bias
    actual = norm(inputs)
    return propagated + (actual - propagated).detach()


def explain_late_fusion_bag(
    model: nn.Module,
    bag: torch.Tensor,
    esm: torch.Tensor,
    class_indices,
) -> dict[int, dict[str, object]]:
    """Return feature-level contextual and sequence relevance for one bag."""
    model.eval()
    raw_bag = bag.detach()
    esm_activation = esm.detach()
    projected_bag, projector_steps = projector_forward(model.ctx_proj, raw_bag)

    n_instances = projected_bag.shape[0]
    batch = torch.zeros(n_instances, dtype=torch.long, device=bag.device)
    ptr = torch.tensor([0, n_instances], dtype=torch.long, device=bag.device)
    attention = model.attention(projected_bag.detach(), batch=batch, ptr=ptr).detach()

    aggregation_input = projected_bag.detach().clone().requires_grad_(True)
    esm_input = esm_activation.detach().clone().requires_grad_(True)
    pooled = (aggregation_input[:, None, :] * attention[:, :, None]).sum(dim=0)
    fused_rule = torch.cat([pooled.flatten(), esm_input], dim=0).unsqueeze(0)
    fused_activation = fused_rule.detach()

    norm_input = fused_activation.detach().clone().requires_grad_(True)
    norm_activation = model.norm(norm_input)
    norm_output = layernorm_rule(model.norm, norm_input)

    classifier_input = norm_activation.detach().clone().requires_grad_(True)
    logits = model.classifier(classifier_input)
    classifier_output = F.linear(classifier_input, model.classifier.weight, None)
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]

    scores = {}
    for class_idx in map(int, class_indices):
        output_relevance = torch.zeros_like(logits)
        output_relevance[0, class_idx] = logits[0, class_idx]
        (norm_relevance,) = lrp_step(
            classifier_output,
            output_relevance,
            (classifier_input,),
            (norm_activation,),
        )
        (fused_relevance,) = lrp_step(
            norm_output,
            norm_relevance,
            (norm_input,),
            (fused_activation,),
        )
        context_relevance, esm_relevance = lrp_step(
            fused_rule,
            fused_relevance,
            (aggregation_input, esm_input),
            (projected_bag, esm_activation),
        )
        for activation, inputs, rule_output in reversed(projector_steps):
            (context_relevance,) = lrp_step(
                rule_output,
                context_relevance,
                (inputs,),
                (activation,),
            )
        context_evidence = context_relevance.sum(dim=1)
        esm_evidence = esm_relevance.sum()
        scores[class_idx] = {
            "prob": float(probabilities[class_idx]),
            "context_evidence": context_evidence.detach().cpu().numpy(),
            "context_positive_evidence": (
                context_evidence.clamp(min=0).detach().cpu().numpy()
            ),
            "context_abs_evidence": (
                context_relevance.abs().sum(dim=1).detach().cpu().numpy()
            ),
            "esm_evidence": float(esm_evidence.detach().cpu()),
            "esm_positive_evidence": float(esm_evidence.clamp(min=0).detach().cpu()),
            "esm_abs_evidence": float(esm_relevance.abs().sum().detach().cpu()),
        }
    return scores
