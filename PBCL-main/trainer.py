import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

if __package__:
    from .metrics import calculate_pbcl_metrics
else:
    from metrics import calculate_pbcl_metrics


LOGGER = logging.getLogger(__name__)
LOSS_NAMES = ("loss", "rec", "sen", "adv", "orth", "edge", "cf")
TEACHER_FIELDS = ("users", "items", "chains", "selected_edges", "utility_target",
                  "sensitivity_target", "utility_mask", "sensitivity_mask")


def _option(args, name, default):
    value = getattr(args, name, None)
    return default if value is None else value


def _integer(value, name, minimum=0):
    try:
        result = int(value)
        valid = float(value) == result and result >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _number(value, name, positive=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    return result


def _sequence(value):
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith("[") else value.replace(",", " ").split()
    return list(value)


def _tree_to(value, device, clone=False):
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device) if clone else value.to(device)
        return result.clone() if clone else result
    if isinstance(value, dict):
        return {key: _tree_to(item, device, clone) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree_to(item, device, clone) for item in value)
    return value


def _take_rows(value, indices):
    if isinstance(value, dict):
        return {key: _take_rows(item, indices) for key, item in value.items()}
    return value[indices]


def _concat_rows(values):
    if isinstance(values[0], dict):
        return {key: _concat_rows([value[key] for value in values]) for key in values[0]}
    return torch.cat(values, dim=0)


def _seed_worker(_worker_id):
    # PyTorch already assigns a worker-specific torch seed.
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _safe_args(args):
    def convert(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return str(value)
    return {key: convert(value) for key, value in vars(args).items()}


class PBCLTrainer:
    """Own three disjoint optimizers and checkpoint complete epoch-boundary state."""

    def __init__(self, model, dataset, args, metric_fn=None):
        self.args, self.dataset = args, dataset
        self.device = torch.device(_option(args, "device", "cpu"))
        self.model = model.to(self.device)
        if not hasattr(model, "main_parameters"):
            raise TypeError("PBCLTrainer requires the PBCL model's parameter-block API.")
        self.behaviors = tuple(dataset.behaviors)
        self.metric_fn = metric_fn
        self.batch_size = _integer(_option(args, "batch_size", 256), "batch_size", 1)
        self.test_batch_size = _integer(_option(args, "test_batch_size", 64), "test_batch_size", 1)
        self.epochs = _integer(_option(args, "epochs", 200), "epochs", 1)
        self.num_workers = _integer(_option(args, "num_workers", 0), "num_workers")
        self.seed = _integer(_option(args, "seed", 2026), "seed")
        self.discriminator_steps = _integer(_option(args, "discriminator_steps", 1), "discriminator_steps", 1)
        self.audit_pretrain_epochs = _integer(_option(args, "audit_pretrain_epochs", 1), "audit_pretrain_epochs")
        self.audit_refresh_interval = _integer(_option(args, "audit_refresh_interval", 50), "audit_refresh_interval")
        self.audit_refresh_epochs = _integer(_option(args, "audit_refresh_epochs", 1), "audit_refresh_epochs")
        self.audit_max_batches = _integer(_option(args, "audit_max_batches", 0), "audit_max_batches")
        # Legacy quota options remain readable for existing configuration files;
        # actual sampling uses one edge_teacher_samples pool and one refresh clock.
        self.audit_history_samples = _integer(_option(args, "audit_history_samples", 1024), "audit_history_samples")
        self.edge_refresh_interval = _integer(_option(args, "edge_refresh_interval", 50), "edge_refresh_interval", 1)
        self.edge_teacher_samples = _integer(_option(args, "edge_teacher_samples", 1024), "edge_teacher_samples", 1)
        self.edge_buy_samples = _integer(_option(args, "edge_buy_samples", 256), "edge_buy_samples")
        self.edge_history_samples = _integer(_option(args, "edge_history_samples", 256), "edge_history_samples")
        self.edge_batch_size = _integer(_option(args, "edge_batch_size", 256), "edge_batch_size", 1)
        self.edge_warmup_steps = _integer(_option(args, "edge_warmup_steps", 0), "edge_warmup_steps")
        self.prototype_refresh_interval = _integer(_option(args, "prototype_refresh_interval", 100),
                                                   "prototype_refresh_interval", 1)
        self.cf_warmup_steps = _integer(_option(args, "cf_warmup_steps", 0), "cf_warmup_steps")
        self.cf_candidate_count = _integer(_option(args, "cf_candidate_count", 2), "cf_candidate_count", 2)
        self.prediction_pair_batch_size = _integer(_option(args, "prediction_pair_batch_size", 4096),
                                                    "prediction_pair_batch_size", 1)
        self.max_grad_norm = _number(_option(args, "max_grad_norm", 0.0), "max_grad_norm")
        self.eval_every = _integer(_option(args, "eval_every", 1), "eval_every", 1)
        # Preserve the paper's epoch budget unless early stopping is requested.
        self.patience = _integer(_option(args, "patience", 0), "patience")
        self.min_delta = _number(_option(args, "min_delta", 0.0), "min_delta")
        self.topk = sorted({_integer(value, "topk", 1) for value in _sequence(_option(args, "topk", [10, 50]))})
        if not self.topk:
            raise ValueError("topk must contain at least one cutoff.")
        self.metrics = []
        for name in _sequence(_option(args, "metrics", ["hit", "ndcg"])):
            name = str(name).lower()
            name = "hit" if name == "hr" else name
            if name not in ("hit", "ndcg", "recall", "precision", "mrr"):
                raise ValueError(f"Unsupported ranking metric: {name}")
            if name not in self.metrics:
                self.metrics.append(name)
        # PBCL always reports the two primary ranking measures.
        self.metrics = list(dict.fromkeys(self.metrics + ["hit", "ndcg"]))
        self.monitor = str(_option(args, "monitor", f"ndcg@{self.topk[0]}")).lower()
        if self.monitor.startswith("hr@"):
            self.monitor = "hit@" + self.monitor.split("@", 1)[1]
        self.monitor_mode = _option(args, "monitor_mode", "max")
        if self.monitor_mode not in ("max", "min"):
            raise ValueError("monitor_mode must be 'max' or 'min'.")
        self.ranking_policy = _option(args, "ranking_policy", "novel")
        if self.ranking_policy not in ("novel", "repeat"):
            raise ValueError("ranking_policy must be 'novel' or 'repeat'.")
        self.save_checkpoints = bool(_option(args, "save_checkpoints", True))
        self.checkpoint_dir = Path(_option(args, "model_path", Path(__file__).resolve().parent / "check_point"))
        self.model_name = str(_option(args, "model_name", "PBCL"))
        if Path(self.model_name).name != self.model_name or any(character in self.model_name for character in "/\\:"):
            raise ValueError("model_name must be a file-name prefix, not a path.")
        self.best_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        self.last_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_last.pth"

        self.main_params = list(model.main_parameters())
        self.margin_params = list(model.chain_learning.utility_predictor.parameters()) + list(
            model.chain_learning.sensitivity_predictor.parameters())
        self.discriminator_params = list(model.discriminator_parameters())
        self.audit_params = list(model.audit_parameters())
        blocks = [set(map(id, block)) for block in (self.main_params, self.discriminator_params, self.audit_params)]
        if any(not block for block in blocks) or any(blocks[a] & blocks[b] for a in range(3) for b in range(a)):
            raise ValueError("The three optimizer blocks must be nonempty and disjoint.")
        if set.union(*blocks) != set(map(id, model.parameters())):
            raise ValueError("Optimizer blocks must cover every model parameter exactly once.")
        learning_rate = _number(_option(args, "lr", 0.001), "lr", True)
        weight_decay = _number(_option(args, "weight_decay", _option(args, "decay", 0.0)), "weight_decay")
        disc_lr = _number(_option(args, "discriminator_lr", learning_rate), "discriminator_lr", True)
        audit_lr = _number(_option(args, "audit_lr", learning_rate), "audit_lr", True)
        disc_decay = _number(_option(args, "discriminator_weight_decay", 0.0), "discriminator_weight_decay")
        audit_decay = _number(_option(args, "audit_weight_decay", 0.0), "audit_weight_decay")
        self.main_optimizer = optim.Adam(self.main_params, lr=learning_rate, weight_decay=weight_decay)
        self.discriminator_optimizer = optim.Adam(self.discriminator_params, lr=disc_lr, weight_decay=disc_decay)
        self.audit_optimizer = optim.Adam(self.audit_params, lr=audit_lr, weight_decay=audit_decay)
        self.optimizer = self.main_optimizer  # Generic compatibility; not a combined optimizer.
        self.train_data = dataset.train_dataset()
        if not len(self.train_data):
            raise ValueError("PBCL training requires observed training-buy samples.")
        self.train_generator = torch.Generator().manual_seed(self.seed)
        self.audit_generator = torch.Generator().manual_seed(self.seed + 1)
        self.evaluation_generator = torch.Generator().manual_seed(self.seed + 2)
        self.sampling_rng = random.Random(self.seed + 3)
        self.global_step, self.next_epoch = 0, 0
        self.last_audit_step = self.last_teacher_step = self.last_prototype_step = -1
        self.audit_initialized = False
        self.teacher_cache = None
        self._teacher_pool = None
        self.best_score, self.best_epoch, self.best_model_state = None, None, None
        self.bad_evaluations = 0
        self.history = []
        self.last_test_result = None
        self._in_epoch = False
        self._epoch_interrupted = False
        self.data_signature = self._dataset_signature()
        self.training_config = self._training_config(learning_rate, weight_decay, disc_lr, disc_decay,
                                                      audit_lr, audit_decay)
        self.model.set_audit_training(False)

    def _make_loader(self, audit=False):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=False, worker_init_fn=_seed_worker,
                          generator=self.audit_generator if audit else self.train_generator,
                          pin_memory=self.device.type == "cuda", persistent_workers=False)

    def _sample_indices(self, size, limit):
        return self.sampling_rng.sample(range(size), min(size, limit))

    def _history_batch(self, limit):
        indices = self._sample_indices(len(self.dataset.training_pair_keys), limit)
        if not indices:
            return None
        keys = self.dataset.training_pair_keys[indices]
        users = torch.as_tensor(keys // (self.dataset.item_count + 1), dtype=torch.long)
        items = torch.as_tensor(keys % (self.dataset.item_count + 1), dtype=torch.long)
        return {"users": users, "items": items,
                "chains": self.dataset.get_batch_chains(users, items)}

    def _buy_batch(self, limit):
        indices = self._sample_indices(len(self.train_data), limit)
        return default_collate([self.train_data[index] for index in indices]) if indices else None

    def _optimizer_step(self, loss, optimizer, parameters):
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("PBCL produced a non-finite or non-scalar optimization loss.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm or float("inf"),
                                              error_if_nonfinite=True)
        optimizer.step()
        return float(norm.detach())

    def _discriminator_step(self, batch):
        self.model.zero_grad(set_to_none=True)
        loss = self.model.discriminator_loss(batch)
        norm = self._optimizer_step(loss, self.discriminator_optimizer, self.discriminator_params)
        return {"disc_loss": float(loss.detach()), "disc_grad_norm": norm}

    def _build_teacher_pool(self):
        """Index eligible query histories using training data only.

        A context is a distinct observed-buy query (u,i,t), excluding that buy,
        or the final training history of a pair with no training buy. These
        contexts form one pool, without fixed buy/auxiliary sampling quotas.
        """
        buys = np.unique(self.dataset.train_events[self.dataset.target_behavior])
        stride = self.dataset.item_count + 1
        buy_keys = np.unique(buys["user"] * stride + buys["item"])
        auxiliary_keys = np.setdiff1d(self.dataset.training_pair_keys, buy_keys, assume_unique=True)
        population = {
            "users": torch.from_numpy(np.concatenate((buys["user"], auxiliary_keys // stride))),
            "items": torch.from_numpy(np.concatenate((buys["item"], auxiliary_keys % stride))),
            "cutoffs": torch.from_numpy(np.concatenate((buys["timestamp"], np.full(len(auxiliary_keys), np.inf)))),
            "buy_mask": torch.cat((torch.ones(len(buys), dtype=torch.bool),
                                   torch.zeros(len(auxiliary_keys), dtype=torch.bool))),
        }
        eligible = []
        for start in range(0, len(population["users"]), 4096):
            batch = _take_rows(population, slice(start, start + 4096))
            chains = self.dataset.get_batch_chains(batch["users"], batch["items"], batch["cutoffs"])
            eligible.append(chains["eligible_edge_mask"].flatten(1).any(dim=1))
        keep = torch.cat(eligible) if eligible else torch.zeros(0, dtype=torch.bool)
        self._teacher_pool = _take_rows(population, keep)
        size = len(self._teacher_pool["users"])
        if size < self.edge_teacher_samples:
            LOGGER.warning("Only %d eligible teacher histories; use all without replacement (requested %d).",
                           size, self.edge_teacher_samples)

    def _sample_teacher_histories(self):
        if self._teacher_pool is None:
            self._build_teacher_pool()
        indices = self._sample_indices(len(self._teacher_pool["users"]), self.edge_teacher_samples)
        if not indices:
            return None
        histories = _take_rows(self._teacher_pool, torch.tensor(indices, dtype=torch.long))
        histories["chains"] = self.dataset.get_batch_chains(
            histories["users"], histories["items"], histories["cutoffs"])
        return histories

    def fit_auditors(self, epochs=None, max_batches=None, histories=None, encoded=None):
        """Fit receiving-behavior auditors on this refresh's sampled histories."""
        epochs = self.audit_refresh_epochs if epochs is None else _integer(epochs, "audit epochs")
        limit = self.audit_max_batches if max_batches is None else _integer(max_batches, "audit max_batches")
        histories = self._sample_teacher_histories() if histories is None else histories
        counts = torch.zeros_like(self.model.auditor_ready, dtype=torch.long)
        total, steps = 0.0, 0
        previous_mode = self.model.training
        self.model.eval()
        self.model.set_audit_training(True)
        try:
            with torch.no_grad():
                encoded = self.model.encode() if encoded is None else encoded
                encoded = _tree_to(encoded, self.device, True)
            if histories is not None:
                for _ in range(epochs):
                    for index, start in enumerate(range(0, len(histories["users"]), self.edge_batch_size)):
                        if limit and index >= limit:
                            break
                        batch = _take_rows(histories, slice(start, start + self.edge_batch_size))
                        self.model.zero_grad(set_to_none=True)
                        audit = self.model.audit_loss(batch["users"], batch["items"], batch["chains"], encoded)
                        if not bool(audit["counts"].any()):
                            continue
                        loss = audit["loss"]
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Non-finite auditor loss.")
                        loss.backward()
                        for behavior, heads in enumerate(self.model.attribute_auditors.heads):
                            for attribute, head in heads.items():
                                column = ("age", "gender", "pop").index(attribute)
                                if not int(audit["counts"][behavior, column]):
                                    for parameter in head.parameters():
                                        parameter.grad = None
                        torch.nn.utils.clip_grad_norm_(self.audit_params, self.max_grad_norm or float("inf"),
                                                       error_if_nonfinite=True)
                        self.audit_optimizer.step()
                        self.model.mark_auditors_ready(audit["counts"])
                        counts += audit["counts"]
                        total += float(loss.detach())
                        steps += 1
        finally:
            self.model.train(previous_mode)
            self.model.set_audit_training(False)
            self.model.zero_grad(set_to_none=True)
        self.audit_initialized = True
        self.last_audit_step = self.global_step
        self.teacher_cache = None
        self.last_teacher_step = -1
        return {"audit_loss": total / steps if steps else 0.0, "audit_steps": steps,
                "head_counts": counts.detach().cpu()}

    def fit_margin_predictors(self, encoded=None):
        """One cache pass updating Q_U/Q_S only; the backbone is detached."""
        if self.teacher_cache is None:
            return {"predictor_loss": 0.0, "predictor_steps": 0}
        previous_mode = self.model.training
        self.model.eval()
        total, steps = 0.0, 0
        try:
            with torch.no_grad():
                encoded = self.model.encode() if encoded is None else encoded
                encoded = _tree_to(encoded, self.device, True)
            indices = self._sample_indices(len(self.teacher_cache["users"]), len(self.teacher_cache["users"]))
            for start in range(0, len(indices), self.edge_batch_size):
                rows = torch.tensor(indices[start:start + self.edge_batch_size], dtype=torch.long)
                teachers = _take_rows(self.teacher_cache, rows)
                self.model.zero_grad(set_to_none=True)
                loss = self.model.edge_distillation_loss(teachers, encoded)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite predictor distillation loss.")
                loss.backward()
                for name, predictor in (("utility", self.model.chain_learning.utility_predictor),
                                        ("sensitivity", self.model.chain_learning.sensitivity_predictor)):
                    if not bool(teachers[f"{name}_mask"].any()):
                        for parameter in predictor.parameters():
                            parameter.grad = None
                torch.nn.utils.clip_grad_norm_(self.margin_params, self.max_grad_norm or float("inf"),
                                               error_if_nonfinite=True)
                self.main_optimizer.step()
                total += float(loss.detach())
                steps += 1
        finally:
            self.model.train(previous_mode)
            self.model.set_audit_training(False)
            self.model.zero_grad(set_to_none=True)
        return {"predictor_loss": total / steps if steps else 0.0, "predictor_steps": steps}

    def refresh_edge_teachers(self, encoded=None):
        """Refresh auditors, detached teachers and margin predictors at one backbone snapshot."""
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                encoded = self.model.encode() if encoded is None else encoded
                encoded = _tree_to(encoded, self.device, True)
            histories = self._sample_teacher_histories()
            epochs = self.audit_refresh_epochs if self.audit_initialized else self.audit_pretrain_epochs
            audit = self.fit_auditors(epochs=epochs, histories=histories, encoded=encoded)
            sources = []
            if histories is not None:
                selected = self.dataset.sample_edge_counterfactual(histories["chains"])
                histories["selected_edges"] = selected
                for start in range(0, len(histories["users"]), self.edge_batch_size):
                    batch = _take_rows(histories, slice(start, start + self.edge_batch_size))
                    buy_mask = batch["buy_mask"]
                    with torch.no_grad():
                        negative_scores = torch.zeros(len(buy_mask), device=self.device)
                        if bool(buy_mask.any()):
                            users = batch["users"][buy_mask]
                            negatives = torch.tensor([self.dataset.sample_negative(int(user)) for user in users],
                                                     dtype=torch.long)
                            negative_chains = self.dataset.get_batch_chains(users, negatives, batch["cutoffs"][buy_mask])
                            negative_scores[buy_mask.to(self.device)] = self.model.score_pairs(
                                users, negatives, negative_chains, encoded=encoded)
                        source = self.model.build_edge_teachers(
                            batch["users"], batch["items"], batch["chains"], negative_scores,
                            buy_mask, batch["selected_edges"], encoded)
                    sources.append(_tree_to({field: source[field] for field in TEACHER_FIELDS}, "cpu", True))
            self.teacher_cache = None
            if sources:
                merged = _concat_rows(sources)
                keep = merged["utility_mask"] | merged["sensitivity_mask"]
                if bool(keep.any()):
                    self.teacher_cache = _take_rows(merged, keep)
            self.last_teacher_step = self.global_step
            predictor = self.fit_margin_predictors(encoded)
            return {"utility_teachers": int(self.teacher_cache["utility_mask"].sum()) if self.teacher_cache else 0,
                    "sensitivity_teachers": int(self.teacher_cache["sensitivity_mask"].sum()) if self.teacher_cache else 0,
                    "sampled_histories": len(histories["users"]) if histories is not None else 0,
                    **audit, **predictor}
        finally:
            self.model.train(previous_mode)
            self.model.set_audit_training(False)

    def _teacher_batch(self):
        if self.teacher_cache is None:
            return None
        indices = self._sample_indices(len(self.teacher_cache["users"]), self.edge_batch_size)
        return _take_rows(self.teacher_cache, torch.tensor(indices, dtype=torch.long))

    def _cf_candidates(self, batch):
        if self.cf_candidate_count == 2:
            return None  # Reuse the model's factual positive/negative outputs.
        users = batch["user"].detach().cpu().long()
        positive, negative = batch["pos_item"].detach().cpu().long(), batch["neg_item"].detach().cpu().long()
        width = min(self.cf_candidate_count, self.dataset.item_count)
        candidates = torch.zeros(len(users), width, dtype=torch.long)
        mask = torch.zeros_like(candidates, dtype=torch.bool)
        for row, (user, pos, neg) in enumerate(zip(users.tolist(), positive.tolist(), negative.tolist())):
            selected = [pos, neg]
            selected_set = set(selected)
            excluded = self.dataset.train_buy_sets.get(user, set())
            # Uniformly sample additional training-buy-unobserved items without replacement.
            attempts = 0
            while len(selected) < width and attempts < 8 * width:
                item = self.sampling_rng.randint(1, self.dataset.item_count)
                attempts += 1
                if item not in excluded and item not in selected_set:
                    selected.append(item)
                    selected_set.add(item)
            if len(selected) < width:
                available = [item for item in range(1, self.dataset.item_count + 1)
                             if item not in excluded and item not in selected_set]
                selected.extend(self.sampling_rng.sample(available, min(width - len(selected), len(available))))
            candidates[row, :len(selected)] = torch.tensor(selected)
            mask[row, :len(selected)] = True
        return {"users": users, "items": candidates, "prediction_times": batch["prediction_time"].detach().cpu(),
                "mask": mask}

    def train_step(self, batch):
        """Complete A updates followed by one main update at the current global step."""
        self.model.train()
        self.model.set_audit_training(False)
        audit_result = {"audit_loss": 0.0, "audit_steps": 0}
        if (self.model.lambda_edge > 0 and self.global_step >= self.edge_warmup_steps and
                (self.last_teacher_step < 0 or self.global_step - self.last_teacher_step >= self.edge_refresh_interval)):
            # Audit and predictor-only updates do not advance the joint-update clock.
            audit_result = self.refresh_edge_teachers()
        batch = _tree_to(batch, self.device)
        disc_losses = []
        for _ in range(self.discriminator_steps):
            disc_losses.append(self._discriminator_step(batch)["disc_loss"])
        self.model.zero_grad(set_to_none=True)
        encoded = self.model.encode()
        compute_cf = self.global_step >= self.cf_warmup_steps
        if compute_cf and self.model.lambda_cf > 0 and (self.last_prototype_step < 0 or
                self.global_step - self.last_prototype_step >= self.prototype_refresh_interval):
            self.model.refresh_prototypes(encoded)
            self.last_prototype_step = self.global_step
        teachers = self._teacher_batch() if self.global_step >= self.edge_warmup_steps else None
        candidates = self._cf_candidates(batch) if compute_cf and self.model.lambda_cf > 0 else None
        losses = self.model.training_losses(batch, edge_teachers=teachers, cf_candidates=candidates,
                                            encoded=encoded, compute_cf=compute_cf)
        norm = self._optimizer_step(losses["loss"], self.main_optimizer, self.main_params)
        self.global_step += 1
        result = {name: float(losses[name].detach()) for name in LOSS_NAMES}
        result.update(disc_loss=sum(disc_losses) / len(disc_losses), main_grad_norm=norm,
                      audit_loss=audit_result["audit_loss"], audit_steps=audit_result["audit_steps"],
                      utility_teachers=int(teachers["utility_mask"].sum()) if teachers is not None else 0,
                      sensitivity_teachers=int(teachers["sensitivity_mask"].sum()) if teachers is not None else 0)
        return result

    def _train_one_epoch(self, loader=None, epoch=None):
        loader = self._make_loader() if loader is None else loader
        sums, examples = {}, 0
        self._in_epoch = True
        self._epoch_interrupted = True
        try:
            for batch in loader:
                count = len(batch["user"])
                values = self.train_step(batch)
                for name, value in values.items():
                    sums[name] = sums.get(name, 0.0) + value * count
                examples += count
        finally:
            self._in_epoch = False
        if not examples:
            raise ValueError("The training loader produced no samples.")
        self._epoch_interrupted = False
        result = {name: value / examples for name, value in sums.items()}
        result["examples"] = examples
        LOGGER.info("PBCL epoch %s step %d: loss=%.6f rec=%.6f edge=%.6f cf=%.6f",
                    epoch, self.global_step, result["loss"], result["rec"], result["edge"], result["cf"])
        return result

    def _is_improvement(self, value):
        if not math.isfinite(value):
            raise ValueError(f"Validation monitor {self.monitor} is not finite.")
        if self.best_score is None:
            return True
        return (value > self.best_score + self.min_delta if self.monitor_mode == "max"
                else value < self.best_score - self.min_delta)

    def train_model(self, evaluate_test=True):
        """Select epochs using validation only; evaluate the selected model on test once."""
        has_validation = bool(self.dataset.validation_interacts)
        self.last_test_result = None
        for epoch in range(self.next_epoch, self.epochs):
            if has_validation and self.patience and self.bad_evaluations >= self.patience:
                break
            train = self._train_one_epoch(epoch=epoch + 1)
            self.next_epoch = epoch + 1
            record = {"epoch": self.next_epoch, "global_step": self.global_step, "train": train}
            improved = False
            if has_validation and (self.next_epoch % self.eval_every == 0 or self.next_epoch == self.epochs):
                validation = self.evaluate(split="validation")
                record["validation"] = validation
                if self.monitor not in validation:
                    raise ValueError(f"Validation did not provide monitor {self.monitor!r}.")
                if self._is_improvement(float(validation[self.monitor])):
                    self.best_score = float(validation[self.monitor])
                    self.best_epoch = self.next_epoch
                    self.bad_evaluations = 0
                    self.best_model_state = _tree_to(self.model.state_dict(), "cpu", True)
                    improved = True
                else:
                    self.bad_evaluations += 1
                LOGGER.info("PBCL validation epoch %d: %s", self.next_epoch, validation)
            self.history.append(record)
            if self.save_checkpoints:
                if improved:
                    self.save_checkpoint(self.best_checkpoint_path)
                self.save_checkpoint(self.last_checkpoint_path)
            if has_validation and self.patience and self.bad_evaluations >= self.patience:
                break
        if evaluate_test and self.dataset.test_interacts:
            # Keep resumable last-epoch weights/optimizers intact after final testing.
            current = _tree_to(self.model.state_dict(), "cpu", True)
            try:
                if has_validation and self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
                self.last_test_result = self.evaluate(split="test")
            finally:
                self.model.load_state_dict(current)
                self.model.set_audit_training(False)
            LOGGER.info("PBCL final test (%s): %s", "validation best" if has_validation else "last epoch",
                        self.last_test_result)
        return {"history": self.history, "best_epoch": self.best_epoch, "best_score": self.best_score,
                "test": self.last_test_result, "next_epoch": self.next_epoch, "global_step": self.global_step}

    @torch.no_grad()
    def evaluate(self, epoch=None, test_batch_size=None, dataset=None, gt_interacts=None,
                 gt_length=None, split=None, return_details=False):
        """Rank the full catalog with one encoded snapshot and training-buy filtering."""
        if split is None:
            split = ("validation" if gt_interacts is self.dataset.validation_interacts or
                     (gt_interacts is None and bool(self.dataset.validation_interacts)) else "test")
        if split not in ("validation", "test"):
            raise ValueError("split must be 'validation' or 'test'.")
        gt_interacts = getattr(self.dataset, f"{split}_interacts") if gt_interacts is None else gt_interacts
        if dataset is None:
            dataset = self.dataset.validate_dataset() if split == "validation" else self.dataset.test_dataset()
        size = self.test_batch_size if test_batch_size is None else _integer(test_batch_size, "test_batch_size", 1)
        loader = DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0,
                            generator=self.evaluation_generator)
        previous_mode = self.model.training
        previous_audit = any(parameter.requires_grad for parameter in self.audit_params)
        self.model.eval()
        self.model.set_audit_training(False)
        users_all, recommendations, positives, lengths = [], [], [], []
        seen_users = set()
        try:
            encoded = self.model.encode()
            for batch_users in loader:
                batch_users = torch.as_tensor(batch_users, device=self.device).long().reshape(-1)
                scores = self.model.full_predict(batch_users, split=split,
                                                 pair_batch_size=self.prediction_pair_batch_size, encoded=encoded)
                if scores.shape != (len(batch_users), self.dataset.item_count + 1):
                    raise ValueError("PBCL full_predict must include exactly one padding column.")
                if bool(torch.isnan(scores).any()) or bool(torch.isposinf(scores).any()):
                    raise FloatingPointError("Non-finite positive/NaN evaluation scores.")
                scores[:, 0] = -torch.inf
                targets = []
                for row, user in enumerate(batch_users.tolist()):
                    if user in seen_users:
                        raise ValueError("This evaluation contract requires one query per user per split.")
                    seen_users.add(user)
                    items = gt_interacts.get(str(user), gt_interacts.get(user, []))
                    target = sorted({int(item) for item in items})
                    if len(target) != 1 or min(target) < 1 or max(target) > self.dataset.item_count:
                        raise ValueError(f"User {user} requires exactly one valid held-out target per split.")
                    trained = self.dataset.train_buy_sets.get(user, set())
                    if self.ranking_policy == "novel":
                        if trained.intersection(target):
                            raise ValueError("Held-out targets overlap training buys under ranking_policy='novel'. "
                                             "Use a consistent novel-item split or explicitly choose 'repeat'.")
                        if trained:
                            scores[row, torch.tensor(sorted(trained), device=self.device)] = -torch.inf
                    targets.append(target)
                width = min(max(self.topk), self.dataset.item_count)
                values, indices = scores.topk(width, dim=1)
                indices = indices.masked_fill(~torch.isfinite(values), 0).cpu().numpy()
                for row, (user, target) in enumerate(zip(batch_users.tolist(), targets)):
                    users_all.append(user)
                    recommendations.append(indices[row])
                    positives.append(np.isin(indices[row], target))
                    lengths.append(len(target))
        finally:
            self.model.train(previous_mode)
            self.model.set_audit_training(previous_audit)
        width = min(max(self.topk), self.dataset.item_count)
        topk_items = np.asarray(recommendations, dtype=np.int64).reshape(-1, width)
        hits = np.asarray(positives, dtype=bool).reshape(-1, width)
        users = np.asarray(users_all, dtype=np.int64)
        ground_truth_lengths = np.asarray(lengths, dtype=np.int64)
        # Recompute counts in the actual user order; legacy gt_length may be misordered.
        result = self.calculate_result(hits, ground_truth_lengths, users, topk_items)
        if return_details:
            return {"metrics": result, "users": users, "topk_items": topk_items,
                    "hits": hits, "gt_lengths": ground_truth_lengths, "split": split}
        return result

    def calculate_result(self, topk_list, gt_len, user_list, topk_idx_list):
        """Use shared PBCL metric definitions unless a callback is supplied."""
        calculator = calculate_pbcl_metrics if self.metric_fn is None else self.metric_fn
        return calculator(topk_list=topk_list, gt_len=gt_len, user_list=user_list,
                          topk_idx_list=topk_idx_list, dataset=self.dataset,
                          topk=self.topk, metrics=self.metrics)

    def _dataset_signature(self):
        digest = hashlib.sha256()
        for name in ("user_gender", "user_age", "item_popularity_label"):
            digest.update(getattr(self.dataset, name).detach().cpu().contiguous().numpy().tobytes())
        for behavior in self.behaviors:
            digest.update(behavior.encode("utf-8"))
            digest.update(self.dataset.train_events[behavior].tobytes())
        evaluation = {"validation": self.dataset.validation_interacts, "test": self.dataset.test_interacts,
                      "cutoffs": self.dataset.evaluation_cutoffs}
        digest.update(json.dumps(evaluation, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return {"users": self.dataset.user_count, "items": self.dataset.item_count,
                "behaviors": list(self.behaviors), "data_sha256": digest.hexdigest(),
                "temporal_scale": float(self.dataset.temporal_scale),
                "evaluation_time_policy": self.dataset.evaluation_time_policy}

    def _training_config(self, lr, decay, disc_lr, disc_decay, audit_lr, audit_decay):
        names = ("batch_size", "num_workers", "seed", "discriminator_steps", "audit_pretrain_epochs",
                 "audit_refresh_interval", "audit_refresh_epochs", "audit_max_batches", "audit_history_samples",
                 "edge_refresh_interval", "edge_teacher_samples", "edge_buy_samples", "edge_history_samples", "edge_batch_size",
                 "edge_warmup_steps", "prototype_refresh_interval", "cf_warmup_steps", "cf_candidate_count",
                 "max_grad_norm", "eval_every", "patience", "min_delta", "topk", "metrics", "monitor",
                 "monitor_mode", "ranking_policy", "test_batch_size", "prediction_pair_batch_size")
        result = {name: getattr(self, name) for name in names}
        result["optimizer"] = dict(lr=lr, decay=decay, disc_lr=disc_lr, disc_decay=disc_decay,
                                   audit_lr=audit_lr, audit_decay=audit_decay)
        result["model"] = {name: getattr(self.model, name) for name in
                           ("embedding_size", "user_rank", "item_rank", "lambda_adv", "lambda_orth",
                            "lambda_edge", "lambda_cf", "lambda_s")}
        chain = self.model.chain_learning
        result["model"].update(gcn_layers=self.model.disentanglement.encoder.layers,
                              chain_layers=chain.chain_layers, utility_weight=chain.utility_weight,
                              sensitivity_weight=chain.sensitivity_weight, empty_chain_policy=chain.empty_chain_policy,
                              cf_temperature=self.model.counterfactual.temperature,
                              sensitivity_attribute_weights=self.model.sensitivity_attribute_weights.detach().cpu().tolist(),
                              classes=dict(self.model.classes))
        return result

    def _rng_state(self):
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "train_generator": self.train_generator.get_state(),
                "audit_generator": self.audit_generator.get_state(),
                "evaluation_generator": self.evaluation_generator.get_state(),
                "sampling": self.sampling_rng.getstate()}

    def _restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        self.train_generator.set_state(state["train_generator"].cpu())
        self.audit_generator.set_state(state["audit_generator"].cpu())
        self.evaluation_generator.set_state(state["evaluation_generator"].cpu())
        self.sampling_rng.setstate(state["sampling"])

    def save_checkpoint(self, path=None):
        """Save at an epoch boundary; no partial DataLoader permutation is serialized."""
        if self._in_epoch or self._epoch_interrupted:
            raise RuntimeError("Save a resumable checkpoint only after completing an epoch.")
        destination = Path(path) if path is not None else self.last_checkpoint_path
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {"format": "PBCLTrainer", "version": 1,
                 "model_state_dict": self.model.state_dict(),
                 "optimizers": {"main": self.main_optimizer.state_dict(),
                                "discriminator": self.discriminator_optimizer.state_dict(),
                                "audit": self.audit_optimizer.state_dict()},
                 "global_step": self.global_step, "next_epoch": self.next_epoch,
                 "last_audit_step": self.last_audit_step, "last_teacher_step": self.last_teacher_step,
                 "last_prototype_step": self.last_prototype_step, "audit_initialized": self.audit_initialized,
                 "teacher_cache": self.teacher_cache, "best_score": self.best_score, "best_epoch": self.best_epoch,
                 "best_model_state": self.best_model_state, "bad_evaluations": self.bad_evaluations,
                 "history": self.history, "rng": self._rng_state(), "data_signature": self.data_signature,
                 "training_config": self.training_config, "args": _safe_args(self.args)}
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                torch.save(state, handle)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def load_checkpoint(self, path, restore_rng=True):
        if self._in_epoch:
            raise RuntimeError("Cannot restore a checkpoint during an epoch.")
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if state.get("format") != "PBCLTrainer" or state.get("version") != 1:
            raise ValueError("Expected a PBCLTrainer checkpoint, not an old-model state dictionary.")
        if state["data_signature"] != self.data_signature:
            raise ValueError("Checkpoint data, behavior order, timestamps, or attribute mapping differs.")
        if state["training_config"] != self.training_config:
            raise ValueError("Resume requires the same effective training/model configuration; epochs may be extended.")
        self.model.load_state_dict(state["model_state_dict"])
        self.main_optimizer.load_state_dict(state["optimizers"]["main"])
        self.discriminator_optimizer.load_state_dict(state["optimizers"]["discriminator"])
        self.audit_optimizer.load_state_dict(state["optimizers"]["audit"])
        for name in ("global_step", "next_epoch", "last_audit_step", "last_teacher_step", "last_prototype_step",
                     "audit_initialized", "teacher_cache", "best_score", "best_epoch", "best_model_state",
                     "bad_evaluations", "history"):
            setattr(self, name, state[name])
        self.teacher_cache = _tree_to(self.teacher_cache, "cpu", True)
        self._epoch_interrupted = False
        self.model.set_audit_training(False)
        if restore_rng:
            self._restore_rng(state["rng"])
        return {"next_epoch": self.next_epoch, "global_step": self.global_step, "best_epoch": self.best_epoch}

    def save_model(self, model=None, path=None):
        """Compatibility entry point that saves the complete PBCL training state."""
        if model is not None and model is not self.model:
            raise ValueError("Cannot combine another model's weights with this trainer's optimizer state.")
        return self.save_checkpoint(path)


Trainer = PBCLTrainer
