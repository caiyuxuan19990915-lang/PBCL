

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


ATTRIBUTES = ("age", "gender", "pop")


def _option(args, name, default):
    value = getattr(args, name, None)
    return default if value is None else value


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _integer(value, name, minimum=0):
    try:
        result = int(value)
        valid = float(value) == result and result >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _mlp(input_size, hidden_size, output_size):
    # Deterministic heads keep factual/counterfactual comparisons at one snapshot.
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, output_size))


def _zero(reference):
    return reference.sum() * 0.0


@contextmanager
def _frozen(module):
    flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(module.parameters(), flags):
            parameter.requires_grad_(enabled)


class BehaviorSpecificParallelEncoding(nn.Module):
    """Section 3.1.1: independent LightGCN graphs, one shared initial ID table."""

    def __init__(self, dataset, embedding_size=64, layers=2):
        super().__init__()
        if embedding_size < 3 or int(layers) != layers or layers < 0:
            raise ValueError("embedding_size must be >=3 and gcn_layers a nonnegative integer.")
        self.behaviors = tuple(dataset.behaviors)
        self.layers = int(layers)
        self.item_offset = dataset.item_offset
        self.embedding = nn.Embedding(dataset.num_nodes, embedding_size)
        keep = torch.ones(dataset.num_nodes, 1)
        keep[0] = keep[self.item_offset] = 0
        self.register_buffer("padding_mask", keep, persistent=False)
        for index, behavior in enumerate(self.behaviors):
            self.register_buffer(f"adjacency_{index}",
                                 dataset.normalized_adjacency[behavior].coalesce(), persistent=False)
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
            self.embedding.weight[self.item_offset].zero_()

    def forward(self):
        initial = self.embedding.weight * self.padding_mask
        outputs = []
        for index in range(len(self.behaviors)):
            adjacency = getattr(self, f"adjacency_{index}")
            hidden, accumulated = initial, initial
            for _ in range(self.layers):
                hidden = torch.sparse.mm(adjacency, hidden)
                accumulated = accumulated + hidden
            # Standard LightGCN aggregation includes layer zero.
            outputs.append(accumulated / (self.layers + 1))
        return torch.stack(outputs, dim=1)


class SensitiveFactorsEncoding(nn.Module):
    """Section 3.1.2: column-normalized shared and behavior-specific factors."""

    def __init__(self, embedding_size, num_behaviors, user_shared_rank=4,
                 user_specific_rank=4, item_shared_rank=4, item_specific_rank=4):
        super().__init__()
        self.num_behaviors = num_behaviors
        self.ranks = {"user": (user_shared_rank, user_specific_rank),
                      "item": (item_shared_rank, item_specific_rank)}
        for side, (shared, specific) in self.ranks.items():
            if any(int(rank) != rank or rank < 0 for rank in (shared, specific)):
                raise ValueError("Sensitive ranks must be nonnegative integers.")
            if not 0 < shared + specific < embedding_size:
                raise ValueError(f"The total {side} sensitive rank must lie in 1..d-1.")
            setattr(self, f"{side}_shared", nn.Parameter(torch.empty(embedding_size, shared)))
            setattr(self, f"{side}_specific", nn.ParameterList([
                nn.Parameter(torch.empty(embedding_size, specific)) for _ in range(num_behaviors)
            ]))
        for parameter in self.parameters():
            if parameter.numel():
                nn.init.xavier_uniform_(parameter)

    def factors(self, side):
        shared = F.normalize(getattr(self, f"{side}_shared"), dim=0, eps=1e-8)
        matrices, penalties = [], []
        for raw in getattr(self, f"{side}_specific"):
            specific = F.normalize(raw, dim=0, eps=1e-8)
            matrix = torch.cat((shared, specific), dim=1)
            matrices.append(matrix)
            # Appendix C.2.4 constrains every column of the concatenated factors.
            identity = torch.eye(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
            penalties.append((matrix.T @ matrix - identity).square().sum())
        return torch.stack(matrices), torch.stack(penalties).sum() / self.num_behaviors

    def forward(self, behavior_embeddings, item_offset):
        result, penalties = {}, []
        for side, embeddings in (("user", behavior_embeddings[:item_offset]),
                                 ("item", behavior_embeddings[item_offset:])):
            factors, orthogonal = self.factors(side)
            sensitive = torch.einsum("nkd,kdr->nkr", embeddings, factors)
            fair = embeddings - torch.einsum("nkr,kdr->nkd", sensitive, factors)
            result[f"{side}_sensitive"] = sensitive
            result[f"{side}_fair"] = fair
            penalties.append(orthogonal)
        result["orth_loss"] = torch.stack(penalties).sum()
        return result


class _AttributeClassifierGroup(nn.Module):
    """Attribute names stay public without colliding with ModuleDict.pop."""

    def __init__(self, user_size, item_size, classes, hidden_size):
        super().__init__()
        self.attributes = tuple(attribute for attribute in ATTRIBUTES if classes[attribute] > 0)
        for attribute in self.attributes:
            self.add_module(f"{attribute}_head", _mlp(item_size if attribute == "pop" else user_size,
                                                     hidden_size, classes[attribute]))

    def __getitem__(self, attribute):
        if attribute not in self.attributes:
            raise KeyError(attribute)
        return getattr(self, f"{attribute}_head")

    def items(self):
        return ((attribute, self[attribute]) for attribute in self.attributes)


class _AttributeHeads(nn.Module):
    """Separate per-behavior classifiers; absent demographic classes have no head."""

    def __init__(self, num_behaviors, user_size, item_size, classes, hidden_size):
        super().__init__()
        self.classes = dict(classes)
        self.heads = nn.ModuleList([
            _AttributeClassifierGroup(user_size, item_size, classes, hidden_size)
            for _ in range(num_behaviors)
        ])


class DualSideAdversarialDisentanglement(nn.Module):
    """Section 3.1: parallel encoding, sensitive factors, predictors, and adversaries."""

    def __init__(self, dataset, embedding_size, gcn_layers, ranks, classes, hidden_size):
        super().__init__()
        self.encoder = BehaviorSpecificParallelEncoding(dataset, embedding_size, gcn_layers)
        self.factors = SensitiveFactorsEncoding(embedding_size, len(dataset.behaviors), **ranks)
        user_rank = ranks["user_shared_rank"] + ranks["user_specific_rank"]
        item_rank = ranks["item_shared_rank"] + ranks["item_specific_rank"]
        self.sensitive_predictors = _AttributeHeads(len(dataset.behaviors), user_rank,
                                                   item_rank, classes, hidden_size)
        self.discriminators = _AttributeHeads(len(dataset.behaviors), embedding_size,
                                             embedding_size, classes, hidden_size)
        self.item_offset = dataset.item_offset

    def forward(self):
        return self.factors(self.encoder(), self.item_offset)


class FairnessAwareBehaviorChainLearning(nn.Module):
    """Section 3.2: margin predictors, shared scalar weights, synchronous propagation."""

    def __init__(self, embedding_size, user_rank, item_rank, hidden_size=64,
                 chain_layers=3, utility_weight=1.0, sensitivity_weight=0.1,
                 empty_chain_policy="target"):
        super().__init__()
        if int(chain_layers) != chain_layers or chain_layers < 0:
            raise ValueError("chain_layers must be a nonnegative integer.")
        if empty_chain_policy not in ("target", "error"):
            raise ValueError("empty_chain_policy must be 'target' or 'error'.")
        self.chain_layers = int(chain_layers)
        self.utility_weight = _nonnegative(utility_weight, "utility_weight")
        self.sensitivity_weight = _nonnegative(sensitivity_weight, "sensitivity_weight")
        self.empty_chain_policy = empty_chain_policy
        self.utility_predictor = _mlp(4 * embedding_size + 1, hidden_size, 1)
        self.sensitivity_predictor = _mlp(4 * embedding_size + 2 * user_rank + 2 * item_rank + 1,
                                         hidden_size, 1)
        # Appendix C.2.6: initial margins are zero before teacher distillation.
        for predictor in (self.utility_predictor, self.sensitivity_predictor):
            nn.init.zeros_(predictor[-1].weight)
            nn.init.zeros_(predictor[-1].bias)

    @staticmethod
    def endpoint_features(user_fair, item_fair, user_sensitive, item_sensitive):
        count = user_fair.shape[1]

        def endpoints(value):
            return (value[:, :, None, :].expand(-1, -1, count, -1),
                    value[:, None, :, :].expand(-1, count, -1, -1))

        uf_source, uf_receiver = endpoints(user_fair)
        if_source, if_receiver = endpoints(item_fair)
        us_source, us_receiver = endpoints(user_sensitive)
        is_source, is_receiver = endpoints(item_sensitive)
        fair = torch.cat((uf_source, if_source, uf_receiver, if_receiver), dim=-1)
        sensitive = torch.cat((us_source, is_source, us_receiver, is_receiver), dim=-1)
        return fair, sensitive

    def predict_margins(self, user_fair, item_fair, user_sensitive, item_sensitive, temporal_weights):
        fair, sensitive = self.endpoint_features(user_fair, item_fair, user_sensitive, item_sensitive)
        temporal = temporal_weights.to(fair.dtype).unsqueeze(-1)
        utility = self.utility_predictor(torch.cat((fair, temporal), dim=-1)).squeeze(-1)
        # Stop gradients only through sensitive inputs, not fair endpoint features.
        sensitivity = self.sensitivity_predictor(
            torch.cat((fair, sensitive.detach(), temporal), dim=-1)
        ).squeeze(-1)
        return utility, sensitivity

    def transition_weights(self, adjacency, temporal_weights, utility, sensitivity):
        logits = temporal_weights.to(utility.dtype) + self.utility_weight * utility
        logits = logits - self.sensitivity_weight * F.relu(sensitivity)
        return adjacency.to(logits.dtype) * torch.sigmoid(logits)

    def propagate(self, user_fair, item_fair, weights, chains, fallback_user, fallback_item):
        mask = chains["node_mask"].unsqueeze(-1)
        initial_user = user_fair * mask
        initial_item = item_fair * mask
        user_states, item_states = initial_user, initial_item
        for _ in range(self.chain_layers):
            # Eq. (9): anchor each residual at layer zero, keeping its gradients.
            user_states = initial_user + torch.bmm(weights.transpose(1, 2), user_states)
            item_states = initial_item + torch.bmm(weights.transpose(1, 2), item_states)
        terminal = chains["terminal_mask"].unsqueeze(-1)
        divisor = terminal.sum(dim=1).clamp_min(1)
        user_readout = (user_states * terminal).sum(dim=1) / divisor
        item_readout = (item_states * terminal).sum(dim=1) / divisor
        has_history = chains["node_mask"].any(dim=1)
        if self.empty_chain_policy == "error" and not bool(has_history.all()):
            raise ValueError("Empty pair history: choose the documented 'target' fallback or supply history.")
        # Supplemental reproduction choice: use fair buy embeddings for empty pairs.
        # This does not add a fictitious observed buy node to either branch.
        user_readout = torch.where(has_history[:, None], user_readout, fallback_user)
        item_readout = torch.where(has_history[:, None], item_readout, fallback_item)
        return {"scores": (user_readout * item_readout).sum(dim=-1),
                "user_states": user_states, "item_states": item_states,
                "user_readout": user_readout, "item_readout": item_readout,
                "weights": weights, "has_history": has_history}

    def forward(self, user_fair, item_fair, user_sensitive, item_sensitive, chains,
                fallback_user, fallback_item, fixed_utility=None, weights_override=None):
        utility, sensitivity = self.predict_margins(user_fair, item_fair, user_sensitive,
                                                   item_sensitive, chains["time_weights"])
        if fixed_utility is not None:
            utility = fixed_utility
        weights = self.transition_weights(chains["adjacency"], chains["time_weights"], utility, sensitivity)
        if weights_override is not None:
            weights = weights_override
        result = self.propagate(user_fair, item_fair, weights, chains, fallback_user, fallback_item)
        result.update(utility_margin=utility, sensitivity_margin=sensitivity)
        return result


class CounterfactualFairnessConstraint(nn.Module):
    """Section 3.3: behavior-wise attribute prototypes and candidate-distribution JS."""

    def __init__(self, num_behaviors, user_rank, item_rank, classes, temperature=1.0):
        super().__init__()
        self.classes = dict(classes)
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("cf_temperature must be finite and positive.")
        self.register_buffer("initialized", torch.tensor(False))
        for attribute in ATTRIBUTES:
            width = item_rank if attribute == "pop" else user_rank
            self.register_buffer(f"{attribute}_prototypes", torch.zeros(num_behaviors, classes[attribute], width))
            self.register_buffer(f"{attribute}_counts", torch.zeros(num_behaviors, classes[attribute], dtype=torch.long))

    @torch.no_grad()
    def update_prototypes(self, encoded, labels, training_masks):
        for attribute in ATTRIBUTES:
            prototypes = getattr(self, f"{attribute}_prototypes")
            counts = getattr(self, f"{attribute}_counts")
            prototypes.zero_()
            counts.zero_()
            side = "item" if attribute == "pop" else "user"
            embeddings = encoded[f"{side}_sensitive"].detach()
            for behavior in range(embeddings.shape[1]):
                targets = labels[attribute][:, behavior] if attribute == "pop" else labels[attribute]
                valid = training_masks[side] & (targets >= 0) & (targets < self.classes[attribute])
                if not bool(valid.any()):
                    continue
                counts[behavior] = torch.bincount(targets[valid], minlength=self.classes[attribute])
                prototypes[behavior].index_add_(0, targets[valid], embeddings[valid, behavior])
                prototypes[behavior].div_(counts[behavior].clamp_min(1).unsqueeze(-1))
        self.initialized.fill_(True)

    def intervene(self, users, items, node_mask, user_sensitive, item_sensitive,
                  attribute, labels, alternatives=None):
        if attribute not in ATTRIBUTES:
            raise ValueError(f"Unknown intervention attribute: {attribute}")
        if not bool(self.initialized):
            raise RuntimeError("Call refresh_prototypes before node counterfactuals.")
        prototypes = getattr(self, f"{attribute}_prototypes")
        available = getattr(self, f"{attribute}_counts") > 0
        shift_user, shift_item = torch.zeros_like(user_sensitive), torch.zeros_like(item_sensitive)
        changed = torch.zeros_like(node_mask)
        selected_labels = torch.full_like(node_mask, -1, dtype=torch.long)
        alternatives = {} if alternatives is None else alternatives
        if attribute in ("age", "gender"):
            for user in users.unique().tolist():
                rows = users == user
                behaviors = node_mask[rows].any(dim=0)
                factual = int(labels[attribute][user])
                if not bool(behaviors.any()) or not 0 <= factual < self.classes[attribute]:
                    continue
                if not bool(available[behaviors, factual].all()):
                    continue
                choices = available[behaviors].all(dim=0)
                choices[factual] = False
                candidates = choices.nonzero(as_tuple=False).flatten()
                if not len(candidates):
                    continue
                alternate = alternatives.get(user)
                if alternate is None:
                    alternate = int(candidates[torch.randint(len(candidates), ()).item()])
                if not 0 <= int(alternate) < len(choices) or not bool(choices[int(alternate)]):
                    raise ValueError("User alternative must be available in every affected behavior.")
                alternate = int(alternate)
                delta = prototypes[:, alternate] - prototypes[:, factual]
                shift_user[rows] = delta.unsqueeze(0) * node_mask[rows, :, None]
                changed[rows] = node_mask[rows]
                selected_labels[rows] = torch.where(node_mask[rows], alternate, -1)
        else:
            for behavior in range(node_mask.shape[1]):
                for item in items[node_mask[:, behavior]].unique().tolist():
                    rows = (items == item) & node_mask[:, behavior]
                    factual = int(labels["pop"][item, behavior])
                    if not 0 <= factual < self.classes[attribute] or not bool(available[behavior, factual]):
                        continue
                    choices = available[behavior].clone()
                    choices[factual] = False
                    candidates = choices.nonzero(as_tuple=False).flatten()
                    if not len(candidates):
                        continue
                    alternate = alternatives.get((item, behavior))
                    if alternate is None:
                        alternate = int(candidates[torch.randint(len(candidates), ()).item()])
                    if not 0 <= int(alternate) < len(choices) or not bool(choices[int(alternate)]):
                        raise ValueError("Item-behavior alternative requires an available prototype.")
                    alternate = int(alternate)
                    shift_item[rows, behavior] = prototypes[behavior, alternate] - prototypes[behavior, factual]
                    changed[rows, behavior] = True
                    selected_labels[rows, behavior] = alternate
        metadata = {"attribute": attribute, "intervention_mask": changed,
                    "alternative_labels": selected_labels, "valid_pairs": changed.any(dim=1)}
        return user_sensitive + shift_user, item_sensitive + shift_item, metadata

    def distribution_loss(self, factual_scores, counterfactual_scores, candidate_mask,
                          valid_groups, group_users=None):
        if factual_scores.ndim != 2 or factual_scores.shape != counterfactual_scores.shape:
            raise ValueError("Distribution scores must have matching [queries,candidates] shapes.")
        valid = valid_groups & (candidate_mask.sum(dim=1) >= 2)
        if not bool(valid.any()):
            return _zero(factual_scores) + _zero(counterfactual_scores)
        mask = candidate_mask[valid]
        floor = torch.finfo(torch.float32).min
        factual = (factual_scores[valid].float() / self.temperature).masked_fill(~mask, floor)
        counterfactual = (counterfactual_scores[valid].float() / self.temperature).masked_fill(~mask, floor)
        log_p, log_q = F.log_softmax(factual, dim=1), F.log_softmax(counterfactual, dim=1)
        log_mean = torch.logaddexp(log_p, log_q) - math.log(2.0)
        js = 0.5 * ((log_p.exp() * (log_p - log_mean)).masked_fill(~mask, 0).sum(dim=1) +
                    (log_q.exp() * (log_q - log_mean)).masked_fill(~mask, 0).sum(dim=1))
        js = js.clamp_min(0)
        if group_users is None:
            return js.mean()
        # Multiple training queries for one user do not multiply that user's weight.
        _, inverse = group_users[valid].unique(return_inverse=True)
        sums = js.new_zeros(int(inverse.max()) + 1).index_add(0, inverse, js)
        counts = torch.bincount(inverse, minlength=len(sums)).to(js.dtype)
        return (sums / counts).mean()


class PBCL(nn.Module):
    """PBCL model and loss primitives, ready for a separate alternating trainer."""

    def __init__(self, args, dataset):
        super().__init__()
        self.dataset = dataset
        self.behaviors = tuple(dataset.behaviors)
        self.num_behaviors = len(self.behaviors)
        self.n_users, self.n_items = dataset.user_count, dataset.item_count
        self.target_behavior_index = dataset.target_behavior_index
        self.embedding_size = _integer(_option(args, "embedding_size", 64), "embedding_size", 3)
        default_rank = min(4, max(1, self.embedding_size // 4))
        ranks = {f"{side}_{kind}_rank": _integer(_option(args, f"{side}_{kind}_rank", default_rank),
                                                f"{side}_{kind}_rank")
                 for side in ("user", "item") for kind in ("shared", "specific")}
        self.user_rank = ranks["user_shared_rank"] + ranks["user_specific_rank"]
        self.item_rank = ranks["item_shared_rank"] + ranks["item_specific_rank"]
        self.classes = {"age": int(dataset.num_age_classes), "gender": int(dataset.num_gender_classes),
                        "pop": int(dataset.num_popularity_groups)}
        head_size = _integer(_option(args, "head_hidden_size", 64), "head_hidden_size", 1)
        margin_size = _integer(_option(args, "margin_hidden_size", 64), "margin_hidden_size", 1)
        self.dual_side_adversarial_disentanglement = DualSideAdversarialDisentanglement(
            dataset, self.embedding_size, _option(args, "gcn_layers", 2), ranks, self.classes, head_size
        )
        self.fairness_aware_behavior_chain_learning = FairnessAwareBehaviorChainLearning(
            self.embedding_size, self.user_rank, self.item_rank, margin_size,
            _option(args, "chain_layers", 3), _option(args, "utility_weight", 1.0),
            _option(args, "sensitivity_weight", 0.1), _option(args, "empty_chain_policy", "target")
        )
        self.counterfactual_fairness_constraint = CounterfactualFairnessConstraint(
            self.num_behaviors, self.user_rank, self.item_rank, self.classes,
            _option(args, "cf_temperature", 1.0)
        )
        self.attribute_auditors = _AttributeHeads(self.num_behaviors, self.embedding_size,
                                                 self.embedding_size, self.classes, head_size)
        self.register_buffer("auditor_ready", torch.zeros(self.num_behaviors, len(ATTRIBUTES), dtype=torch.bool))
        self.set_audit_training(False)
        for name, default in (("lambda_adv", 0.01), ("lambda_orth", 0.001),
                              ("lambda_edge", 1.0), ("lambda_cf", 0.1), ("lambda_s", 1.0)):
            setattr(self, name, _nonnegative(_option(args, name, default), name))
        attribute_weights = _option(args, "sensitivity_attribute_weights", {})
        weights = [_nonnegative(attribute_weights.get(attribute, 1.0), attribute) for attribute in ATTRIBUTES]
        if not any(weights):
            raise ValueError("At least one sensitivity attribute weight must be positive.")
        self.register_buffer("sensitivity_attribute_weights", torch.tensor(weights))
        for name, value in (("user_age", dataset.user_age), ("user_gender", dataset.user_gender),
                            ("item_popularity_label", dataset.item_popularity_label),
                            ("training_user_mask", dataset.user_behavior_mask.any(dim=1)),
                            ("training_item_mask", dataset.item_behavior_mask.any(dim=1))):
            self.register_buffer(name, value.clone(), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def disentanglement(self):
        return self.dual_side_adversarial_disentanglement

    @property
    def chain_learning(self):
        return self.fairness_aware_behavior_chain_learning

    @property
    def counterfactual(self):
        return self.counterfactual_fairness_constraint

    def main_parameters(self):
        excluded = {id(parameter) for parameter in self.discriminator_parameters()}
        excluded.update(id(parameter) for parameter in self.audit_parameters())
        return (parameter for parameter in self.parameters() if id(parameter) not in excluded)

    def discriminator_parameters(self):
        return self.disentanglement.discriminators.parameters()

    def audit_parameters(self):
        return self.attribute_auditors.parameters()

    def set_audit_training(self, enabled):
        self.attribute_auditors.requires_grad_(bool(enabled))
        self.attribute_auditors.train(bool(enabled))

    @torch.no_grad()
    def mark_auditors_ready(self, trained_counts):
        """Call only after auditor optimizer steps, then freeze before teachers."""
        counts = torch.as_tensor(trained_counts, device=self.device)
        if counts.shape != self.auditor_ready.shape:
            raise ValueError("Auditor counts require [num_behaviors,3] in age/gender/pop order.")
        self.auditor_ready |= counts > 0

    def encode(self):
        """Encode once per optimizer update; never reuse an old training autograd graph."""
        return self.disentanglement()

    def _labels(self):
        return {"age": self.user_age, "gender": self.user_gender, "pop": self.item_popularity_label}

    def _ids(self, values, side):
        value = torch.as_tensor(values, device=self.device)
        if value.is_floating_point() and (not bool(torch.isfinite(value).all()) or not bool((value == value.floor()).all())):
            raise ValueError("Entity IDs must be finite integers.")
        value = value.long().reshape(-1)
        upper = self.n_users if side == "user" else self.n_items
        if bool(((value < 1) | (value > upper)).any()):
            raise ValueError(f"{side} IDs must lie in 1..{upper}.")
        return value

    def _chains_to_device(self, chains):
        result = {name: torch.as_tensor(value, device=self.device) for name, value in chains.items()}
        if result["node_mask"].ndim == 1:
            result = {name: value.unsqueeze(0) for name, value in result.items()}
        for name in ("node_mask", "terminal_mask", "adjacency", "eligible_edge_mask", "has_history"):
            result[name] = result[name].bool()
        return result

    def _pair_inputs(self, users, items, encoded):
        return {"user_fair": encoded["user_fair"][users], "item_fair": encoded["item_fair"][items],
                "user_sensitive": encoded["user_sensitive"][users],
                "item_sensitive": encoded["item_sensitive"][items]}

    def score_pairs(self, users, items, chains=None, cutoffs=None, encoded=None, return_details=False):
        users, items = self._ids(users, "user"), self._ids(items, "item")
        if users.shape != items.shape:
            raise ValueError("Pair users and items must have equal lengths.")
        if encoded is None:
            encoded = self.encode()
        if chains is None:
            chains = self.dataset.get_batch_chains(users, items, cutoffs)
        chains = self._chains_to_device(chains)
        if chains["node_mask"].shape != (len(users), self.num_behaviors):
            raise ValueError("Chain batch and behavior dimensions do not match the pairs.")
        inputs = self._pair_inputs(users, items, encoded)
        result = self.chain_learning(**inputs, chains=chains,
                                     fallback_user=inputs["user_fair"][:, self.target_behavior_index],
                                     fallback_item=inputs["item_fair"][:, self.target_behavior_index])
        if not return_details:
            return result["scores"]
        result.update(inputs)
        result.update(users=users, items=items, chains=chains)
        return result

    def attribute_loss(self, encoded, block="predictor", user_ids=None, item_ids=None,
                       detach_embeddings=False):
        if block not in ("predictor", "discriminator"):
            raise ValueError("block must be 'predictor' or 'discriminator'.")
        network = self.disentanglement.sensitive_predictors if block == "predictor" else self.disentanglement.discriminators
        suffix = "sensitive" if block == "predictor" else "fair"
        user_ids = (torch.arange(1, self.n_users + 1, device=self.device) if user_ids is None
                    else self._ids(user_ids, "user").unique())
        item_ids = (torch.arange(1, self.n_items + 1, device=self.device) if item_ids is None
                    else self._ids(item_ids, "item").unique())
        total = _zero(encoded["user_fair"]) + _zero(encoded["item_fair"])
        # Even an entirely masked discriminator batch has a backward-safe zero.
        total = total + sum((_zero(parameter) for parameter in network.parameters()),
                            total.new_zeros(()))
        for behavior, heads in enumerate(network.heads):
            for attribute, head in heads.items():
                side, ids = ("item", item_ids) if attribute == "pop" else ("user", user_ids)
                labels = self._labels()[attribute]
                targets = labels[ids, behavior] if attribute == "pop" else labels[ids]
                training = self.training_item_mask[ids] if side == "item" else self.training_user_mask[ids]
                valid = training & (targets >= 0) & (targets < self.classes[attribute])
                if not bool(valid.any()):
                    continue
                embeddings = encoded[f"{side}_{suffix}"][ids[valid], behavior]
                if detach_embeddings:
                    embeddings = embeddings.detach()
                total = total + F.cross_entropy(head(embeddings), targets[valid])
        return total

    def discriminator_loss(self, batch, encoded=None):
        if encoded is None:
            with torch.no_grad():
                encoded = self.encode()
        # The disconnected zero must also be detached in the discriminator step.
        encoded = {key: value.detach() for key, value in encoded.items()}
        return self.attribute_loss(encoded, "discriminator", batch["user"],
                                   torch.cat((batch["pos_item"], batch["neg_item"])), True)

    def audit_loss(self, users, items, chains, encoded=None):
        """Train only auditor heads on detached factual receiving-behavior states."""
        if not any(parameter.requires_grad for parameter in self.audit_parameters()):
            raise RuntimeError("Call set_audit_training(True) before audit_loss.")
        with torch.no_grad():
            factual = self.score_pairs(users, items, chains, encoded=encoded, return_details=True)
        counts = torch.zeros_like(self.auditor_ready, dtype=torch.long)
        total = sum((_zero(parameter) for parameter in self.audit_parameters()),
                    factual["scores"].new_zeros(()))
        for behavior, heads in enumerate(self.attribute_auditors.heads):
            for attribute, head in heads.items():
                labels = self._labels()[attribute]
                targets = (labels[factual["items"], behavior] if attribute == "pop"
                           else labels[factual["users"]])
                receiving = factual["chains"]["eligible_edge_mask"].any(dim=1)
                valid = receiving[:, behavior] & (targets >= 0)
                valid &= targets < self.classes[attribute]
                counts[behavior, ATTRIBUTES.index(attribute)] = valid.sum()
                if bool(valid.any()):
                    states = factual["item_states"] if attribute == "pop" else factual["user_states"]
                    total = total + F.cross_entropy(head(states[valid, behavior].detach()), targets[valid])
        return {"loss": total, "counts": counts}

    @torch.no_grad()
    def refresh_prototypes(self, encoded=None):
        self.counterfactual.update_prototypes(self.encode() if encoded is None else encoded,
                                             self._labels(), {"user": self.training_user_mask,
                                                              "item": self.training_item_mask})

    @staticmethod
    def _interleave(first, second):
        return torch.stack((first, second), dim=1).flatten(0, 1)

    def _training_pairs(self, batch, encoded):
        users = self._ids(batch["user"], "user")
        positives, negatives = self._ids(batch["pos_item"], "item"), self._ids(batch["neg_item"], "item")
        if users.shape != positives.shape or users.shape != negatives.shape or bool((positives == negatives).any()):
            raise ValueError("BPR users, positives, and distinct negatives must have equal lengths.")
        positive_chains = self._chains_to_device(batch["pos_chain"])
        negative_chains = self._chains_to_device(batch["neg_chain"])
        chains = {key: self._interleave(value, negative_chains[key]) for key, value in positive_chains.items()}
        items = self._interleave(positives, negatives)
        factual = self.score_pairs(users.repeat_interleave(2), items, chains=chains,
                                   encoded=encoded, return_details=True)
        return users, positives, negatives, factual

    def counterfactual_loss(self, users, candidate_items, prediction_times=None, candidate_mask=None,
                            attribute=None, encoded=None, factual=None, alternatives=None):
        users = self._ids(users, "user")
        candidates = torch.as_tensor(candidate_items, device=self.device)
        if candidates.ndim != 2 or candidates.shape[0] != len(users):
            raise ValueError("candidate_items requires [queries,candidates].")
        mask = candidates != 0 if candidate_mask is None else torch.as_tensor(candidate_mask, device=self.device).bool()
        if mask.shape != candidates.shape:
            raise ValueError("candidate_mask must match candidate_items.")
        items = self._ids(candidates[mask], "item")
        flat_users = users[:, None].expand_as(candidates)[mask]
        if encoded is None:
            encoded = self.encode()
        if not bool(self.counterfactual.initialized):
            self.refresh_prototypes(encoded)
        if factual is None:
            if prediction_times is None:
                raise ValueError("Training counterfactual candidates require explicit prediction_times.")
            times = torch.as_tensor(prediction_times, dtype=torch.float64, device=self.device).reshape(-1)
            if len(times) != len(users):
                raise ValueError("prediction_times requires one cutoff per candidate query.")
            flat_times = times[:, None].expand_as(candidates)[mask]
            factual = self.score_pairs(flat_users, items, cutoffs=flat_times, encoded=encoded, return_details=True)
        elif not torch.equal(factual["users"], flat_users) or not torch.equal(factual["items"], items):
            raise ValueError("Reused factual outputs must describe exactly the same ordered candidate pairs.")
        if attribute is None:
            available = [name for name in ATTRIBUTES if self.classes[name] >= 2]
            attribute = available[torch.randint(len(available), ()).item()] if available else None
        if attribute is None:
            metadata = {"attribute": None, "valid_pairs": torch.zeros(len(items), device=self.device, dtype=torch.bool)}
            counterfactual = factual
        else:
            user_sensitive, item_sensitive, metadata = self.counterfactual.intervene(
                flat_users, items, factual["chains"]["node_mask"], factual["user_sensitive"],
                factual["item_sensitive"], attribute, self._labels(), alternatives
            )
            counterfactual = self.chain_learning(
                factual["user_fair"], factual["item_fair"], user_sensitive, item_sensitive,
                factual["chains"], factual["user_fair"][:, self.target_behavior_index],
                factual["item_fair"][:, self.target_behavior_index], fixed_utility=factual["utility_margin"]
            )
        factual_scores = factual["scores"].new_zeros(mask.shape).masked_scatter(mask, factual["scores"])
        cf_scores = counterfactual["scores"].new_zeros(mask.shape).masked_scatter(mask, counterfactual["scores"])
        changed = torch.zeros_like(mask).masked_scatter(mask, metadata["valid_pairs"])
        valid_groups = changed.any(dim=1) & (mask.sum(dim=1) >= 2)
        loss = self.counterfactual.distribution_loss(factual_scores, cf_scores, mask, valid_groups, users)
        return {"loss": loss, "attribute": attribute, "valid_groups": valid_groups,
                "factual_scores": factual_scores, "counterfactual_scores": cf_scores,
                "factual": factual, "counterfactual": counterfactual, "intervention": metadata,
                "candidate_mask": mask}

    @torch.no_grad()
    def build_edge_teachers(self, users, items, chains, negative_scores=None,
                            buy_supervision_mask=None, selected_edges=None, encoded=None):
        """Measure edge deletion at a fixed snapshot; unlabeled histories have no ΔU."""
        if any(parameter.requires_grad for parameter in self.audit_parameters()):
            raise RuntimeError("Freeze attribute auditors before generating edge teachers.")
        factual = self.score_pairs(users, items, chains, encoded=encoded, return_details=True)
        chains = factual["chains"]
        count = len(factual["users"])
        edges = (self.dataset.sample_edge_counterfactual(chains) if selected_edges is None
                 else torch.as_tensor(selected_edges))
        edges = edges.to(device=self.device, dtype=torch.long)
        if edges.shape != (count, 2):
            raise ValueError("selected_edges requires [pairs,2].")
        eligible = (edges >= 0).all(dim=1)
        rows = torch.arange(count, device=self.device)
        safe = edges.clamp(min=0, max=self.num_behaviors - 1)
        if bool((edges[eligible] >= self.num_behaviors).any()) or bool(
                (~chains["eligible_edge_mask"][rows[eligible], safe[eligible, 0], safe[eligible, 1]]).any()):
            raise ValueError("Selected teacher edge must be an existing auxiliary-source connection.")
        weights = factual["weights"].clone()
        weights[rows[eligible], safe[eligible, 0], safe[eligible, 1]] = 0
        # Keep every other edge unchanged; do not renormalize after deletion.
        deleted = self.chain_learning.propagate(
            factual["user_fair"], factual["item_fair"], weights, chains,
            factual["user_fair"][:, self.target_behavior_index],
            factual["item_fair"][:, self.target_behavior_index]
        )
        utility_target = factual["scores"].new_zeros(count)
        utility_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
        if buy_supervision_mask is not None:
            supervision = torch.as_tensor(buy_supervision_mask, device=self.device).bool().reshape(-1)
            if len(supervision) != count:
                raise ValueError("buy_supervision_mask requires one flag per pair.")
            utility_mask = eligible & supervision
        if bool(utility_mask.any()):
            if negative_scores is None:
                raise ValueError("Observed-buy utility teachers require the same factual negative scores.")
            negatives = torch.as_tensor(negative_scores, device=self.device).reshape(-1)
            if len(negatives) != count:
                raise ValueError("negative_scores requires one unchanged score per pair.")
            utility_target = F.softplus(negatives - deleted["scores"]) - F.softplus(negatives - factual["scores"])
        attribute_targets = utility_target.new_zeros(count, len(ATTRIBUTES))
        attribute_mask = torch.zeros_like(attribute_targets, dtype=torch.bool)
        for behavior, heads in enumerate(self.attribute_auditors.heads):
            receiving = eligible & (safe[:, 1] == behavior)
            for attribute, head in heads.items():
                index = ATTRIBUTES.index(attribute)
                if not bool(self.auditor_ready[behavior, index]):
                    continue
                targets = (self.item_popularity_label[factual["items"], behavior] if attribute == "pop"
                           else self._labels()[attribute][factual["users"]])
                valid = receiving & (targets >= 0) & (targets < self.classes[attribute])
                # Different pairs can select different receiving behaviors.
                # Accumulate those disjoint rows instead of overwriting earlier heads.
                attribute_mask[:, index] |= valid
                if bool(valid.any()):
                    state = "item_states" if attribute == "pop" else "user_states"
                    factual_loss = F.cross_entropy(head(factual[state][valid, behavior]), targets[valid], reduction="none")
                    deleted_loss = F.cross_entropy(head(deleted[state][valid, behavior]), targets[valid], reduction="none")
                    attribute_targets[valid, index] = deleted_loss - factual_loss
        required = self.sensitivity_attribute_weights > 0
        sensitivity_mask = eligible & attribute_mask[:, required].all(dim=1)
        sensitivity_target = (attribute_targets.clamp_min(0) * self.sensitivity_attribute_weights).sum(dim=1)
        return {"users": factual["users"], "items": factual["items"], "chains": chains,
                "selected_edges": edges, "utility_target": utility_target.detach(),
                "sensitivity_target": sensitivity_target.detach(), "utility_mask": utility_mask,
                "sensitivity_mask": sensitivity_mask, "attribute_targets": attribute_targets.detach(),
                "attribute_mask": attribute_mask, "factual_scores": factual["scores"],
                "deleted_scores": deleted["scores"], "factual_weights": factual["weights"],
                "deleted_weights": weights}

    @torch.no_grad()
    def edge_teachers_from_batch(self, batch, encoded=None, selected_edges=None):
        """The supplied batch must contain genuine observed training-buy events."""
        if encoded is None:
            encoded = self.encode()
        negatives = self.score_pairs(batch["user"], batch["neg_item"], batch["neg_chain"], encoded=encoded)
        return self.build_edge_teachers(
            batch["user"], batch["pos_item"], batch["pos_chain"], negatives,
            torch.ones_like(torch.as_tensor(batch["user"]), dtype=torch.bool), selected_edges, encoded
        )

    def edge_distillation_loss(self, teachers, encoded):
        if teachers is None:
            return _zero(encoded["user_fair"])
        users, items = self._ids(teachers["users"], "user"), self._ids(teachers["items"], "item")
        inputs = self._pair_inputs(users, items, encoded)
        temporal = self._chains_to_device(teachers["chains"])["time_weights"]
        utility, sensitivity = self.chain_learning.predict_margins(**inputs, temporal_weights=temporal)
        edges = torch.as_tensor(teachers["selected_edges"], device=self.device).long().clamp_min(0)
        rows = torch.arange(len(users), device=self.device)
        total = _zero(utility) + _zero(sensitivity)
        for name, predictions, coefficient in (("utility", utility, 1.0), ("sensitivity", sensitivity, self.lambda_s)):
            valid = torch.as_tensor(teachers[f"{name}_mask"], device=self.device).bool()
            if bool(valid.any()):
                target = torch.as_tensor(teachers[f"{name}_target"], device=self.device).detach()
                measured = predictions[rows[valid], edges[valid, 0], edges[valid, 1]]
                total = total + coefficient * F.mse_loss(measured, target[valid])
        return total

    def training_losses(self, batch, edge_teachers=None, cf_candidates=None, cf_attribute=None,
                        encoded=None, compute_cf=True, return_details=False):
        if encoded is None:
            encoded = self.encode()
        users, positives, negatives, factual = self._training_pairs(batch, encoded)
        scores = factual["scores"].reshape(-1, 2)
        rec = F.softplus(scores[:, 1] - scores[:, 0]).mean() if len(users) else _zero(scores)
        items = torch.cat((positives, negatives))
        sen = self.attribute_loss(encoded, "predictor", users, items)
        # Freeze A parameters during this forward, while preserving dL_adv/dE_f.
        with _frozen(self.disentanglement.discriminators):
            adv = self.attribute_loss(encoded, "discriminator", users, items)
        edge = self.edge_distillation_loss(edge_teachers, encoded)
        cf = _zero(scores)
        cf_result = None
        if compute_cf and self.lambda_cf > 0 and len(users):
            if cf_candidates is None:
                cf_result = self.counterfactual_loss(
                    users, torch.stack((positives, negatives), dim=1), batch["prediction_time"],
                    attribute=cf_attribute, encoded=encoded, factual=factual
                )
            else:
                cf_result = self.counterfactual_loss(
                    cf_candidates["users"], cf_candidates["items"], cf_candidates["prediction_times"],
                    cf_candidates.get("mask"), cf_attribute, encoded
                )
            cf = cf_result["loss"]
        orth = encoded["orth_loss"]
        loss = rec + sen - self.lambda_adv * adv + self.lambda_orth * orth
        loss = loss + self.lambda_edge * edge + self.lambda_cf * cf
        result = {"loss": loss, "rec": rec, "sen": sen, "adv": adv, "orth": orth,
                  "edge": edge, "cf": cf, "pos_scores": scores[:, 0], "neg_scores": scores[:, 1]}
        if return_details:
            result.update(encoded=encoded, factual=factual, counterfactual=cf_result)
        return result

    def forward(self, batch, **kwargs):
        return self.training_losses(batch, **kwargs)

    def predict(self, users, items, cutoffs=None, chains=None, encoded=None):
        return self.score_pairs(users, items, chains, cutoffs, encoded)

    @torch.no_grad()
    def predict_candidates(self, users, candidate_items, split="test", cutoffs=None, encoded=None):
        users = self._ids(users, "user")
        candidates = torch.as_tensor(candidate_items, device=self.device)
        if candidates.ndim != 2 or candidates.shape[0] != len(users):
            raise ValueError("candidate_items requires [users,candidates].")
        mask = candidates != 0
        items = self._ids(candidates[mask], "item")
        flat_users = users[:, None].expand_as(candidates)[mask]
        if cutoffs is None:
            chains = self.dataset.get_evaluation_chains(flat_users, items, split)
        else:
            times = torch.as_tensor(cutoffs, device=self.device, dtype=torch.float64).reshape(-1)
            if len(times) != len(users):
                raise ValueError("cutoffs requires one value per query user.")
            chains = self.dataset.get_batch_chains(flat_users, items, times[:, None].expand_as(candidates)[mask])
        scores = self.score_pairs(flat_users, items, chains, encoded=encoded)
        return scores.new_full(mask.shape, -torch.inf).masked_scatter(mask, scores)

    @torch.no_grad()
    def full_predict(self, batch_users, split="test", pair_batch_size=4096, cutoffs=None, encoded=None):
        """Score pair-specific chains in bounded chunks; output includes padding column 0."""
        users = self._ids(batch_users, "user")
        if int(pair_batch_size) != pair_batch_size or pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be a positive integer.")
        if encoded is None:
            encoded = self.encode()
        scores = encoded["user_fair"].new_full((len(users), self.n_items + 1), -torch.inf)
        times = None
        if cutoffs is not None:
            times = torch.as_tensor(cutoffs, dtype=torch.float64, device=self.device).reshape(-1)
            if len(times) != len(users):
                raise ValueError("cutoffs requires one value per user.")
        for start in range(0, len(users) * self.n_items, int(pair_batch_size)):
            indices = torch.arange(start, min(start + int(pair_batch_size), len(users) * self.n_items), device=self.device)
            user_rows, items = torch.div(indices, self.n_items, rounding_mode="floor"), indices % self.n_items + 1
            pair_users = users[user_rows]
            chains = (self.dataset.get_evaluation_chains(pair_users, items, split) if times is None
                      else self.dataset.get_batch_chains(pair_users, items, times[user_rows]))
            scores[user_rows, items] = self.score_pairs(pair_users, items, chains, encoded=encoded)
        return scores
