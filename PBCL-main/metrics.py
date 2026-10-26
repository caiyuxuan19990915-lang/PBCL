#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _integers(value, name, ndim=1, minimum=0):
    array = _numpy(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s).")
    if (not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating)
            or not np.all(np.isfinite(array)) or np.any(array < minimum)):
        raise ValueError(f"{name} must contain finite integers >= {minimum}.")
    # Float64 rounds INT64_MAX up to 2**63; reject before an overflowing cast.
    if np.issubdtype(array.dtype, np.floating):
        invalid = np.any(array >= 2.0 ** 63) or np.any(array != np.floor(array))
    else:
        invalid = np.any(array > np.iinfo(np.int64).max)
    if invalid:
        raise ValueError(f"{name} must contain integers within the int64 range.")
    return array.astype(np.int64, copy=False)


def _positive_integer(value, name):
    values = _integers([value], name, minimum=1)
    return int(values[0])


def _hits(value):
    array = _numpy(value)
    if array.ndim != 2 or not np.all((array == 0) | (array == 1)):
        raise ValueError("pos_index must be a two-dimensional binary hit matrix.")
    return array.astype(bool, copy=False)


def _ranking_inputs(pos_index, pos_len):
    hits = _hits(pos_index)
    lengths = _integers(pos_len, "pos_len", minimum=1)
    if len(lengths) != len(hits):
        raise ValueError("pos_len and pos_index must have aligned user dimensions.")
    if np.any(hits.sum(axis=1) > lengths):
        raise ValueError("A ranking cannot contain more hits than ground-truth items.")
    return hits, lengths


def hit_(pos_index, pos_len):
    """Per-query HR at every prefix; any relevant item counts as a hit."""
    hits, _ = _ranking_inputs(pos_index, pos_len)
    return (np.cumsum(hits, axis=1) > 0).astype(float)


def recall_(pos_index, pos_len):
    """Per-query recall; divide by all relevant items for that query."""
    hits, lengths = _ranking_inputs(pos_index, pos_len)
    return np.cumsum(hits, axis=1) / lengths[:, None]


def precision_(pos_index, pos_len):
    """Per-query precision at every prefix rank."""
    hits, _ = _ranking_inputs(pos_index, pos_len)
    return np.cumsum(hits, axis=1) / np.arange(1, hits.shape[1] + 1)


def ndcg_(pos_index, pos_len):
    """Binary-relevance NDCG with IDCG using min(number of positives, K)."""
    hits, lengths = _ranking_inputs(pos_index, pos_len)
    width = hits.shape[1]
    if not width:
        return np.zeros(hits.shape, dtype=float)
    discount = 1.0 / np.log2(np.arange(width) + 2.0)
    dcg = np.cumsum(hits * discount, axis=1)
    ideal_sizes = np.minimum(lengths[:, None], np.arange(1, width + 1))
    idcg = np.cumsum(discount)[ideal_sizes - 1]
    return dcg / idcg


def mrr_(pos_index, pos_len):
    """Reciprocal rank of the first hit, or zero if no hit has occurred."""
    hits, _ = _ranking_inputs(pos_index, pos_len)
    if not hits.shape[1]:
        return np.zeros(hits.shape, dtype=float)
    reciprocal = hits / np.arange(1, hits.shape[1] + 1)
    return np.maximum.accumulate(reciprocal, axis=1)


def map_(pos_index, pos_len):
    """Per-query AP at each prefix, normalized by min(positive count, K)."""
    hits, lengths = _ranking_inputs(pos_index, pos_len)
    ranks = np.arange(1, hits.shape[1] + 1)
    precision = np.cumsum(hits, axis=1) / ranks
    return np.cumsum(precision * hits, axis=1) / np.minimum(lengths[:, None], ranks)


def group_ndcg_gap_(per_user_ndcg, labels):
    """Maximum minus minimum group-mean NDCG, excluding unknown labels.

    Fewer than two represented groups makes the disparity undefined. This is
    the explicit multi-group convention used by this PBCL implementation.
    """
    values = _numpy(per_user_ndcg).astype(float)
    labels = _integers(labels, "labels", minimum=-1)
    if values.ndim != 1 or len(values) != len(labels):
        raise ValueError("NDCG values and labels must be aligned one-dimensional arrays.")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1 + 1e-12):
        raise ValueError("Per-user NDCG values must be finite and between zero and one.")
    means = [values[labels == label].mean() for label in np.unique(labels[labels >= 0])]
    return float(max(means) - min(means)) if len(means) >= 2 else float("nan")


def gini_index_(topk_idx_list, item_count):
    """Exposure Gini over all N real catalog items, including zero exposures.

    Count each occupied recommendation slot once without position discount.
    Padding is ignored. No exposure yields NaN; one exposed item out of four
    catalog items yields 0.75, rather than the zero from dropping unseen items.
    """
    size = _positive_integer(item_count, "item_count")
    items = _integers(topk_idx_list, "topk_idx_list", ndim=2)
    if np.any(items > size):
        raise ValueError("Recommendation item IDs must be between 0 and item_count.")
    counts = np.bincount(items.reshape(-1), minlength=size + 1)[1:].astype(np.float64)
    total = counts.sum()
    if not total:
        return float("nan")
    ordered = np.sort(counts)
    gini = 2 * np.dot(np.arange(1, size + 1), ordered) / (size * total) - (size + 1) / size
    return float(np.clip(gini, 0.0, 1.0))


def balanced_accuracy_(trues, preds):
    """Macro recall over classes represented in independent probe ground truth.

    Inputs are hard class labels, not logits. Unknown ground-truth labels are
    excluded. Missing true classes are excluded from the macro average; no
    valid ground truth yields NaN. This does not train or select a probe.
    """
    truth = _integers(trues, "trues", minimum=-1)
    predictions = _integers(preds, "preds", minimum=-1)
    if truth.shape != predictions.shape:
        raise ValueError("Probe ground truth and predictions must be aligned.")
    valid = truth >= 0
    truth, predictions = truth[valid], predictions[valid]
    if np.any(predictions < 0):
        raise ValueError("A valid probe example requires a nonnegative predicted class.")
    if not len(truth):
        return float("nan")
    recalls = [(predictions[truth == label] == label).mean() for label in np.unique(truth)]
    return float(np.mean(recalls))


def stratified_hr_(pos_index, demographic_labels, popularity_labels, cutoff=10):
    """HR grouped by target-item buy popularity and a demographic attribute.

    Supply one demographic label and one TRUE target item's training-derived
    buy-popularity label per query, as in single-target leave-one-out testing.
    Never derive a stratum from the recommended item. Unknown labels are
    excluded; only nonempty cells are returned, with their sample counts.
    """
    hits = _hits(pos_index)
    demographics = _integers(demographic_labels, "demographic_labels", minimum=-1)
    popularity = _integers(popularity_labels, "popularity_labels", minimum=-1)
    cutoff = _positive_integer(cutoff, "cutoff")
    if len(demographics) != len(hits) or len(popularity) != len(hits):
        raise ValueError("Hits, demographic labels, and target popularity must align by query.")
    if np.any(hits.sum(axis=1) > 1):
        raise ValueError("Target-popularity stratification requires single-target queries.")
    per_user = hits[:, :cutoff].any(axis=1).astype(float)
    valid = (demographics >= 0) & (popularity >= 0)
    records = []
    for pop in np.unique(popularity[valid]):
        for group in np.unique(demographics[valid & (popularity == pop)]):
            mask = valid & (popularity == pop) & (demographics == group)
            records.append({"popularity_group": int(pop), "demographic_group": int(group),
                            "hr": float(per_user[mask].mean()), "count": int(mask.sum())})
    return records


# These functions share the legacy per-query/per-prefix ranking interface.
# Fairness helpers have distinct inputs and are deliberately not in this map.
metrics_dict = {"hit": hit_, "hr": hit_, "ndcg": ndcg_, "recall": recall_,
                "precision": precision_, "mrr": mrr_, "map": map_}


def calculate_pbcl_metrics(*, topk_list, gt_len, user_list, topk_idx_list,
                           dataset, topk, metrics):
    """Trainer callback for ranking quality, demographic gaps, and exposure.

    Preserve the existing PBCL result keys and requested-K convention. If K
    exceeds catalog size, use available ranks but retain K as the precision
    denominator. Probe BA and target-popularity strata require extra inputs
    and are exposed separately instead of being inferred from rankings.
    """
    hits, lengths = _ranking_inputs(topk_list, gt_len)
    items = _integers(topk_idx_list, "topk_idx_list", ndim=2)
    users = _integers(user_list, "user_list", minimum=1)
    size = _positive_integer(dataset.item_count, "item_count")
    if items.shape != hits.shape or len(users) != len(hits):
        raise ValueError("Evaluation arrays must have aligned user/rank dimensions.")
    if np.any(items > size) or np.any(lengths > size):
        raise ValueError("Item IDs and ground-truth lengths must fit the catalog.")
    if np.any(hits & (items == 0)):
        raise ValueError("Padding item 0 cannot count as a relevant recommendation.")
    if len(np.unique(users)) != len(users):
        raise ValueError("Evaluation requires one query per user per split.")
    cutoffs = sorted({_positive_integer(value, "topk") for value in topk})
    names = list(dict.fromkeys("hit" if str(value).lower() == "hr" else str(value).lower()
                              for value in metrics))
    if not cutoffs or not names or any(name not in metrics_dict for name in names):
        raise ValueError("Use nonempty topk and supported ranking metric names.")
    if items.shape[1] > size:
        raise ValueError("A recommendation ranking cannot be wider than the catalog.")
    labels = {}
    for attribute in ("gender", "age"):
        table = _integers(getattr(dataset, f"user_{attribute}"), f"user_{attribute}", minimum=-1)
        if len(users) and users.max() >= len(table):
            raise ValueError(f"User IDs are outside the {attribute} label table.")
        labels[attribute] = table[users]
    result = {}
    for cutoff in cutoffs:
        current = hits[:, :cutoff]
        rank_width = current.shape[1]
        discount = 1.0 / np.log2(np.arange(rank_width) + 2.0)
        dcg = (current * discount).sum(axis=1)
        ideal_lengths = np.minimum(lengths, cutoff)
        ideal = np.array([np.sum(1.0 / np.log2(np.arange(length) + 2.0)) for length in ideal_lengths])
        ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
        found = current.sum(axis=1)
        per_user = {"hit": (found > 0).astype(float), "ndcg": ndcg,
                    "recall": found / lengths, "precision": found / cutoff,
                    "mrr": np.max(current / (np.arange(rank_width) + 1), axis=1) if rank_width
                    else np.zeros(len(users))}
        if "map" in names:
            precision = np.cumsum(current, axis=1) / np.arange(1, rank_width + 1)
            per_user["map"] = np.divide((precision * current).sum(axis=1), ideal_lengths)
        for name in names:
            result[f"{name}@{cutoff}"] = float(per_user[name].mean()) if len(users) else float("nan")
        for attribute in ("gender", "age"):
            result[f"{attribute}_ndcg_gap@{cutoff}"] = group_ndcg_gap_(ndcg, labels[attribute])
        result[f"gini@{cutoff}"] = gini_index_(items[:, :cutoff], size)
    return result
