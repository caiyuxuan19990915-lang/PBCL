
import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _as_numpy(value, dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _timestamp(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric timestamp at {context}: {value!r}") from error
    if not math.isfinite(result) or abs(result) > 2 ** 53:
        raise ValueError(f"Timestamp at {context} must be finite and exactly representable; "
                         "convert nanoseconds to seconds or milliseconds first.")
    return result


def _label_key(value):
    if value is None:
        return ""
    result = str(value).strip().lower()
    try:
        number = float(result)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return result


class PBCLBehaviorHistory:
    """Compact sorted timestamp lists indexed by a user-item pair key."""

    def __init__(self, events, item_count):
        keys = events["user"] * (item_count + 1) + events["item"]
        order = np.lexsort((events["timestamp"], keys))
        self.timestamps = events["timestamp"][order]
        self.pair_keys, starts = np.unique(keys[order], return_index=True)
        self.offsets = np.append(starts, len(order))

    def latest_before(self, pair_keys, cutoffs):
        result = np.full(len(pair_keys), np.nan, dtype=np.float64)
        if not len(self.pair_keys):
            return result
        positions = np.searchsorted(self.pair_keys, pair_keys)
        valid = positions < len(self.pair_keys)
        valid[valid] &= self.pair_keys[positions[valid]] == pair_keys[valid]
        rows = np.flatnonzero(valid)
        for row in rows:
            position = positions[row]
            start, end = self.offsets[position:position + 2]
            # Strictly earlier events exclude both the target and same-time events.
            offset = np.searchsorted(self.timestamps[start:end], cutoffs[row], side="left")
            if offset:
                result[row] = self.timestamps[start + offset - 1]
        return result

    def latest(self, pair_keys):
        result = np.full(len(pair_keys), np.nan, dtype=np.float64)
        if not len(self.pair_keys):
            return result
        positions = np.searchsorted(self.pair_keys, pair_keys)
        valid = positions < len(self.pair_keys)
        valid[valid] &= self.pair_keys[positions[valid]] == pair_keys[valid]
        result[valid] = self.timestamps[self.offsets[positions[valid] + 1] - 1]
        return result


class PBCLEvaluationData(Dataset):
    """Return users in exactly the order used by the ground-truth lengths."""

    def __init__(self, users):
        self.users = [int(user) for user in users]

    def __getitem__(self, index):
        return self.users[index]

    def __len__(self):
        return len(self.users)


class PBCLTrainData(Dataset):
    """One observed training buy and one training-buy-unobserved negative."""

    def __init__(self, dataset):
        self.data = dataset
        self.events = dataset.train_events[dataset.target_behavior]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = self.events[index]
        user, positive = int(event["user"]), int(event["item"])
        cutoff = float(event["timestamp"])
        negative = self.data.sample_negative(user)
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "pos_item": torch.tensor(positive, dtype=torch.long),
            "neg_item": torch.tensor(negative, dtype=torch.long),
            "prediction_time": torch.tensor(cutoff, dtype=torch.float64),
            "gender": self.data.user_gender[user],
            "age": self.data.user_age[user],
            "gender_mask": self.data.gender_mask[user],
            "age_mask": self.data.age_mask[user],
            "pos_popularity": self.data.item_popularity_label[positive],
            "neg_popularity": self.data.item_popularity_label[negative],
            # Both candidates use the same prediction time and training-only history.
            "pos_chain": self.data.get_chain(user, positive, cutoff),
            "neg_chain": self.data.get_chain(user, negative, cutoff),
        }


class PBCLDataSet:
    """Prepare PBCL graphs, demographic supervision, BPR data, and chains.

    The graph is a static training-only graph. Per-event time truncation applies
    to chain inputs; it does not rebuild LightGCN graphs for every training buy.
    No validation/test labels enter training sampling or popularity statistics.
    """

    EVENT_DTYPE = np.dtype([("user", "i8"), ("item", "i8"), ("timestamp", "f8")])

    def __init__(self, args):
        self.args = args
        self.path = Path(args.data_path).expanduser()
        requested = getattr(args, "behaviors", None) or ["view", "favorite", "cart", "buy"]
        if isinstance(requested, str):
            requested = requested.split(",")
        self.behavior_sources = {}
        for raw in requested:
            raw = str(raw).strip().lower()
            behavior = self._canonical_behavior(raw)
            if behavior in self.behavior_sources:
                raise ValueError(f"Duplicate behavior after alias normalization: {behavior}")
            self.behavior_sources[behavior] = raw
        self.behaviors = list(self.behavior_sources)
        self.behavior_to_index = {behavior: index for index, behavior in enumerate(self.behaviors)}
        self.num_behaviors = len(self.behaviors)
        self.target_behavior = "buy"
        if self.target_behavior not in self.behavior_to_index:
            raise ValueError("PBCL requires the target behavior 'buy'.")
        self.target_behavior_index = self.behavior_to_index[self.target_behavior]
        count = _read_json(self.path / "count.txt")
        self.user_count, self.item_count = int(count["user"]), int(count["item"])
        if self.user_count <= 0 or self.item_count <= 0:
            raise ValueError("count.txt must specify positive user and item counts.")
        self.num_nodes = self.user_count + self.item_count + 2
        self.item_offset = self.user_count + 1
        self.evaluation_time_policy = getattr(args, "evaluation_time_policy", "require")
        if self.evaluation_time_policy not in ("require", "all_training_history"):
            raise ValueError("evaluation_time_policy must be 'require' or 'all_training_history'.")

        self.train_events = self._load_training_events()
        self._build_training_graphs()
        self._load_demographics()
        self._build_popularity()
        self.pair_histories = {
            behavior: PBCLBehaviorHistory(events, self.item_count)
            for behavior, events in self.train_events.items()
        }
        self.training_pair_keys = np.unique(np.concatenate([
            history.pair_keys for history in self.pair_histories.values()
        ]))
        self.temporal_scale, self.temporal_scale_is_fallback = self._get_temporal_scale()
        self.tau_T = self.temporal_scale

        # Loading held-out labels last makes their separation from training explicit.
        self.validation_interacts = self._load_ground_truth("validation")
        self.test_interacts = self._load_ground_truth("test")
        self.validation_users = [int(user) for user in self.validation_interacts]
        self.test_users = [int(user) for user in self.test_interacts]
        self.validation_gt_length = np.array([
            len(items) for items in self.validation_interacts.values()
        ], dtype=np.int64)
        self.test_gt_length = np.array([
            len(items) for items in self.test_interacts.values()
        ], dtype=np.int64)
        self.evaluation_cutoffs = {
            split: self._load_evaluation_cutoffs(split) for split in ("validation", "test")
        }

    @staticmethod
    def _canonical_behavior(name):
        name = str(name).strip().lower()
        return "favorite" if name == "collect" else name

    def _resolve(self, filename):
        path = Path(filename)
        return path if path.is_absolute() else self.path / path

    def _find_file(self, configured, candidates):
        if configured:
            path = self._resolve(configured)
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return next((self.path / name for name in candidates if (self.path / name).is_file()), None)

    def _check_id(self, value, side, context):
        try:
            result = int(value)
            if float(value) != result:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid {side} ID at {context}: {value!r}") from error
        upper = self.user_count if side == "user" else self.item_count
        if not 1 <= result <= upper:
            raise ValueError(f"{side} ID {result} at {context} is outside 1..{upper}; "
                             "apply the same ID mapping to every input file.")
        return result

    def _csv_rows(self, path, required):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
                raise ValueError(f"{path} requires CSV columns {', '.join(required)}.")
            for number, row in enumerate(reader, 2):
                yield number, row

    def _load_training_events(self):
        rows = {behavior: [] for behavior in self.behaviors}
        unified = self._find_file(getattr(self.args, "train_file", None), ["train_interactions.csv"])

        def append(user, item, time, behavior, context):
            behavior = self._canonical_behavior(behavior)
            if behavior not in rows:
                raise ValueError(f"Unknown behavior {behavior!r} at {context}.")
            rows[behavior].append((self._check_id(user, "user", context),
                                   self._check_id(item, "item", context),
                                   _timestamp(time, context)))

        if unified:
            for number, row in self._csv_rows(unified, ["user_id", "item_id", "behavior", "timestamp"]):
                append(row["user_id"], row["item_id"], row["timestamp"], row["behavior"],
                       f"{unified}:{number}")
        else:
            overrides = getattr(self.args, "behavior_files", {}) or {}
            timestamp_files = getattr(self.args, "timestamp_files", {}) or {}
            for behavior in self.behaviors:
                names = list(dict.fromkeys([self.behavior_sources[behavior], behavior] +
                                           (["collect"] if behavior == "favorite" else [])))
                path = self._find_file(overrides.get(behavior),
                                       [f"{name}.{extension}" for name in names for extension in ("txt", "csv")])
                if path is None:
                    raise FileNotFoundError(f"Missing timestamped training file for {behavior} in {self.path}.")
                if path.suffix.lower() == ".csv":
                    for number, row in self._csv_rows(path, ["user_id", "item_id", "timestamp"]):
                        append(row["user_id"], row["item_id"], row["timestamp"], behavior, f"{path}:{number}")
                    continue
                sidecar_path = self._find_file(timestamp_files.get(behavior),
                                               [f"{name}_timestamps.json" for name in names])
                sidecar = _read_json(sidecar_path) if sidecar_path else None
                loaded_pairs = set()
                with path.open(encoding="utf-8-sig") as handle:
                    for number, line in enumerate(handle, 1):
                        if not line.strip() or line.lstrip().startswith("#"):
                            continue
                        fields = line.split()
                        context = f"{path}:{number}"
                        if len(fields) == 3:
                            append(*fields, behavior, context)
                        elif len(fields) == 2 and sidecar is not None:
                            user = self._check_id(fields[0], "user", context)
                            item = self._check_id(fields[1], "item", context)
                            if (user, item) in loaded_pairs:
                                continue
                            loaded_pairs.add((user, item))
                            times = sidecar.get(str(user), {}).get(str(item))
                            if times is None or times == []:
                                raise ValueError(f"Missing timestamps for ({user}, {item}) in {sidecar_path}.")
                            for time in times if isinstance(times, list) else [times]:
                                append(user, item, time, behavior, context)
                        else:
                            raise ValueError(f"{context}: expected 'user_id item_id timestamp'. "
                                             "For two-column files, provide <behavior>_timestamps.json; "
                                             "PBCL never fabricates timestamps or behavior order.")
        result = {}
        for behavior, values in rows.items():
            events = np.array(values, dtype=self.EVENT_DTYPE)
            events.sort(order=["user", "item", "timestamp"])
            result[behavior] = events
        if not len(result[self.target_behavior]):
            raise ValueError("PBCL BPR training requires at least one observed training buy.")
        return result

    def _build_training_graphs(self):
        self.edge_index, self.normalized_adjacency = {}, {}
        self.train_behavior_dict = {}
        user_degrees, item_degrees, all_edges = [], [], []
        for behavior, events in self.train_events.items():
            pairs = np.unique(np.column_stack((events["user"], events["item"])), axis=0)
            users, items = pairs[:, 0], pairs[:, 1]
            user_degree = np.bincount(users, minlength=self.user_count + 1)
            item_degree = np.bincount(items, minlength=self.item_count + 1)
            user_degrees.append(torch.from_numpy(user_degree))
            item_degrees.append(torch.from_numpy(item_degree))
            grouped = {}
            for user, item in pairs:
                grouped.setdefault(str(user), []).append(int(item))
            self.train_behavior_dict[behavior] = grouped
            shifted = items + self.item_offset
            edges = torch.from_numpy(np.stack((np.concatenate((users, shifted)),
                                               np.concatenate((shifted, users))))).long()
            self.edge_index[behavior] = edges
            degree = torch.cat((torch.from_numpy(user_degree), torch.from_numpy(item_degree))).float()
            inverse_sqrt = degree.clamp_min(1).rsqrt()
            values = inverse_sqrt[edges[0]] * inverse_sqrt[edges[1]]
            self.normalized_adjacency[behavior] = torch.sparse_coo_tensor(
                edges, values, (self.num_nodes, self.num_nodes)
            ).coalesce()
            all_edges.append(edges)
        self.user_behavior_degree = torch.stack(user_degrees, dim=1)
        self.item_behavior_degree = torch.stack(item_degrees, dim=1)
        self.user_behavior_mask = self.user_behavior_degree > 0
        self.item_behavior_mask = self.item_behavior_degree > 0
        # Retain spelling aliases used by the previous project, without old losses.
        self.user_behaviour_degree = self.user_behavior_degree
        self.item_behaviour_degree = self.item_behavior_degree
        self.all_edge_index = torch.unique(torch.cat(all_edges, dim=1), dim=1)
        combined = {}
        for behavior in self.behaviors:
            for user, items in self.train_behavior_dict[behavior].items():
                combined.setdefault(user, set()).update(items)
        self.train_behavior_dict["all"] = {user: sorted(items) for user, items in combined.items()}
        self.global_interacts = self.train_behavior_dict["all"]
        self.train_buy_sets = {
            int(user): set(items) for user, items in self.train_behavior_dict[self.target_behavior].items()
        }

    def _attribute_records(self, path):
        if path.suffix.lower() in (".json", ".txt"):
            records = _read_json(path)
            if not isinstance(records, dict):
                raise ValueError(f"{path} must contain a JSON object keyed by mapped user IDs.")
            return records
        user_field = getattr(self.args, "user_id_field", "user_id")
        return {row[user_field]: row for _, row in self._csv_rows(path, [user_field])}

    def _load_demographics(self):
        profile = self._find_file(getattr(self.args, "profile_file", None),
                                  ["user_profile.json", "user_profile.csv", "user_info.json", "user_info.csv"])
        raw = {"gender": {}, "age": {}}
        if profile:
            for user, record in self._attribute_records(profile).items():
                user = self._check_id(user, "user", profile)
                if not isinstance(record, dict):
                    raise ValueError(f"Profile for user {user} in {profile} must contain named attributes.")
                for attribute in raw:
                    field = getattr(self.args, f"{attribute}_field", attribute)
                    value = record.get(field)
                    if attribute == "age" and field == "age" and "age" not in record:
                        value = record.get("age_range")
                    raw[attribute][user] = value
        for attribute in raw:
            separate = self._find_file(getattr(self.args, f"{attribute}_file", None),
                                       [f"user_{attribute}.json", f"user_{attribute}.txt",
                                        f"{attribute}.json", f"{attribute}.txt", f"user_{attribute}.csv"])
            if separate:
                for user, value in self._attribute_records(separate).items():
                    user = self._check_id(user, "user", separate)
                    if isinstance(value, dict):
                        value = value.get(getattr(self.args, f"{attribute}_field", attribute))
                    raw[attribute][user] = value
            if not profile and not separate and not getattr(self.args, "allow_missing_demographics", False):
                raise FileNotFoundError(f"Missing {attribute} data in {self.path}. Set profile_file or "
                                        f"{attribute}_file to the dataset's actual file name.")
            labels, mapping, classes = self._encode_attribute(raw[attribute], attribute)
            setattr(self, f"user_{attribute}", labels)
            setattr(self, f"{attribute}_mask", labels >= 0)
            setattr(self, f"{attribute}_value_map", mapping)
            setattr(self, f"num_{attribute}_classes", classes)

    def _encode_attribute(self, values, attribute):
        invalid = {"", "-1", "nan", "none", "null", "unknown", "n/a"}
        configured = getattr(self.args, f"{attribute}_unknown_values", []) or []
        if isinstance(configured, str):
            configured = configured.split(",")
        invalid.update(_label_key(value) for value in configured)
        normalized = {user: _label_key(value) for user, value in values.items()}
        for value in set(normalized.values()):
            try:
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    invalid.add(value)
            except ValueError:
                pass
        mapping = getattr(self.args, f"{attribute}_value_map", None)
        boundaries = getattr(self.args, "age_boundaries", None) if attribute == "age" else None
        if mapping is not None:
            mapping = {_label_key(key): int(value) for key, value in mapping.items()}
            used = sorted({value for value in mapping.values() if value >= 0})
            if used != list(range(len(used))):
                raise ValueError(f"{attribute}_value_map must use contiguous labels starting at 0.")
            classes = len(used)
        elif boundaries is not None:
            boundaries = np.asarray(boundaries, dtype=np.float64)
            if boundaries.ndim != 1 or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0):
                raise ValueError("age_boundaries must be finite, strictly increasing numeric boundaries.")
            mapping, classes = {}, len(boundaries) + 1
            for value in set(normalized.values()) - invalid:
                try:
                    age = float(value)
                except ValueError as error:
                    raise ValueError("age_boundaries requires numeric ages.") from error
                if not math.isfinite(age) or age < 0:
                    invalid.add(value)
                else:
                    mapping[value] = int(np.searchsorted(boundaries, age, side="right"))
        elif attribute == "gender":
            # Zero is a valid gender label. Other source codes require an explicit map.
            mapping = {"0": 0, "1": 1, "female": 0, "male": 1, "f": 0, "m": 1}
            classes = 2
        else:
            categories = set(normalized.values()) - invalid
            def category_key(value):
                try:
                    return 0, float(value)
                except ValueError:
                    return 1, value
            mapping = {value: index for index, value in enumerate(sorted(categories, key=category_key))}
            classes = len(mapping)
        labels = torch.full((self.user_count + 1,), -1, dtype=torch.long)
        for user, value in normalized.items():
            if value not in invalid:
                label = mapping.get(value, -1)
                if label >= 0:
                    labels[user] = label
        return labels, mapping, classes

    def _build_popularity(self):
        self.num_popularity_groups = int(getattr(self.args, "num_popularity_groups", 4))
        if self.num_popularity_groups < 2:
            raise ValueError("num_popularity_groups must be at least 2.")
        self.popularity_count_mode = getattr(self.args, "popularity_count_mode", "users")
        if self.popularity_count_mode not in ("users", "events"):
            raise ValueError("popularity_count_mode must be 'users' or 'events'.")
        counts = self.item_behavior_degree.clone()
        if self.popularity_count_mode == "events":
            counts = torch.from_numpy(np.stack([
                np.bincount(self.train_events[behavior]["item"], minlength=self.item_count + 1)
                for behavior in self.behaviors
            ], axis=1))
        self.item_popularity_count = counts
        labels = torch.full_like(counts, -1)
        self.popularity_bin_edges = {}
        quantiles = np.arange(1, self.num_popularity_groups) / self.num_popularity_groups
        for index, behavior in enumerate(self.behaviors):
            values = counts[1:, index].numpy()
            edges = np.quantile(values, quantiles)
            # Equal counts always share a state, even if this leaves empty groups.
            labels[1:, index] = torch.from_numpy(np.searchsorted(edges, values, side="left")).long()
            self.popularity_bin_edges[behavior] = edges
        self.item_popularity_label = labels
        self.item_popularity_mask = labels >= 0

    @staticmethod
    def _chain_arrays(times):
        mask = ~np.isnan(times)
        timestamps = np.where(mask, times, 0.0)
        deltas = timestamps[:, None, :] - timestamps[:, :, None]
        forward = mask[:, :, None] & mask[:, None, :] & (deltas > 0)
        nearest = np.where(forward, deltas, np.inf).min(axis=2, keepdims=True)
        adjacency = forward & (deltas == nearest)
        terminal_time = np.where(mask, timestamps, -np.inf).max(axis=1, keepdims=True)
        terminal = mask & (timestamps == terminal_time)
        return mask, timestamps, adjacency, np.where(adjacency, deltas, 0.0), terminal

    def _get_temporal_scale(self):
        override = getattr(self.args, "temporal_scale", None)
        if override is not None:
            override = float(override)
            if not math.isfinite(override) or override <= 0:
                raise ValueError("temporal_scale must be finite and positive.")
            return override, False
        intervals = []
        # Reproduction choice: estimate the median on final training-only pair chains.
        # No held-out timestamps or labels participate in this statistic.
        for start in range(0, len(self.training_pair_keys), 65536):
            keys = self.training_pair_keys[start:start + 65536]
            times = np.stack([self.pair_histories[b].latest(keys) for b in self.behaviors], axis=1)
            _, _, adjacency, deltas, _ = self._chain_arrays(times)
            intervals.append(deltas[adjacency])
        positive = np.concatenate(intervals)
        if len(positive):
            return float(np.median(positive)), False
        warnings.warn("No positive training transition intervals; temporal_scale=1.0 is an explicit "
                      "fallback and does not create any chain edges.", RuntimeWarning)
        return 1.0, True

    def get_batch_chains(self, users, items, cutoffs=None):
        """Build [batch,K,K] source-to-receiver chains; None uses all training events."""
        users = self._query_ids(users, "user")
        items = self._query_ids(items, "item")
        if len(users) != len(items):
            raise ValueError("users and items must have equal lengths.")
        if np.any((users < 1) | (users > self.user_count)) or np.any((items < 1) | (items > self.item_count)):
            raise ValueError("Chain queries require real one-based user and item IDs.")
        if cutoffs is None:
            cutoffs = np.full(len(users), np.inf)
        else:
            cutoffs = _as_numpy(cutoffs, np.float64)
            if cutoffs.ndim == 0:
                cutoffs = np.full(len(users), cutoffs.item())
            cutoffs = cutoffs.reshape(-1)
            if len(cutoffs) != len(users) or np.isnan(cutoffs).any():
                raise ValueError("cutoffs must be a scalar or one non-NaN timestamp per pair.")
        keys = users * (self.item_count + 1) + items
        times = np.stack([self.pair_histories[b].latest_before(keys, cutoffs) for b in self.behaviors], axis=1)
        mask, timestamps, adjacency, deltas, terminal = self._chain_arrays(times)
        weights = np.where(adjacency, np.exp(-deltas / self.temporal_scale), 0.0).astype(np.float32)
        eligible = adjacency.copy()
        eligible[:, self.target_behavior_index, :] = False
        return {
            "node_mask": torch.from_numpy(mask),
            "timestamps": torch.from_numpy(timestamps),
            "adjacency": torch.from_numpy(adjacency),
            "time_deltas": torch.from_numpy(deltas),
            "time_weights": torch.from_numpy(weights),
            "terminal_mask": torch.from_numpy(terminal),
            "has_history": torch.from_numpy(mask.any(axis=1)),
            "eligible_edge_mask": torch.from_numpy(eligible),
        }

    def _query_ids(self, values, side):
        try:
            values = _as_numpy(values, np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{side} query IDs must be integers.") from error
        upper = self.user_count if side == "user" else self.item_count
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{side} query IDs must be finite integers, without truncation.")
        if np.any((values < 1) | (values > upper)):
            raise ValueError(f"{side} query IDs must be in 1..{upper}.")
        return values.astype(np.int64)

    def get_chain(self, user, item, cutoff=None):
        """Empty histories remain empty; the model must define their scoring fallback."""
        return {key: value[0] for key, value in self.get_batch_chains([user], [item], cutoff).items()}

    def sample_edge_counterfactual(self, chains):
        """Select one eligible connection uniformly within each chain (Definition 1)."""
        eligible = chains["eligible_edge_mask"]
        if eligible.ndim == 2:
            eligible = eligible.unsqueeze(0)
        selected = torch.full((len(eligible), 2), -1, dtype=torch.long)
        for row, candidates in enumerate(eligible):
            edges = candidates.nonzero(as_tuple=False)
            if not len(edges):
                continue
            selected[row] = edges[torch.randint(len(edges), ()).item()].cpu()
        return selected

    def sample_negative(self, user):
        """Sample uniformly outside training buys, with a bounded rejection loop."""
        user = self._check_id(user, "user", "negative sampling")
        excluded = self.train_buy_sets.get(user, set())
        if len(excluded) >= self.item_count:
            raise ValueError(f"User {user} has bought every item; no valid BPR negative exists.")
        for _ in range(32):
            item = int(torch.randint(1, self.item_count + 1, ()).item())
            if item not in excluded:
                return item
        # Exact complement fallback avoids hanging for dense users.
        candidates = np.setdiff1d(np.arange(1, self.item_count + 1), np.fromiter(excluded, dtype=np.int64))
        return int(candidates[torch.randint(len(candidates), ()).item()])

    def _load_ground_truth(self, split):
        path = self._resolve(getattr(self.args, f"{split}_file", None) or f"{split}_dict.txt")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} ground truth: {path}. Use an empty JSON object if unused.")
        result = {}
        for raw_user, raw_items in _read_json(path).items():
            user = self._check_id(raw_user, "user", path)
            if not isinstance(raw_items, list):
                raise ValueError(f"{path}: user {user} requires a list of item IDs.")
            items = sorted({self._check_id(item, "item", path) for item in raw_items})
            if items:
                result[str(user)] = items
        return dict(sorted(result.items(), key=lambda pair: int(pair[0])))

    def _load_evaluation_cutoffs(self, split):
        path = self._find_file(getattr(self.args, f"{split}_time_file", None), [f"{split}_timestamps.json"])
        if path is None:
            if getattr(self, f"{split}_users"):
                if self.evaluation_time_policy == "require":
                    raise ValueError(f"Missing {split} prediction timestamps. Provide {split}_timestamps.json "
                                     "or explicitly set evaluation_time_policy='all_training_history' only "
                                     "if all training events precede the evaluation queries.")
                warnings.warn(f"{split}: using all training history by explicit configuration; "
                              "strict per-query chronology cannot be checked without prediction timestamps.",
                              RuntimeWarning)
            return None
        result = {}
        for user, value in _read_json(path).items():
            user = self._check_id(user, "user", path)
            result[user] = _timestamp(value, path)
        for user in getattr(self, f"{split}_users"):
            if user not in result:
                raise ValueError(f"Missing {split} prediction time for user {user} in {path}.")
        return result

    def get_evaluation_chains(self, users, items, split="test"):
        if split not in self.evaluation_cutoffs:
            raise ValueError("split must be 'validation' or 'test'.")
        times = self.evaluation_cutoffs[split]
        users = self._query_ids(users, "user")
        cutoffs = None if times is None else [times[int(user)] for user in users]
        return self.get_batch_chains(users, items, cutoffs)

    def train_dataset(self):
        return PBCLTrainData(self)

    def behavior_dataset(self):
        """Import-stage compatibility only: batches now follow the PBCL dictionary contract."""
        return self.train_dataset()

    def validate_dataset(self):
        return PBCLEvaluationData(self.validation_users)

    def test_dataset(self):
        return PBCLEvaluationData(self.test_users)


# Keep old imports working while main/model/trainer are replaced in later steps.
DataSet = PBCLDataSet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a PBCL dataset")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--behaviors", nargs="+", default=["view", "favorite", "cart", "buy"])
    parser.add_argument("--train_file")
    parser.add_argument("--profile_file")
    parser.add_argument("--gender_file")
    parser.add_argument("--age_file")
    parser.add_argument("--gender_unknown_values", nargs="*", default=[])
    parser.add_argument("--age_unknown_values", nargs="*", default=[])
    parser.add_argument("--age_boundaries", nargs="*", type=float)
    parser.add_argument("--num_popularity_groups", type=int, default=4)
    parser.add_argument("--popularity_count_mode", choices=["users", "events"], default="users")
    parser.add_argument("--temporal_scale", type=float)
    parser.add_argument("--evaluation_time_policy", choices=["require", "all_training_history"], default="require")
    parser.add_argument("--seed", type=int, default=2026)
    configuration = parser.parse_args()
    torch.manual_seed(configuration.seed)
    data = PBCLDataSet(configuration)
    print(f"PBCL: {data.user_count} users, {data.item_count} items, {len(data.train_dataset())} buy events")
    print(f"Behaviors: {data.behaviors}; temporal scale: {data.temporal_scale:g}")
    print(f"Valid gender labels: {int(data.gender_mask.sum())}; valid age labels: {int(data.age_mask.sum())}")
