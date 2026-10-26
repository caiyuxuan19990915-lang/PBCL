#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PBCL entry script. Edit the explicit defaults below to configure a run."""

import argparse
import json
from pathlib import Path


if __name__ == '__main__':

    parser = argparse.ArgumentParser('PBCL Model Configuration', add_help=False, allow_abbrev=False)
    parser.add_argument('-h', '--help', action='help', help='Show all PBCL options and exit')

    # ---------------- Dataset & Input Files ----------------
    parser.add_argument('--data_name', type=str, default='ijcai', help='Dataset: ijcai, jdata, or fliggy')
    parser.add_argument('--data_root', type=str, default=str(PROJECT_DIR / 'data'), help='Data root')
    parser.add_argument('--data_path', type=str, default='', help='Empty string uses data_root/data_name')
    parser.add_argument('--behaviors', type=str, nargs='+', default=['view', 'favorite', 'cart', 'buy'], help='Configured behaviors; collect is an alias of favorite')
    parser.add_argument('--train_file', type=str, default='', help='Empty string discovers unified CSV or per-behavior files')
    parser.add_argument('--behavior_files', type=json.loads, default={}, help='Behavior files')
    parser.add_argument('--timestamp_files', type=json.loads, default={}, help='Timestamp files')
    parser.add_argument('--validation_file', type=str, default='', help='Validation file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--validation_time_file', type=str, default='', help='Validation time file')
    parser.add_argument('--test_time_file', type=str, default='', help='Test time file')
    parser.add_argument('--evaluation_time_policy', type=str, default='require', help='Evaluation time policy')

    # ---------------- User Attributes & Item Popularity ----------------
    parser.add_argument('--profile_file', type=str, default='', help='Empty string discovers the user profile file')
    parser.add_argument('--gender_file', type=str, default='', help='Optional separate gender file; empty string uses the profile')
    parser.add_argument('--age_file', type=str, default='', help='Optional separate age file; empty string uses the profile')
    parser.add_argument('--user_id_field', type=str, default='user_id', help='User id field')
    parser.add_argument('--gender_field', type=str, default='gender', help='Gender field')
    parser.add_argument('--age_field', type=str, default='age', help='Age field')
    parser.add_argument('--gender_value_map', type=json.loads, default={}, help='JSON object; empty object retains automatic gender encoding')
    parser.add_argument('--age_value_map', type=json.loads, default={}, help='JSON object; empty object retains automatic categorical age encoding')
    parser.add_argument('--gender_unknown_values', type=str, nargs='*', default=[], help='Gender unknown values')
    parser.add_argument('--age_unknown_values', type=str, nargs='*', default=[], help='Age unknown values')
    parser.add_argument('--age_boundaries', type=float, nargs='+', default=[], help='Raw-age bin edges; empty list retains categorical encoding')
    parser.add_argument('--allow_missing_demographics', action=argparse.BooleanOptionalAction, default=False, help='Allow missing demographics')
    parser.add_argument('--num_popularity_groups', type=int, default=4, help='Num popularity groups')
    parser.add_argument('--popularity_count_mode', type=str, default='users', help='Popularity count mode')
    parser.add_argument('--temporal_scale', type=float, default=0.0, help='0 estimates the median positive training interval; positive values override it')

    # ---------------- Basic Recommendation & GNN Hyperparameters ----------------
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of parallel LightGCN layers per behavior')
    parser.add_argument('--chain_layers', type=int, default=3, help='Number of synchronous behavior-chain propagation layers')
    parser.add_argument('--user_shared_rank', type=int, default=4, help='User shared rank')
    parser.add_argument('--user_specific_rank', type=int, default=4, help='User specific rank')
    parser.add_argument('--item_shared_rank', type=int, default=4, help='Item shared rank')
    parser.add_argument('--item_specific_rank', type=int, default=4, help='Item specific rank')
    parser.add_argument('--head_hidden_size', type=int, default=64, help='Head hidden size')
    parser.add_argument('--margin_hidden_size', type=int, default=64, help='Margin hidden size')
    parser.add_argument('--empty_chain_policy', type=str, default='target', help='Empty chain policy')

    # ---------------- PBCL Fairness & Counterfactual Hyperparameters ----------------
    parser.add_argument('--utility_weight', type=float, default=1.0, help='Utility margin weight alpha in transition control')
    parser.add_argument('--sensitivity_weight', type=float, default=0.1, help='Sensitivity weight beta; JDATA preset is 0.01')
    parser.add_argument('--lambda_adv', type=float, default=0.01, help='Adversarial loss weight; JDATA preset is 0.001')
    parser.add_argument('--lambda_orth', type=float, default=0.001, help='Full concatenated-factor orthogonality penalty weight')
    parser.add_argument('--lambda_edge', type=float, default=1.0, help='Edge margin distillation weight')
    parser.add_argument('--lambda_cf', type=float, default=0.1, help='Node counterfactual JS weight; JDATA preset is 1.0')
    parser.add_argument('--lambda_s', type=float, default=1.0, help='Sensitivity term weight within edge distillation')
    parser.add_argument('--cf_temperature', type=float, default=1.0, help='Shared factual/counterfactual softmax temperature')
    parser.add_argument('--sensitivity_attribute_weights', type=json.loads, default={'age': 1.0, 'gender': 1.0, 'pop': 1.0}, help='JSON object containing age, gender, and pop weights')

    # ---------------- Training & Alternating Optimization ----------------
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of the main optimizer')
    parser.add_argument('--discriminator_lr', type=float, default=0.001, help='Independent learning rate of the adversarial optimizer')
    parser.add_argument('--audit_lr', type=float, default=0.001, help='Independent learning rate of attribute auditors')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--discriminator_weight_decay', type=float, default=0.0, help='Discriminator weight decay')
    parser.add_argument('--audit_weight_decay', type=float, default=0.0, help='Audit weight decay')
    parser.add_argument('--discriminator_steps', type=int, default=1, help='Discriminator steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs, including completed resumed epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers')
    parser.add_argument('--max_grad_norm', type=float, default=0.0, help='Gradient clipping threshold; 0 disables clipping')

    # ---------------- Auditor, Edge Teacher & Prototype Schedules ----------------
    parser.add_argument('--audit_pretrain_epochs', type=int, default=1, help='Auditor passes before the first teacher refresh')
    parser.add_argument('--audit_refresh_interval', type=int, default=50, help='Legacy compatibility option; auditors refresh with edge teachers')
    parser.add_argument('--audit_refresh_epochs', type=int, default=1, help='Audit refresh epochs')
    parser.add_argument('--audit_max_batches', type=int, default=0, help='Auditor batch limit; 0 uses all batches')
    parser.add_argument('--audit_history_samples', type=int, default=1024, help='Legacy compatibility option; the unified teacher pool uses edge_teacher_samples')
    parser.add_argument('--edge_refresh_interval', type=int, default=50, help='Ordinary joint updates between complete teacher refreshes')
    parser.add_argument('--edge_teacher_samples', type=int, default=1024, help='Eligible histories sampled uniformly without replacement at each refresh')
    parser.add_argument('--edge_buy_samples', type=int, default=256, help='Legacy compatibility option; no fixed buy-history quota is used')
    parser.add_argument('--edge_history_samples', type=int, default=256, help='Legacy compatibility option; no fixed auxiliary-history quota is used')
    parser.add_argument('--edge_batch_size', type=int, default=256, help='Edge batch size')
    parser.add_argument('--edge_warmup_steps', type=int, default=0, help='Edge warmup steps')
    parser.add_argument('--prototype_refresh_interval', type=int, default=100, help='Prototype refresh interval')
    parser.add_argument('--cf_warmup_steps', type=int, default=0, help='Cf warmup steps')
    parser.add_argument('--cf_candidate_count', type=int, default=2, help='2 reuses the BPR positive/negative pair; larger values add candidates')

    # ---------------- Validation, Metrics & Final Evaluation ----------------
    parser.add_argument('--topk', type=str, nargs='+', default=[10, 50], help='Ranking cutoffs: 10 50, 10,50, or [10,50]')
    parser.add_argument('--metrics', type=str, nargs='+', default=['hit', 'ndcg'], help='hit/HR, ndcg, recall, precision, mrr')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')
    parser.add_argument('--prediction_pair_batch_size', type=int, default=4096, help='Prediction pair batch size')
    parser.add_argument('--ranking_policy', type=str, default='novel', help='novel excludes training buys; repeat permits them')
    parser.add_argument('--eval_every', type=int, default=1, help='Eval every')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience; 0 runs the full epoch budget')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Min delta')
    parser.add_argument('--monitor', type=str, default='ndcg@10', help='Validation metric; follows the smallest K unless explicitly set')
    parser.add_argument('--monitor_mode', type=str, default='max', help='Monitor mode')
    parser.add_argument('--evaluate_test', action=argparse.BooleanOptionalAction, default=True, help='Test the selected model after training')
    parser.add_argument('--eval_split', type=str, default='test', help='Eval split')
    parser.add_argument('--eval_weights', type=str, default='selected', help='selected uses validation-best weights; current uses checkpoint weights')

    # ---------------- Execution, Reproducibility & Saving ----------------
    parser.add_argument('--mode', type=str, default='train', help='train, resume, or evaluate')
    parser.add_argument('--config', type=str, default='', help='Optional flat JSON configuration path')
    parser.add_argument('--checkpoint', type=str, default='', help='Required checkpoint path for resume/evaluate')
    parser.add_argument('--restore_rng', action=argparse.BooleanOptionalAction, default=True, help='Restore checkpoint random states during resume')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--seed', type=int, default=2026, help='Seed')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False, help='Deterministic')
    parser.add_argument('--torch_threads', type=int, default=4, help='Number of Torch CPU threads')
    parser.add_argument('--model_name', type=str, default='PBCL', help='PBCL checkpoint name prefix')
    parser.add_argument('--model_path', type=str, default='', help='Empty string uses the generated run directory/checkpoints')
    parser.add_argument('--save_checkpoints', action=argparse.BooleanOptionalAction, default=True, help='Save checkpoints')
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_DIR / 'results'), help='Output dir')
    parser.add_argument('--run_name', type=str, default='', help='Empty string generates a unique PBCL run name')
    parser.add_argument('--log_level', type=str, default='INFO', help='Log level')

    # ---------------- Resolve Configuration & Start PBCL ----------------
    args = prepare_args(parser)
    report = run(args)
