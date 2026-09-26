#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import shutil
import copy
import json
import yaml
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import random
import numpy as np

from utility import Datasets
from models.CFDGN import CFDGN, MultiCBR



PROTOCOL_SEED = 2024


def set_global_seed(seed=PROTOCOL_SEED):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0", type=str)
    parser.add_argument("-d", "--dataset", default="NetEase", type=str)
    parser.add_argument("-m", "--model", default="MultiCBR", type=str)
    parser.add_argument("-i", "--info", default="", type=str)
    parser.add_argument(
        "-s", "--seed",
        default=PROTOCOL_SEED,
        type=int,
        help="random seed for reproducibility"
    )
    return parser.parse_args()


def print_gfr_config(conf, log_path):
    items = sorted((k, v) for k, v in conf.items() if k.startswith("gfr_"))
    lines = ["", "=" * 96, "CFDGN QUERY-VALUE MULTI-INTEREST + DECOUPLED SCALE-INVARIANT CONFIG"]
    lines += [f"{k} = {v}" for k, v in items]
    lines.append("=" * 96)
    for line in lines:
        print(line)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line + "\n")


def update_gfr_diag_accumulator(model, diag_sum, diag_count):
    diag = model.get_diagnostic_info() if hasattr(model, "get_diagnostic_info") else {}
    for k, v in diag.items():
        if isinstance(v, (int, float)):
            diag_sum[k] = diag_sum.get(k, 0.0) + float(v)
    return diag_sum, diag_count + (1 if diag else 0), diag


def print_gfr_diagnostics(epoch, batch_anchor, diag_sum, diag_count, last_diag, log_path, run):
    if diag_count <= 0:
        return
    avg = {k: v / diag_count for k, v in diag_sum.items()}
    ordered = [
        "main_bpr", "residual_bpr", "raw_struct_rank", "calibration_loss",
        "residual_rank_help", "residual_preserve",
        "residual_preserve_amplitude", "residual_preserve_margin",
        "main_need_weight", "main_preserve_weight", "main_error_weight_floor", "raw_rank_margin",
        "residual_scale", "main_score_scale", "deploy_multiplier", "normalized_main_gap",
        "relation_loss", "main_gap", "raw_residual_gap", "residual_gap", "final_gap",
        "main_wrong_frac", "correction_rate", "damage_rate", "raw_residual_abs_mean",
        "raw_residual_std_mean", "residual_abs_mean", "residual_std_mean", "residual_sat95",
        "repo_main_gap", "repo_struct_gap", "repo_struct_win", "score_facet",
        "score_bundle", "score_type", "score_comp",
        "interest_selector_entropy", "interest_selector_top1", "composition_reliability",
        "struct_neg_count", "history_valid_frac", "role_entropy", "role_diversity",
        "role_diversity_target", "role_diversity_loss_diag",
        "role_affinity_pred", "role_affinity_target",
        "fixed_ui_target_mean", "fixed_ui_target_std",
        "role_certainty", "relation_compatibility",
        "type_sim", "type_comp", "type_noise", "source_prior", "source_goal",
        "source_context", "source_ppmi", "ppmi_role_shift", "ppmi_local_active",
        "ppmi_gate", "pcl_item", "pcl_bundle", "pcl_user",
        "orth_item", "orth_bundle", "orth_user",
    ]
    lines = ["", f"[GFR-FINAL-DIAG] epoch={epoch} interval_batches={diag_count}"]
    for i in range(0, len(ordered), 4):
        keys = ordered[i:i+4]
        vals = " | ".join(f"{k}={avg.get(k, float('nan')):.6f}" for k in keys)
        lines.append("[GFR-FINAL-DIAG] " + vals)
    for line in lines:
        print(line)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line + "\n")
    for k, v in avg.items():
        run.add_scalar("gfr_final_diag/" + k, v, batch_anchor)


def main():
    all_conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")
    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    assert paras["model"] in ["MultiCBR", "CFDGN"]

    conf = all_conf[dataset_name.split("_")[0] if "_" in dataset_name else dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]

    # Exact original MultiCBR single-run RNG protocol.  No -s argument is
    # exposed/required; the historical repository default was seed=2024.
    protocol_seed = int(paras.get("seed", PROTOCOL_SEED))
    conf["seed"] = protocol_seed
    set_global_seed(protocol_seed)

    # Datasets/negative sampling receive the same seed as the original run.
    dataset = Datasets(conf)
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)

    for lr, l2_reg, UB_ratio, UI_ratio, BI_ratio, embedding_size, num_layers, c_lambda, c_temp in product(
        conf["lrs"], conf["l2_regs"], conf["UB_ratios"], conf["UI_ratios"], conf["BI_ratios"],
        conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"]
    ):
        base_log = f"./log/{conf['dataset']}/{conf['model']}"
        base_run = f"./runs/{conf['dataset']}/{conf['model']}"
        base_model = f"./checkpoints/{conf['dataset']}/{conf['model']}/model"
        base_conf = f"./checkpoints/{conf['dataset']}/{conf['model']}/conf"
        for p in (base_log, base_run, base_model, base_conf):
            os.makedirs(p, exist_ok=True)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size
        conf["UB_ratio"] = UB_ratio
        conf["UI_ratio"] = UI_ratio
        conf["BI_ratio"] = BI_ratio
        conf["num_layers"] = num_layers
        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp

        settings = []
        if conf["info"]:
            settings.append(conf["info"])
        settings.append(f"Seed_{protocol_seed}")
        settings.append(conf["aug_type"])
        if conf["aug_type"] == "ED":
            settings.append(str(conf["ed_interval"]))
        if conf["aug_type"] == "OP":
            assert UB_ratio == UI_ratio == BI_ratio == 0
        settings += [
            f"Neg_{conf['neg_num']}", str(conf["batch_size_train"]), str(lr), str(l2_reg),
            str(embedding_size), str(UB_ratio), str(UI_ratio), str(BI_ratio), str(num_layers),
            "_".join([
                str(conf["fusion_weights"]["modal_weight"]),
                str(conf["fusion_weights"]["UB_layer"]),
                str(conf["fusion_weights"]["UI_layer"]),
                str(conf["fusion_weights"]["BI_layer"]),
            ]),
            str(c_lambda), str(c_temp),
        ]
        setting = "_".join(settings)
        log_path = base_log + "/" + setting
        run_path = base_run + "/" + setting
        checkpoint_model_path = base_model + "/" + setting
        checkpoint_conf_path = base_conf + "/" + setting
        run = SummaryWriter(run_path)

        # Match the original MultiCBR protocol exactly for every hyperparameter
        # product-run: reset model-init RNG and the independent TRAIN loader /
        # negative-sampling generator before constructing the model.
        set_global_seed(protocol_seed)
        if hasattr(dataset, "reset_train_generator"):
            dataset.reset_train_generator(protocol_seed)

        if conf["model"] == "MultiCBR":
            model = MultiCBR(conf, dataset.graphs).to(device)
        else:
            model = CFDGN(conf, dataset.graphs).to(device)
            print_gfr_config(conf, log_path)
            gfr_diag_sum, gfr_diag_count, gfr_last_diag = {}, 0, {}
            main_named, structural_named = model.get_parameter_ownership()
            lines = [
                f"[GFR-FINAL] mode={conf['dataset']} QUERY-VALUE MULTI-INTEREST + DECOUPLED SCALE-INVARIANT CORRECTION",
                f"[GFR-FINAL] Main tensors={len(main_named)} | CFDGN tensors={len(structural_named)}",
                "[GFR-FINAL] CFDGN losses have no gradient path to MultiCBR backbone; structural init is Torch-RNG neutral",
                "[GFR-FINAL] Train=Deploy score: Final = Main + alpha*sigma_Main*raw_struct; alpha is dimensionless and TRAIN-learned",
                "[GFR-FINAL] sigma_Main = RMS within-user full-catalog Main-score std from the non-augmented TRAIN graph; no eval labels",
                "[GFR-FINAL] Main-preserving confidence/trust-region is computed in Main-normalized units",
                "[GFR-FINAL] raw Structural ranking and deploy calibration use separate gradient paths; calibration cannot collapse raw structure",
                f"[GFR-FINAL] gfr_main_error_weight_floor={conf.get('gfr_main_error_weight_floor', 0.10)} is active on TRAIN structural ranking/calibration",
                "[GFR-FINAL] candidate-specific interest selection uses target-independent UI queries; values remain history-aggregated",
                "[GFR-FINAL] composition preference is calibrated by TRAIN-history effective sample size; no dataset-specific threshold",
            ]
            for line in lines:
                print(line)
            with open(log_path, "a") as f:
                for line in lines:
                    f.write(line + "\n")

        # CRITICAL paired-protocol reset: CFDGN has extra structural
        # modules, so after construction reset the global RNG exactly as the
        # historical MultiCBR trainer did.  This pairs subsequent Noise
        # augmentation / stochastic training streams while leaving the user's
        # command line unchanged.
        set_global_seed(protocol_seed)
        seed_line = (
            f"[PROTOCOL-SEED] internal fixed seed={protocol_seed}; "
            "TRAIN loader/negative sampler/model init/post-init RNG are paired"
        )
        print(seed_line)
        with open(log_path, "a") as f:
            f.write(seed_line + "\n")

        # MultiCBR keeps the official Adam construction.  CFDGN's
        # residual_scale_logit is a bounded deploy calibration scalar and MUST
        # NOT receive L2/weight decay: its initialized logit is negative, so
        # decay toward zero would mechanically increase alpha toward
        # alpha_max/2 even without useful task evidence.
        if conf["model"] == "CFDGN":
            normal_params = []
            scale_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name == "residual_scale_logit":
                    scale_params.append(p)
                else:
                    normal_params.append(p)
            if len(scale_params) != 1:
                raise RuntimeError(
                    "CFDGN must expose exactly one residual_scale_logit"
                )
            optimizer = optim.Adam(
                [
                    {"params": normal_params, "weight_decay": conf["l2_reg"]},
                    {"params": scale_params, "weight_decay": 0.0},
                ],
                lr=lr,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=conf["l2_reg"]
            )

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])
        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        gfr_selector_state = init_gfr_selector_state(checkpoint_model_path) if conf["model"] == "CFDGN" else None

        for epoch in range(conf["epochs"]):
            epoch_anchor = epoch * batch_cnt
            if conf["model"] == "CFDGN":
                # Refresh once per epoch from the current non-augmented TRAIN
                # graph.  This scalar is detached and label-free; it only puts
                # the structural correction in the same numerical units as Main.
                model.eval()
                with torch.no_grad():
                    current_main_scale = model.refresh_main_score_scale()
                run.add_scalar(
                    "gfr_main_score_scale_epoch",
                    current_main_scale.detach(),
                    epoch,
                )
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i
                ED_drop = conf["aug_type"] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0

                if conf["model"] == "MultiCBR":
                    bpr_loss, c_loss = model(batch, ED_drop=ED_drop)
                    loss = bpr_loss + conf["c_lambda"] * c_loss
                    loss.backward()
                    pbar.set_description(
                        "epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f"
                        % (epoch, loss.detach(), bpr_loss.detach(), c_loss.detach())
                    )
                    run.add_scalar("loss_bpr", bpr_loss.detach(), batch_anchor)
                    run.add_scalar("loss_c", c_loss.detach(), batch_anchor)
                    run.add_scalar("loss", loss.detach(), batch_anchor)
                else:
                    out = model(batch, ED_drop=ED_drop)
                    losses = out["loss"]
                    loss = losses["total"]
                    loss.backward()
                    pbar.set_description(
                        "epoch: %d, total: %.4f, main: %.4f, struct: %.4f, rel: %.4f"
                        % (
                            epoch, loss.detach(), losses["main_bpr"].detach(),
                            losses["residual_bpr"].detach(),
                            losses["relation"].detach(),
                        )
                    )
                    for k, v in losses.items():
                        run.add_scalar("loss_" + k, v.detach(), batch_anchor)
                    gfr_diag_sum, gfr_diag_count, new_diag = update_gfr_diag_accumulator(
                        model, gfr_diag_sum, gfr_diag_count
                    )
                    if new_diag:
                        gfr_last_diag = new_diag

                optimizer.step()

                if (batch_anchor + 1) % test_interval_bs == 0:
                    if conf["model"] == "CFDGN":
                        print_gfr_diagnostics(
                            epoch, batch_anchor, gfr_diag_sum, gfr_diag_count,
                            gfr_last_diag, log_path, run
                        )
                        gfr_diag_sum, gfr_diag_count, gfr_last_diag = {}, 0, {}
                        # Structural representation is expensive; parameters are
                        # unchanged between Val/Test, so compute it exactly once.
                        model.eval()
                        with torch.no_grad():
                            eval_state = model.get_multi_modal_representations(test=True)
                        # Final experimental protocol: Test is sealed during
                        # checkpoint selection. Validation evaluates the exact
                        # full-catalog score used at deployment.
                        val_final, val_main, val_unified_diag = test_cfdgn_unified(
                            model, dataset.val_loader, conf, eval_state
                        )
                        metrics = {
                            "val": val_final,
                            "val_main": val_main,
                            "val_unified_diag": val_unified_diag,
                            "test": None,
                        }
                    else:
                        metrics = {
                            "val": test(model, dataset.val_loader, conf),
                            "test": test(model, dataset.test_loader, conf),
                        }

                    best_metrics, best_perform, best_epoch, gfr_selector_state = log_metrics(
                        conf, model, metrics, run, log_path,
                        checkpoint_model_path, checkpoint_conf_path,
                        epoch, batch_anchor,
                        best_metrics, best_perform, best_epoch,
                        gfr_selector_state,
                    )

        if conf["model"] == "CFDGN":
            finalize_cfdgn_once(
                conf, model, dataset, checkpoint_model_path,
                checkpoint_conf_path, best_epoch, log_path, run,
            )
            cleanup_gfr_selector_state(gfr_selector_state)
        else:
            print_final_best(conf, best_perform, best_epoch, log_path)
        run.close()


def finalize_cfdgn_once(conf, model, dataset, checkpoint_model_path,
                                checkpoint_conf_path, best_epoch, log_path, run):
    """Load the complete-Final Val-selected checkpoint; Test exactly once."""
    if not os.path.isfile(checkpoint_model_path):
        raise RuntimeError("No Validation-selected CFDGN checkpoint exists")

    state = torch.load(checkpoint_model_path, map_location=conf["device"])
    model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        eval_state = model.get_multi_modal_representations(test=True)
        # Re-evaluate Validation for reproducibility only; there is no extra
        # hyperparameter selection after checkpoint freeze.
        val_metrics, val_main, val_diag = test_cfdgn_unified(
            model, dataset.val_loader, conf, eval_state
        )

        saved_conf = dict(conf)
        saved_conf["gfr_selected_epoch"] = int(best_epoch)
        saved_conf["gfr_epoch_selection"] = "CompleteFinal_Val_sum_RN_at20_and40"
        saved_conf["gfr_deployment"] = "full_catalog_Main_plus_scale_invariant_calibrated_structural_correction"
        saved_conf["gfr_posthoc_fusion"] = "none"
        del saved_conf["device"]
        json.dump(saved_conf, open(checkpoint_conf_path, "w"))

        # Test is touched exactly once, with no lambda/candidate selection.
        test_metrics, test_main, test_diag = test_cfdgn_unified(
            model, dataset.test_loader, conf, eval_state
        )

    banner = "=" * 96
    lines = [
        "", banner,
        "CFDGN QUERY-VALUE MULTI-INTEREST + DECOUPLED SCALE-INVARIANT — VAL-SELECTED / TEST-ONCE",
        "Selected epoch: %d" % best_epoch,
        (
            "Val residual std=%.6f abs=%.6f sat95OfCap=%.6f | "
            "Test std=%.6f abs=%.6f sat95OfCap=%.6f"
            % (
                float(val_diag["residual_std"]),
                float(val_diag["residual_abs_mean"]),
                float(val_diag["residual_sat95"]),
                float(test_diag["residual_std"]),
                float(test_diag["residual_abs_mean"]),
                float(test_diag["residual_sat95"]),
            )
        ),
        (
            "Val residual positive-minus-false=%.6f | "
            "Test=%.6f | noHistory=%.4f/%.4f"
            % (
                float(val_diag["residual_pos_false_gap"]),
                float(test_diag["residual_pos_false_gap"]),
                float(val_diag["no_history_frac"]),
                float(test_diag["no_history_frac"]),
            )
        ),
    ]
    for topk in conf["topk"]:
        lines.append(
            "TOP %d: Val Main=%.5f/%.5f Final=%.5f/%.5f d=%+.5f/%+.5f | "
            "Test Main=%.5f/%.5f Final=%.5f/%.5f d=%+.5f/%+.5f"
            % (
                topk,
                val_main["recall"][topk], val_main["ndcg"][topk],
                val_metrics["recall"][topk], val_metrics["ndcg"][topk],
                val_metrics["recall"][topk] - val_main["recall"][topk],
                val_metrics["ndcg"][topk] - val_main["ndcg"][topk],
                test_main["recall"][topk], test_main["ndcg"][topk],
                test_metrics["recall"][topk], test_metrics["ndcg"][topk],
                test_metrics["recall"][topk] - test_main["recall"][topk],
                test_metrics["ndcg"][topk] - test_main["ndcg"][topk],
            )
        )
        run.add_scalar("final_once/Val_recall_%d" % topk, val_metrics["recall"][topk], 0)
        run.add_scalar("final_once/Val_ndcg_%d" % topk, val_metrics["ndcg"][topk], 0)
        run.add_scalar("final_once/Test_recall_%d" % topk, test_metrics["recall"][topk], 0)
        run.add_scalar("final_once/Test_ndcg_%d" % topk, test_metrics["ndcg"][topk], 0)
    lines.append(banner)

    for line in lines:
        print(line)
    with open(log_path, "a") as log:
        for line in lines:
            log.write(line + "\n")


def print_final_best(conf, best_perform, best_epoch, log_path):
    banner = "=" * 88
    if conf["model"] == "MultiCBR":
        header = "TRAINING FINISHED - FINAL VALIDATION-SELECTED BEST"
    else:
        header = "TRAINING FINISHED - FINAL VALIDATION-SELECTED BEST"
    lines = ["", banner, header, "Best epoch: %d" % best_epoch]

    if not best_perform["val"]:
        lines.append("No best checkpoint was recorded.")
    else:
        for topk in sorted(best_perform["val"]):
            lines.append(best_perform["val"][topk])
            lines.append(best_perform["test"][topk])

    lines.append(banner)

    for line in lines:
        print(line)

    with open(log_path, "a") as log:
        for line in lines:
            log.write(line + "\n")


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}

    for topk in conf["topk"]:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0

    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}
    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" % (m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" % (m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (
        curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk]
    )
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (
        curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk]
    )

    log = open(log_path, "a")
    log.write("%s\n" % val_str)
    log.write("%s\n" % test_str)
    log.close()

    print(val_str)
    print(test_str)


def init_gfr_selector_state(checkpoint_model_path):
    """Validation selector for the complete deployable CFDGN model.

    CFDGN is evaluated as MultiCBR + innovation, so internal Main is
    attribution-only and never gates checkpoint eligibility.  The primary score
    jointly covers the paper-critical @20 and @40 Recall/NDCG metrics; @10 is a
    deterministic exact-tie breaker.  Test remains sealed.
    """
    return {
        "best_joint_20_40": float("-inf"),
        "best_tie_10": float("-inf"),
        "selected_epoch": None,
    }


def cleanup_gfr_selector_state(state):
    # No temporary candidate frontier exists in the decoupled selector.
    return



def _log_metrics_multicbr_official(conf, model, metrics, log_path,
                                    checkpoint_model_path, checkpoint_conf_path,
                                    epoch, best_metrics, best_perform, best_epoch):
    """
    EXACT original MultiCBR best-checkpoint rule.

    IMPORTANT: CFDGN selector changes must never alter this branch.
    """
    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" % topk_)

    # EXACT official best-checkpoint rule.
    if (
        metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_]
        and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]
    ):
        torch.save(model.state_dict(), checkpoint_model_path)

        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))

        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for topk in conf["topk"]:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = (
                "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f"
                % (
                    curr_time, best_epoch, topk,
                    best_metrics["test"]["recall"][topk],
                    best_metrics["test"]["ndcg"][topk],
                )
            )
            best_perform["val"][topk] = (
                "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f"
                % (
                    curr_time, best_epoch, topk,
                    best_metrics["val"]["recall"][topk],
                    best_metrics["val"]["ndcg"][topk],
                )
            )

            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()
    return best_metrics, best_perform, best_epoch

def _log_metrics_cfdgn_unified_final(
        conf, model, metrics, log_path,
        checkpoint_model_path, checkpoint_conf_path,
        epoch, batch_anchor, best_metrics, best_perform, best_epoch,
        state):
    """Select the complete Final model on Validation @20+@40 jointly."""
    final = metrics["val"]
    main = metrics["val_main"]
    r20 = float(final["recall"][20]); n20 = float(final["ndcg"][20])
    r40 = float(final["recall"][40]); n40 = float(final["ndcg"][40])
    joint = r20 + n20 + r40 + n40
    tie10 = float(final["recall"].get(10, 0.0)) + float(final["ndcg"].get(10, 0.0))
    best_joint = float(state["best_joint_20_40"])
    best_tie = float(state["best_tie_10"])
    eps = 1e-12
    improve = (
        joint > best_joint + eps
        or (abs(joint - best_joint) <= eps and tie10 > best_tie + eps)
    )

    print(
        "[GFR-BEST-VAL] Complete-Final joint selector: "
        "Final R20/N20=%.6f/%.6f R40/N40=%.6f/%.6f | "
        "joint=%.6f tie10=%.6f | Main20=%.6f/%.6f | "
        "bestJoint=%.6f | improve=%s"
        % (
            r20, n20, r40, n40, joint, tie10,
            float(main["recall"][20]), float(main["ndcg"][20]),
            best_joint, "YES" if improve else "NO",
        )
    )

    if not improve:
        return best_metrics, best_perform, best_epoch, state

    torch.save(model.state_dict(), checkpoint_model_path)
    dump_conf = dict(conf)
    dump_conf["gfr_epoch_selection"] = "CompleteFinal_Val_sum_RN_at20_and40"
    dump_conf["gfr_deployment"] = "full_catalog_Main_plus_calibrated_structural_correction"
    dump_conf["gfr_posthoc_fusion"] = "none"
    del dump_conf["device"]
    json.dump(dump_conf, open(checkpoint_conf_path, "w"))

    state["best_joint_20_40"] = joint
    state["best_tie_10"] = tie10
    state["selected_epoch"] = int(epoch)
    best_epoch = int(epoch)
    best_metrics = {"val": copy.deepcopy(final), "test": {}}
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_perform = {"val": {}, "test": {}}
    for topk in conf["topk"]:
        best_perform["val"][topk] = (
            "%s, Complete-Final-selected epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f"
            % (curr_time, best_epoch, topk,
               final["recall"][topk], final["ndcg"][topk])
        )
        print(best_perform["val"][topk])

    with open(log_path, "a") as log:
        for topk in conf["topk"]:
            log.write(best_perform["val"][topk] + "\n")
    return best_metrics, best_perform, best_epoch, state


def write_gfr_val_log(run, log_path, topk, step, val_scores):
    """CFDGN in-run logging: Validation only. Test remains sealed."""
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for m, val_score in val_scores.items():
        run.add_scalar("%s_%d/Val" % (m, topk), val_score[topk], step)
    val_str = "%s, Top_%d, Val: recall: %f, ndcg: %f" % (
        curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk]
    )
    with open(log_path, "a") as log:
        log.write(val_str + "\n")
    print(val_str)


def write_gfr_val_main_compare(run, log_path, topk, step, final_scores, main_scores):
    """Log final-vs-isolated-Main validation attribution without touching Test."""
    dr = float(final_scores["recall"][topk] - main_scores["recall"][topk])
    dn = float(final_scores["ndcg"][topk] - main_scores["ndcg"][topk])
    run.add_scalar("gfr_attribution_%d/Main_Val_recall" % topk, main_scores["recall"][topk], step)
    run.add_scalar("gfr_attribution_%d/Main_Val_ndcg" % topk, main_scores["ndcg"][topk], step)
    run.add_scalar("gfr_attribution_%d/FinalMinusMain_recall" % topk, dr, step)
    run.add_scalar("gfr_attribution_%d/FinalMinusMain_ndcg" % topk, dn, step)
    line = (
        "[GFR-ATTR-VAL] TOP %d: Main R=%.6f N=%.6f | Final R=%.6f N=%.6f | dR=%+.6f dN=%+.6f"
        % (topk, main_scores["recall"][topk], main_scores["ndcg"][topk],
           final_scores["recall"][topk], final_scores["ndcg"][topk], dr, dn)
    )
    print(line)
    with open(log_path, "a") as log:
        log.write(line + "\n")


def write_gfr_unified_diag(run, log_path, step, diag):
    """Full-catalog calibrated-correction diagnostics on Validation."""
    lines = []
    line = (
        "[GFR-CORR-VAL] alpha=%.6f | sigmaMain=%.6f | alphaSigma=%.6f | "
        "correction_std=%.6f | abs=%.6f | sat95OfCap=%.6f | "
        "posMinusFalse=%.6f | noHistory=%.4f"
        % (
            float(diag.get("residual_scale", float("nan"))),
            float(diag.get("main_score_scale", float("nan"))),
            float(diag.get("deploy_multiplier", float("nan"))),
            float(diag.get("residual_std", float("nan"))),
            float(diag.get("residual_abs_mean", float("nan"))),
            float(diag.get("residual_sat95", float("nan"))),
            float(diag.get("residual_pos_false_gap", float("nan"))),
            float(diag.get("no_history_frac", float("nan"))),
        )
    )
    print(line); lines.append(line)

    line2 = (
        "[GFR-TYPE-VAL] mean(sim/comp/noise)=%.4f/%.4f/%.4f | "
        "argmax=%.4f/%.4f/%.4f | typeH=%.4f | D=%.4f targetD=%.4f "
        "roleAffDiag=%.4f/%.4f S=%.4f C=%.4f"
        % (
            float(diag.get("type_sim_mean", float("nan"))),
            float(diag.get("type_comp_mean", float("nan"))),
            float(diag.get("type_noise_mean", float("nan"))),
            float(diag.get("type_sim_argmax", float("nan"))),
            float(diag.get("type_comp_argmax", float("nan"))),
            float(diag.get("type_noise_argmax", float("nan"))),
            float(diag.get("type_entropy", float("nan"))),
            float(diag.get("role_diversity", float("nan"))),
            float(diag.get("role_diversity_target", float("nan"))),
            float(diag.get("role_affinity_pred", float("nan"))),
            float(diag.get("role_affinity_target", float("nan"))),
            float(diag.get("role_certainty", float("nan"))),
            float(diag.get("relation_compatibility", float("nan"))),
        )
    )
    print(line2); lines.append(line2)

    line_sem = (
        "[GFR-TYPE-SEM] valid=%.4f | std(sim/comp/noise)=%.5f/%.5f/%.5f | margin=%.5f | "
        "softD(sim/comp/noise)=%.4f/%.4f/%.4f | "
        "softC(sim/comp/noise)=%.4f/%.4f/%.4f | semanticOrder=%s"
        % (
            float(diag.get("type_valid_frac", float("nan"))),
            float(diag.get("type_std_sim", float("nan"))),
            float(diag.get("type_std_comp", float("nan"))),
            float(diag.get("type_std_noise", float("nan"))),
            float(diag.get("type_margin", float("nan"))),
            float(diag.get("type_soft_d_sim", float("nan"))),
            float(diag.get("type_soft_d_comp", float("nan"))),
            float(diag.get("type_soft_d_noise", float("nan"))),
            float(diag.get("type_soft_c_sim", float("nan"))),
            float(diag.get("type_soft_c_comp", float("nan"))),
            float(diag.get("type_soft_c_noise", float("nan"))),
            "PASS" if float(diag.get("type_semantic_order", 0.0)) >= 0.5 else "FAIL",
        )
    )
    print(line_sem); lines.append(line_sem)

    line3 = (
        "[GFR-PPMI-VAL] local_active=%.4f | active_role_shift=%.6f | "
        "active_component_std(facet/bundle/type/comp)=%.5f/%.5f/%.5f/%.5f | "
        "actual_contrib_std=%.5f/%.5f/%.5f/%.5f"
        % (
            float(diag.get("ppmi_local_active", float("nan"))),
            float(diag.get("ppmi_active_role_shift", float("nan"))),
            float(diag.get("component_std_facet", float("nan"))),
            float(diag.get("component_std_bundle", float("nan"))),
            float(diag.get("component_std_type", float("nan"))),
            float(diag.get("component_std_comp", float("nan"))),
            float(diag.get("component_contrib_std_facet", float("nan"))),
            float(diag.get("component_contrib_std_bundle", float("nan"))),
            float(diag.get("component_contrib_std_type", float("nan"))),
            float(diag.get("component_contrib_std_comp", float("nan"))),
        )
    )
    print(line3); lines.append(line3)

    with open(log_path, "a") as log:
        for x in lines:
            log.write(x + "\n")
    for k, v in diag.items():
        if isinstance(v, (int, float)):
            run.add_scalar("gfr_unified_val/" + k, float(v), step)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path,
                checkpoint_conf_path, epoch, batch_anchor,
                best_metrics, best_perform, best_epoch,
                gfr_selector_state=None):
    if conf["model"] == "MultiCBR":
        # STRICTLY preserve original MultiCBR reporting/selection behavior.
        for topk in conf["topk"]:
            write_log(run, log_path, topk, batch_anchor, metrics)
        best_metrics, best_perform, best_epoch = _log_metrics_multicbr_official(
            conf, model, metrics, log_path,
            checkpoint_model_path, checkpoint_conf_path,
            epoch, best_metrics, best_perform, best_epoch
        )
        return best_metrics, best_perform, best_epoch, gfr_selector_state

    if conf["model"] == "CFDGN":
        if gfr_selector_state is None:
            raise RuntimeError("CFDGN selector state was not initialized")
        if "test" in metrics and metrics["test"] is not None:
            raise RuntimeError("CFDGN in-run selector must not receive Test metrics")
        if "val_main" not in metrics:
            raise RuntimeError("CFDGN attribution requires isolated Main validation metrics")
        for topk in conf["topk"]:
            write_gfr_val_log(run, log_path, topk, batch_anchor, metrics["val"])
            write_gfr_val_main_compare(
                run, log_path, topk, batch_anchor, metrics["val"], metrics["val_main"]
            )
        if "val_unified_diag" not in metrics:
            raise RuntimeError("CFDGN unified full-catalog diagnostics missing")
        write_gfr_unified_diag(
            run, log_path, batch_anchor, metrics["val_unified_diag"]
        )
        return _log_metrics_cfdgn_unified_final(
            conf, model, metrics, log_path,
            checkpoint_model_path, checkpoint_conf_path,
            epoch, batch_anchor,
            best_metrics, best_perform, best_epoch,
            gfr_selector_state
        )

    raise ValueError("Unimplemented model %s" % conf["model"])


# The following evaluation functions are copied from the official MultiCBR train.py
# without metric-definition changes.
def test(model, dataloader, conf, propagate_result=None):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    with torch.no_grad():
        rs = (
            model.get_multi_modal_representations(test=True)
            if propagate_result is None else propagate_result
        )
        for users, ground_truth_u_b, train_mask_u_b in dataloader:
            pred_b = model.evaluate(rs, users.to(device))
            pred_b -= 1e8 * train_mask_u_b.to(device)
            tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics

def _fresh_gfr_metric(conf):
    return {
        "recall": {int(k): [0.0, 0] for k in conf["topk"]},
        "ndcg": {int(k): [0.0, 0] for k in conf["topk"]},
    }


def _reduce_gfr_metric(tmp):
    out = {"recall": {}, "ndcg": {}}
    for metric in ("recall", "ndcg"):
        for k, pair in tmp[metric].items():
            out[metric][k] = float(pair[0]) / max(int(pair[1]), 1)
    return out


def test_cfdgn_unified(model, dataloader, conf, propagate_result):
    """Evaluate exact full-catalog Main and unified Final scores.

    Test/Validation semantics are identical.  The only dataset-specific input is
    the official TRAIN mask supplied by the repository dataloader.  No candidate
    retrieval or score calibration occurs here.
    """
    if not isinstance(propagate_result, dict):
        raise TypeError("CFDGN unified evaluation requires dict representation state")
    tmp_final = _fresh_gfr_metric(conf)
    tmp_main = _fresh_gfr_metric(conf)
    device = conf["device"]
    model.eval()

    residual_sum = 0.0
    residual_sq_sum = 0.0
    residual_abs_sum = 0.0
    residual_sat95_sum = 0.0
    residual_scale = float(model._residual_scale().detach().cpu())
    main_score_scale = float(model._main_score_scale().detach().cpu())
    deploy_multiplier = residual_scale * main_score_scale
    residual_sat95_threshold = 0.95 * deploy_multiplier
    residual_count = 0
    pos_false_gap_sum = 0.0
    pos_false_gap_count = 0
    no_history_count = 0
    user_count = 0
    comp_sum = torch.zeros(4, dtype=torch.float64)
    comp_sq_sum = torch.zeros(4, dtype=torch.float64)
    comp_count = 0

    with torch.no_grad():
        for users, ground_truth_u_b, train_mask_u_b in dataloader:
            users_dev = users.to(device)
            mask_dev = train_mask_u_b.to(device=device, dtype=torch.bool)
            out = model.evaluate_unified(
                propagate_result, users_dev, mask_dev
            )
            tmp_final = get_metrics(
                tmp_final, ground_truth_u_b, out["final"], conf["topk"]
            )
            tmp_main = get_metrics(
                tmp_main, ground_truth_u_b, out["main"], conf["topk"]
            )

            residual = out["residual"]
            has = out["has_history"].to(torch.bool)
            visible = (~mask_dev) & has.unsqueeze(-1)
            if bool(visible.any()):
                rv = residual[visible]
                residual_sum += float(rv.sum().item())
                residual_sq_sum += float(rv.square().sum().item())
                residual_abs_sum += float(rv.abs().sum().item())
                residual_sat95_sum += float(
                    (rv.abs() >= residual_sat95_threshold).sum().item()
                )
                residual_count += int(rv.numel())

            # Diagnostic only: average residual preference for held-out positives
            # versus TRAIN-unseen false bundles.  This is NOT an optimization
            # target and does not relabel unobserved bundles during training.
            gt = ground_truth_u_b.to(device=device, dtype=torch.bool)
            for r in range(residual.shape[0]):
                if not bool(has[r]):
                    continue
                pos = gt[r] & (~mask_dev[r])
                false = (~gt[r]) & (~mask_dev[r])
                if bool(pos.any()) and bool(false.any()):
                    gap = residual[r, pos].mean() - residual[r, false].mean()
                    pos_false_gap_sum += float(gap.item())
                    pos_false_gap_count += 1

            no_history_count += int((~has).sum().item())
            user_count += int(has.numel())
            comp_sum += out["component_sum"].detach().cpu().double()
            comp_sq_sum += out["component_sq_sum"].detach().cpu().double()
            comp_count += int(out["component_count"])

    final_metrics = _reduce_gfr_metric(tmp_final)
    main_metrics = _reduce_gfr_metric(tmp_main)

    if residual_count > 0:
        mean = residual_sum / residual_count
        var = max(residual_sq_sum / residual_count - mean * mean, 0.0)
        residual_std = var ** 0.5
        residual_abs = residual_abs_sum / residual_count
        residual_sat95 = residual_sat95_sum / residual_count
    else:
        residual_std = residual_abs = residual_sat95 = 0.0

    if comp_count > 0:
        cm = comp_sum / float(comp_count)
        cv = (comp_sq_sum / float(comp_count) - cm.square()).clamp_min(0.0)
        cs = cv.sqrt()
    else:
        cs = torch.zeros(4, dtype=torch.float64)

    # Bundle-structural diagnostics are global properties of the frozen
    # representation state, so compute them exactly once rather than repeating
    # them per user batch. Paper-facing type diagnostics exclude singleton/
    # invalid bundles, whose TypeHead output is intentionally uniform.
    bs = propagate_result["bundle_state"]
    type_valid = bs["type_valid"].detach().cpu().to(torch.bool)
    tp_all = bs["type_prob"].detach().cpu().double()
    tp = tp_all[type_valid]
    am = tp.argmax(dim=-1) if tp.numel() else torch.empty(0, dtype=torch.long)
    type_argmax = torch.bincount(am, minlength=3).double() / max(tp.shape[0], 1)
    type_mean = tp.mean(dim=0) if tp.numel() else torch.zeros(3, dtype=torch.float64)
    type_entropy = float((
        -(tp.clamp_min(1e-12) * tp.clamp_min(1e-12).log()).sum(dim=-1)
        / torch.log(torch.tensor(3.0, dtype=torch.float64))
    ).mean().item()) if tp.numel() else 0.0
    type_std = tp.std(dim=0, unbiased=False) if tp.numel() else torch.zeros(3, dtype=torch.float64)
    if tp.numel():
        top2 = torch.topk(tp, k=2, dim=-1).values
        type_margin = float((top2[:, 0] - top2[:, 1]).mean().item())
        d_valid = bs["diversity"].detach().cpu().double()[type_valid]
        c_valid = bs["compatibility"].detach().cpu().double()[type_valid]
        mass = tp.sum(dim=0).clamp_min(1e-12)
        soft_d = (tp * d_valid.unsqueeze(-1)).sum(dim=0) / mass
        soft_c = (tp * c_valid.unsqueeze(-1)).sum(dim=0) / mass
        semantic_order_ok = bool(
            soft_d[0] < soft_d[1]
            and soft_d[0] < soft_d[2]
            and soft_c[2] < soft_c[0]
            and soft_c[2] < soft_c[1]
        )
    else:
        type_margin = 0.0
        soft_d = torch.zeros(3, dtype=torch.float64)
        soft_c = torch.zeros(3, dtype=torch.float64)
        semantic_order_ok = False

    rav = bs["role_affinity_target_bundle_valid"].detach().cpu().to(torch.bool)
    if bool(rav.any()):
        role_aff_pred = float(bs["role_affinity_pred_mean"].detach().cpu()[rav].mean().item())
        role_aff_tgt = float(bs["role_affinity_target_mean"].detach().cpu()[rav].mean().item())
    else:
        role_aff_pred = role_aff_tgt = 0.0

    ppmi_active = bs["ppmi_active_fraction"].detach().cpu()
    ppmi_shift = bs["ppmi_role_shift"].detach().cpu()
    active = ppmi_active > 0
    active_shift = float(ppmi_shift[active].mean().item()) if bool(active.any()) else 0.0

    component_contrib_std = cs * (0.25 * deploy_multiplier)
    diag = {
        "residual_scale": residual_scale,
        "main_score_scale": main_score_scale,
        "deploy_multiplier": deploy_multiplier,
        "residual_std": float(residual_std),
        "residual_abs_mean": float(residual_abs),
        "residual_sat95": float(residual_sat95),
        "residual_pos_false_gap": float(pos_false_gap_sum / max(pos_false_gap_count, 1)),
        "no_history_frac": float(no_history_count / max(user_count, 1)),
        "component_std_facet": float(cs[0]),
        "component_std_bundle": float(cs[1]),
        "component_std_type": float(cs[2]),
        "component_std_comp": float(cs[3]),
        "component_contrib_std_facet": float(component_contrib_std[0]),
        "component_contrib_std_bundle": float(component_contrib_std[1]),
        "component_contrib_std_type": float(component_contrib_std[2]),
        "component_contrib_std_comp": float(component_contrib_std[3]),
        "type_sim_mean": float(type_mean[0]),
        "type_comp_mean": float(type_mean[1]),
        "type_noise_mean": float(type_mean[2]),
        "type_sim_argmax": float(type_argmax[0]),
        "type_comp_argmax": float(type_argmax[1]),
        "type_noise_argmax": float(type_argmax[2]),
        "type_entropy": float(type_entropy),
        "type_valid_frac": float(type_valid.double().mean().item()) if type_valid.numel() else 0.0,
        "type_std_sim": float(type_std[0]),
        "type_std_comp": float(type_std[1]),
        "type_std_noise": float(type_std[2]),
        "type_margin": float(type_margin),
        "type_soft_d_sim": float(soft_d[0]),
        "type_soft_d_comp": float(soft_d[1]),
        "type_soft_d_noise": float(soft_d[2]),
        "type_soft_c_sim": float(soft_c[0]),
        "type_soft_c_comp": float(soft_c[1]),
        "type_soft_c_noise": float(soft_c[2]),
        "type_semantic_order": 1.0 if semantic_order_ok else 0.0,
        "role_diversity": float(bs["diversity"][bs["type_valid"]].mean().detach().cpu())
            if bool(bs["type_valid"].any()) else 0.0,
        "role_diversity_target": float(
            bs["semantic_diversity_target"][bs["semantic_diversity_target_valid"]].mean().detach().cpu()
            if bool(bs["semantic_diversity_target_valid"].any()) else 0.0
        ),
        "role_affinity_pred": role_aff_pred,
        "role_affinity_target": role_aff_tgt,
        "role_certainty": float(bs["certainty"].mean().detach().cpu()),
        "relation_compatibility": float(bs["compatibility"].mean().detach().cpu()),
        "ppmi_local_active": float(ppmi_active.mean().item()),
        "ppmi_active_role_shift": active_shift,
    }
    return final_metrics, main_metrics, diag


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}

    for topk in topks:
        # pred is on GPU during evaluation, while ground-truth tensors returned
        # by the official BundleTestDataset remain on CPU.  Keep the official
        # metric computation on CPU and move only the ranking indices to CPU.
        _, col_indice = torch.topk(pred, topk)
        col_indice = col_indice.cpu()
        row_indice = (
            torch.zeros_like(col_indice)
            + torch.arange(pred.shape[0], dtype=torch.long).view(-1, 1)
        )
        is_hit = grd[
            row_indice.view(-1), col_indice.view(-1)
        ].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()
    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit / torch.log2(
            torch.arange(2, topk + 2, device=device, dtype=torch.float)
        )
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1

    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()
    return [nomina, denorm]


if __name__ == "__main__":
    main()
