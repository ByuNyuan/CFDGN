from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_bpr_loss(pred):
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs))
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class MultiCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        self.fusion_weights = conf['fusion_weights']

        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)

        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()


    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.conf["UB_ratio"], True)
        self.UI_dropout = nn.Dropout(self.conf["UI_ratio"], True)
        self.BI_dropout = nn.Dropout(self.conf["BI_ratio"], True)
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }


    def init_noise_eps(self):
        self.UB_eps = self.conf["UB_ratio"]
        self.UI_eps = self.conf["UI_ratio"]
        self.BI_eps = self.conf["BI_ratio"]
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def init_fusion_weights(self):
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
            "The number of layer fusion weights does not correspond to number of layers"

        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)


    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)


    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)


    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.conf["aug_type"] == "Noise" and not test:
                random_noise = torch.rand_like(features).to(self.device)
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        if self.conf["aug_type"] == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.conf["aug_type"] == "Noise" and not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature


    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        users_feature = torch.stack(users_feature, dim=0)
        bundles_feature = torch.stack(bundles_feature, dim=0)

        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        return users_rep, bundles_rep


    def get_multi_modal_representations(self, test=False):
        if test:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        if test:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
        else:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)

        if test:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
        else:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]

        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)

        return users_rep, bundles_rep


    def cal_c_loss(self, pos, aug):
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1, 0))

        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)

        c_losses = [u_view_cl, b_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        users, bundles = batch
        users_rep, bundles_rep = self.get_multi_modal_representations()

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores

EPS = 1e-12


def _normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    if p.shape[-1] < 2:
        raise ValueError("categorical dimension must be >=2")
    q = p.clamp_min(EPS)
    return (-(q * q.log()).sum(dim=-1) / math.log(float(p.shape[-1]))).clamp(0.0, 1.0)


def _center_rms_normalize(x: torch.Tensor) -> torch.Tensor:
    y = x - x.mean(dim=-1, keepdim=True)
    rms = torch.sqrt((y * y).mean(dim=-1, keepdim=True) + EPS)
    return y / rms


def _to_sparse_tensor(A: sp.spmatrix) -> torch.Tensor:
    coo = A.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, tuple(A.shape)).coalesce()


def _row_normalize_csr(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocsr().astype(np.float64)
    s = np.asarray(A.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(s, dtype=np.float64)
    nz = s > 0
    inv[nz] = 1.0 / s[nz]
    out = sp.diags(inv) @ A
    out.eliminate_zeros()
    return out.tocsr()


def build_fixed_ui_bundle_diversity_target(
    ui_graph: sp.spmatrix,
    bi_graph: sp.spmatrix,
    train_bundle_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    U = ui_graph.tocsr().astype(np.float64).copy()
    B = bi_graph.tocsr().astype(np.float64).copy()
    train_bundle_mask = np.asarray(train_bundle_mask, dtype=np.bool_).reshape(-1)
    if train_bundle_mask.shape[0] != B.shape[0]:
        raise ValueError("train_bundle_mask length must equal number of bundles")
    if U.nnz:
        U.data[:] = 1.0
    if B.nnz:
        B.data[:] = 1.0

    item_deg = np.asarray(U.sum(axis=0)).reshape(-1)
    valid_item = item_deg > 0.0
    inv_sqrt = np.zeros_like(item_deg, dtype=np.float64)
    inv_sqrt[valid_item] = 1.0 / np.sqrt(item_deg[valid_item])
    U_norm = (U @ sp.diags(inv_sqrt)).tocsr()

    B_valid = (B @ sp.diags(valid_item.astype(np.float64))).tocsr()
    n = np.asarray(B_valid.sum(axis=1)).reshape(-1)

    agg = (B_valid @ U_norm.T).tocsr()
    sq_norm = np.asarray(agg.multiply(agg).sum(axis=1)).reshape(-1)
    denom = n * (n - 1.0)
    raw_mean_cos = np.zeros(B.shape[0], dtype=np.float64)
    good = n >= 2.0
    raw_mean_cos[good] = (sq_norm[good] - n[good]) / np.maximum(denom[good], 1.0)
    raw_mean_cos = np.clip(raw_mean_cos, 0.0, 1.0)

    calibration_good = good & train_bundle_mask
    reference = np.sort(raw_mean_cos[calibration_good])
    if reference.size == 0:
        raise RuntimeError("no TRAIN-observed bundle has >=2 items with U-I evidence")

    values = raw_mean_cos[good]
    left = np.searchsorted(reference, values, side="left").astype(np.float64)
    right = np.searchsorted(reference, values, side="right").astype(np.float64)
    percentile = (left + right) / (2.0 * float(reference.size))
    percentile = np.clip(percentile, 0.0, 1.0)

    target = np.zeros(B.shape[0], dtype=np.float32)
    target[good] = (1.0 - percentile).astype(np.float32)

    train_target = target[calibration_good]
    q10, q50, q90 = np.quantile(reference, [0.10, 0.50, 0.90])
    stats = {
        "valid_bundle_fraction": float(np.mean(good)) if good.size else 0.0,
        "train_calibration_bundle_fraction": float(np.mean(calibration_good)) if good.size else 0.0,
        "target_mean_valid": float(target[good].mean()) if np.any(good) else 0.0,
        "target_std_valid": float(target[good].std()) if np.any(good) else 0.0,
        "target_mean_train": float(train_target.mean()),
        "target_std_train": float(train_target.std()),
        "raw_cos_mean_train": float(reference.mean()),
        "raw_cos_q10_train": float(q10),
        "raw_cos_q50_train": float(q50),
        "raw_cos_q90_train": float(q90),
        "ui_items_with_evidence": int(valid_item.sum()),
    }
    return target, good.astype(np.bool_), stats


def build_fixed_ui_edge_role_affinity_target(
    ui_graph: sp.spmatrix,
    bi_graph: sp.spmatrix,
    train_bundle_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    U = ui_graph.tocsr().astype(np.float64).copy()
    B = bi_graph.tocsr().astype(np.float64).copy()
    train_bundle_mask = np.asarray(train_bundle_mask, dtype=np.bool_).reshape(-1)
    if train_bundle_mask.shape[0] != B.shape[0]:
        raise ValueError("train_bundle_mask length must equal number of bundles")
    if U.nnz:
        U.data[:] = 1.0
    if B.nnz:
        B.data[:] = 1.0

    item_deg = np.asarray(U.sum(axis=0)).reshape(-1)
    valid_item = item_deg > 0.0
    inv_sqrt = np.zeros_like(item_deg, dtype=np.float64)
    inv_sqrt[valid_item] = 1.0 / np.sqrt(item_deg[valid_item])
    U_norm = (U @ sp.diags(inv_sqrt)).tocsc()

    raw = np.zeros(B.nnz, dtype=np.float64)
    good = np.zeros(B.nnz, dtype=np.bool_)
    edge_bundle = np.repeat(np.arange(B.shape[0], dtype=np.int64), np.diff(B.indptr))

    for b in range(B.shape[0]):
        st, en = int(B.indptr[b]), int(B.indptr[b + 1])
        if en - st < 2:
            continue
        items = B.indices[st:en]
        local_valid = valid_item[items]
        if int(local_valid.sum()) < 2:
            continue
        valid_local_pos = np.flatnonzero(local_valid)
        valid_items = items[valid_local_pos]
        X = U_norm[:, valid_items]
        centroid = X.sum(axis=1)
        dots = np.asarray(X.T @ centroid).reshape(-1)
        n = float(valid_items.size)
        affinity = np.clip((dots - 1.0) / max(n - 1.0, 1.0), 0.0, 1.0)
        pos = st + valid_local_pos
        raw[pos] = affinity
        good[pos] = True

    calibration_good = good & train_bundle_mask[edge_bundle]
    reference = np.sort(raw[calibration_good])
    if reference.size == 0:
        raise RuntimeError("no TRAIN-observed B-I edge has a valid strict U-I context")

    values = raw[good]
    left = np.searchsorted(reference, values, side="left").astype(np.float64)
    right = np.searchsorted(reference, values, side="right").astype(np.float64)
    percentile = np.clip((left + right) / (2.0 * float(reference.size)), 0.0, 1.0)

    target = np.zeros(B.nnz, dtype=np.float32)
    target[good] = percentile.astype(np.float32)
    train_target = target[calibration_good]
    q10, q50, q90 = np.quantile(reference, [0.10, 0.50, 0.90])
    stats = {
        "valid_edge_fraction": float(good.mean()) if good.size else 0.0,
        "train_calibration_edge_fraction": float(calibration_good.mean()) if good.size else 0.0,
        "target_mean_train": float(train_target.mean()),
        "target_std_train": float(train_target.std()),
        "raw_affinity_mean_train": float(reference.mean()),
        "raw_affinity_q10_train": float(q10),
        "raw_affinity_q50_train": float(q50),
        "raw_affinity_q90_train": float(q90),
    }
    return target, good, stats


def build_ppmi_relation_prior(
    bi_graph: sp.spmatrix,
    num_items: int,
    topk: int = 20,
    min_count: int = 2,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, float]]:
    B = bi_graph.tocsr().astype(np.float64).copy()
    if B.nnz:
        B.data[:] = 1.0

    C = (B.T @ B).tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()

    if C.nnz == 0:
        empty = sp.csr_matrix((num_items, num_items), dtype=np.float64)
        return empty, C, {
            "nnz": 0,
            "isolated": int(num_items),
            "topk": int(topk),
            "min_count": int(min_count),
        }

    total = float(C.data.sum())
    row_mass = np.asarray(C.sum(axis=1)).reshape(-1)
    p_i = row_mass / max(total, EPS)

    rows, cols, vals = [], [], []
    kept_support = []
    for i in range(num_items):
        st, en = C.indptr[i], C.indptr[i + 1]
        if st == en or p_i[i] <= 0.0:
            continue
        js = C.indices[st:en]
        cs = C.data[st:en]
        p_ij = cs / total
        denom = p_i[i] * p_i[js]
        ppmi = np.maximum(np.log((p_ij + EPS) / (denom + EPS)), 0.0)
        keep = np.flatnonzero((cs >= float(min_count)) & (ppmi > 0.0))
        if keep.size == 0:
            continue
        if keep.size > int(topk):
            keep = keep[np.argpartition(-ppmi[keep], int(topk) - 1)[: int(topk)]]
        keep = keep[np.lexsort((js[keep], -ppmi[keep]))]
        rows.extend([i] * int(keep.size))
        cols.extend(js[keep].astype(np.int64).tolist())
        vals.extend(ppmi[keep].astype(np.float64).tolist())
        kept_support.extend(cs[keep].astype(np.int64).tolist())

    A = sp.coo_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(num_items, num_items),
    ).tocsr()
    A = A.maximum(A.T).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()

    if A.diagonal().any():
        raise RuntimeError("PPMI self-edge seal failed")

    support = np.asarray(kept_support, dtype=np.int64)
    row_nnz = A.getnnz(axis=1)
    return A, C, {
        "nnz": int(A.nnz),
        "isolated": int(np.sum(row_nnz == 0)),
        "topk": int(topk),
        "min_count": int(min_count),
        "support_mean": float(support.mean()) if support.size else 0.0,
        "support_min": int(support.min()) if support.size else 0,
    }


def build_strict_current_bundle_relation(
    A_ppmi: sp.csr_matrix,
    bi_graph: sp.spmatrix,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict[str, float]]:
    B = bi_graph.tocsr()
    rows, cols, vals = [], [], []
    active = np.zeros(B.nnz, dtype=bool)
    linked = np.zeros(B.nnz, dtype=np.int64)

    edge = 0
    for b in range(B.shape[0]):
        members = B.indices[B.indptr[b]:B.indptr[b + 1]]
        member_set = set(map(int, members.tolist()))
        for i in members.tolist():
            st, en = A_ppmi.indptr[i], A_ppmi.indptr[i + 1]
            local_cols, local_vals = [], []
            for j, w in zip(A_ppmi.indices[st:en], A_ppmi.data[st:en]):
                j = int(j)
                if j == int(i) or j not in member_set or float(w) <= 0.0:
                    continue
                local_cols.append(j)
                local_vals.append(float(w))
            if local_vals:
                z = max(float(np.sum(local_vals)), EPS)
                for j, w in zip(local_cols, local_vals):
                    rows.append(edge)
                    cols.append(j)
                    vals.append(w / z)
                active[edge] = True
                linked[edge] = len(local_vals)
            edge += 1

    if edge != B.nnz:
        raise RuntimeError("B-I CSR edge accounting mismatch")

    R = sp.coo_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(B.nnz, B.shape[1]),
    ).tocsr()

    if R.nnz:
        coo = R.tocoo()
        edge_b = np.repeat(np.arange(B.shape[0], dtype=np.int64), np.diff(B.indptr))
        edge_i = B.indices.astype(np.int64)
        for rr, cc in zip(coo.row.tolist(), coo.col.tolist()):
            b = int(edge_b[rr])
            i = int(edge_i[rr])
            if int(cc) == i:
                raise RuntimeError("strict B\\{i} self leakage")
            members = B.indices[B.indptr[b]:B.indptr[b + 1]]
            if int(cc) not in set(map(int, members.tolist())):
                raise RuntimeError("PPMI relation introduced non-member")
        sums = np.asarray(R.sum(axis=1)).reshape(-1)
        err = float(np.max(np.abs(sums[active] - 1.0))) if active.any() else 0.0
        if err > 2e-6:
            raise RuntimeError(f"strict relation row-normalization error={err}")
    else:
        err = 0.0

    return R, active, {
        "active_fraction": float(active.mean()) if active.size else 0.0,
        "active_edges": int(active.sum()),
        "relation_nnz": int(R.nnz),
        "linked_mean_active": float(linked[active].mean()) if active.any() else 0.0,
        "row_norm_error": err,
    }


def build_bundle_ppmi_evidence(
    A_ppmi: sp.csr_matrix,
    bi_graph: sp.spmatrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = bi_graph.tocsr()
    coverage = np.zeros(B.shape[0], dtype=np.float32)
    strength = np.zeros(B.shape[0], dtype=np.float32)
    evidence = np.zeros(B.shape[0], dtype=np.float32)

    for b in range(B.shape[0]):
        members = B.indices[B.indptr[b]:B.indptr[b + 1]]
        n = len(members)
        if n < 2:
            continue
        total_pairs = n * (n - 1) / 2.0
        sub = A_ppmi[members][:, members].tocoo()
        mask = sub.row < sub.col
        vals = sub.data[mask]
        linked = int(vals.size)
        coverage[b] = float(linked / max(total_pairs, 1.0))
        if linked:
            strength[b] = float(np.mean(1.0 - np.exp(-vals)))
            evidence[b] = coverage[b] * strength[b]

    return coverage, strength, evidence


def build_relation_pair_bank(
    A_ppmi: sp.csr_matrix,
    C_cooccur: sp.csr_matrix,
    num_items: int,
    bank_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upper = sp.triu(A_ppmi, k=1).tocoo()
    pos_i = upper.row.astype(np.int64)
    pos_j = upper.col.astype(np.int64)

    if pos_i.size == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z, z, z

    rng = np.random.default_rng(int(seed))
    max_bank = min(int(bank_size), int(pos_i.size))
    if pos_i.size > max_bank:
        idx = rng.choice(pos_i.size, size=max_bank, replace=False)
        pos_i = pos_i[idx]
        pos_j = pos_j[idx]

    need = int(pos_i.size)
    neg_i_parts, neg_j_parts = [], []
    got = 0
    while got < need:
        draw = max(4096, 3 * (need - got))
        ii = rng.integers(0, num_items, size=draw, dtype=np.int64)
        jj = rng.integers(0, num_items, size=draw, dtype=np.int64)
        neq = ii != jj
        ii, jj = ii[neq], jj[neq]
        if ii.size == 0:
            continue
        co = np.asarray(C_cooccur[ii, jj]).reshape(-1)
        keep = co == 0
        ii, jj = ii[keep], jj[keep]
        if ii.size:
            take = min(int(ii.size), need - got)
            neg_i_parts.append(ii[:take])
            neg_j_parts.append(jj[:take])
            got += take

    neg_i = np.concatenate(neg_i_parts).astype(np.int64)
    neg_j = np.concatenate(neg_j_parts).astype(np.int64)
    return pos_i, pos_j, neg_i, neg_j


def _masked_bpr(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if score.ndim != 2 or score.shape[1] < 2:
        raise ValueError("score must be [B,1+neg]")
    mask = mask.to(torch.bool)
    if bool(mask.any()):
        return cal_bpr_loss(score[mask])
    return score.sum() * 0.0


class PPMIFacetEnricher(nn.Module):
    def __init__(self, weight: float, tau: float):
        super().__init__()
        if weight < 0.0:
            raise ValueError("PPMI enrichment weight must be >=0")
        if tau <= 0.0:
            raise ValueError("PPMI enrichment tau must be >0")
        self.weight = float(weight)
        self.tau = float(tau)

    def forward(self, facets: torch.Tensor, ppmi_row_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, k, d = facets.shape
        flat = facets.reshape(n, k * d)
        msg = torch.sparse.mm(ppmi_row_norm, flat).reshape(n, k, d)
        msg_norm = F.normalize(msg, p=2, dim=-1, eps=EPS)
        gate = torch.sigmoid((facets * msg_norm).sum(dim=-1) / self.tau)
        has_msg = (torch.linalg.vector_norm(msg, dim=-1) > EPS).to(gate.dtype)
        gate = gate * has_msg
        enriched = F.normalize(
            facets + self.weight * gate.unsqueeze(-1) * msg_norm,
            p=2, dim=-1, eps=EPS,
        )
        return enriched, gate


class ContextualRoleRouter(nn.Module):
    def __init__(
        self,
        k: int,
        tau: float,
        prior_weight: float,
        specialization_weight: float,
        context_weight_legacy: float,
        ppmi_weight: float,
    ):
        super().__init__()
        self.k = int(k)
        self.tau = float(tau)
        if self.k < 2 or self.tau <= 0.0:
            raise ValueError("router requires K>=2 and tau>0")

        intrinsic_weight = float(prior_weight) + float(context_weight_legacy)
        base = torch.tensor(
            [intrinsic_weight, float(specialization_weight), float(ppmi_weight)],
            dtype=torch.float32,
        )
        if bool((base < 0).any()) or float(base.sum()) <= 0.0:
            raise ValueError("router source weights must be non-negative with positive sum")
        if float(ppmi_weight) <= 0.0:
            raise ValueError("active PPMI must have a fixed positive router contribution")
        self.register_buffer("base_source_weight", base, persistent=True)

    def forward(
        self,
        item_activation: torch.Tensor,
        plain_context_activation: torch.Tensor,
        ppmi_context_activation: torch.Tensor,
        has_other: torch.Tensor,
        has_ppmi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if item_activation.ndim != 2 or item_activation.shape[1] != self.k:
            raise ValueError("item_activation must be [E,K]")
        if plain_context_activation.shape != item_activation.shape:
            raise ValueError("plain_context_activation must match item_activation")
        if ppmi_context_activation.shape != item_activation.shape:
            raise ValueError("ppmi_context_activation must match item_activation")

        prior = _center_rms_normalize(
            torch.log1p(item_activation.clamp_min(0.0))
        )
        ctx_occ = _center_rms_normalize(
            torch.log1p(plain_context_activation.clamp_min(0.0))
        )
        pctx_occ = _center_rms_normalize(
            torch.log1p(ppmi_context_activation.clamp_min(0.0))
        )

        specialization = prior - ctx_occ
        ppmi_specialization = prior - pctx_occ
        specialization = torch.where(
            has_other.unsqueeze(-1), specialization, torch.zeros_like(specialization)
        )
        ppmi_specialization = torch.where(
            has_ppmi.unsqueeze(-1), ppmi_specialization, torch.zeros_like(ppmi_specialization)
        )

        sources = torch.stack([
            _center_rms_normalize(prior),
            _center_rms_normalize(specialization),
            _center_rms_normalize(ppmi_specialization),
        ], dim=1)

        avail = torch.stack([
            torch.ones_like(has_other),
            has_other,
            has_ppmi,
        ], dim=-1).to(item_activation.dtype)
        w = self.base_source_weight.to(item_activation.dtype).view(1, 3) * avail
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(EPS)
        role_logits = (w.unsqueeze(-1) * sources).sum(dim=1)
        role = torch.softmax(role_logits / self.tau, dim=-1)

        avail0 = avail.clone()
        avail0[:, 2] = 0.0
        w0 = self.base_source_weight.to(item_activation.dtype).view(1, 3) * avail0
        w0 = w0 / w0.sum(dim=-1, keepdim=True).clamp_min(EPS)
        logits0 = (w0.unsqueeze(-1) * sources).sum(dim=1)
        role0 = torch.softmax(logits0 / self.tau, dim=-1)
        ppmi_role_shift = 0.5 * torch.abs(role - role0).sum(dim=-1)
        ppmi_role_shift = torch.where(
            has_ppmi, ppmi_role_shift, torch.zeros_like(ppmi_role_shift)
        )

        w4 = torch.stack([
            w[:, 0], w[:, 1], torch.zeros_like(w[:, 0]), w[:, 2]
        ], dim=-1)
        return {
            "role": role,
            "source_weight": w4,
            "source_norm": sources,
            "role_logits": role_logits,
            "ppmi_role_shift": ppmi_role_shift,
        }


class BundleTypeHead(nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        if tau <= 0.0:
            raise ValueError("type tau must be >0")
        self.tau = float(tau)

    def forward(
        self,
        diversity: torch.Tensor,
        compatibility: torch.Tensor,
        certainty: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        d = diversity.clamp(0.0, 1.0)
        c = compatibility.clamp(0.0, 1.0)
        s = certainty.clamp(0.0, 1.0)
        evidence = torch.stack([
            s * (1.0 - d) * c,
            s * d * c,
            s * d * (1.0 - c),
        ], dim=-1)
        prob = torch.softmax(evidence / self.tau, dim=-1)
        uniform = torch.full_like(prob, 1.0 / 3.0)
        return torch.where(valid.unsqueeze(-1), prob, uniform)


class StructuralScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "component_weight",
            torch.full((4,), 0.25, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        s_facet: torch.Tensor,
        s_bundle: torch.Tensor,
        s_type: torch.Tensor,
        s_comp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        components = torch.stack([
            s_facet.clamp(-1.0, 1.0),
            s_bundle.clamp(-1.0, 1.0),
            (2.0 * s_type - 1.0).clamp(-1.0, 1.0),
            (2.0 * s_comp - 1.0).clamp(-1.0, 1.0),
        ], dim=-1)
        w = self.component_weight.to(components.dtype)
        score = (components * w).sum(dim=-1).clamp(-1.0, 1.0)
        return score, components


class CFDGN(MultiCBR):
    def __init__(self, conf: Dict, raw_graph):
        super().__init__(conf, raw_graph)

        _gfr_cpu_rng_after_main = torch.get_rng_state().clone()
        _gfr_cuda_rng_after_main = (
            [x.clone() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        )

        self.K = int(conf.get("gfr_num_facets", 4))
        self.d_f = int(conf.get("gfr_facet_dim", 64))
        if self.K != 4:
            raise ValueError("Final CFDGN contract requires K=4")
        if self.embedding_size != 128:
            raise ValueError("Final CFDGN contract requires backbone embedding_size=128")
        if self.d_f < self.K:
            raise ValueError("gfr_facet_dim must be >=K")

        self.role_tau = float(conf.get("gfr_role_tau", 0.50))
        self.history_tau = float(conf.get("gfr_history_tau", 0.50))
        self.match_tau = float(conf.get("gfr_match_tau", 0.50))
        self.type_tau = float(conf.get("gfr_type_tau", 0.25))
        self.pcl_tau = float(conf.get("gfr_pcl_tau", 0.20))
        self.pcl_label_smoothing = float(conf.get("gfr_pcl_label_smoothing", 0.05))
        self.ppmi_topk = int(conf.get("gfr_ppmi_topk", 20))
        self.ppmi_min_count = int(conf.get("gfr_ppmi_min_count", 2))
        self.ppmi_enrich_weight = float(conf.get("gfr_ppmi_enrich_weight", 0.10))
        self.ppmi_enrich_tau = float(conf.get("gfr_ppmi_enrich_tau", 0.50))
        self.role_prior_weight = float(conf.get("gfr_role_prior_weight", 0.20))
        self.role_goal_weight = float(conf.get("gfr_role_goal_weight", 0.45))
        self.role_context_weight = float(conf.get("gfr_role_context_weight", 0.20))
        self.role_ppmi_weight = float(conf.get("gfr_role_ppmi_weight", 0.15))
        self.relation_role_residual_weight = float(conf.get("gfr_relation_role_residual_weight", 0.0))
        if abs(self.relation_role_residual_weight) > 1e-12:
            raise ValueError(
                "D/C independence requires gfr_relation_role_residual_weight=0"
            )
        self.relation_dim = int(conf.get("gfr_relation_dim", 64))
        self.relation_bank_size = int(conf.get("gfr_relation_bank_size", 65536))
        self.relation_batch_size = int(conf.get("gfr_relation_batch_size", 2048))
        self.relation_bank_seed = int(conf.get("gfr_relation_bank_seed", 314159))
        self.eval_bundle_chunk = int(conf.get("gfr_eval_bundle_chunk", 512))
        if self.eval_bundle_chunk <= 0:
            raise ValueError("gfr_eval_bundle_chunk must be positive")
        self.struct_hard_k = int(conf.get("gfr_struct_hard_k", 0))
        if self.struct_hard_k != 0:
            raise ValueError(
                "MAIN-PRESERVING-CORRECTION requires gfr_struct_hard_k=0; "
                "unobserved Main-hard bundles cannot be assumed negative"
            )

        self.lambda_residual = float(conf.get("gfr_lambda_residual", 0.10))
        if self.lambda_residual <= 0.0:
            raise ValueError("gfr_lambda_residual must be >0")
        self.main_error_weight_floor = float(
            conf.get("gfr_main_error_weight_floor", 0.10)
        )
        if not (0.0 <= self.main_error_weight_floor < 1.0):
            raise ValueError(
                "require 0 <= gfr_main_error_weight_floor < 1"
            )
        self.residual_scale_max = float(conf.get("gfr_residual_scale_max", 0.50))
        self.residual_scale_init = float(conf.get("gfr_residual_scale_init", 0.10))
        if not (0.0 < self.residual_scale_init < self.residual_scale_max <= 1.0):
            raise ValueError(
                "require 0 < gfr_residual_scale_init < gfr_residual_scale_max <= 1"
            )
        q0 = self.residual_scale_init / self.residual_scale_max
        self._residual_scale_logit_init = math.log(q0 / (1.0 - q0))
        self.lambda_relation = float(conf.get("gfr_lambda_relation", 0.05))

        self.lambda_role_semantic = float(conf.get("gfr_lambda_role_semantic", 0.0))
        if abs(self.lambda_role_semantic) > 1e-12:
            raise ValueError(
                "MAIN-PRESERVING-CORRECTION requires gfr_lambda_role_semantic=0; "
                "co-user percentiles are diagnostics, not latent-role labels"
            )
        self.lambda_pcl_item = float(conf.get("gfr_lambda_pcl_item", 0.001))
        self.lambda_pcl_bundle = float(conf.get("gfr_lambda_pcl_bundle", 0.001))
        self.lambda_pcl_user = float(conf.get("gfr_lambda_pcl_user", 0.001))
        self.lambda_orth_item = float(conf.get("gfr_lambda_orth_item", 0.001))
        self.lambda_orth_bundle = float(conf.get("gfr_lambda_orth_bundle", 0.001))
        self.lambda_orth_user = float(conf.get("gfr_lambda_orth_user", 0.001))

        self._gfr_backbone_param_names = tuple(
            name for name, p in self.named_parameters() if p.requires_grad
        )
        self._gfr_backbone_param_name_set = set(self._gfr_backbone_param_names)

        self.residual_scale_logit = nn.Parameter(
            torch.tensor(self._residual_scale_logit_init, dtype=torch.float32)
        )

        ub = raw_graph[0].tocsr()
        ui = raw_graph[1].tocsr()
        bi = raw_graph[2].tocsr()
        if ub.shape != (self.num_users, self.num_bundles):
            raise ValueError("unexpected U-B shape")
        if ui.shape != (self.num_users, self.num_items):
            raise ValueError("unexpected U-I shape")
        if bi.shape != (self.num_bundles, self.num_items):
            raise ValueError("unexpected B-I shape")

        self.register_buffer("gfr_ub_indptr", torch.from_numpy(ub.indptr.astype(np.int64)), persistent=False)
        self.register_buffer("gfr_ub_indices", torch.from_numpy(ub.indices.astype(np.int64)), persistent=False)
        self.register_buffer("gfr_bi_indptr", torch.from_numpy(bi.indptr.astype(np.int64)), persistent=False)
        self.register_buffer("gfr_bi_indices", torch.from_numpy(bi.indices.astype(np.int64)), persistent=False)
        self.register_buffer(
            "gfr_user_degree",
            torch.from_numpy(np.diff(ub.indptr).astype(np.int64)),
            persistent=False,
        )
        pair_keys = (
            np.repeat(np.arange(ub.shape[0], dtype=np.int64), np.diff(ub.indptr))
            * np.int64(self.num_bundles)
            + ub.indices.astype(np.int64)
        )
        if np.unique(pair_keys).size != pair_keys.size:
            raise RuntimeError("duplicate TRAIN U-B edge")
        pair_keys.sort()
        self.register_buffer(
            "gfr_ub_pair_keys_sorted", torch.from_numpy(pair_keys), persistent=False
        )
        self.register_buffer(
            "gfr_bundle_degree",
            torch.from_numpy(np.diff(bi.indptr).astype(np.int64)),
            persistent=False,
        )

        train_bundle_mask = np.asarray(ub.getnnz(axis=0)).reshape(-1) > 0
        fixed_ui_div, fixed_ui_valid, fixed_ui_stats = build_fixed_ui_bundle_diversity_target(
            ui, bi, train_bundle_mask
        )
        self.register_buffer(
            "gfr_bundle_fixed_ui_diversity_target",
            torch.from_numpy(fixed_ui_div),
            persistent=True,
        )
        self.register_buffer(
            "gfr_bundle_fixed_ui_diversity_valid",
            torch.from_numpy(fixed_ui_valid),
            persistent=True,
        )
        self.fixed_ui_diversity_stats = fixed_ui_stats

        edge_role_target, edge_role_valid, edge_role_stats = (
            build_fixed_ui_edge_role_affinity_target(ui, bi, train_bundle_mask)
        )
        self.register_buffer(
            "gfr_bi_edge_role_affinity_target",
            torch.from_numpy(edge_role_target),
            persistent=True,
        )
        self.register_buffer(
            "gfr_bi_edge_role_affinity_valid",
            torch.from_numpy(edge_role_valid.astype(np.bool_)),
            persistent=True,
        )
        self.fixed_ui_edge_role_stats = edge_role_stats

        train_bundle_mask = np.asarray(ub.sum(axis=0)).reshape(-1) > 0
        bi_ppmi = (sp.diags(train_bundle_mask.astype(np.float64)) @ bi).tocsr()
        A_ppmi, C_cooccur, ppmi_stats = build_ppmi_relation_prior(
            bi_ppmi, self.num_items, self.ppmi_topk, self.ppmi_min_count
        )
        ppmi_stats["train_bundle_count"] = int(train_bundle_mask.sum())
        ppmi_stats["all_bundle_count"] = int(self.num_bundles)
        A_norm = _row_normalize_csr(A_ppmi)
        R_local, active_local, local_stats = build_strict_current_bundle_relation(A_ppmi, bi)
        coverage, strength, evidence = build_bundle_ppmi_evidence(A_ppmi, bi)
        self.register_buffer(
            "gfr_train_bundle_mask",
            torch.from_numpy(train_bundle_mask.astype(np.bool_)),
            persistent=False,
        )

        self.register_buffer("gfr_ppmi_row_norm", _to_sparse_tensor(A_norm), persistent=False)
        self.register_buffer("gfr_ppmi_raw", _to_sparse_tensor(A_ppmi), persistent=False)
        self.register_buffer("gfr_rel_indptr", torch.from_numpy(R_local.indptr.astype(np.int64)), persistent=False)
        self.register_buffer("gfr_rel_indices", torch.from_numpy(R_local.indices.astype(np.int64)), persistent=False)
        self.register_buffer("gfr_rel_values", torch.from_numpy(R_local.data.astype(np.float32)), persistent=False)
        self.register_buffer("gfr_rel_active", torch.from_numpy(active_local.astype(np.bool_)), persistent=False)
        self.register_buffer("gfr_bundle_ppmi_coverage", torch.from_numpy(coverage), persistent=False)
        self.register_buffer("gfr_bundle_ppmi_strength", torch.from_numpy(strength), persistent=False)
        self.register_buffer("gfr_bundle_ppmi_evidence", torch.from_numpy(evidence), persistent=False)
        self.ppmi_stats = ppmi_stats
        self.ppmi_local_stats = local_stats

        pos_i, pos_j, neg_i, neg_j = build_relation_pair_bank(
            A_ppmi, C_cooccur, self.num_items,
            self.relation_bank_size, self.relation_bank_seed,
        )
        self.register_buffer("gfr_rel_pos_i", torch.from_numpy(pos_i), persistent=False)
        self.register_buffer("gfr_rel_pos_j", torch.from_numpy(pos_j), persistent=False)
        self.register_buffer("gfr_rel_neg_i", torch.from_numpy(neg_i), persistent=False)
        self.register_buffer("gfr_rel_neg_j", torch.from_numpy(neg_j), persistent=False)

        self.item_encoder = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.embedding_size),
        )
        self.facet_projector = nn.Linear(self.embedding_size, self.K * self.d_f, bias=False)
        self.ppmi_enricher = PPMIFacetEnricher(self.ppmi_enrich_weight, self.ppmi_enrich_tau)
        self.router = ContextualRoleRouter(
            self.K, self.role_tau,
            self.role_prior_weight, self.role_goal_weight,
            self.role_context_weight, self.role_ppmi_weight,
        )

        self.relation_item_proj = nn.Linear(self.embedding_size, self.relation_dim, bias=False)
        self.relation_scale_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.relation_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.bundle_semantic_proj = nn.Sequential(
            nn.Linear(self.K * self.d_f + self.K + 3 + 4, self.d_f, bias=False),
            nn.LayerNorm(self.d_f),
        )
        self.user_query = nn.Linear(self.embedding_size, self.K * self.d_f, bias=False)

        self.structural_scorer = StructuralScorer()
        self.type_head = BundleTypeHead(self.type_tau)

        for m in (
            self.item_encoder[0], self.facet_projector,
            self.relation_item_proj,
            self.bundle_semantic_proj[0], self.user_query,
        ):
            nn.init.xavier_normal_(m.weight)

        anchor_seed = int(conf.get("gfr_anchor_seed", 12345))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(anchor_seed)
        raw = torch.randn(self.d_f, self.K, generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        anchors = q[:, :self.K].T.contiguous()
        self.register_buffer("facet_anchors", anchors, persistent=True)

        self.register_buffer("gfr_train_steps", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer(
            "gfr_main_score_scale", torch.ones((), dtype=torch.float32),
            persistent=False,
        )
        self._last_diag: Dict[str, float] = {}

        torch.set_rng_state(_gfr_cpu_rng_after_main)
        if _gfr_cuda_rng_after_main is not None:
            torch.cuda.set_rng_state_all(_gfr_cuda_rng_after_main)
        if not torch.equal(torch.get_rng_state(), _gfr_cpu_rng_after_main):
            raise RuntimeError("CFDGN RNG-neutral initialization seal failed")

        print(
            f"[CFDGN-DECOUPLED-SCALE-INVARIANT] K={self.K} d_f={self.d_f} | "
            f"PPMI topK={self.ppmi_topk} minCount={self.ppmi_min_count} "
            f"localActive={100.0*local_stats['active_fraction']:.2f}% | "
            f"repoNegOnly=True unifiedFullCatalog=True | "
            f"fixedUI-D(trainRankMean={fixed_ui_stats['target_mean_train']:.3f},"
            f"trainRankStd={fixed_ui_stats['target_std_train']:.3f}) | "
            f"edgeRole(trainRankMean={edge_role_stats['target_mean_train']:.3f},"
            f"trainRankStd={edge_role_stats['target_std_train']:.3f}) | "
            "contrastive contextual router + independent D/C axes | "
            f"Main-preserving alphaInit={self.residual_scale_init:.3f} "
            f"alphaMax={self.residual_scale_max:.3f} | "
            "Main-gradient/RNG-isolated structural branch | no mined-unseen negative"
        )

    def _views(self, test: bool = False) -> Dict[str, torch.Tensor]:
        if test:
            ub_g = self.UB_propagation_graph_ori
            ui_g = self.UI_propagation_graph_ori
            ui_a = self.UI_aggregation_graph_ori
            bi_g = self.BI_propagation_graph_ori
            bi_a = self.BI_aggregation_graph_ori
        else:
            ub_g = self.UB_propagation_graph
            ui_g = self.UI_propagation_graph
            ui_a = self.UI_aggregation_graph
            bi_g = self.BI_propagation_graph
            bi_a = self.BI_aggregation_graph

        UB_u, UB_b = self.propagate(ub_g, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        UI_u, UI_i = self.propagate(ui_g, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
        UI_b = self.aggregate(bi_a, UI_i, "BI", test)
        BI_b, BI_i = self.propagate(bi_g, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
        BI_u = self.aggregate(ui_a, BI_i, "UI", test)
        users_main, bundles_main = self.fuse_users_bundles_feature(
            [UB_u, UI_u, BI_u], [UB_b, UI_b, BI_b]
        )
        return {
            "users_main": users_main,
            "bundles_main": bundles_main,
            "ui_users": UI_u,
            "ui_items": UI_i,
            "bi_items_main_only": BI_i,
        }

    @staticmethod
    def _select_csr_rows(
        indptr: torch.Tensor,
        indices: torch.Tensor,
        rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if rows.ndim != 1:
            rows = rows.reshape(-1)
        counts = indptr[rows + 1] - indptr[rows]
        total = int(counts.sum().item())
        if total == 0:
            z = torch.empty(0, dtype=torch.long, device=rows.device)
            return z, z, z, counts
        local_group = torch.repeat_interleave(
            torch.arange(rows.numel(), device=rows.device, dtype=torch.long), counts
        )
        base = torch.repeat_interleave(indptr[rows], counts)
        prior = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
        offset = torch.arange(total, device=rows.device, dtype=torch.long) - prior
        positions = base + offset
        values = indices[positions]
        return positions, values, local_group, counts

    def _structural_item_state(self, views: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_item = self.items_feature.detach()
        ui_item = views["ui_items"].detach()
        token = self.item_encoder(torch.cat([raw_item, ui_item], dim=-1))
        raw_facets = self.facet_projector(token).view(
            self.num_items, self.K, self.d_f
        )
        facet_activation = raw_facets.norm(dim=-1)
        base = F.normalize(raw_facets, p=2, dim=-1, eps=EPS)
        enriched, ppmi_gate = self.ppmi_enricher(base, self.gfr_ppmi_row_norm)
        relation_base = F.normalize(self.relation_item_proj(token), p=2, dim=-1, eps=EPS)
        return {
            "token": token,
            "base_facets": base,
            "facets": enriched,
            "facet_activation": facet_activation,
            "ppmi_gate": ppmi_gate,
            "relation_base": relation_base,
        }

    def _ppmi_activation_context_selected_edges(
        self,
        item_activation: torch.Tensor,
        edge_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if item_activation.ndim != 2 or item_activation.shape[1] != self.K:
            raise ValueError("item_activation must be [num_items,K]")
        n = int(edge_positions.numel())
        out = torch.zeros(
            n, self.K,
            dtype=item_activation.dtype, device=item_activation.device,
        )
        if n == 0:
            return out, torch.zeros(0, dtype=torch.bool, device=item_activation.device)

        starts = self.gfr_rel_indptr[edge_positions]
        ends = self.gfr_rel_indptr[edge_positions + 1]
        counts = ends - starts
        active = counts > 0
        total = int(counts.sum().item())
        if total == 0:
            return out, active

        local_group = torch.repeat_interleave(
            torch.arange(n, device=item_activation.device, dtype=torch.long), counts
        )
        base = torch.repeat_interleave(starts, counts)
        prior = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
        offset = torch.arange(total, device=item_activation.device, dtype=torch.long) - prior
        rel_pos = base + offset
        cols = self.gfr_rel_indices[rel_pos]
        vals = self.gfr_rel_values[rel_pos].to(item_activation.dtype)
        out.index_add_(0, local_group, vals.view(-1, 1) * item_activation[cols])
        return out, active

    def _bundle_state_selected(
        self,
        bundle_ids: torch.Tensor,
        item_state: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if bundle_ids.ndim != 1:
            bundle_ids = bundle_ids.reshape(-1)
        if bundle_ids.numel() == 0:
            raise ValueError("bundle_ids cannot be empty")
        if not bool(torch.all(bundle_ids[1:] >= bundle_ids[:-1])) if bundle_ids.numel() > 1 else False:
            raise ValueError("bundle_ids must be sorted")

        edge_pos, edge_item, group, counts = self._select_csr_rows(
            self.gfr_bi_indptr, self.gfr_bi_indices, bundle_ids
        )
        m = int(bundle_ids.numel())
        e = int(edge_item.numel())
        token = item_state["token"]
        facets = item_state["facets"]

        if e == 0:
            raise RuntimeError("selected bundles contain no B-I members")

        edge_facets = facets[edge_item]
        degree = counts.to(token.dtype)
        has_other = degree[group] > 1.0

        edge_activation = item_state["facet_activation"][edge_item]
        activation_sum = torch.zeros(
            m, self.K, dtype=edge_activation.dtype, device=edge_activation.device
        )
        activation_sum.index_add_(0, group, edge_activation)
        plain_context_activation = (
            activation_sum[group] - edge_activation
        ) / (degree[group] - 1.0).clamp_min(1.0).unsqueeze(-1)
        plain_context_activation = torch.where(
            has_other.unsqueeze(-1), plain_context_activation,
            torch.zeros_like(plain_context_activation),
        )

        ppmi_context_activation, has_ppmi = (
            self._ppmi_activation_context_selected_edges(
                item_state["facet_activation"], edge_pos
            )
        )
        route = self.router(
            edge_activation, plain_context_activation, ppmi_context_activation,
            has_other, has_ppmi,
        )
        role = route["role"]

        mass = torch.zeros(m, self.K, dtype=role.dtype, device=role.device)
        mass.index_add_(0, group, role)
        attn = role / mass[group].clamp_min(EPS)

        bf = torch.zeros(m, self.K, self.d_f, dtype=facets.dtype, device=facets.device)
        bf.index_add_(0, group, attn.unsqueeze(-1) * facets[edge_item])

        bf_sem = torch.zeros_like(bf)
        bf_sem.index_add_(0, group, attn.detach().unsqueeze(-1) * facets[edge_item])

        rho = mass / degree.clamp_min(1.0).unsqueeze(-1)

        edge_h = _normalized_entropy(role)
        mean_h = torch.zeros(m, dtype=role.dtype, device=role.device)
        mean_h.index_add_(0, group, edge_h)
        mean_h = mean_h / degree.clamp_min(1.0)
        rho_h = _normalized_entropy(rho)
        js_natural = (rho_h - mean_h).clamp_min(0.0) * math.log(float(self.K))
        max_roles = torch.minimum(
            degree, torch.full_like(degree, float(self.K))
        )
        max_js = torch.log(max_roles.clamp_min(2.0))
        diversity = torch.where(
            degree >= 2.0,
            (js_natural / max_js.clamp_min(EPS)).clamp(0.0, 1.0),
            torch.zeros_like(js_natural),
        )
        certainty = (1.0 - mean_h).clamp(0.0, 1.0)

        context_role = (mass[group] - role) / (degree[group] - 1.0).clamp_min(1.0).unsqueeze(-1)
        context_role = torch.where(
            has_other.unsqueeze(-1), context_role, torch.zeros_like(context_role)
        )
        role_affinity_pred = (role * context_role).sum(dim=-1).clamp(0.0, 1.0)
        role_affinity_target = (
            self.gfr_bi_edge_role_affinity_target[edge_pos]
            .to(dtype=role.dtype, device=role.device)
            .detach()
        )
        role_affinity_target_valid = (
            self.gfr_bi_edge_role_affinity_valid[edge_pos]
            .to(device=role.device)
            .detach()
        )
        role_aff_pred_sum = torch.zeros(m, dtype=role.dtype, device=role.device)
        role_aff_pred_sum.index_add_(0, group, role_affinity_pred)
        role_affinity_pred_mean = role_aff_pred_sum / degree.clamp_min(1.0)
        role_aff_tgt_sum = torch.zeros(m, dtype=role.dtype, device=role.device)
        role_aff_tgt_cnt = torch.zeros(m, dtype=role.dtype, device=role.device)
        role_aff_tgt_sum.index_add_(
            0, group, role_affinity_target * role_affinity_target_valid.to(role.dtype)
        )
        role_aff_tgt_cnt.index_add_(
            0, group, role_affinity_target_valid.to(role.dtype)
        )
        role_affinity_target_mean = role_aff_tgt_sum / role_aff_tgt_cnt.clamp_min(1.0)
        role_affinity_target_bundle_valid = role_aff_tgt_cnt > 0

        semantic_diversity_target = (
            self.gfr_bundle_fixed_ui_diversity_target[bundle_ids]
            .to(dtype=diversity.dtype, device=diversity.device)
            .detach()
        )
        semantic_diversity_target_valid = (
            self.gfr_bundle_fixed_ui_diversity_valid[bundle_ids]
            .to(device=diversity.device)
            .detach()
        )

        edge_rel = item_state["relation_base"][edge_item]
        rel_sum = torch.zeros(m, self.relation_dim, dtype=edge_rel.dtype, device=edge_rel.device)
        rel_sum.index_add_(0, group, edge_rel)
        pair_denom = degree * (degree - 1.0)
        mean_pair_dot = (
            rel_sum.square().sum(dim=-1) - degree
        ) / pair_denom.clamp_min(1.0)
        valid_type = degree >= 2.0
        mean_pair_dot = torch.where(valid_type, mean_pair_dot, torch.zeros_like(mean_pair_dot))
        rel_scale = F.softplus(self.relation_scale_raw) + 1e-4
        c_sem = torch.sigmoid(rel_scale * mean_pair_dot + self.relation_bias)

        ppmi_cov = self.gfr_bundle_ppmi_coverage[bundle_ids].to(c_sem.dtype)
        ppmi_ev = self.gfr_bundle_ppmi_evidence[bundle_ids].to(c_sem.dtype)
        compatibility = 1.0 - (1.0 - c_sem) * (1.0 - ppmi_ev)
        compatibility = torch.where(valid_type, compatibility.clamp(0.0, 1.0), torch.full_like(compatibility, 0.5))

        type_prob = self.type_head(diversity, compatibility, certainty, valid_type)
        ppmi_shift_sum = torch.zeros(m, dtype=role.dtype, device=role.device)
        ppmi_shift_sum.index_add_(0, group, route["ppmi_role_shift"])
        ppmi_role_shift = ppmi_shift_sum / degree.clamp_min(1.0)
        ppmi_active_sum = torch.zeros(m, dtype=role.dtype, device=role.device)
        ppmi_active_sum.index_add_(0, group, has_ppmi.to(role.dtype))
        ppmi_active_fraction = ppmi_active_sum / degree.clamp_min(1.0)

        z_input = torch.cat([
            bf.reshape(m, self.K * self.d_f),
            rho,
            type_prob,
            diversity.unsqueeze(-1),
            compatibility.unsqueeze(-1),
            certainty.unsqueeze(-1),
            ppmi_cov.unsqueeze(-1),
        ], dim=-1)
        z_bundle = F.normalize(self.bundle_semantic_proj(z_input), p=2, dim=-1, eps=EPS)

        return {
            "ids": bundle_ids,
            "facets": bf,
            "facets_sem": bf_sem,
            "rho": rho,
            "diversity": diversity,
            "certainty": certainty,
            "semantic_diversity_target": semantic_diversity_target,
            "semantic_diversity_target_valid": semantic_diversity_target_valid,
            "role_affinity_pred": role_affinity_pred,
            "role_affinity_target": role_affinity_target,
            "role_affinity_target_valid": role_affinity_target_valid,
            "role_affinity_pred_mean": role_affinity_pred_mean,
            "role_affinity_target_mean": role_affinity_target_mean,
            "role_affinity_target_bundle_valid": role_affinity_target_bundle_valid,
            "compatibility": compatibility,
            "type_prob": type_prob,
            "type_valid": valid_type,
            "ppmi_coverage": ppmi_cov,
            "ppmi_role_shift": ppmi_role_shift,
            "ppmi_active_fraction": ppmi_active_fraction,
            "z_bundle": z_bundle,
            "edge_positions": edge_pos,
            "edge_items": edge_item,
            "edge_group": group,
            "role": role,
            "source_weight": route["source_weight"],
            "source_norm": route["source_norm"],
            "member_attn": attn,
        }

    @staticmethod
    def _map_sorted_ids(sorted_ids: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        query = query.contiguous()
        idx = torch.searchsorted(sorted_ids, query)
        if bool((idx >= sorted_ids.numel()).any()):
            raise RuntimeError("ID lookup out of range")
        if not bool(torch.equal(sorted_ids[idx], query)):
            raise RuntimeError("ID lookup missing selected entity")
        return idx

    def _assert_train_positive_membership(
        self, users: torch.Tensor, positive_bundles: torch.Tensor
    ) -> None:
        keys = users.to(torch.long) * int(self.num_bundles) + positive_bundles.to(torch.long)
        pos = torch.searchsorted(self.gfr_ub_pair_keys_sorted, keys.contiguous())
        ok = pos < self.gfr_ub_pair_keys_sorted.numel()
        safe = pos.clamp_max(max(0, self.gfr_ub_pair_keys_sorted.numel() - 1))
        ok = ok & (self.gfr_ub_pair_keys_sorted[safe] == keys)
        if not bool(ok.all()):
            raise RuntimeError("training positive is not a TRAIN U-B history edge")

    def _train_ub_membership(
        self, users: torch.Tensor, bundles: torch.Tensor
    ) -> torch.Tensor:
        keys = users.to(torch.long) * int(self.num_bundles) + bundles.to(torch.long)
        shape = keys.shape
        flat = keys.contiguous().reshape(-1)
        pos = torch.searchsorted(self.gfr_ub_pair_keys_sorted, flat)
        ok = pos < self.gfr_ub_pair_keys_sorted.numel()
        safe = pos.clamp_max(max(0, self.gfr_ub_pair_keys_sorted.numel() - 1))
        ok = ok & (self.gfr_ub_pair_keys_sorted[safe] == flat)
        return ok.view(shape)

    @torch.no_grad()
    def _select_structural_candidates(
        self,
        users: torch.Tensor,
        sampled_bundles: torch.Tensor,
        users_main: Optional[torch.Tensor] = None,
        bundles_main: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sampled_bundles.ndim != 2 or sampled_bundles.shape[1] < 2:
            raise ValueError("structural ranking requires [positive, repository negative]")
        pos = sampled_bundles[:, 0]
        easy = sampled_bundles[:, 1]
        self._assert_train_positive_membership(users, pos)
        if bool(self._train_ub_membership(users, easy).any()):
            raise RuntimeError("repository negative overlaps TRAIN U-B positive")
        return sampled_bundles[:, :2]

    def _batch_history_meta(self, users: torch.Tensor) -> Dict[str, torch.Tensor]:
        unique_users, inv = torch.unique(users, sorted=True, return_inverse=True)
        _, hist_b, hist_group, counts = self._select_csr_rows(
            self.gfr_ub_indptr, self.gfr_ub_indices, unique_users
        )
        return {
            "unique_users": unique_users,
            "inverse": inv,
            "history_bundles": hist_b,
            "history_group": hist_group,
            "history_counts": counts,
        }

    @staticmethod
    def _grouped_exp_stats(
        logits: torch.Tensor,
        group: torch.Tensor,
        n_groups: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = logits.shape[-1]
        maxv = torch.full(
            (n_groups, k), -torch.inf,
            dtype=logits.dtype, device=logits.device,
        )
        idx = group.unsqueeze(-1).expand(-1, k)
        maxv.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)
        expw = torch.exp(logits - maxv[group])
        z = torch.zeros(n_groups, k, dtype=logits.dtype, device=logits.device)
        z.index_add_(0, group, expw)
        return expw, z

    def _user_state_training(
        self,
        users: torch.Tensor,
        positive_bundles: torch.Tensor,
        history_meta: Mapping[str, torch.Tensor],
        bundle_state: Mapping[str, torch.Tensor],
        ui_users: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        unique_users = history_meta["unique_users"]
        inv = history_meta["inverse"]
        hist_b = history_meta["history_bundles"]
        hist_group = history_meta["history_group"]
        n_u = int(unique_users.numel())

        q = F.normalize(
            self.user_query(ui_users.detach()[unique_users]).view(n_u, self.K, self.d_f),
            p=2, dim=-1, eps=EPS,
        )
        hist_local = self._map_sorted_ids(bundle_state["ids"], hist_b)
        hist_f = bundle_state["facets"][hist_local]
        hist_f_norm = F.normalize(hist_f, p=2, dim=-1, eps=EPS)
        logits = (q[hist_group] * hist_f_norm).sum(dim=-1) / self.history_tau
        expw, z = self._grouped_exp_stats(logits, hist_group, n_u)

        sum_exp2 = torch.zeros(
            n_u, self.K, dtype=expw.dtype, device=expw.device
        )
        sum_exp2.index_add_(0, hist_group, expw.square())

        sum_f = torch.zeros(n_u, self.K, self.d_f, dtype=hist_f.dtype, device=hist_f.device)
        sum_f.index_add_(0, hist_group, expw.unsqueeze(-1) * hist_f)
        hist_type = bundle_state["type_prob"][hist_local]
        sum_type = torch.zeros(n_u, self.K, 3, dtype=hist_f.dtype, device=hist_f.device)
        sum_type.index_add_(0, hist_group, expw.unsqueeze(-1) * hist_type.unsqueeze(1))
        hist_rho = bundle_state["rho"][hist_local]
        sum_rho = torch.zeros(n_u, self.K, self.K, dtype=hist_f.dtype, device=hist_f.device)
        sum_rho.index_add_(0, hist_group, expw.unsqueeze(-1) * hist_rho.unsqueeze(1))

        hist_f_sem = bundle_state["facets_sem"][hist_local]
        sum_f_sem = torch.zeros_like(sum_f)
        sum_f_sem.index_add_(0, hist_group, expw.detach().unsqueeze(-1) * hist_f_sem)

        pos_local = self._map_sorted_ids(bundle_state["ids"], positive_bundles)
        pos_f = bundle_state["facets"][pos_local]
        pos_f_norm = F.normalize(pos_f, p=2, dim=-1, eps=EPS)
        pos_logit = (q[inv] * pos_f_norm).sum(dim=-1) / self.history_tau

        k = self.K
        maxv = torch.full(
            (n_u, k), -torch.inf,
            dtype=logits.dtype, device=logits.device,
        )
        idx = hist_group.unsqueeze(-1).expand(-1, k)
        maxv.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)
        pos_exp = torch.exp(pos_logit - maxv[inv])

        denom = z[inv] - pos_exp
        has_history = denom.min(dim=-1).values > 1e-8
        denom_safe = denom.clamp_min(1e-8)

        loo_exp2 = (sum_exp2[inv] - pos_exp.square()).clamp_min(1e-8)
        n_eff = denom_safe.square() / loo_exp2
        pref_reliability = n_eff / (n_eff + float(self.K))
        pref_reliability = torch.where(
            has_history[:, None],
            pref_reliability.clamp(0.0, 1.0),
            torch.zeros_like(pref_reliability),
        )

        user_f = (sum_f[inv] - pos_exp.unsqueeze(-1) * pos_f) / denom_safe.unsqueeze(-1)
        user_f = torch.where(has_history[:, None, None], user_f, torch.zeros_like(user_f))

        pos_type = bundle_state["type_prob"][pos_local]
        type_pref = (
            sum_type[inv] - pos_exp.unsqueeze(-1) * pos_type.unsqueeze(1)
        ) / denom_safe.unsqueeze(-1)
        type_pref = torch.where(has_history[:, None, None], type_pref, torch.zeros_like(type_pref))

        pos_rho = bundle_state["rho"][pos_local]
        rho_pref = (
            sum_rho[inv] - pos_exp.unsqueeze(-1) * pos_rho.unsqueeze(1)
        ) / denom_safe.unsqueeze(-1)
        rho_pref = torch.where(has_history[:, None, None], rho_pref, torch.zeros_like(rho_pref))

        pos_f_sem = bundle_state["facets_sem"][pos_local]
        user_f_sem = (
            sum_f_sem[inv] - pos_exp.detach().unsqueeze(-1) * pos_f_sem
        ) / denom_safe.detach().unsqueeze(-1)
        user_f_sem = torch.where(has_history[:, None, None], user_f_sem, torch.zeros_like(user_f_sem))

        return {
            "facets": user_f,
            "facets_sem": user_f_sem,
            "selector_query": q[inv],
            "type_pref": type_pref,
            "rho_pref": rho_pref,
            "pref_reliability": pref_reliability,
            "has_history": has_history,
        }

    def _user_state_full(
        self,
        bundle_state: Mapping[str, torch.Tensor],
        ui_users: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        users = torch.arange(self.num_users, device=ui_users.device, dtype=torch.long)
        _, hist_b, hist_group, counts = self._select_csr_rows(
            self.gfr_ub_indptr, self.gfr_ub_indices, users
        )
        q = F.normalize(
            self.user_query(ui_users.detach()).view(self.num_users, self.K, self.d_f),
            p=2, dim=-1, eps=EPS,
        )
        hist_local = self._map_sorted_ids(bundle_state["ids"], hist_b)
        hist_f = bundle_state["facets"][hist_local]
        logits = (
            q[hist_group] * F.normalize(hist_f, p=2, dim=-1, eps=EPS)
        ).sum(dim=-1) / self.history_tau
        expw, z = self._grouped_exp_stats(logits, hist_group, self.num_users)
        denom = z.clamp_min(1e-8)

        sum_exp2 = torch.zeros(
            self.num_users, self.K, dtype=expw.dtype, device=expw.device
        )
        sum_exp2.index_add_(0, hist_group, expw.square())
        n_eff = denom.square() / sum_exp2.clamp_min(1e-8)
        pref_reliability = n_eff / (n_eff + float(self.K))

        sum_f = torch.zeros(self.num_users, self.K, self.d_f, dtype=hist_f.dtype, device=hist_f.device)
        sum_f.index_add_(0, hist_group, expw.unsqueeze(-1) * hist_f)
        uf = sum_f / denom.unsqueeze(-1)

        hist_type = bundle_state["type_prob"][hist_local]
        sum_type = torch.zeros(self.num_users, self.K, 3, dtype=hist_f.dtype, device=hist_f.device)
        sum_type.index_add_(0, hist_group, expw.unsqueeze(-1) * hist_type.unsqueeze(1))
        type_pref = sum_type / denom.unsqueeze(-1)

        hist_rho = bundle_state["rho"][hist_local]
        sum_rho = torch.zeros(self.num_users, self.K, self.K, dtype=hist_f.dtype, device=hist_f.device)
        sum_rho.index_add_(0, hist_group, expw.unsqueeze(-1) * hist_rho.unsqueeze(1))
        rho_pref = sum_rho / denom.unsqueeze(-1)

        has = counts > 0
        uf = torch.where(has[:, None, None], uf, torch.zeros_like(uf))
        type_pref = torch.where(has[:, None, None], type_pref, torch.zeros_like(type_pref))
        rho_pref = torch.where(has[:, None, None], rho_pref, torch.zeros_like(rho_pref))
        pref_reliability = torch.where(
            has[:, None],
            pref_reliability.clamp(0.0, 1.0),
            torch.zeros_like(pref_reliability),
        )
        return {
            "facets": uf,
            "selector_query": q,
            "type_pref": type_pref,
            "rho_pref": rho_pref,
            "pref_reliability": pref_reliability,
            "has_history": has,
        }

    def _residual_scale(self) -> torch.Tensor:
        return self.residual_scale_max * torch.sigmoid(self.residual_scale_logit)

    def _main_score_scale(self) -> torch.Tensor:
        return self.gfr_main_score_scale.clamp_min(EPS)

    @torch.no_grad()
    def _set_main_score_scale_from_representations(
        self,
        users_main: torch.Tensor,
        bundles_main: torch.Tensor,
    ) -> torch.Tensor:
        if users_main.ndim != 2 or bundles_main.ndim != 2:
            raise ValueError("Main representations must be rank-2 tensors")
        if users_main.shape[1] != bundles_main.shape[1]:
            raise ValueError("Main user/bundle dimensions do not match")
        if users_main.shape[0] <= 0 or bundles_main.shape[0] <= 0:
            raise ValueError("Main score scale requires non-empty representations")

        u = users_main.detach().to(dtype=torch.float32)
        b = bundles_main.detach().to(dtype=torch.float32)
        bc = b - b.mean(dim=0, keepdim=True)
        cov_b = torch.mm(bc.t(), bc) / float(b.shape[0])
        user_var = torch.einsum("ud,df,uf->u", u, cov_b, u).clamp_min(0.0)
        sigma = user_var.mean().sqrt().clamp_min(EPS)
        self.gfr_main_score_scale.copy_(
            sigma.to(
                device=self.gfr_main_score_scale.device,
                dtype=self.gfr_main_score_scale.dtype,
            )
        )
        return self._main_score_scale()

    @torch.no_grad()
    def refresh_main_score_scale(self) -> torch.Tensor:
        views = self._views(test=True)
        return self._set_main_score_scale_from_representations(
            views["users_main"], views["bundles_main"]
        )

    def _deploy_residual_multiplier(self) -> torch.Tensor:
        return self._residual_scale() * self._main_score_scale()

    @staticmethod
    def _decoupled_structural_objective(
        main_margin_det: torch.Tensor,
        raw_struct_scores: torch.Tensor,
        residual_scale: torch.Tensor,
        main_score_scale: torch.Tensor,
        main_error_weight_floor: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if raw_struct_scores.ndim != 2 or raw_struct_scores.shape[1] != 2:
            raise ValueError("raw_struct_scores must be [B,2] for repo pos/neg")
        if main_margin_det.ndim != 1 or main_margin_det.shape[0] != raw_struct_scores.shape[0]:
            raise ValueError("main margin/raw structural score shape mismatch")
        if not (0.0 <= float(main_error_weight_floor) < 1.0):
            raise ValueError("main_error_weight_floor must be in [0,1)")

        score_scale = main_score_scale.detach().to(
            device=raw_struct_scores.device, dtype=raw_struct_scores.dtype
        ).clamp_min(EPS)
        alpha = residual_scale.to(
            device=raw_struct_scores.device, dtype=raw_struct_scores.dtype
        )

        m = main_margin_det.detach() / score_scale
        soft_need = torch.sigmoid(-m)

        floor_t = raw_struct_scores.new_tensor(float(main_error_weight_floor))
        rank_weight = floor_t + (1.0 - floor_t) * soft_need
        preserve_weight = 1.0 - rank_weight

        raw_margin = raw_struct_scores[:, 0] - raw_struct_scores[:, 1]
        raw_rank_term = F.softplus(-raw_margin)
        raw_rank_loss = (rank_weight.detach() * raw_rank_term).mean()

        raw_cal = raw_struct_scores.detach()
        correction_norm = alpha * raw_cal
        correction_margin = correction_norm[:, 0] - correction_norm[:, 1]

        correction_rank_term = F.softplus(-correction_margin)
        calibration_rank = (rank_weight * correction_rank_term).mean()

        amplitude_term = correction_norm.square().mean(dim=1)
        margin_term = correction_margin.square()
        preserve_term = 0.5 * (amplitude_term + margin_term)
        calibration_preserve = (preserve_weight * preserve_term).mean()
        calibration_loss = calibration_rank + calibration_preserve

        total = raw_rank_loss + calibration_loss
        aux = {
            "need_weight": rank_weight.mean(),
            "preserve_weight": preserve_weight.mean(),
            "raw_rank_loss": raw_rank_loss,
            "calibration_loss": calibration_loss,
            "rank_help": calibration_rank,
            "preserve_penalty": calibration_preserve,
            "preserve_amplitude": (preserve_weight * amplitude_term).mean(),
            "preserve_margin": (preserve_weight * margin_term).mean(),
            "raw_rank_margin": raw_margin.mean(),
            "correction_margin_normalized": correction_margin.mean(),
            "main_margin_normalized": m,
        }
        return total, aux

    @staticmethod
    def _assemble_struct_components(
        m: torch.Tensor,
        s_facet: torch.Tensor,
        s_bundle: torch.Tensor,
        s_type: torch.Tensor,
        s_comp: torch.Tensor,
        diversity: torch.Tensor,
        compatibility: torch.Tensor,
        certainty: torch.Tensor,
        type_prob: torch.Tensor,
        ppmi_coverage: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "facet_match_by_interest": m,
            "s_facet": s_facet,
            "s_bundle": s_bundle,
            "s_type": s_type,
            "s_comp": s_comp,
            "diversity": diversity,
            "compatibility": compatibility,
            "certainty": certainty,
            "type_prob": type_prob,
            "ppmi_coverage": ppmi_coverage,
        }

    def _struct_components_training(
        self,
        user_state: Mapping[str, torch.Tensor],
        bundle_state: Mapping[str, torch.Tensor],
        sampled_bundles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        local = self._map_sorted_ids(bundle_state["ids"], sampled_bundles)
        bf = bundle_state["facets"][local]
        uf = user_state["facets"]
        uf_n = F.normalize(uf, p=2, dim=-1, eps=EPS)
        bf_n = F.normalize(bf, p=2, dim=-1, eps=EPS)
        m = torch.einsum("bkd,bckd->bck", uf_n, bf_n)
        selector = F.normalize(
            user_state["selector_query"], p=2, dim=-1, eps=EPS
        )
        selector_match = torch.einsum("bkd,bckd->bck", selector, bf_n)
        omega = torch.softmax(selector_match / self.match_tau, dim=-1)
        s_facet = (omega * m).sum(dim=-1)

        z_u = torch.einsum(
            "bck,bkd->bcd", omega, F.normalize(uf, p=2, dim=-1, eps=EPS)
        )
        z_b = bundle_state["z_bundle"][local]
        s_bundle = (
            F.normalize(z_u, p=2, dim=-1, eps=EPS)
            * F.normalize(z_b, p=2, dim=-1, eps=EPS)
        ).sum(dim=-1)

        p_b = bundle_state["type_prob"][local]
        type_by_interest = torch.einsum(
            "bkt,bct->bck", user_state["type_pref"], p_b
        )
        s_type = (omega * type_by_interest).sum(dim=-1)

        rho_b = bundle_state["rho"][local]
        rho_pref = F.normalize(user_state["rho_pref"], p=2, dim=-1, eps=EPS)
        rho_b_n = F.normalize(rho_b, p=2, dim=-1, eps=EPS)
        comp_by_interest = torch.einsum("bkj,bcj->bck", rho_pref, rho_b_n)
        rel = user_state["pref_reliability"].unsqueeze(1)
        comp_by_interest = 0.5 + rel * (comp_by_interest - 0.5)
        s_comp = (omega * comp_by_interest).sum(dim=-1)

        out = self._assemble_struct_components(
            m, s_facet, s_bundle, s_type, s_comp,
            bundle_state["diversity"][local],
            bundle_state["compatibility"][local],
            bundle_state["certainty"][local],
            p_b,
            bundle_state["ppmi_coverage"][local],
        )
        out["selector_match_by_interest"] = selector_match
        out["interest_weight"] = omega
        out["composition_reliability"] = user_state["pref_reliability"]
        return out

    def _score_training(
        self,
        main_candidate_score: torch.Tensor,
        user_state: Mapping[str, torch.Tensor],
        bundle_state: Mapping[str, torch.Tensor],
        sampled_bundles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        components = self._struct_components_training(
            user_state, bundle_state, sampled_bundles
        )
        raw_residual, four = self.structural_scorer(
            components["s_facet"], components["s_bundle"],
            components["s_type"], components["s_comp"],
        )
        alpha = self._residual_scale().to(raw_residual.dtype)
        main_scale = self._main_score_scale().to(raw_residual.dtype)
        deploy_multiplier = alpha * main_scale
        residual = deploy_multiplier * raw_residual
        final = main_candidate_score.detach() + residual
        return {
            "raw_residual": raw_residual,
            "residual": residual,
            "residual_scale": alpha,
            "main_score_scale": main_scale,
            "deploy_multiplier": deploy_multiplier,
            "final": final,
            "components": components,
            "four_components": four,
        }

    def _struct_score_catalog_chunk(
        self,
        user_state: Mapping[str, torch.Tensor],
        bundle_state: Mapping[str, torch.Tensor],
        users: torch.Tensor,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= int(start) < int(end) <= self.num_bundles):
            raise ValueError("invalid catalog chunk")
        users = users.reshape(-1)
        uf = user_state["facets"][users]
        bf = bundle_state["facets"][start:end]
        uf_n = F.normalize(uf, p=2, dim=-1, eps=EPS)
        bf_n = F.normalize(bf, p=2, dim=-1, eps=EPS)
        m = torch.einsum("ukd,mkd->umk", uf_n, bf_n)
        selector = F.normalize(
            user_state["selector_query"][users], p=2, dim=-1, eps=EPS
        )
        selector_match = torch.einsum("ukd,mkd->umk", selector, bf_n)
        omega = torch.softmax(selector_match / self.match_tau, dim=-1)
        s_facet = (omega * m).sum(dim=-1)

        z_u = torch.einsum("umk,ukd->umd", omega, uf_n)
        z_b = bundle_state["z_bundle"][start:end]
        s_bundle = (
            F.normalize(z_u, p=2, dim=-1, eps=EPS)
            * F.normalize(z_b, p=2, dim=-1, eps=EPS).unsqueeze(0)
        ).sum(dim=-1)

        p_b = bundle_state["type_prob"][start:end]
        type_by_interest = torch.einsum(
            "ukt,mt->umk", user_state["type_pref"][users], p_b
        )
        s_type = (omega * type_by_interest).sum(dim=-1)

        rho_pref = F.normalize(
            user_state["rho_pref"][users], p=2, dim=-1, eps=EPS
        )
        rho_b = F.normalize(
            bundle_state["rho"][start:end], p=2, dim=-1, eps=EPS
        )
        comp_by_interest = torch.einsum("ukj,mj->umk", rho_pref, rho_b)
        rel = user_state["pref_reliability"][users].unsqueeze(1)
        comp_by_interest = 0.5 + rel * (comp_by_interest - 0.5)
        s_comp = (omega * comp_by_interest).sum(dim=-1)

        raw_residual, four = self.structural_scorer(
            s_facet, s_bundle, s_type, s_comp
        )
        residual = (
            self._deploy_residual_multiplier().to(raw_residual.dtype)
            * raw_residual
        )
        return residual, four

    def _struct_components_candidates(
        self,
        user_state: Mapping[str, torch.Tensor],
        bundle_state: Mapping[str, torch.Tensor],
        users: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != users.numel():
            raise ValueError("candidate_ids must be [num_users, M]")

        uf = user_state["facets"][users]
        bf = bundle_state["facets"][candidate_ids]
        uf_n = F.normalize(uf, p=2, dim=-1, eps=EPS)
        bf_n = F.normalize(bf, p=2, dim=-1, eps=EPS)
        m = torch.einsum("ukd,umkd->umk", uf_n, bf_n)
        selector = F.normalize(
            user_state["selector_query"][users], p=2, dim=-1, eps=EPS
        )
        selector_match = torch.einsum("ukd,umkd->umk", selector, bf_n)
        omega = torch.softmax(selector_match / self.match_tau, dim=-1)
        s_facet = (omega * m).sum(dim=-1)

        z_u = torch.einsum(
            "umk,ukd->umd", omega, F.normalize(uf, p=2, dim=-1, eps=EPS)
        )
        z_b = bundle_state["z_bundle"][candidate_ids]
        s_bundle = (
            F.normalize(z_u, p=2, dim=-1, eps=EPS)
            * F.normalize(z_b, p=2, dim=-1, eps=EPS)
        ).sum(dim=-1)

        p_b = bundle_state["type_prob"][candidate_ids]
        type_by_interest = torch.einsum(
            "ukt,umt->umk", user_state["type_pref"][users], p_b
        )
        s_type = (omega * type_by_interest).sum(dim=-1)

        rho_pref = F.normalize(
            user_state["rho_pref"][users], p=2, dim=-1, eps=EPS
        )
        rho_b = F.normalize(
            bundle_state["rho"][candidate_ids], p=2, dim=-1, eps=EPS
        )
        comp_by_interest = torch.einsum("ukj,umj->umk", rho_pref, rho_b)
        rel = user_state["pref_reliability"][users].unsqueeze(1)
        comp_by_interest = 0.5 + rel * (comp_by_interest - 0.5)
        s_comp = (omega * comp_by_interest).sum(dim=-1)

        out = self._assemble_struct_components(
            m, s_facet, s_bundle, s_type, s_comp,
            bundle_state["diversity"][candidate_ids],
            bundle_state["compatibility"][candidate_ids],
            bundle_state["certainty"][candidate_ids],
            p_b,
            bundle_state["ppmi_coverage"][candidate_ids],
        )
        out["selector_match_by_interest"] = selector_match
        out["interest_weight"] = omega
        out["composition_reliability"] = user_state["pref_reliability"][users]
        return out

    def _score_eval_candidates(
        self,
        main_candidate_score: torch.Tensor,
        user_state: Mapping[str, torch.Tensor],
        bundle_state: Mapping[str, torch.Tensor],
        users: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        components = self._struct_components_candidates(
            user_state, bundle_state, users, candidate_ids
        )
        raw_residual, four = self.structural_scorer(
            components["s_facet"], components["s_bundle"],
            components["s_type"], components["s_comp"],
        )
        alpha = self._residual_scale().to(raw_residual.dtype)
        main_scale = self._main_score_scale().to(raw_residual.dtype)
        deploy_multiplier = alpha * main_scale
        residual = deploy_multiplier * raw_residual
        return {
            "raw_residual": raw_residual,
            "residual": residual,
            "residual_scale": alpha,
            "main_score_scale": main_scale,
            "deploy_multiplier": deploy_multiplier,
            "final": main_candidate_score.detach() + residual,
            "components": components,
            "four_components": four,
        }


    def _anchor_loss(self, z: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return self.facet_anchors.sum() * 0.0
        z = F.normalize(z, p=2, dim=-1, eps=EPS)
        a = F.normalize(self.facet_anchors, p=2, dim=-1, eps=EPS)
        logits = torch.einsum("nkd,qd->nkq", z, a) / self.pcl_tau
        target = torch.arange(self.K, device=z.device).view(1, self.K).expand(z.shape[0], -1)
        return F.cross_entropy(
            logits.reshape(-1, self.K), target.reshape(-1),
            label_smoothing=self.pcl_label_smoothing,
        )

    @staticmethod
    def _orth_loss(z: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return z.sum() * 0.0
        q = F.normalize(z, p=2, dim=-1, eps=EPS)
        gram = torch.matmul(q, q.transpose(-1, -2))
        k = gram.shape[-1]
        eye = torch.eye(k, dtype=gram.dtype, device=gram.device)
        off = 1.0 - eye
        return (gram.square() * off).sum() / (off.sum() * gram.shape[0]).clamp_min(1.0)

    def _relation_loss(self, item_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        n = int(self.gfr_rel_pos_i.numel())
        if n == 0:
            return item_state["relation_base"].sum() * 0.0
        bs = min(self.relation_batch_size, n)
        start = (int(self.gfr_train_steps.item()) * bs) % n
        idx = (torch.arange(bs, device=self.device, dtype=torch.long) + start) % n
        pi, pj = self.gfr_rel_pos_i[idx], self.gfr_rel_pos_j[idx]
        ni, nj = self.gfr_rel_neg_i[idx], self.gfr_rel_neg_j[idx]
        r = item_state["relation_base"]
        scale = F.softplus(self.relation_scale_raw) + 1e-4
        pos_logit = scale * (r[pi] * r[pj]).sum(dim=-1) + self.relation_bias
        neg_logit = scale * (r[ni] * r[nj]).sum(dim=-1) + self.relation_bias
        return 0.5 * (
            F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))
            + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
        )

    def _semantic_losses(
        self,
        item_state: Mapping[str, torch.Tensor],
        bundle_state: Mapping[str, torch.Tensor],
        user_state: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        unique_items = torch.unique(bundle_state["edge_items"], sorted=True)
        item_z = item_state["facets"][unique_items]
        valid_b = bundle_state["type_valid"] | (self.gfr_bundle_degree[bundle_state["ids"]] > 0)
        bundle_z = bundle_state["facets_sem"][valid_b]
        user_z = user_state["facets_sem"][user_state["has_history"]]

        pcl_i = self._anchor_loss(item_z)
        pcl_b = self._anchor_loss(bundle_z)
        pcl_u = self._anchor_loss(user_z)
        orth_i = self._orth_loss(item_z)
        orth_b = self._orth_loss(bundle_z)
        orth_u = self._orth_loss(user_z)
        train_edge = self.gfr_train_bundle_mask[bundle_state["ids"]][
            bundle_state["edge_group"]
        ]
        valid_role_edge = (
            bundle_state["role_affinity_target_valid"] & train_edge
        )
        if bool(valid_role_edge.any()):
            role_sem = F.smooth_l1_loss(
                bundle_state["role_affinity_pred"][valid_role_edge],
                bundle_state["role_affinity_target"][valid_role_edge].detach(),
            )
        else:
            role_sem = bundle_state["role_affinity_pred"].sum() * 0.0

        valid_role_bundle = (
            bundle_state["type_valid"]
            & bundle_state["semantic_diversity_target_valid"]
            & self.gfr_train_bundle_mask[bundle_state["ids"]]
        )
        if bool(valid_role_bundle.any()):
            role_div_diag = F.smooth_l1_loss(
                bundle_state["diversity"][valid_role_bundle],
                bundle_state["semantic_diversity_target"][valid_role_bundle].detach(),
            )
        else:
            role_div_diag = bundle_state["diversity"].sum() * 0.0
        return {
            "pcl_item": pcl_i,
            "pcl_bundle": pcl_b,
            "pcl_user": pcl_u,
            "orth_item": orth_i,
            "orth_bundle": orth_b,
            "orth_user": orth_u,
            "role_semantic_consistency": role_sem,
            "role_diversity_consistency_diag": role_div_diag,
        }

    def forward(self, batch, ED_drop: bool = False):
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        users, sampled_bundles = batch
        users = users.reshape(-1)
        if sampled_bundles.ndim != 2 or sampled_bundles.shape[0] != users.shape[0]:
            raise ValueError("sampled_bundles must be [B,1+neg]")
        positive = sampled_bundles[:, 0]
        self._assert_train_positive_membership(users, positive)

        views = self._views(test=False)
        main_u = views["users_main"][users].unsqueeze(1).expand(-1, sampled_bundles.shape[1], -1)
        main_b = views["bundles_main"][sampled_bundles]
        main_bpr, c_loss = self.cal_loss(main_u, main_b)
        main_score = (main_u * main_b).sum(dim=-1)

        structural_bundles = self._select_structural_candidates(
            users, sampled_bundles, views["users_main"], views["bundles_main"]
        )
        structural_users_main = views["users_main"].detach()[users]
        structural_bundles_main = views["bundles_main"].detach()
        structural_main_score = (
            structural_users_main.unsqueeze(1)
            * structural_bundles_main[structural_bundles]
        ).sum(dim=-1)

        history_meta = self._batch_history_meta(users)
        selected_b = torch.unique(
            torch.cat([
                sampled_bundles.reshape(-1),
                structural_bundles.reshape(-1),
                history_meta["history_bundles"],
            ]),
            sorted=True,
        )
        item_state = self._structural_item_state(views)
        bundle_state = self._bundle_state_selected(selected_b, item_state)
        user_state = self._user_state_training(
            users, positive, history_meta, bundle_state, views["ui_users"]
        )
        scored = self._score_training(
            structural_main_score,
            user_state, bundle_state, structural_bundles
        )

        residual_margin = scored["residual"][:, 0] - scored["residual"][:, 1]
        raw_residual_margin = (
            scored["raw_residual"][:, 0] - scored["raw_residual"][:, 1]
        )
        main_margin_det = (
            structural_main_score[:, 0] - structural_main_score[:, 1]
        ).detach()
        valid_struct = user_state["has_history"].to(torch.bool)
        if bool(valid_struct.any()):
            residual_bpr, residual_aux = self._decoupled_structural_objective(
                main_margin_det[valid_struct],
                scored["raw_residual"][valid_struct, :2],
                scored["residual_scale"],
                scored["main_score_scale"],
                self.main_error_weight_floor,
            )
        else:
            z = scored["residual"].sum() * 0.0
            residual_bpr = z
            residual_aux = {
                "need_weight": z,
                "preserve_weight": z,
                "raw_rank_loss": z,
                "calibration_loss": z,
                "rank_help": z,
                "preserve_penalty": z,
                "preserve_amplitude": z,
                "preserve_margin": z,
                "raw_rank_margin": z,
                "correction_margin_normalized": z,
                "main_margin_normalized": main_margin_det,
            }
        final_margin_detmain = main_margin_det + residual_margin

        relation_loss = self._relation_loss(item_state)
        sem = self._semantic_losses(item_state, bundle_state, user_state)

        pcl_weighted = (
            self.lambda_pcl_item * sem["pcl_item"]
            + self.lambda_pcl_bundle * sem["pcl_bundle"]
            + self.lambda_pcl_user * sem["pcl_user"]
        )
        orth_weighted = (
            self.lambda_orth_item * sem["orth_item"]
            + self.lambda_orth_bundle * sem["orth_bundle"]
            + self.lambda_orth_user * sem["orth_user"]
        )
        total = (
            main_bpr
            + self.conf["c_lambda"] * c_loss
            + self.lambda_relation * relation_loss
            + self.lambda_residual * residual_bpr
            + pcl_weighted
            + orth_weighted
        )

        losses = {
            "main_bpr": main_bpr,
            "cl": c_loss,
            "relation": relation_loss,
            "residual_bpr": residual_bpr,
            "raw_struct_rank": residual_aux["raw_rank_loss"],
            "calibration_loss": residual_aux["calibration_loss"],
            "residual_rank_help": residual_aux["rank_help"],
            "residual_preserve": residual_aux["preserve_penalty"],
            "role_affinity_pseudo_diag": sem["role_semantic_consistency"],
            **sem,
            "pcl_weighted": pcl_weighted,
            "orth_weighted": orth_weighted,
            "total": total,
        }

        if not all(bool(torch.isfinite(v)) for v in losses.values()):
            raise RuntimeError("CFDGN non-finite loss")

        with torch.no_grad():
            valid = user_state["has_history"]
            if bool(valid.any()):
                s = scored["residual"][valid]
                raw_s = scored["raw_residual"][valid]
                fs = scored["final"][valid]
                ms = structural_main_score[valid]
                struct_gap = (s[:, 0] - s[:, 1:].mean(dim=1)).mean()
                raw_struct_gap = (raw_s[:, 0] - raw_s[:, 1:].mean(dim=1)).mean()
                main_gap = (ms[:, 0] - ms[:, 1:].mean(dim=1)).mean()
                final_gap = (fs[:, 0] - fs[:, 1:].mean(dim=1)).mean()
                struct_abs_mean = s.abs().mean()
                struct_std_mean = s.std(dim=1, unbiased=False).mean()
                raw_struct_abs_mean = raw_s.abs().mean()
                raw_struct_std_mean = raw_s.std(dim=1, unbiased=False).mean()
                deploy_cap_det = (
                    self._deploy_residual_multiplier().detach().to(s.dtype)
                )
                struct_sat95 = (
                    s.abs() >= (0.95 * deploy_cap_det).clamp_min(EPS)
                ).float().mean()
                easy_main_gap = (
                    main_score[valid, 0] - main_score[valid, 1:].mean(dim=1)
                ).mean()
                repo_main_gap = (ms[:, 0] - ms[:, 1]).mean()
                repo_struct_gap = (s[:, 0] - s[:, 1]).mean()
                repo_struct_win = (s[:, 0] > s[:, 1]).float().mean()
                mm = ms[:, 0] - ms[:, 1]
                fm = fs[:, 0] - fs[:, 1]
                main_wrong = mm <= 0
                main_right = ~main_wrong
                correction_rate = (
                    (fm[main_wrong] > 0).float().mean()
                    if bool(main_wrong.any()) else total.detach() * 0.0
                )
                damage_rate = (
                    (fm[main_right] <= 0).float().mean()
                    if bool(main_right.any()) else total.detach() * 0.0
                )
                main_wrong_frac = main_wrong.float().mean()
            else:
                z = total.detach() * 0.0
                struct_gap = raw_struct_gap = main_gap = final_gap = z
                struct_abs_mean = struct_std_mean = raw_struct_abs_mean = raw_struct_std_mean = struct_sat95 = z
                easy_main_gap = repo_main_gap = repo_struct_gap = repo_struct_win = z
                correction_rate = damage_rate = main_wrong_frac = z
            four = scored["four_components"]
            comp_mean = four.mean(dim=(0, 1))
            iw = scored["components"]["interest_weight"]
            pref_rel = scored["components"]["composition_reliability"]
            if bool(valid.any()):
                iw_valid = iw[valid]
                selector_entropy = _normalized_entropy(iw_valid).mean()
                selector_top1 = iw_valid.max(dim=-1).values.mean()
                composition_reliability = pref_rel[valid].mean()
            else:
                zdiag = total.detach() * 0.0
                selector_entropy = selector_top1 = composition_reliability = zdiag
            type_mean = bundle_state["type_prob"].mean(dim=0)
            src_mean = bundle_state["source_weight"].mean(dim=0)
            self._last_diag = {
                "main_bpr": float(main_bpr.detach().cpu()),
                "residual_bpr": float(residual_bpr.detach().cpu()),
                "raw_struct_rank": float(residual_aux["raw_rank_loss"].detach().cpu()),
                "calibration_loss": float(residual_aux["calibration_loss"].detach().cpu()),
                "residual_rank_help": float(residual_aux["rank_help"].detach().cpu()),
                "residual_preserve": float(residual_aux["preserve_penalty"].detach().cpu()),
                "residual_preserve_amplitude": float(residual_aux["preserve_amplitude"].detach().cpu()),
                "residual_preserve_margin": float(residual_aux["preserve_margin"].detach().cpu()),
                "main_need_weight": float(residual_aux["need_weight"].detach().cpu()),
                "main_preserve_weight": float(residual_aux["preserve_weight"].detach().cpu()),
                "main_error_weight_floor": float(self.main_error_weight_floor),
                "raw_rank_margin": float(residual_aux["raw_rank_margin"].detach().cpu()),
                "residual_scale": float(self._residual_scale().detach().cpu()),
                "main_score_scale": float(self._main_score_scale().detach().cpu()),
                "deploy_multiplier": float(
                    self._deploy_residual_multiplier().detach().cpu()
                ),
                "normalized_main_gap": float(
                    (main_gap / self._main_score_scale().detach()).cpu()
                ),
                "relation_loss": float(relation_loss.detach().cpu()),
                "main_gap": float(main_gap.detach().cpu()),
                "raw_residual_gap": float(raw_struct_gap.detach().cpu()),
                "residual_gap": float(struct_gap.detach().cpu()),
                "final_gap": float(final_gap.detach().cpu()),
                "main_wrong_frac": float(main_wrong_frac.detach().cpu()),
                "correction_rate": float(correction_rate.detach().cpu()),
                "damage_rate": float(damage_rate.detach().cpu()),
                "raw_residual_abs_mean": float(raw_struct_abs_mean.detach().cpu()),
                "raw_residual_std_mean": float(raw_struct_std_mean.detach().cpu()),
                "residual_abs_mean": float(struct_abs_mean.detach().cpu()),
                "residual_std_mean": float(struct_std_mean.detach().cpu()),
                "residual_sat95": float(struct_sat95.detach().cpu()),
                "easy_main_gap": float(easy_main_gap.detach().cpu()),
                "repo_main_gap": float(repo_main_gap.detach().cpu()),
                "repo_struct_gap": float(repo_struct_gap.detach().cpu()),
                "repo_struct_win": float(repo_struct_win.detach().cpu()),
                "score_facet": float(comp_mean[0].detach().cpu()),
                "score_bundle": float(comp_mean[1].detach().cpu()),
                "score_type": float(comp_mean[2].detach().cpu()),
                "score_comp": float(comp_mean[3].detach().cpu()),
                "interest_selector_entropy": float(selector_entropy.detach().cpu()),
                "interest_selector_top1": float(selector_top1.detach().cpu()),
                "composition_reliability": float(composition_reliability.detach().cpu()),
                "struct_neg_count": float(max(0, structural_bundles.shape[1] - 1)),
                "history_valid_frac": float(valid.float().mean().detach().cpu()),
                "role_entropy": float(_normalized_entropy(bundle_state["role"]).mean().detach().cpu()),
                "role_diversity": float(bundle_state["diversity"].mean().detach().cpu()),
                "role_diversity_target": float(
                    bundle_state["semantic_diversity_target"][bundle_state["semantic_diversity_target_valid"]].mean().detach().cpu()
                    if bool(bundle_state["semantic_diversity_target_valid"].any())
                    else 0.0
                ),
                "fixed_ui_target_mean": float(self.fixed_ui_diversity_stats["target_mean_train"]),
                "fixed_ui_target_std": float(self.fixed_ui_diversity_stats["target_std_train"]),
                "role_affinity_pseudo_diag": float(sem["role_semantic_consistency"].detach().cpu()),
                "role_diversity_loss_diag": float(sem["role_diversity_consistency_diag"].detach().cpu()),
                "role_affinity_pred": float(bundle_state["role_affinity_pred"].mean().detach().cpu()),
                "role_affinity_target": float(
                    bundle_state["role_affinity_target"][bundle_state["role_affinity_target_valid"]].mean().detach().cpu()
                    if bool(bundle_state["role_affinity_target_valid"].any()) else 0.0
                ),
                "role_certainty": float(bundle_state["certainty"].mean().detach().cpu()),
                "relation_compatibility": float(bundle_state["compatibility"].mean().detach().cpu()),
                "type_sim": float(type_mean[0].detach().cpu()),
                "type_comp": float(type_mean[1].detach().cpu()),
                "type_noise": float(type_mean[2].detach().cpu()),
                "source_prior": float(src_mean[0].detach().cpu()),
                "source_goal": float(src_mean[1].detach().cpu()),
                "source_context": float(src_mean[2].detach().cpu()),
                "source_ppmi": float(src_mean[3].detach().cpu()),
                "ppmi_role_shift": float(bundle_state["ppmi_role_shift"].mean().detach().cpu()),
                "ppmi_local_active": float(bundle_state["ppmi_active_fraction"].mean().detach().cpu()),
                "ppmi_gate": float(item_state["ppmi_gate"].mean().detach().cpu()),
                "pcl_item": float(sem["pcl_item"].detach().cpu()),
                "pcl_bundle": float(sem["pcl_bundle"].detach().cpu()),
                "pcl_user": float(sem["pcl_user"].detach().cpu()),
                "orth_item": float(sem["orth_item"].detach().cpu()),
                "orth_bundle": float(sem["orth_bundle"].detach().cpu()),
                "orth_user": float(sem["orth_user"].detach().cpu()),
            }
            self.gfr_train_steps.add_(1)

        return {
            "scores": {
                "main": main_score,
                "structural_main": structural_main_score,
                "structural_bundles": structural_bundles,
                "raw_residual": scored["raw_residual"],
                "residual": scored["residual"],
                "residual_scale": scored["residual_scale"],
                "final": scored["final"],
            },
            "bundle": {
                "rho": bundle_state["rho"],
                "role_diversity": bundle_state["diversity"],
                "role_diversity_target": bundle_state["semantic_diversity_target"],
                "role_diversity_target_valid": bundle_state["semantic_diversity_target_valid"],
                "role_certainty": bundle_state["certainty"],
                "relation_compatibility": bundle_state["compatibility"],
                "type_prob": bundle_state["type_prob"],
                "type_valid": bundle_state["type_valid"],
                "ppmi_coverage": bundle_state["ppmi_coverage"],
            },
            "role": {
                "prob": bundle_state["role"],
                "source_weight": bundle_state["source_weight"],
                "source_norm": bundle_state["source_norm"],
            },
            "loss": losses,
        }

    def get_multi_modal_representations(self, test: bool = False):
        if not test:
            v = self._views(test=False)
            return v["users_main"], v["bundles_main"]

        views = self._views(test=True)
        self._set_main_score_scale_from_representations(
            views["users_main"], views["bundles_main"]
        )
        item_state = self._structural_item_state(views)
        all_b = torch.arange(
            self.num_bundles, device=self.device, dtype=torch.long
        )
        bundle_state = self._bundle_state_selected(all_b, item_state)
        user_state = self._user_state_full(bundle_state, views["ui_users"])
        return {
            "users_main": views["users_main"],
            "bundles_main": views["bundles_main"],
            "bundle_state": bundle_state,
            "user_state": user_state,
        }

    def evaluate_unified(
        self,
        propagate_result: Mapping[str, object],
        users: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(propagate_result, dict):
            raise TypeError("CFDGN unified evaluation requires dict state")
        users = users.reshape(-1)
        if train_mask.shape != (users.numel(), self.num_bundles):
            raise ValueError("train_mask must be [num_users_in_batch, num_bundles]")

        um = propagate_result["users_main"][users]
        bm = propagate_result["bundles_main"]
        main_raw = torch.mm(um, bm.t())
        mask = train_mask.to(device=main_raw.device, dtype=torch.bool)
        main = main_raw.masked_fill(mask, -torch.inf)

        user_state = propagate_result["user_state"]
        bundle_state = propagate_result["bundle_state"]
        has_history = user_state["has_history"][users].to(torch.bool)

        residual = torch.zeros_like(main_raw)
        component_sum = torch.zeros(
            4, dtype=main_raw.dtype, device=main_raw.device
        )
        component_sq_sum = torch.zeros_like(component_sum)
        component_count = 0
        for start in range(0, self.num_bundles, self.eval_bundle_chunk):
            end = min(self.num_bundles, start + self.eval_bundle_chunk)
            r, four = self._struct_score_catalog_chunk(
                user_state, bundle_state, users, start, end
            )
            residual[:, start:end] = r
            if bool(has_history.any()):
                four_active = four[has_history]
                component_sum += four_active.sum(dim=(0, 1))
                component_sq_sum += four_active.square().sum(dim=(0, 1))
                component_count += int(four_active.shape[0] * four_active.shape[1])

        residual = torch.where(
            has_history.unsqueeze(-1), residual, torch.zeros_like(residual)
        )
        final = (main_raw + residual).masked_fill(mask, -torch.inf)

        if component_count > 0:
            mean = component_sum / float(component_count)
            var = component_sq_sum / float(component_count) - mean.square()
            component_std = var.clamp_min(0.0).sqrt()
        else:
            component_std = torch.zeros_like(component_sum)

        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError("non-finite unified structural residual")
        if not bool(torch.equal(torch.isneginf(main), torch.isneginf(final))):
            raise RuntimeError("TRAIN mask changed between Main and Final")

        return {
            "main": main,
            "final": final,
            "residual": residual,
            "residual_scale": self._residual_scale().detach(),
            "main_score_scale": self._main_score_scale().detach(),
            "deploy_multiplier": self._deploy_residual_multiplier().detach(),
            "has_history": has_history,
            "component_std": component_std,
            "component_sum": component_sum,
            "component_sq_sum": component_sq_sum,
            "component_count": int(component_count),
        }

    def evaluate_rerank_components(self, *args, **kwargs):
        raise RuntimeError(
            "post-hoc Top-M rerank is removed; use evaluate_unified(...)"
        )

    def compose_rerank(self, *args, **kwargs):
        raise RuntimeError(
            "post-hoc lambda composition is removed in DECOUPLED-SCALE-INVARIANT"
        )

    def evaluate_rerank(self, *args, **kwargs):
        raise RuntimeError(
            "post-hoc rerank is removed; use evaluate_unified(...)"
        )

    def evaluate(self, propagate_result, users):
        if not isinstance(propagate_result, dict):
            return super().evaluate(propagate_result, users)
        raise RuntimeError(
            "CFDGN DECOUPLED-SCALE-INVARIANT requires evaluate_unified(..., train_mask). "
            "TRAIN masking is part of the exact deployment score."
        )


    def get_diagnostic_info(self) -> Dict[str, float]:
        return dict(self._last_diag)

    def get_parameter_ownership(self):
        main, structural = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (main if name in self._gfr_backbone_param_name_set else structural).append((name, p))
        return main, structural
