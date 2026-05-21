import dataloader
import torch
from collections import defaultdict
from dataloader import BasicDataset
from torch import nn
from torch.nn import functional as F
import scipy.sparse as sp
import numpy as np
import os
import ode as ode
try:
    from torch_scatter import scatter_add
except ImportError:
    def scatter_add(src, index, dim=0, dim_size=None):
        if dim != 0:
            raise NotImplementedError("Fallback scatter_add only supports dim=0.")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        return out.index_add(dim, index, src)


class GraphODEEncoder(nn.Module):
    """Graph ODE based encoder that evolves node embeddings through a continuous-time dynamic."""

    def __init__(self, graph, latent_dim, data_name, t_end=1.0, solver="euler"):
        super().__init__()
        self.graph = graph
        self.latent_dim = latent_dim
        self.data_name = data_name
        self.t_end = t_end
        self.solver = solver
        self.ode_block = ode.ODEblock(
            ode.ODEFunc(graph, latent_dim, data_name),
            t=torch.tensor([0.0, t_end]),
            solver=solver,
        )

    def forward(self, x: torch.Tensor, t_eval=None, return_traj: bool = False) -> torch.Tensor:
        return self.ode_block(x, t_eval=t_eval, return_traj=return_traj)


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class CDE_CF(BasicModel):
    def __init__(self, args, dataset: BasicDataset):
        super(CDE_CF, self).__init__()
        self.args = args
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def _resolve_dataset_artifact_path(self, base_path, stem_suffix=None, allow_fallback=True):
        """Prefer dataset-specific artifacts when they exist.

        Example:
        - base_path=./data/item_text_emb.npy
        - data_name=Beauty
        -> try ./data/Beauty_item_text_emb.npy first, then base_path
        """
        if not base_path:
            return base_path

        norm_path = os.path.normpath(base_path)
        dir_name = os.path.dirname(norm_path) or "."
        base_name = os.path.basename(norm_path)
        stem, ext = os.path.splitext(base_name)

        candidate_names = []
        if stem_suffix and stem.endswith(stem_suffix):
            prefix = stem[: -len(stem_suffix)]
            if prefix:
                candidate_names.append(f"{self.args.data_name}_{stem_suffix}{ext}")
        candidate_names.append(f"{self.args.data_name}_{base_name}")

        for candidate_name in candidate_names:
            candidate_path = os.path.join(dir_name, candidate_name)
            if os.path.exists(candidate_path):
                return candidate_path

        if allow_fallback and os.path.exists(norm_path):
            return norm_path
        return norm_path

    def __init_weight(self):
        self.device = self.args.device
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.session_propagation = False

        # keep compatibility with helper functions
        self.num_user = self.num_users
        self.num_item = self.num_items

        self.latent_dim = self.args.recdim

        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # ---- Text embeddings: item-side trainable semantic gate and MV-DDI bridge evidence ----
        self.use_td_init = bool(getattr(self.args, "use_td_init", 1))
        self.td_init_target = str(getattr(self.args, "td_init_target", "item")).lower()
        self.td_init_normalize = bool(getattr(self.args, "td_init_normalize", 1))
        self.td_gate_init = float(getattr(self.args, "td_gate_init", 0.2))
        self.td_tail_gate = bool(getattr(self.args, "td_tail_gate", 1))
        self.td_gate_floor = float(np.clip(getattr(self.args, "td_gate_floor", 0.05), 0.0, 1.0))
        self.td_debug = bool(getattr(self.args, "td_debug", 0))
        self.td_debug_every = max(int(getattr(self.args, "td_debug_every", 200)), 1)
        self._td_forward_calls = 0
        self.text_emb_dim = int(getattr(self.args, "text_emb_dim", 384))
        self.text_emb_path = self._resolve_dataset_artifact_path(
            getattr(self.args, "text_emb_path", "./data/item_text_emb.npy"),
            stem_suffix="item_text_emb",
        )
        self.item_text_asins_path = self._resolve_dataset_artifact_path(
            os.path.join(os.path.dirname(self.text_emb_path), "item_text_asins.txt"),
            stem_suffix="item_text_asins",
        )
        self.td_semantic_adapter = None
        self.td_item_gate = None
        self._td_initial_params = {}
        self.td_init_item_enabled = False
        self.item_text_emb = None
        self.item_text_semantic = None
        need_text_emb = (
            self.use_td_init
            or (
                bool(getattr(self.args, "use_semantic_bridge", 1))
                and bool(getattr(self.args, "bridge_use_semantic_view", 1))
            )
        )
        if need_text_emb:
            if os.path.exists(self.text_emb_path):
                text_emb = np.load(self.text_emb_path)
                # expected shape: [num_items, text_dim]
                if text_emb.shape[0] != self.num_items:
                    print(
                        f"[Text] Disable text features: embedding rows {text_emb.shape[0]} != num_items {self.num_items} "
                        f"for dataset {self.args.data_name}. File: {self.text_emb_path}"
                    )
                    self.use_td_init = False
                else:
                    self.item_text_emb = torch.tensor(text_emb, dtype=torch.float32)
                    print(f"[Text] Loaded item text emb {text_emb.shape} from {self.text_emb_path}")
                    if self.use_td_init:
                        self.item_text_semantic = self._build_item_text_semantic(text_emb)
                        adapter_dropout = float(getattr(self.args, "td_adapter_dropout", 0.0))
                        self.td_semantic_adapter = nn.Sequential(
                            nn.Linear(self.item_text_semantic.shape[1], self.latent_dim),
                            nn.LayerNorm(self.latent_dim),
                            nn.GELU(),
                            nn.Dropout(adapter_dropout),
                        )
                        self.td_item_gate = nn.Sequential(
                            nn.Linear(self.latent_dim * 2, self.latent_dim),
                            nn.Sigmoid(),
                        )
                        self._init_semantic_modules(self.td_semantic_adapter, self.td_item_gate)
                        self._init_item_gate_bias()
            else:
                print(f"[Text] requested text features but embedding file not found: {self.text_emb_path}. Disable text features.")
                self.use_td_init = False

        if self.use_td_init:
            self._prepare_item_semantic_gated_fusion()
            self._capture_td_initial_params()

        self.f = nn.Sigmoid()
        self.num_nodes = self.num_users + self.num_items

        # Hyper-parameters for building the enhanced I-I graph
        self.knn_k = getattr(self.args, "knn_k", 10)
        self.num_of_session_k = getattr(self.args, "num_of_session_k", 10)
        self.degree_n = getattr(self.args, "degree_n", 3)
        self.gamma = getattr(self.args, "gamma", 0.2)
        self.tau = getattr(self.args, "tau", 1.0)

        # item co-occurrence dict
        self.item_graph_dict = self.dataset.item_graph_dict

        # Metadata is used only as optional MV-DDI candidate evidence.
        self.related_edges_path = self._resolve_dataset_artifact_path(
            getattr(self.args, "related_edges_path", "./data/item_related_edges.npy"),
            stem_suffix="item_related_edges",
        )
        self.related_type_weight = {
            "also_bought": float(getattr(self.args, "related_w_also_bought", 0.8)),
            "also_viewed": float(getattr(self.args, "related_w_also_viewed", 0.5)),
            "bought_together": float(getattr(self.args, "related_w_bought_together", 1.0)),
        }

        self.use_tcrc = bool(getattr(self.args, "use_tcrc", 0))
        self.use_tcrc_pop_calibration = bool(getattr(self.args, "use_tcrc_pop_calibration", 0))
        self._init_tcrc_statistics()

        # Build augmented graph for Graph ODE.
        self.Graph = self._build_augmented_graph(self.dataset.UserItemNet)

        # Graph ODE encoder.
        self.encoder = GraphODEEncoder(
            self.Graph,
            self.latent_dim,
            self.args.data_name,
            t_end=getattr(self.args, "t", 1.0),
            solver=getattr(self.args, "solver", "euler"),
        )
        self.odeblock = self.encoder.ode_block

    @staticmethod
    def _l2_normalize_tensor(x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-12)

    @staticmethod
    def _format_tensor_stats(name, x):
        x = x.detach()
        if x.numel() == 0:
            return f"{name}=empty"
        return (
            f"{name}: mean={float(x.mean()):.6f}, std={float(x.std()):.6f}, "
            f"norm={float(x.norm(dim=-1).mean()):.6f}, "
            f"min={float(x.min()):.6f}, max={float(x.max()):.6f}"
        )

    @staticmethod
    def _init_semantic_modules(*modules):
        for module in modules:
            if module is None:
                continue
            for _, param in module.named_parameters():
                try:
                    nn.init.xavier_normal_(param.data)
                except ValueError:
                    pass

    def _init_item_gate_bias(self):
        if self.td_item_gate is None:
            return
        gate_init = float(np.clip(self.td_gate_init, 1e-4, 1.0 - 1e-4))
        linear = self.td_item_gate[0] if isinstance(self.td_item_gate, nn.Sequential) else self.td_item_gate
        if isinstance(linear, nn.Linear) and linear.bias is not None:
            bias = float(np.log(gate_init / (1.0 - gate_init)))
            nn.init.constant_(linear.bias, bias)
            print(f"[TD-Gate] initialized item gate bias to {bias:.4f} for target gate mean {gate_init:.4f}")

    def _capture_td_initial_params(self):
        self._td_initial_params = {}
        for prefix, module in (
            ("semantic_adapter", self.td_semantic_adapter),
            ("item_gate", self.td_item_gate),
        ):
            if module is None:
                continue
            for name, param in module.named_parameters():
                self._td_initial_params[f"{prefix}.{name}"] = param.detach().cpu().clone()

    def _build_item_text_semantic(self, text_emb):
        semantic = np.asarray(text_emb, dtype=np.float32)
        if bool(getattr(self.args, "td_init_normalize", 1)):
            semantic = semantic / (np.linalg.norm(semantic, axis=1, keepdims=True) + 1e-12)
        print(f"[TD-Gate] Built item text semantic embedding: {semantic.shape}")
        return torch.tensor(semantic, dtype=torch.float32)

    def _get_td_item_semantic(self, device):
        if self.item_text_semantic is not None and self.td_semantic_adapter is not None:
            return self.td_semantic_adapter(self.item_text_semantic.to(device))
        return None

    def _prepare_item_semantic_gated_fusion(self):
        if self.item_text_emb is None or self.td_semantic_adapter is None:
            print("[TD-Gate] Disabled: need item text embeddings and semantic adapter.")
            self.use_td_init = False
            return

        target = self.td_init_target
        if target != "item":
            print(f"[TD-Gate] target={target} requested, but current method applies semantic fusion to item only.")
            target = "item"

        self.td_init_target = "item"
        self.td_init_item_enabled = True
        print(
            f"[TD-Gate] trainable gated fusion enabled: target={target}, semantic_dim={self.item_text_semantic.shape[1]}, "
            f"normalize={int(self.td_init_normalize)}, gate_init={float(np.clip(self.td_gate_init, 1e-4, 1.0 - 1e-4)):.4f}, "
            f"tail_gate={int(self.td_tail_gate)}, gate_floor={self.td_gate_floor:.4f}"
        )
        if self.td_debug:
            with torch.no_grad():
                semantic = self.item_text_semantic
                item_id = self.embedding_item.weight.detach().cpu()
                print("[TD-Gate][Init] " + self._format_tensor_stats("raw_text_semantic", semantic))
                print("[TD-Gate][Init] " + self._format_tensor_stats("item_id", item_id))
                print(
                    f"[TD-Gate][Init] adapter={self.td_semantic_adapter}, "
                    f"gate={self.td_item_gate}, dropout={float(getattr(self.args, 'td_adapter_dropout', 0.0))}"
                )

    def _init_tcrc_statistics(self):
        item_degree = np.asarray(self.dataset.UserItemNet.sum(axis=0)).reshape(-1).astype(np.float32)
        if item_degree.shape[0] != self.num_items:
            item_degree = np.resize(item_degree, self.num_items).astype(np.float32)
        log_degree = np.log1p(item_degree).astype(np.float32)
        log_dmax = float(log_degree.max()) if log_degree.size > 0 else 0.0
        log_dmin = float(log_degree.min()) if log_degree.size > 0 else 0.0
        denom = max(log_dmax - log_dmin, 1e-12)
        if log_dmax <= log_dmin:
            item_tau = np.zeros_like(log_degree, dtype=np.float32)
        else:
            item_tau = ((log_dmax - log_degree) / denom).astype(np.float32)
        item_tau = np.clip(item_tau, 0.0, 1.0).astype(np.float32)
        pop_pref_item = (log_degree / (log_dmax + 1e-12)).astype(np.float32) if log_dmax > 0 else np.zeros_like(log_degree)
        user_pop_pref = np.zeros(self.num_users, dtype=np.float32)
        all_pos = self.dataset.allPos
        for u in range(self.num_users):
            pos_items = all_pos[u]
            if len(pos_items) > 0:
                user_pop_pref[u] = float(pop_pref_item[np.asarray(pos_items, dtype=np.int64)].mean())
        gamma0 = float(getattr(self.args, "tcrc_gamma0", 0.05))
        user_gamma = (gamma0 * (1.0 - user_pop_pref)).astype(np.float32)
        self.register_buffer("tcrc_item_tau", torch.tensor(item_tau, dtype=torch.float32), persistent=False)
        self.register_buffer("tcrc_item_log_pop", torch.tensor(log_degree, dtype=torch.float32), persistent=False)
        self.register_buffer("tcrc_user_gamma", torch.tensor(user_gamma, dtype=torch.float32), persistent=False)
        if self.use_tcrc or self.use_tcrc_pop_calibration:
            print(f"[TCRC] item degree quantiles: {np.quantile(item_degree, [0.0, 0.5, 0.8, 0.9, 1.0])}")
            print(f"[TCRC] tail strength quantiles: {np.quantile(item_tau, [0.0, 0.5, 0.8, 0.9, 1.0])}")
            print(f"[TCRC] user gamma quantiles: {np.quantile(user_gamma, [0.0, 0.5, 0.8, 0.9, 1.0])}")

    # ------------------------ Embeddings ------------------------
    def _get_td_gated_embeddings(self):
        """Return trainable ID embeddings dynamically fused with semantic gates."""
        user_id_emb = self.embedding_user.weight
        item_id_emb = self.embedding_item.weight
        if not self.use_td_init or self.td_semantic_adapter is None or self.item_text_semantic is None:
            return user_id_emb, item_id_emb

        item_semantic_adapted = self._get_td_item_semantic(item_id_emb.device)
        if item_semantic_adapted is None:
            return user_id_emb, item_id_emb
        if self.td_init_normalize:
            item_semantic_adapted = self._l2_normalize_tensor(item_semantic_adapted)

        item_emb = item_id_emb
        if self.td_init_item_enabled and self.td_item_gate is not None:
            item_gate = self.td_item_gate(torch.cat([item_semantic_adapted, item_id_emb], dim=-1))
            if self.td_gate_floor > 0.0:
                item_gate = self.td_gate_floor + (1.0 - self.td_gate_floor) * item_gate
            if self.td_tail_gate and hasattr(self, "tcrc_item_tau"):
                item_gate = item_gate * self.tcrc_item_tau.to(item_gate.device).unsqueeze(1)
            item_emb = item_gate * item_semantic_adapted + (1.0 - item_gate) * item_id_emb
            if self.td_debug and self.training:
                self._td_forward_calls += 1
                if self._td_forward_calls == 1 or self._td_forward_calls % self.td_debug_every == 0:
                    delta = item_emb - item_id_emb
                    cos = F.cosine_similarity(item_semantic_adapted, item_id_emb, dim=-1)
                    print(
                        f"[TD-Gate][Forward step={self._td_forward_calls}] "
                        + self._format_tensor_stats("semantic_adapted", item_semantic_adapted)
                    )
                    print(
                        f"[TD-Gate][Forward step={self._td_forward_calls}] "
                        + self._format_tensor_stats("item_gate", item_gate)
                    )
                    print(
                        f"[TD-Gate][Forward step={self._td_forward_calls}] "
                        + self._format_tensor_stats("fused_minus_id", delta)
                        + f", cos_sem_id_mean={float(cos.mean()):.6f}"
                    )

        return user_id_emb, item_emb

    def log_td_gradients(self, step=None):
        if not self.td_debug or not self.use_td_init:
            return
        tag = f" step={step}" if step is not None else ""
        parts = []
        for name, module in (
            ("semantic_adapter", self.td_semantic_adapter),
            ("item_gate", self.td_item_gate),
        ):
            if module is None:
                continue
            grad_norms = []
            param_norms = []
            for param in module.parameters():
                param_norms.append(float(param.detach().norm()))
                if param.grad is not None:
                    grad_norms.append(float(param.grad.detach().norm()))
            grad_sum = sum(grad_norms)
            param_sum = sum(param_norms)
            parts.append(f"{name}: grad_norm={grad_sum:.8f}, param_norm={param_sum:.8f}")
        if parts:
            print(f"[TD-Gate][Grad{tag}] " + " | ".join(parts))

    def log_td_state(self, tag="state"):
        if not self.td_debug or not self.use_td_init:
            return
        if self.td_semantic_adapter is None or self.td_item_gate is None or self.item_text_semantic is None:
            print(f"[TD-Gate][State {tag}] semantic modules are not available")
            return

        was_training = self.training
        self.eval()
        with torch.no_grad():
            device = self.embedding_item.weight.device
            item_id = self.embedding_item.weight
            semantic_adapted = self._get_td_item_semantic(device)
            if semantic_adapted is None:
                print(f"[TD-Gate][State {tag}] semantic_adapted is None")
                if was_training:
                    self.train()
                return
            if self.td_init_normalize:
                semantic_adapted = self._l2_normalize_tensor(semantic_adapted)
            raw_gate = self.td_item_gate(torch.cat([semantic_adapted, item_id], dim=-1))
            floor_gate = raw_gate
            if self.td_gate_floor > 0.0:
                floor_gate = self.td_gate_floor + (1.0 - self.td_gate_floor) * raw_gate
            effective_gate = floor_gate
            if self.td_tail_gate and hasattr(self, "tcrc_item_tau"):
                tau = self.tcrc_item_tau.to(device).unsqueeze(1)
                effective_gate = floor_gate * tau
            fused = effective_gate * semantic_adapted + (1.0 - effective_gate) * item_id
            delta = fused - item_id
            cos = F.cosine_similarity(semantic_adapted, item_id, dim=-1)

            print(f"[TD-Gate][State {tag}] " + self._format_tensor_stats("semantic_adapted", semantic_adapted))
            print(f"[TD-Gate][State {tag}] " + self._format_tensor_stats("raw_item_gate", raw_gate))
            print(f"[TD-Gate][State {tag}] " + self._format_tensor_stats("floor_item_gate", floor_gate))
            print(f"[TD-Gate][State {tag}] " + self._format_tensor_stats("effective_item_gate", effective_gate))
            print(
                f"[TD-Gate][State {tag}] "
                + self._format_tensor_stats("fused_minus_id", delta)
                + f", cos_sem_id_mean={float(cos.mean()):.6f}"
            )
            if self.td_tail_gate and hasattr(self, "tcrc_item_tau"):
                print(
                    f"[TD-Gate][State {tag}] "
                    + self._format_tensor_stats("tail_tau", self.tcrc_item_tau.to(device))
                )

            update_parts = []
            for prefix, module in (
                ("semantic_adapter", self.td_semantic_adapter),
                ("item_gate", self.td_item_gate),
            ):
                if module is None:
                    continue
                update_norm = 0.0
                base_norm = 0.0
                param_norm = 0.0
                for name, param in module.named_parameters():
                    key = f"{prefix}.{name}"
                    param_cpu = param.detach().cpu()
                    param_norm += float(param_cpu.norm())
                    if key in self._td_initial_params:
                        base = self._td_initial_params[key]
                        update_norm += float((param_cpu - base).norm())
                        base_norm += float(base.norm())
                rel_update = update_norm / (base_norm + 1e-12)
                update_parts.append(
                    f"{prefix}: update_norm={update_norm:.8f}, rel_update={rel_update:.8f}, param_norm={param_norm:.8f}"
                )
            if update_parts:
                print(f"[TD-Gate][ParamDelta {tag}] " + " | ".join(update_parts))

        if was_training:
            self.train()

    def _get_item_emb0(self):
        """Return item embeddings used as the graph input."""
        return self._get_td_gated_embeddings()[1]

    def build_I_I_graph(self):
        weighted_binary_relations, co_adj = self.get_co_occurrence_item()
        self.co_adj = co_adj
        self.weighted_binary_relations = weighted_binary_relations
        item_sim = self.build_sim(self.embedding_item.weight.detach()).to('cpu')
        session_enhanced = self.build_session_tree(item_sim)
        del item_sim
        session_adj = self.build_non_zero_graph(session_enhanced, is_sparse=False, norm_type='sym').coalesce().to(self.device)
        self.session_adj = session_adj.to(self.device)
        del session_adj, session_enhanced

    def computer(self, t_eval=None, return_traj: bool = False):
        """Compute node embeddings through Graph ODE on the augmented graph.

        If return_traj=True, returns trajectory [T, n_users, d], [T, n_items, d].
        """
        users_emb, items_emb = self._get_td_gated_embeddings()
        all_emb = torch.cat([users_emb, items_emb])

        all_emb_ode = self.encoder(all_emb, t_eval=t_eval, return_traj=return_traj)
        if return_traj:
            # [T, N, d]
            users_traj = all_emb_ode[:, : self.num_users, :]
            items_traj = all_emb_ode[:, self.num_users :, :]
            return users_traj, items_traj

        users, items = torch.split(all_emb_ode, [self.num_users, self.num_items])
        if self.session_propagation:
            h = items
            for _ in range(self.args.n_layers):
                h = torch.sparse.mm(self.session_adj, h)
            items = items + h
            del h
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        raw_scores = torch.matmul(users_emb, items_emb.t())
        if self.use_tcrc_pop_calibration:
            user_gamma = self.tcrc_user_gamma.to(raw_scores.device)[users.long()]
            item_log_pop = self.tcrc_item_log_pop.to(raw_scores.device)
            raw_scores = raw_scores - user_gamma.unsqueeze(1) * item_log_pop.unsqueeze(0)
        rating = self.f(raw_scores)
        return rating

    # ------------------------ Losses ------------------------
    def bpr_loss(self, users, pos, neg):
        full_users, full_items = self.computer()
        users_emb = full_users[users.long()]
        pos_emb = full_items[pos.long()]
        neg_emb = full_items[neg.long()]

        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos)
        negEmb0 = self.embedding_item(neg)

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        if self.use_tcrc:
            item_tau = self.tcrc_item_tau.to(pos_scores.device)
            pos_tau = item_tau[pos.long()]
            neg_tau = item_tau[neg.long()]
            pos_weight = 1.0 + float(getattr(self.args, "tcrc_lambda_t", 0.5)) * pos_tau
            neg_weight = torch.clamp(1.0 - float(getattr(self.args, "tcrc_lambda_n", 0.5)) * neg_tau, min=0.0)
            margin = float(getattr(self.args, "tcrc_margin0", 0.0)) + float(getattr(self.args, "tcrc_margin1", 0.1)) * pos_tau
            bpr_loss = torch.mean(pos_weight * neg_weight * F.softplus(neg_scores - pos_scores + margin))
        else:
            bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return bpr_loss, reg_loss

    # ------------------------ Graph build ------------------------
    def _build_augmented_graph(self, ui_spmat):
        """Build the base U-I graph and optional MV-DDI I-I bridge edges."""
        n_users, n_items = self.num_users, self.num_items
        n_total = n_users + n_items

        # U-I block
        R = ui_spmat.tocoo().astype(np.float32)
        ui_rows = R.row
        ui_cols = R.col
        ui_vals = R.data.astype(np.float32)
        base_ui_vals = ui_vals

        # Build adjacency in COO (symmetric)
        rows = []
        cols = []
        vals = []

        # add U->I and I->U
        rows.append(ui_rows)
        cols.append(ui_cols + n_users)
        vals.append(base_ui_vals)

        rows.append(ui_cols + n_users)
        cols.append(ui_rows)
        vals.append(base_ui_vals)

        ui_edges = len(ui_rows)
        raw_ui_rows = [rows[0], rows[1]]
        raw_ui_cols = [cols[0], cols[1]]
        raw_ui_vals = [vals[0], vals[1]]

        use_bridge = bool(getattr(self.args, "use_semantic_bridge", 1))
        use_ddi_target = bool(getattr(self.args, "use_ddi_target", 1))
        use_semantic_view = bool(getattr(self.args, "bridge_use_semantic_view", 1))
        use_cf_view = bool(getattr(self.args, "bridge_use_cf_view", 1))
        use_meta_view = bool(getattr(self.args, "bridge_use_meta_view", 1))

        partial_graph = None
        if use_bridge and use_ddi_target:
            partial_graph = self._build_temp_norm_graph(raw_ui_rows, raw_ui_cols, raw_ui_vals, n_total)

        # Multi-view DDI bridge edges for diffusion-deficient long-tail items
        bridge_edges = 0
        if use_bridge:
            if use_semantic_view and self.item_text_emb is None:
                print("[Bridge] Disabled: need valid item_text_emb.npy")
            elif use_ddi_target and partial_graph is not None:
                item_ddi, _ = self._compute_ddi_scores(partial_graph)
                print(f"[Bridge] DDI quantiles: {np.quantile(item_ddi, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])}")
                base_ddi_threshold = float(getattr(self.args, "ddi_threshold", 0.5))
                ddi_target_quantile = float(getattr(self.args, "ddi_target_quantile", 0.0))
                effective_ddi_threshold = base_ddi_threshold
                if 0.0 < ddi_target_quantile < 1.0:
                    quantile_threshold = float(np.quantile(item_ddi, ddi_target_quantile))
                    effective_ddi_threshold = max(base_ddi_threshold, quantile_threshold)
                    print(
                        f"[Bridge] adaptive DDI target threshold: max(abs={base_ddi_threshold:.6f}, "
                        f"q{ddi_target_quantile:.2f}={quantile_threshold:.6f}) = {effective_ddi_threshold:.6f}"
                    )
                else:
                    print(f"[Bridge] fixed DDI target threshold: {effective_ddi_threshold:.6f}")
                br_rows, br_cols, br_vals = self._build_multiview_ddi_bridge_edges(
                    item_ddi=item_ddi,
                    ddi_threshold=effective_ddi_threshold,
                    top_k_sem=int(getattr(self.args, "bridge_k_sem", 20)),
                    top_k_cf=int(getattr(self.args, "bridge_k_cf", 20)),
                    top_k_meta=int(getattr(self.args, "bridge_k_meta", 20)),
                    top_k_bridge=int(getattr(self.args, "bridge_k", 5)),
                    alpha_sem=float(getattr(self.args, "bridge_alpha_sem", 0.5)),
                    alpha_cf=float(getattr(self.args, "bridge_alpha_cf", 0.3)),
                    alpha_meta=float(getattr(self.args, "bridge_alpha_meta", 0.2)),
                    agreement_lambda=float(getattr(self.args, "bridge_bonus_gamma", 0.2)),
                    cf_mode=str(getattr(self.args, "bridge_cf_mode", "ppmi")),
                    anchor_mode=str(getattr(self.args, "bridge_anchor_mode", "degree")),
                    anchor_sigma=float(getattr(self.args, "bridge_anchor_sigma", 1.0)),
                    anchor_gamma=float(getattr(self.args, "bridge_anchor_gamma", 0.5)),
                    use_ddi_target=use_ddi_target,
                    use_semantic_view=use_semantic_view,
                    use_cf_view=use_cf_view,
                    use_meta_view=use_meta_view,
                )
                if len(br_rows) > 0:
                    rows.append(br_rows)
                    cols.append(br_cols)
                    vals.append(br_vals)
                    bridge_edges = len(br_rows) // 2
            elif not use_ddi_target:
                print("[Bridge] DDI target selection disabled: bridge targets are all items.")
                item_ddi = np.zeros(self.num_items, dtype=np.float32)
                br_rows, br_cols, br_vals = self._build_multiview_ddi_bridge_edges(
                    item_ddi=item_ddi,
                    ddi_threshold=float(getattr(self.args, "ddi_threshold", 0.5)),
                    top_k_sem=int(getattr(self.args, "bridge_k_sem", 20)),
                    top_k_cf=int(getattr(self.args, "bridge_k_cf", 20)),
                    top_k_meta=int(getattr(self.args, "bridge_k_meta", 20)),
                    top_k_bridge=int(getattr(self.args, "bridge_k", 5)),
                    alpha_sem=float(getattr(self.args, "bridge_alpha_sem", 0.5)),
                    alpha_cf=float(getattr(self.args, "bridge_alpha_cf", 0.3)),
                    alpha_meta=float(getattr(self.args, "bridge_alpha_meta", 0.2)),
                    agreement_lambda=float(getattr(self.args, "bridge_bonus_gamma", 0.2)),
                    cf_mode=str(getattr(self.args, "bridge_cf_mode", "ppmi")),
                    anchor_mode=str(getattr(self.args, "bridge_anchor_mode", "degree")),
                    anchor_sigma=float(getattr(self.args, "bridge_anchor_sigma", 1.0)),
                    anchor_gamma=float(getattr(self.args, "bridge_anchor_gamma", 0.5)),
                    use_ddi_target=use_ddi_target,
                    use_semantic_view=use_semantic_view,
                    use_cf_view=use_cf_view,
                    use_meta_view=use_meta_view,
                )
                if len(br_rows) > 0:
                    rows.append(br_rows)
                    cols.append(br_cols)
                    vals.append(br_vals)
                    bridge_edges = len(br_rows) // 2

        all_r = np.concatenate(rows)
        all_c = np.concatenate(cols)
        all_v = np.concatenate(vals)
        if bool(getattr(self.args, "use_ode_weight_calibration", 1)):
            all_v = self._calibrate_weights_for_ode(all_v, bridge_edges=bridge_edges)

        # build sparse adjacency
        adj = sp.coo_matrix((all_v, (all_r, all_c)), shape=(n_total, n_total), dtype=np.float32)

        # symmetric normalization
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat)

        graph = self._convert_sp_mat_to_sp_tensor(norm_adj)

        print(
            f"[Graph Build] U-I edges: {ui_edges}, I-I(Bridge): {bridge_edges}. "
            f"Total nnz before norm: {adj.nnz}"
        )
        return graph.coalesce().to(self.device)

    def _build_temp_norm_graph(self, rows, cols, vals, n_total):
        all_r = np.concatenate(rows)
        all_c = np.concatenate(cols)
        all_v = np.concatenate(vals)
        adj = sp.coo_matrix((all_v, (all_r, all_c)), shape=(n_total, n_total), dtype=np.float32)
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        return self._convert_sp_mat_to_sp_tensor(d_mat.dot(adj).dot(d_mat)).coalesce().to(self.device)

    def _compute_ddi_scores(self, graph):
        with torch.no_grad():
            users_emb = self.embedding_user.weight.detach()
            items_emb = self.embedding_item.weight.detach()
            x0 = torch.cat([users_emb, items_emb], dim=0).to(graph.device)
            ax0 = torch.spmm(graph, x0)
            a2x0 = torch.spmm(graph, ax0)
            alpha_hat = float(getattr(self.args, "ddi_alpha_hat", 0.5))
            ddi = 1.0 - alpha_hat * a2x0.norm(dim=1) / (x0.norm(dim=1) + 1e-8)
        ddi_np = ddi.detach().cpu().numpy().astype(np.float32)
        return ddi_np[self.num_users:], ddi_np[:self.num_users]

    def _load_metadata_neighbors(self):
        if not os.path.exists(self.related_edges_path):
            return {}
        if not os.path.exists(self.item_text_asins_path):
            return {}
        with open(self.item_text_asins_path, "r", encoding="utf-8") as f:
            asins = [line.strip() for line in f if line.strip()]
        if len(asins) != self.num_items:
            return {}
        asin2iid = {a: idx for idx, a in enumerate(asins)}
        rel = np.load(self.related_edges_path, allow_pickle=True)
        max_w = max(max(self.related_type_weight.values()), 1e-9)
        meta_neighbors = defaultdict(dict)
        for (a, b, t) in rel:
            if a in asin2iid and b in asin2iid:
                ia = asin2iid[a]
                ib = asin2iid[b]
                w = float(self.related_type_weight.get(str(t), 1.0)) / max_w
                meta_neighbors[ia][ib] = max(w, meta_neighbors[ia].get(ib, 0.0))
                meta_neighbors[ib][ia] = max(w, meta_neighbors[ib].get(ia, 0.0))
        return {
            int(i): [(int(j), float(w), "metadata") for j, w in sorted(neighbors.items(), key=lambda x: x[1], reverse=True)]
            for i, neighbors in meta_neighbors.items()
        }

    @staticmethod
    def _row_max_normalize(score_dict):
        pos_scores = {
            int(k): float(v)
            for k, v in score_dict.items()
            if float(v) > 0.0 and np.isfinite(float(v))
        }
        if not pos_scores:
            return {}
        max_score = max(pos_scores.values())
        if max_score <= 0.0:
            return {}
        return {k: float(v / max_score) for k, v in pos_scores.items()}

    def _semantic_candidate_scores(self, text, item_id, top_k):
        if top_k <= 0 or self.num_items <= 1:
            return {}
        sim = text[item_id] @ text.T
        sim[item_id] = -1.0
        k = min(int(top_k), self.num_items - 1)
        if k <= 0:
            return {}
        candidate_ids = np.argpartition(sim, -k)[-k:]
        return {
            int(j): float(sim[j])
            for j in candidate_ids
            if int(j) != int(item_id) and float(sim[j]) > 0.0
        }

    def _cf_candidate_scores(self, item_id, ui_mat, item_degree, total_interactions, top_k, mode):
        if top_k <= 0 or self.num_items <= 1:
            return {}
        col_i = ui_mat.getcol(int(item_id))
        if col_i.nnz == 0:
            return {}
        di = float(item_degree[item_id])
        if di <= 0.0:
            return {}

        co_scores = col_i.T.dot(ui_mat).tocoo()
        eps = 1e-9
        raw_scores = {}
        use_ppmi = str(mode).lower() not in {"cosine", "normalized", "norm"}
        for j, c in zip(co_scores.col, co_scores.data):
            j = int(j)
            if j == int(item_id):
                continue
            c = float(c)
            dj = float(item_degree[j])
            if c <= 0.0 or dj <= 0.0:
                continue
            if use_ppmi:
                ratio = (c * total_interactions) / (di * dj + eps)
                score = max(0.0, float(np.log(max(ratio, eps))))
            else:
                score = c / np.sqrt(di * dj + eps)
            if score > 0.0:
                raw_scores[j] = score

        if not raw_scores:
            return {}
        return dict(sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[: int(top_k)])

    @staticmethod
    def _compute_anchor_reliability(item_ddi, item_degree, mode="degree", sigma=1.0):
        mode = str(mode).lower()
        if mode in {"none", "off", "flat"}:
            return np.ones_like(item_ddi, dtype=np.float32)

        ddi_term = np.clip(1.0 - item_ddi, 0.0, 1.0)
        log_deg = np.log1p(np.maximum(item_degree, 0.0))
        if mode in {"mid", "median", "medium"}:
            positive_logs = log_deg[item_degree > 0.0]
            center = float(np.median(positive_logs)) if positive_logs.size > 0 else 0.0
            sigma = max(float(sigma), 1e-6)
            degree_term = np.exp(-np.abs(log_deg - center) / sigma)
        else:
            max_log = max(float(log_deg.max()), 1e-9)
            degree_term = log_deg / max_log
        return (ddi_term * degree_term).astype(np.float32)

    def _build_multiview_ddi_bridge_edges(
        self,
        item_ddi,
        ddi_threshold,
        top_k_sem,
        top_k_cf,
        top_k_meta,
        top_k_bridge,
        alpha_sem,
        alpha_cf,
        alpha_meta,
        agreement_lambda,
        cf_mode,
        anchor_mode,
        anchor_sigma,
        anchor_gamma,
        use_ddi_target=True,
        use_semantic_view=True,
        use_cf_view=True,
        use_meta_view=True,
    ):
        text = None
        if use_semantic_view and self.item_text_emb is not None:
            text = self.item_text_emb.detach().cpu().numpy().astype(np.float32)
            text = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-12)
        ui_mat = self.dataset.UserItemNet.tocsc()
        item_degree = np.asarray(self.dataset.UserItemNet.sum(axis=0)).reshape(-1).astype(np.float32)
        total_interactions = float(item_degree.sum()) + 1e-9
        meta_neighbors = self._load_metadata_neighbors() if use_meta_view else {}
        ddi_raw = np.nan_to_num(item_ddi.astype(np.float32), nan=1.0, posinf=1.0, neginf=0.0)
        ddi_weight = np.clip(ddi_raw, 0.0, 1.0)
        anchor_reliability = self._compute_anchor_reliability(
            ddi_weight,
            item_degree,
            mode=anchor_mode,
            sigma=anchor_sigma,
        )
        longtail_ids = np.where(ddi_raw > ddi_threshold)[0] if use_ddi_target else np.arange(self.num_items)
        rows, cols, vals = [], [], []
        view_hit_counts = np.zeros(3, dtype=np.int64)
        agreement_values = []
        selected_weights = []
        enabled_view_count = int(use_semantic_view) + int(use_cf_view) + int(use_meta_view)
        enabled_view_count = max(enabled_view_count, 1)
        if use_ddi_target:
            print(f"[MV-DDI Bridge] diffusion-deficient items: {len(longtail_ids)} / {self.num_items}")
        print(
            f"[MV-DDI Bridge] views sem/cf/meta="
            f"{int(use_semantic_view)}/{int(use_cf_view)}/{int(use_meta_view)}"
        )
        if use_ddi_target:
            print(
                "[MV-DDI Bridge] raw DDI/view/anchor scores rank top-k; "
                f"anchor reliability is target-wise normalized with gamma={float(anchor_gamma):.4f} for final weights."
            )
        else:
            print("[MV-DDI Bridge] final edge weights multiply anchor reliability, multi-view score, and agreement bonus.")
        for i in longtail_ids:
            sem_scores = {}
            if use_semantic_view and text is not None:
                sem_scores = self._row_max_normalize(
                    self._semantic_candidate_scores(text, int(i), top_k_sem)
                )
            cf_scores = {}
            if use_cf_view:
                cf_scores = self._row_max_normalize(
                    self._cf_candidate_scores(int(i), ui_mat, item_degree, total_interactions, top_k_cf, cf_mode)
                )
            meta_scores = {}
            if use_meta_view:
                for j, w, _ in meta_neighbors.get(int(i), [])[: max(int(top_k_meta), 0)]:
                    if j != i:
                        meta_scores[int(j)] = max(float(w), meta_scores.get(int(j), 0.0))
                meta_scores = self._row_max_normalize(meta_scores)

            candidates = set(sem_scores.keys()) | set(cf_scores.keys()) | set(meta_scores.keys())
            if not candidates:
                continue
            scored = []
            for j in candidates:
                if j < 0 or j >= self.num_items or j == i:
                    continue
                s_sem = float(sem_scores.get(j, 0.0))
                s_cf = float(cf_scores.get(j, 0.0))
                s_meta = float(meta_scores.get(j, 0.0))
                mv_score = alpha_sem * s_sem + alpha_cf * s_cf + alpha_meta * s_meta
                if mv_score <= 0.0:
                    continue
                view_count = int(s_sem > 0.0) + int(s_cf > 0.0) + int(s_meta > 0.0)
                agreement = view_count / float(enabled_view_count)
                if use_ddi_target:
                    target_strength = float(ddi_weight[int(i)])
                else:
                    target_strength = 1.0
                anchor_score = float(anchor_reliability[j])
                base_weight = target_strength * mv_score * (1.0 + agreement_lambda * agreement)
                raw_score = base_weight * anchor_score
                if raw_score > 0:
                    view_hit_counts += np.asarray([s_sem > 0.0, s_cf > 0.0, s_meta > 0.0], dtype=np.int64)
                    agreement_values.append(agreement)
                    scored.append((int(j), float(raw_score), float(base_weight), float(anchor_score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored[: max(int(top_k_bridge), 0)]
            anchor_denominator = 1.0
            if use_ddi_target and selected:
                anchor_denominator = max(max(anchor for _, _, _, anchor in selected), 1e-12)
            gamma = max(float(anchor_gamma), 1e-6)
            for j, raw_w, base_w, anchor_score in selected:
                if use_ddi_target:
                    compensated_anchor = float(np.power(anchor_score / anchor_denominator, gamma))
                    w = float(base_w * compensated_anchor)
                else:
                    w = float(raw_w)
                rows.extend([int(i) + self.num_users, j + self.num_users])
                cols.extend([j + self.num_users, int(i) + self.num_users])
                vals.extend([w, w])
                selected_weights.append(float(w))
        if len(agreement_values) > 0:
            print(
                f"[MV-DDI Bridge] candidate view hits sem/cf/meta={view_hit_counts.tolist()}, "
                f"avg_agreement={float(np.mean(agreement_values)):.4f}"
            )
        if len(selected_weights) > 0:
            weight_arr = np.asarray(selected_weights, dtype=np.float32)
            print(
                "[MV-DDI Bridge] selected w_ij stats: "
                f"count={weight_arr.size}, "
                f"mean={float(weight_arr.mean()):.8f}, "
                f"var={float(weight_arr.var()):.8f}, "
                f"std={float(weight_arr.std()):.8f}, "
                f"min={float(weight_arr.min()):.8f}, "
                f"q25={float(np.quantile(weight_arr, 0.25)):.8f}, "
                f"median={float(np.quantile(weight_arr, 0.5)):.8f}, "
                f"q75={float(np.quantile(weight_arr, 0.75)):.8f}, "
                f"q90={float(np.quantile(weight_arr, 0.90)):.8f}, "
                f"q95={float(np.quantile(weight_arr, 0.95)):.8f}, "
                f"q99={float(np.quantile(weight_arr, 0.99)):.8f}, "
                f"max={float(weight_arr.max()):.8f}"
            )
        try:
            print(f"[MV-DDI Bridge] anchor reliability quantiles: {np.quantile(anchor_reliability, [0.0, 0.5, 0.9, 0.99, 1.0])}")
        except Exception:
            pass
        print(f"[MV-DDI Bridge] added item bridge edges: {len(rows) // 2}")
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=np.float32)

    def _calibrate_weights_for_ode(self, all_v, bridge_edges):
        all_v = all_v.astype(np.float32, copy=True)
        if len(all_v) > 0:
            p99 = float(np.percentile(all_v, 99))
            if p99 > 0:
                all_v = np.clip(all_v, 0.0, p99 * 2.0).astype(np.float32)
                print(f"[ODE-Calib] clipped weights with upper={p99 * 2.0:.6f}, bridge_edges={bridge_edges}")
        return all_v

    def _convert_sp_mat_to_sp_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # -------------- original helper functions for I-I graph --------------
    def build_sim(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_session_tree(self, sim_matrix):
        session_enhanced_z = torch.zeros_like(sim_matrix).to('cpu')
        for i in range(self.num_item):
            n_order_relationships = self.find_weighted_n_order_relationships(
                self.weighted_binary_relations, idx=i, n=self.degree_n
            )
            _, cols3 = torch.topk(sim_matrix[i], self.knn_k)
            session_enhanced_z[i, cols3] += sim_matrix[i, cols3] / 2
            for order, relation in n_order_relationships.items():
                enhanced_clos = []
                values = []
                for tail, value in relation:
                    enhanced_clos.append(tail)
                    values.append(value ** self.tau)
                order_coefficient = self.gamma * np.exp(-1 * (order - 1))
                coefficient = torch.tensor([order_coefficient * x for x in values], dtype=torch.float32).to(sim_matrix.device)
                session_enhanced_z[i, enhanced_clos] += coefficient * sim_matrix[i, enhanced_clos]
        return session_enhanced_z

    def get_co_occurrence_item(self):
        graph_co = self.item_graph_dict
        indices = []
        result = []
        weighted_binary_relations = {}
        for indx, v in graph_co.items():
            length = self.num_of_session_k if len(v[0]) >= self.num_of_session_k else len(v[0])
            indices.append(np.full(length, indx))
            result.append(v[0][: self.num_of_session_k])

            session_dict = {}
            for i in range(length):
                session_dict[v[0][i]] = v[1][i]
            weighted_binary_relations[indx] = session_dict

        indices = torch.IntTensor(np.concatenate(indices)).to(self.device)
        result = torch.IntTensor(np.concatenate(result)).to(self.device)

        indices = torch.stack((torch.flatten(indices), torch.flatten(result)), 0).to(torch.int64).to(self.device)
        adj_size = torch.Size([len(graph_co), len(graph_co)])
        return weighted_binary_relations, self.compute_normalized_laplacian(indices, adj_size)

    def find_weighted_n_order_relationships(self, graph, idx, n):
        order_dict = defaultdict(list)
        visited = set()
        visited.add(idx)
        current_level_nodes = [(idx, 0)]
        for order in range(1, n + 1):
            next_level_nodes = []
            for node, _ in current_level_nodes:
                for neighbor, weight in graph.get(node, {}).items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level_nodes.append((neighbor, weight))
            if next_level_nodes:
                if order > 1:
                    next_level_nodes = sorted(next_level_nodes, key=lambda x: x[1], reverse=True)[: int(len(current_level_nodes) / ((order - 1) * 2))]
                order_dict[order] = next_level_nodes
            current_level_nodes = next_level_nodes

        return dict(order_dict)

    def build_non_zero_graph(self, adj, is_sparse=True, norm_type='sym'):
        nonzero_indices = adj.nonzero()
        i = nonzero_indices.T
        v = adj[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        edge_index, edge_weight = self.get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_sparse_laplacian(self, edge_index, edge_weight, num_nodes, normalization='none'):
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg[deg <= 0] = 1e-7
        if normalization == 'sym':
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'rw':
            deg_inv = 1.0 / deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            edge_weight = deg_inv[row] * edge_weight
        return edge_index, edge_weight
