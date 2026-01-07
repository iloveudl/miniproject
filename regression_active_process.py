import numpy as np
import torch
import random
from torch import nn
from tqdm.auto import tqdm
from acquisition import select_top_k, select_batch_diverse
from process_data_regression import DataRegression
from regression_utils import (
    fit_last_layer_analytic,
    fit_last_layer_mfvi,
    fit_last_layer_vi,
    fit_last_layer_vi_full,
    acq_trace_analytic,
    acq_trace_mfvi,
    acq_trace_vi,
    acq_trace_vi_full,
)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RegressionTrainer:

    def __init__(self, cfg, device, output_dim):
        self.cfg = cfg
        self.device = device
        self.lr = float(cfg.lr)
        self.wd = float(cfg.weight_decay)
        self.epochs = int(cfg.epochs)
        self.sigma2 = float(cfg.sigma) ** 2
        self.s2 = float(cfg.s) ** 2
        self.output_dim = int(output_dim)

    def train_feature_extractor_once(self, model, loader):
        sample_x, _ = next(iter(loader))
        sample_x = sample_x.to(self.device, non_blocking=True)
        with torch.no_grad():
            feat = model.forward_features(sample_x)
        model.head = nn.Linear(feat.shape[1], self.output_dim, bias=False).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.MSELoss()
        model.train()
        for ep in range(self.epochs):
            for x, y in tqdm(loader, desc=f"train {ep+1}/{self.epochs}", leave=False):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                opt.zero_grad()
                phi = model.forward_features(x)
                out = model.head(phi)
                criterion(out, y).backward()
                opt.step()

        for p in model.parameters():
            p.requires_grad = False

    def normalize_phi(self, phi):
        phi_norm = torch.norm(phi, p=2, dim=1, keepdim=True)
        phi_norm_sq = phi_norm ** 2
        phi_normalized = phi / phi_norm.clamp_min(1e-12)
        return phi_normalized, phi_norm_sq


    @torch.no_grad()
    def extract(self, model, loader):
        model.eval()
        phis, ys = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if y.dim() > 2:
                y = y.view(y.shape[0], -1)
            phi = model.forward_features(x)
            if getattr(self.cfg, "normalize_features", False):
                phi, _ = self.normalize_phi(phi)
            phi_aug = torch.cat([phi, torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)], dim=1)
            phis.append(phi_aug)
            ys.append(y)
        return torch.cat(phis), torch.cat(ys)

    @torch.no_grad()
    def rmse(self, model, loader, W):
        model.eval()
        se = torch.tensor(0.0, device=self.device)
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if y.dim() > 2:
                y = y.view(y.shape[0], -1)
            phi = model.forward_features(x)
            if getattr(self.cfg, "normalize_features", False):
                phi, _ = self.normalize_phi(phi)
            phi_aug = torch.cat([phi, torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)], dim=1)
            pred = phi_aug @ W
            se += ((pred - y) ** 2).sum()
            n += y.numel()
        return float((se.item() / n) ** 0.5)

    def fit_last_layer(self, Phi, Y, method):
        if method == "analytic":
            W, L = fit_last_layer_analytic(Phi, Y, self.sigma2, self.s2)
            return {"method": method, "W": W, "L": L}
        if method == "mfvi":
            W, v_diag, L = fit_last_layer_mfvi(Phi, Y, self.sigma2, self.s2)
            return {"method": method, "W": W, "v_diag": v_diag, "L": L}
        if method == "vi":
            vi_steps = int(getattr(self.cfg, "vi_steps", getattr(self.cfg, "inference_steps", 400)))
            W, u_diag, V, V_eigenvalues = fit_last_layer_vi(
                Phi,
                Y,
                self.sigma2,
                self.s2,
                steps=vi_steps,
                lr_cov=float(self.cfg.vi_lr),
            )
            return {"method": method, "W": W, "u_diag": u_diag, "eigvals_V": V_eigenvalues}
        if method == "vi_full":
            vi_steps = int(getattr(self.cfg, "vi_steps", getattr(self.cfg, "inference_steps", 400)))
            W, U, V, V_eigenvalues = fit_last_layer_vi_full(
                Phi, Y, self.sigma2,self.s2, steps=vi_steps, lr_cov=float(self.cfg.vi_lr),
            )
            return {"method": method, "W": W, "U": U, "eigvals_V": V_eigenvalues}
        raise ValueError(f"Unknown inference: {method}")

    @torch.no_grad()
    def score_pool(self, model, loader, params, D, return_embeddings=False):
        method = params["method"]
        acq_mode = str(getattr(self.cfg, "regression_af", "trace")).strip().lower()
        scores = []
        embs = [] if return_embeddings else None
        for x, _ in tqdm(loader, desc="scoring", leave=False):
            x = x.to(self.device, non_blocking=True)
            phi = model.forward_features(x)
            if getattr(self.cfg, "normalize_features", False):
                phi, _ = self.normalize_phi(phi)
            phi_aug = torch.cat([phi, torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)], dim=1)
            if acq_mode == "trace":
                trace_map = {
                    "analytic": lambda: acq_trace_analytic(phi_aug, params["L"], D),
                    "mfvi": lambda: acq_trace_mfvi(phi_aug, params["v_diag"], D),
                    "vi": lambda: acq_trace_vi(phi_aug, params["u_diag"], params["eigvals_V"]),
                    "vi_full": lambda: acq_trace_vi_full(phi_aug, params["U"], params["eigvals_V"]),
                }
                fn = trace_map.get(method)
                if fn is None:
                    raise ValueError(f"Unknown inference: {method}")
                sc = fn()
            elif acq_mode == "random":
                sc = torch.ones(phi_aug.shape[0], device=phi_aug.device, dtype=phi_aug.dtype)
            else:
                raise ValueError(f"Unknown regression_af: {acq_mode}")
            scores.append(sc)
            if return_embeddings:
                embs.append(phi)
        score_vec = torch.cat(scores, dim=0)
        if return_embeddings:
            return score_vec, torch.cat(embs, dim=0)
        return score_vec

    def infer_last_layer(self, model, train_loader, method):
        Phi, Y = self.extract(model, train_loader)
        params = self.fit_last_layer(Phi, Y, method)
        return params



class RegressionAL:
    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg
        self.rng = data.numpy_generator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = int(cfg.acquisition_size)
        self.n_steps = int(cfg.acquire_steps)
        self.D = int(getattr(data, "output_dim", 10))
        self.diversity_lambda = float(getattr(cfg, "diversity_lambda", 0.0))

        method = str(cfg.inference_method).strip().lower()
        if method not in ("analytic", "mfvi", "vi", "vi_full"):
            raise ValueError(f"Unknown inference: {method}")
        self.method = method

        acq = str(getattr(cfg, "regression_af", "trace")).strip().lower()
        if acq not in ("trace", "random"):
            raise ValueError("regression_af must be one of: trace, random")
        self.acq_mode = acq

        self.trainer = RegressionTrainer(cfg, self.device, output_dim=self.D)

    def run(self):
        train_idx = list(self.data.train_indexes)
        pool_idx = list(self.data.pool_indexes)
        n_init = len(train_idx)
        history = []

        pool_ood_ratio = float(getattr(self.data, "pool_ood_ratio", 0.0))
        is_ood = getattr(self.data, "is_ood", None)
        ood_cum_count = 0

        init_train_loader, _ = self.data.build_loaders(train_idx, pool_idx)
        model = self.data.make_model().to(self.device)
        self.trainer.train_feature_extractor_once(model, init_train_loader)

        for step in range(self.n_steps + 1):
            n_acquired = len(train_idx) - n_init

            train_loader, pool_loader = self.data.build_loaders(train_idx, pool_idx)

            params = self.trainer.infer_last_layer(model, train_loader, self.method)
            rmse = self.trainer.rmse(model, self.data.test_loader, params["W"])

            if step == self.n_steps or not pool_idx:
                history.append(
                    {
                        "step": step,
                        "labels": len(train_idx),
                        "labels_from_pool": n_acquired,
                        "test_rmse": rmse,
                        "K": params["W"].shape[0],
                        "inference_method": self.method,
                        "acquisition": self.acq_mode,
                        "pool_ood_ratio": pool_ood_ratio,
                        "ood_in_batch_rate": 0.0,
                        "ood_cum_count": ood_cum_count,
                        "ood_cum_rate": ood_cum_count / max(1, n_acquired),
                    }
                )
                break

            if self.acq_mode == "random":
                k = min(self.batch_size, len(pool_idx))
                selected = self.rng.choice(pool_idx, size=k, replace=False).tolist()
            else:
                use_diverse = self.diversity_lambda > 0.0
                scores = self.trainer.score_pool(
                    model,
                    pool_loader,
                    params,
                    self.D,
                    return_embeddings=use_diverse,
                )
                if use_diverse:
                    score_vec, embs = scores
                    selected = select_batch_diverse(
                        score_vec, embs, pool_idx, self.batch_size, self.diversity_lambda, self.rng
                    )
                else:
                    selected = select_top_k(scores, pool_idx, self.batch_size, self.rng)

            ood_batch = 0
            ood_frac = 0.0
            if is_ood is not None and selected:
                flags = is_ood[torch.as_tensor(selected, dtype=torch.long)]
                ood_batch = int(flags.sum().item())
                ood_frac = float(flags.float().mean().item())
            ood_cum_count += ood_batch
            labels_after = len(train_idx) + len(selected)
            labels_from_pool_after = n_acquired + len(selected)
            ood_cum_frac = ood_cum_count / max(1, labels_from_pool_after)

            history.append(
                {
                    "step": step,
                    "labels": len(train_idx),
                    "labels_from_pool": n_acquired,
                    "labels_after": labels_after,
                    "test_rmse": rmse,
                    "K": params["W"].shape[0],
                    "inference_method": self.method,
                    "acquisition": self.acq_mode,
                    "pool_ood_ratio": pool_ood_ratio,
                    "ood_in_batch_rate": ood_frac,
                    "ood_cum_count": ood_cum_count,
                    "ood_cum_rate": ood_cum_frac,
                }
            )

            self.data.apply_label_noise(selected)
            train_idx.extend(selected)
            sel = set(selected)
            pool_idx = [i for i in pool_idx if i not in sel]

        is_ood = getattr(self.data, "is_ood", None)
        if is_ood is not None:
            all_train_idx = [i for i in (train_idx + pool_idx) if not is_ood[i]]
        else:
            all_train_idx = train_idx + pool_idx
        if all_train_idx:
            all_train_loader, _ = self.data.build_loaders(all_train_idx, [])
            params_full = self.trainer.infer_last_layer(model, all_train_loader, self.method)
            upper_bound_rmse = self.trainer.rmse(model, self.data.test_loader, params_full["W"])
        else:
            upper_bound_rmse = None

        for entry in history:
            entry["upper_bound_rmse"] = upper_bound_rmse

        return history


def run_active_process_regression(cfg):
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    return RegressionAL(DataRegression(cfg, rng), cfg).run()