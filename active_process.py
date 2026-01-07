import numpy as np
import torch
import random

from model_utils import ModelTrainer
from acquisition import score_pool, select_top_k, select_batch_diverse
from process_data import Data


class ActiveLearning:
    def __init__(self, data, trainer, cfg, retune_steps=(0, 10, 30, 60, 90)):
        self.data = data
        self.trainer = trainer
        self.cfg = cfg
        self.rng = data.numpy_generator
        self.retune_steps = retune_steps

        self.batch_size = int(cfg.active_learning_points)
        self.n_steps = int(cfg.active_learning_iters)
        self.acq_fn = cfg.acquisition_function.strip().lower()
        self.diversity_lambda = float(getattr(cfg, "diversity_lambda", 0.0))
        self.dropout_acq = cfg.use_dropout_in_acquisition
        self.dropout_eval = cfg.use_dropout_in_eval
        self.wd = float(cfg.weight_decay)
        self.t_pool = int(cfg.t_pool)
        self.t_test = int(cfg.t_test)
        self.lr = float(cfg.lr)

    def run(self):
        train_idx = list(self.data.train_indexes)
        pool_idx = list(self.data.pool_indexes)
        n_init = len(train_idx)
        wd = self.wd
        history = []

        base = self.trainer.make_model()
        init_state = {k: v.cpu().clone() for k, v in base.state_dict().items()}

        for step in range(self.n_steps + 1):
            n_acq = len(train_idx) - n_init

            if step in self.retune_steps:
                tl, _ = self.data.build_loaders(train_idx, pool_idx)
                wd = self.trainer.tune_weight_decay(tl, self.data.val_loader, init_state)
                del tl

            train_loader, pool_loader = self.data.build_loaders(train_idx, pool_idx)

            model = self.trainer.make_model()
            model.load_state_dict(init_state)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=wd)

            self.trainer.fit_with_optimizer(model, train_loader, opt)

            from model_utils import evaluate
            acc, loss = evaluate(model, self.data.test_loader, self.trainer.device, self.t_test, self.dropout_eval)
            history.append({
                "step": step,
                "acquired": len(train_idx),
                "acquired_from_pool": n_acq,
                "test_accuracy": acc,
                "test_loss": loss,
                "attempt": int(self.cfg.seed),
                "weight_decay_used": wd,
            })

            if step == self.n_steps:
                break

            selected = self.acquire(model, pool_loader, pool_idx)

            self.data.apply_label_noise(selected)
            train_idx.extend(selected)
            pool_idx = [i for i in pool_idx if i not in set(selected)]
            
            del train_loader, pool_loader

        return history

    def acquire(self, model, loader, pool_idx):
        if self.acq_fn == "random":
            return self.rng.choice(pool_idx, size=self.batch_size, replace=False).tolist()
        use_diverse = self.diversity_lambda > 0.0
        scores = score_pool(model, loader, self.trainer.device, self.t_pool, self.acq_fn, use_dropout=self.dropout_acq, use_tqdm=True, return_embeddings=use_diverse)
        if use_diverse:
            score_vec, embs = scores
            return select_batch_diverse(score_vec, embs, pool_idx, self.batch_size, self.diversity_lambda, self.rng)
        return select_top_k(scores, pool_idx, self.batch_size, self.rng)


def run_active_process(cfg):
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    data = Data(cfg, rng)
    trainer = ModelTrainer(cfg)
    return ActiveLearning(data, trainer, cfg).run()
