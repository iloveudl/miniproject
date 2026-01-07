import torch
from torch import nn
from tqdm.auto import tqdm

from models import CNN


def train_epoch(model, loader, opt, criterion, device, desc):
    model.train()
    for x, y in tqdm(loader, desc, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        criterion(model(x), y).backward()
        opt.step()


def mc_forward(model, x, T, use_dropout):
    if use_dropout and T > 1:
        B = x.size(0)
        x_rep = x.unsqueeze(0).expand(T, -1, -1, -1, -1)
        x_rep = x_rep.reshape(T * B, *x.shape[1:])
        logits = model(x_rep).view(T, B, -1)
        probs = logits.softmax(dim=-1).mean(dim=0)
        return logits, probs
    logits = model(x)
    return logits, logits.softmax(dim=-1)



def evaluate(model, loader, device, T=1, use_dropout=True):
    model.eval()
    if use_dropout and T > 1:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = torch.tensor(0, device=device, dtype=torch.long)
    total_loss = torch.tensor(0.0, device=device)

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B = x.size(0)

            _, probs = mc_forward(model, x, T, use_dropout)

            pred = probs.argmax(dim=-1)
            correct += (pred == y).sum()
            total += B

            idx = torch.arange(B, device=device)
            total_loss += (-probs[idx, y].clamp_min(1e-9).log()).sum()

    return correct.item() / total.item(), total_loss.item() / total.item()


class ModelTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = float(cfg.lr)
        self.wd = float(cfg.weight_decay)
        self.epochs = int(cfg.epochs)

    def make_model(self):
        return CNN().to(self.device)

    def fit_with_optimizer(self, model, loader, opt):
        criterion = nn.CrossEntropyLoss()
        for ep in range(self.epochs):
            train_epoch(model, loader, opt, criterion, self.device, f"epoch {ep+1}/{self.epochs}")


    def tune_weight_decay(self, train_loader, val_loader, init_state):
        if val_loader is None:
            return self.wd

        best_wd, best_acc = self.wd, -1.0
        for wd in self.cfg.weight_decay_grid:
            wd = float(wd)
            model = self.make_model()
            model.load_state_dict(init_state)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=wd)
            crit = nn.CrossEntropyLoss()

            for ep in range(self.epochs):
                train_epoch(model, train_loader, opt, crit, self.device, f"wd={wd} ep={ep+1}")

            acc, _ = evaluate(model, val_loader, self.device, self.cfg.t_test, self.cfg.use_dropout_in_eval)
            if acc > best_acc:
                best_acc, best_wd = acc, wd

        return best_wd
