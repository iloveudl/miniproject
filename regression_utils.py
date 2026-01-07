import torch
from torch import nn

@torch.no_grad()
def fit_last_layer_analytic(Phi, Y, sigma2, s2):
    identity = torch.eye(Phi.shape[1], device=Phi.device, dtype=Phi.dtype)
    precision_matrix = Phi.T @ Phi / sigma2 + identity / s2
    L = torch.linalg.cholesky(precision_matrix)
    W = torch.cholesky_solve(Phi.T @ Y / sigma2, L)
    return W, L


@torch.no_grad()
def fit_last_layer_mfvi(Phi, Y, sigma2, s2):
    W, L = fit_last_layer_analytic(Phi, Y, sigma2, s2)

    phi_squared_sum = (Phi ** 2).sum(dim=0)
    v_diag = 1.0 / (1.0 / s2 + (1.0 / sigma2) * phi_squared_sum).clamp_min(1e-12)

    return W, v_diag, L

def softplus(x):
    return torch.nn.functional.softplus(x) + 1e-8


def make_positive(L_raw):
    L = torch.tril(L_raw)
    d = torch.diagonal(L)
    return L - torch.diag(d) + torch.diag(softplus(d) + 1e-6)


def similarity(mat):
    mat = 0.5 * (mat + mat.T)
    return torch.linalg.eigvalsh(mat).clamp_min(1e-12)


def fit_last_layer_vi(Phi, Y, sigma2, s2, steps=400, lr_cov=5e-3):
    device, dtype = Phi.device, Phi.dtype
    N, K = Phi.shape
    D = Y.shape[1]

    identity = torch.eye(K, device=device, dtype=dtype)
    W0 = torch.linalg.solve(Phi.T @ Phi + (sigma2 / s2) * identity, Phi.T @ Y)

    steps = int(steps)
    if steps < 800:
        steps = 800

    W = W0.clone()
    u_raw = nn.Parameter(torch.zeros(K, device=device, dtype=dtype))
    L_raw = nn.Parameter(torch.eye(D, device=device, dtype=dtype))
    param_groups = [{"params": [u_raw, L_raw], "lr": lr_cov}]
    opt = torch.optim.Adam(param_groups)

    phi_squared_sum = (Phi ** 2).sum(dim=0)
    log_normalization = -0.5 * N * D * torch.log(torch.tensor(2 * torch.pi * sigma2, device=device))

    for step in range(steps):
        opt.zero_grad()

        u = softplus(u_raw)
        L = make_positive(L_raw)
        V = L @ L.T

        variance_part = torch.trace(V) * (u * phi_squared_sum).sum()
        loglikelihood = log_normalization - 0.5 / sigma2 * (((Y - Phi @ W) ** 2).sum() + variance_part)

        logdet_V = 2 * torch.diagonal(L).log().sum()
        logdet_U = u.log().sum()
        kl = 0.5 * (
            (torch.trace(V) * u.sum() + (W ** 2).sum()) / s2
            - K * D + K * D * torch.log(torch.tensor(s2, device=device))
            - K * logdet_V - D * logdet_U
        )

        (-loglikelihood + kl).backward()
        opt.step()

    with torch.no_grad():
        u = softplus(u_raw)
        L = make_positive(L_raw)
        V = L @ L.T
        V_eigenvalues = similarity(V)

    return W, u.detach(), V.detach(), V_eigenvalues.detach()


def fit_last_layer_vi_full(Phi, Y, sigma2, s2, steps=400, lr_cov=5e-3):
    N, K = Phi.shape
    D = Y.shape[1]

    identity = torch.eye(K, device=Phi.device, dtype=Phi.dtype)
    W0 = torch.linalg.solve(Phi.T @ Phi + (sigma2 / s2) * identity, Phi.T @ Y)

    steps = int(steps)
    if steps < 800:
        steps = 800

    W = W0.clone()
    U_raw = nn.Parameter(torch.eye(K, device=Phi.device, dtype=Phi.dtype))
    L_raw = nn.Parameter(torch.eye(D, device=Phi.device, dtype=Phi.dtype))
    param_groups = [{"params": [U_raw, L_raw], "lr": lr_cov}]
    opt = torch.optim.Adam(param_groups)

    phi_cov = Phi.T @ Phi
    log_normalization = -0.5 * N * D * torch.log(torch.tensor(2 * torch.pi * sigma2, device=Phi.device))

    for step in range(steps):
        opt.zero_grad()

        U_chol = make_positive(U_raw)
        U = U_chol @ U_chol.T
        L = make_positive(L_raw)
        V = L @ L.T

        variance_part = torch.trace(V) * torch.trace(U @ phi_cov)
        loglikelihood = log_normalization - 0.5 / sigma2 * (((Y - Phi @ W) ** 2).sum() + variance_part)

        logdet_V = 2 * torch.diagonal(L).log().sum()
        logdet_U = 2 * torch.diagonal(U_chol).log().sum()
        kl = 0.5 * (
            (torch.trace(V) * torch.trace(U) + (W ** 2).sum()) / s2
            - K * D + K * D * torch.log(torch.tensor(s2, device=Phi.device))
            - K * logdet_V - D * logdet_U
        )

        (-loglikelihood + kl).backward()
        opt.step()

    with torch.no_grad():
        U_chol = make_positive(U_raw)
        L = make_positive(L_raw)
        V = L @ L.T
        U = U_chol @ U_chol.T
        V_eigenvalues = similarity(V)

    return W, U.detach(), V.detach(), V_eigenvalues.detach()


@torch.no_grad()
def acq_trace_analytic(Phi, L, D):
    cholesky_solution = torch.cholesky_solve(Phi.T, L)
    var = (Phi.T * cholesky_solution).sum(dim=0).clamp_min(1e-12)
    return D * var


@torch.no_grad()
def acq_trace_mfvi(Phi, v_diag, D):
    var = ((Phi ** 2) @ v_diag).clamp_min(1e-12)
    return D * var


@torch.no_grad()
def acq_trace_vi(Phi, u_diag, eigvals_V):
    var = ((Phi ** 2) @ u_diag).clamp_min(1e-12)
    trace_V = eigvals_V.sum()
    return var * trace_V


@torch.no_grad()
def acq_trace_vi_full(Phi, U, eigvals_V):
    var = (Phi * (Phi @ U)).sum(dim=1).clamp_min(1e-12)
    trace_V = eigvals_V.sum()
    return var * trace_V

