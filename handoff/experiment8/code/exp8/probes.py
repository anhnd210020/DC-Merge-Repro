"""Linear probes: multinomial logistic regression on standardized features.

The L2 penalty is chosen from a fixed grid by accuracy on a fixed 20% hold-out of the training
images, then the probe is refit on all training images. Full-batch L-BFGS in fp32 from a zero
initialization, so fits are deterministic.
"""
import numpy as np
import torch


def standardize(train, test):
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True).clamp_min(1e-6)
    return (train - mu) / sd, (test - mu) / sd


def _fit(X, y, n_cls, lam, iters):
    d = X.shape[1]
    W = torch.zeros(d, n_cls, device=X.device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(n_cls, device=X.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, tolerance_grad=1e-6,
                            tolerance_change=1e-9, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + lam * (W * W).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return W.detach(), b.detach()


def fit_probe(X_train, y_train, X_test, y_test, n_cls, lambdas, holdout=0.2, iters=200, seed=0):
    """Returns (chosen lambda, per-image test correctness as a bool numpy array)."""
    X_train, X_test = standardize(X_train.float(), X_test.float())
    y_train = torch.as_tensor(y_train, device=X_train.device, dtype=torch.long)
    y_test = torch.as_tensor(y_test, device=X_train.device, dtype=torch.long)
    n = X_train.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    n_ho = max(1, int(round(holdout * n)))
    ho = torch.as_tensor(np.sort(perm[:n_ho]), device=X_train.device)
    tr = torch.as_tensor(np.sort(perm[n_ho:]), device=X_train.device)
    best, best_acc = lambdas[0], -1.0
    for lam in lambdas:
        W, b = _fit(X_train[tr], y_train[tr], n_cls, lam, iters)
        acc = ((X_train[ho] @ W + b).argmax(1) == y_train[ho]).float().mean().item()
        if acc > best_acc + 1e-12:
            best, best_acc = lam, acc
    W, b = _fit(X_train, y_train, n_cls, best, iters)
    correct = ((X_test @ W + b).argmax(1) == y_test).cpu().numpy()
    return best, correct
