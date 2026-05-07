import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

@dataclass
class FitResult:
    trace: str
    distribution: str
    n: int
    params: str
    ks_stat: float
    ks_pvalue: float
    r2: float
    loglik: float
    aic: float

def fit_lognormal(x: np.ndarray) -> Tuple[float, float]:
    logx = np.log(x)
    mu = float(np.mean(logx))
    sigma = float(np.std(logx, ddof=1))
    return (mu, sigma)

def fit_gamma(x: np.ndarray) -> Tuple[float, float]:
    shape, _, scale = stats.gamma.fit(x, floc=0)
    return (float(shape), float(scale))

def fit_exponential(x: np.ndarray) -> float:
    return float(1.0 / np.mean(x))

def empirical_cdf(x_sorted: np.ndarray) -> np.ndarray:
    n = len(x_sorted)
    return (np.arange(1, n + 1) - 0.5) / n

def r_squared_cdf(x_sorted: np.ndarray, fitted_cdf_at_x: np.ndarray) -> float:
    emp = empirical_cdf(x_sorted)
    ss_res = float(np.sum((emp - fitted_cdf_at_x) ** 2))
    ss_tot = float(np.sum((emp - np.mean(emp)) ** 2))
    if ss_tot == 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot

def evaluate(trace_label: str, x: np.ndarray) -> List[FitResult]:
    x = np.asarray(x, dtype=float)
    x = x[(x > 0) & np.isfinite(x)]
    n = len(x)
    x_sorted = np.sort(x)
    results: List[FitResult] = []
    mu, sigma = fit_lognormal(x)
    cdf_ln = stats.lognorm.cdf(x_sorted, s=sigma, scale=math.exp(mu))
    pdf_ln = stats.lognorm.pdf(x, s=sigma, scale=math.exp(mu))
    ks_ln = stats.kstest(x, 'lognorm', args=(sigma, 0, math.exp(mu)))
    ll_ln = float(np.sum(np.log(np.clip(pdf_ln, 1e-300, None))))
    results.append(FitResult(trace=trace_label, distribution='lognormal', n=n, params=f'mu={mu:.4f}, sigma={sigma:.4f}', ks_stat=float(ks_ln.statistic), ks_pvalue=float(ks_ln.pvalue), r2=r_squared_cdf(x_sorted, cdf_ln), loglik=ll_ln, aic=2 * 2 - 2 * ll_ln))
    shape, scale = fit_gamma(x)
    cdf_g = stats.gamma.cdf(x_sorted, a=shape, scale=scale)
    pdf_g = stats.gamma.pdf(x, a=shape, scale=scale)
    ks_g = stats.kstest(x, 'gamma', args=(shape, 0, scale))
    ll_g = float(np.sum(np.log(np.clip(pdf_g, 1e-300, None))))
    results.append(FitResult(trace=trace_label, distribution='gamma', n=n, params=f'shape={shape:.4f}, scale={scale:.4f}', ks_stat=float(ks_g.statistic), ks_pvalue=float(ks_g.pvalue), r2=r_squared_cdf(x_sorted, cdf_g), loglik=ll_g, aic=2 * 2 - 2 * ll_g))
    rate = fit_exponential(x)
    cdf_e = stats.expon.cdf(x_sorted, scale=1.0 / rate)
    pdf_e = stats.expon.pdf(x, scale=1.0 / rate)
    ks_e = stats.kstest(x, 'expon', args=(0, 1.0 / rate))
    ll_e = float(np.sum(np.log(np.clip(pdf_e, 1e-300, None))))
    results.append(FitResult(trace=trace_label, distribution='exponential', n=n, params=f'rate={rate:.6f}', ks_stat=float(ks_e.statistic), ks_pvalue=float(ks_e.pvalue), r2=r_squared_cdf(x_sorted, cdf_e), loglik=ll_e, aic=2 * 1 - 2 * ll_e))
    return results
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.frameon': False, 'lines.linewidth': 1.6})
DIST_COLORS = {'empirical': 'black', 'lognormal': '#d62728', 'gamma': '#1f77b4', 'exponential': '#7f7f7f'}
DIST_STYLES = {'empirical': '-', 'lognormal': '--', 'gamma': ':', 'exponential': '-.'}

def plot_cdf_overlay(traces: Dict[str, np.ndarray], outdir: Path):
    n_panels = len(traces)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.2), squeeze=False)
    axes = axes.ravel()
    for ax, (label, x) in zip(axes, traces.items()):
        x = np.asarray(x, dtype=float)
        x = x[(x > 0) & np.isfinite(x)]
        x_sorted = np.sort(x)
        emp = empirical_cdf(x_sorted)
        mu, sigma = fit_lognormal(x)
        shape, scale = fit_gamma(x)
        rate = fit_exponential(x)
        grid = np.logspace(np.log10(x_sorted[0]), np.log10(x_sorted[-1]), 500)
        ax.plot(x_sorted, emp, color=DIST_COLORS['empirical'], linestyle=DIST_STYLES['empirical'], label='Empirical')
        ax.plot(grid, stats.lognorm.cdf(grid, s=sigma, scale=math.exp(mu)), color=DIST_COLORS['lognormal'], linestyle=DIST_STYLES['lognormal'], label=f'Log-normal ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})')
        ax.plot(grid, stats.gamma.cdf(grid, a=shape, scale=scale), color=DIST_COLORS['gamma'], linestyle=DIST_STYLES['gamma'], label='Gamma')
        ax.plot(grid, stats.expon.cdf(grid, scale=1.0 / rate), color=DIST_COLORS['exponential'], linestyle=DIST_STYLES['exponential'], label='Exponential')
        ax.set_xscale('log')
        ax.set_xlabel('Inter-turn interval (s, log scale)')
        ax.set_ylabel('Cumulative probability')
        ax.set_title(label)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
        ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'cdf_overlay.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'cdf_overlay.png', bbox_inches='tight', dpi=200)
    plt.close(fig)

def plot_qq(traces: Dict[str, np.ndarray], outdir: Path):
    n_panels = len(traces)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.4), squeeze=False)
    axes = axes.ravel()
    for ax, (label, x) in zip(axes, traces.items()):
        x = np.asarray(x, dtype=float)
        x = x[(x > 0) & np.isfinite(x)]
        mu, sigma = fit_lognormal(x)
        n = len(x)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical = stats.lognorm.ppf(probs, s=sigma, scale=math.exp(mu))
        empirical_q = np.sort(x)
        ax.scatter(theoretical, empirical_q, s=6, alpha=0.4, color=DIST_COLORS['lognormal'], edgecolors='none')
        lo = min(theoretical[0], empirical_q[0])
        hi = max(theoretical[-1], empirical_q[-1])
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Theoretical log-normal quantile (s)')
        ax.set_ylabel('Empirical quantile (s)')
        ax.set_title(label)
        ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outdir / 'qq_lognormal.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'qq_lognormal.png', bbox_inches='tight', dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', action='append', required=True, help='format: label:path/to/intervals.csv (use multiple times)')
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--max-samples', type=int, default=None, help='Random subsample (helps when n is huge and KS p-value is dominated by sample size; reproducible via --seed)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    traces: Dict[str, np.ndarray] = {}
    for spec in args.trace:
        if ':' not in spec:
            ap.error(f'--trace must be label:path, got {spec!r}')
        label, path = spec.split(':', 1)
        df = pd.read_csv(path)
        x = df['interval_sec'].to_numpy()
        x = x[(x > 0) & np.isfinite(x)]
        if args.max_samples is not None and len(x) > args.max_samples:
            idx = rng.choice(len(x), args.max_samples, replace=False)
            x = x[idx]
        traces[label] = x
        print(f'[{label}] using n={len(x):,} intervals')
    all_results: List[FitResult] = []
    for label, x in traces.items():
        all_results.extend(evaluate(label, x))
    fits_df = pd.DataFrame([asdict(r) for r in all_results])
    fits_path = args.outdir / 'fits.csv'
    fits_df.to_csv(fits_path, index=False)
    print(f'\n=== Fit results ===')
    pd.set_option('display.float_format', lambda v: f'{v:.4g}')
    print(fits_df.to_string(index=False))
    print(f'\nWrote {fits_path}')
    plot_cdf_overlay(traces, args.outdir)
    plot_qq(traces, args.outdir)
    print(f'Wrote {args.outdir}/cdf_overlay.pdf, qq_lognormal.pdf (and .png copies)')
if __name__ == '__main__':
    main()
