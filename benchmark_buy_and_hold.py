"""Buy-and-hold benchmarks for a completed run, on the same panel the model saw.

Builds a passive portfolio from the prediction panel itself -- every stock the
model could have traded, held throughout -- and reports it alongside the
strategy portfolios using the same conventions as backtest.py, so the numbers
drop straight into a table next to the decile results.

Reported for each portfolio: mean monthly excess return, standard deviation,
annualised Sharpe ratio, and CAPM / Fama-French alphas with Newey-West t-stats.

Two passive portfolios are built, and they are not the same kind of object:

  VW  value-weighted by market_cap. This is genuinely buy-and-hold: a
      cap-weighted portfolio self-rebalances, because a stock's price move
      changes your holding's value and its market weight by the same factor.
      Turnover comes only from issuance, buybacks, and entry/exit.

  EW  the cross-sectional mean, i.e. equal weights restored at the start of
      every month. That is the 1/N strategy of DeMiguel, Garlappi and Uppal
      (2009), NOT an equal-weighted buy-and-hold. Left to drift, equal weights
      stop being equal within a year; holding them at 1/N costs roughly 100%
      annual turnover, selling winners and buying losers. Reported here as a
      naive-diversification benchmark, and its turnover matters once
      transaction costs enter the picture.

The Fama-French market factor is loaded as an independent check. The panel is
the GKX universe (stocks with the full characteristic set), not all of CRSP, so
the VW series should track mktrf closely without matching it exactly. A large
divergence would point at the market_cap weight being timed wrongly -- weights
must be known at t to earn the return realised at t+1.

Usage:
    python benchmark_buy_and_hold.py outputs/<run_dir>
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import annualized_sharpe_ratio

try:
    from scipy import stats as _scipy_stats
except ImportError:            # pragma: no cover - depends on the environment
    _scipy_stats = None

RUN = Path(sys.argv[1] if len(sys.argv) > 1
           else "outputs/nn1_cache_shufflefix_1987_2016")
# ff5.csv carries all six factors (mktrf, smb, hml, rmw, cma, umd) from
# 1963-07, so it covers the whole 1987-2016 test window. ff3.csv lacks rmw and
# cma and is used only as a fallback, in which case FF6 is skipped.
FF = Path("data_in_case_needed/ff5.csv")
FF_FALLBACK = Path("data_csv/ff3.csv")
PREFIX = "nn1"

FACTOR_MODELS = [
    ("CAPM", ["mktrf"]),
    ("FF3", ["mktrf", "smb", "hml"]),
    ("FF6", ["mktrf", "smb", "hml", "rmw", "cma", "umd"]),
]


def newey_west_tstat(y, X, lags=None):
    """OLS with Newey-West standard errors. Returns (coefs, tstats).

    Monthly portfolio returns are mildly autocorrelated and heteroskedastic, so
    plain OLS standard errors overstate alpha's significance. Lag length follows
    the usual 4*(T/100)^(2/9) rule.
    """
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        u_t, u_l = resid[lag:, None] * X[lag:], resid[:-lag, None] * X[:-lag]
        G = u_t.T @ u_l
        S += w * (G + G.T)

    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    tstats = beta / np.sqrt(np.diag(cov))
    return beta, tstats, two_sided_p(tstats, df=n - k)


def two_sided_p(tstats, df):
    """Two-sided p-values for t statistics.

    Uses Student-t when scipy is present. Without it, falls back to the normal
    distribution: at the sample sizes here (df above 300) the two agree to
    within a few tenths of a percent, which never changes a conclusion.
    """
    tstats = np.atleast_1d(tstats)
    if _scipy_stats is not None:
        return 2 * _scipy_stats.t.sf(np.abs(tstats), df)
    return np.array([math.erfc(abs(t) / math.sqrt(2)) for t in tstats])


def performance_profile(returns):
    """Cumulative and tail statistics for a monthly excess-return series.

    Wealth compounds the excess return, so it reads as growth *above cash* and
    stays comparable across the long-only and self-financing portfolios alike.
    Drawdown is measured on that same series.
    """
    r = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return {
        "growth_of_1": wealth[-1],
        "max_drawdown_pct": (wealth / peak - 1.0).min() * 100,
        "best_month_pct": r.max() * 100,
        "worst_month_pct": r.min() * 100,
        "pct_months_positive": (r > 0).mean() * 100,
    }


# ---------------------------------------------------------------- load panel
panel = pd.read_parquet(RUN / f"{PREFIX}_predictions.parquet",
                        columns=["YYYYMM", "permno", "market_cap", "excess_ret"])
print(f"run    : {RUN}")
print(f"panel  : {len(panel):,} stock-months, "
      f"{panel.YYYYMM.min()}-{panel.YYYYMM.max()}, "
      f"{panel.permno.nunique():,} stocks\n")

# Value-weighting done as sum(w*r)/sum(w) on whole columns rather than via
# groupby.apply: vectorised over 1.5M rows instead of a Python call per month,
# and free of the include_groups keyword, which only exists in pandas >= 2.2.
panel["_wr"] = panel.excess_ret * panel.market_cap
by_month = panel.groupby("YYYYMM")
market = pd.DataFrame({
    "vw": by_month._wr.sum() / by_month.market_cap.sum(),
    "ew": by_month.excess_ret.mean(),
    "n": by_month.size(),
}).reset_index()
panel.drop(columns="_wr", inplace=True)

# ------------------------------------------------------- strategy portfolios
# nn1_long_short.csv is already wide: YYYYMM, deciles "1".."10", and the
# spread. (nn1_decile_returns.csv carries the same realised returns in long
# format alongside the predicted ones, so it is not needed here.)
longshort = pd.read_csv(RUN / f"{PREFIX}_long_short.csv")
series = market.merge(longshort[["YYYYMM", "1", "10", "long_short_10_1"]],
                      on="YYYYMM")

# --------------------------------------------------------- Fama-French merge
ff_path = FF if FF.exists() else FF_FALLBACK
ff = pd.read_csv(ff_path)
ff["YYYYMM"] = (pd.to_datetime(ff["dateff"]).dt.year * 100
                + pd.to_datetime(ff["dateff"]).dt.month)
available = [c for c in ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
             if c in ff.columns]
series = series.merge(ff[["YYYYMM"] + available], on="YYYYMM")

models = [(name, cols) for name, cols in FACTOR_MODELS
          if all(c in available for c in cols)]
print(f"factors: {ff_path.name} -> {', '.join(c for c in available if c != 'rf')}")
fitted = {name for name, _ in models}
if len(models) < len(FACTOR_MODELS):
    skipped = [name for name, _ in FACTOR_MODELS if name not in fitted]
    missing = sorted({c for _, cols in FACTOR_MODELS for c in cols}
                     - set(available))
    print(f"         missing {missing} -> {', '.join(skipped)} skipped")
print(f"months : {len(series)}  (after merging Fama-French)\n")

# -------------------------------------------------------------------- report
rows = []
portfolios = [
    ("Buy & hold, value-weighted", "vw"),
    ("1/N, rebalanced monthly", "ew"),
    ("FF market factor (mktrf)", "mktrf"),
    ("Long only, decile 10", "10"),
    ("Short leg, decile 1", "1"),
    ("Long-short, 10 minus 1", "long_short_10_1"),
]
for label, col in portfolios:
    r = series[col].to_numpy()
    row = {
        "portfolio": label,
        "mean_pct": r.mean() * 100,
        "sd_pct": r.std(ddof=1) * 100,
        "sharpe": annualized_sharpe_ratio(pd.Series(r)),
        **performance_profile(r),
    }
    for name, cols in models:
        beta, tstat, pval = newey_west_tstat(r, series[cols].to_numpy())
        key = name.lower()
        row[f"{key}_alpha_pct"] = beta[0] * 100
        row[f"{key}_t"] = tstat[0]
        row[f"{key}_p"] = pval[0]
        for j, factor in enumerate(cols, start=1):
            row[f"{key}_b_{factor}"] = beta[j]
    rows.append(row)

out = pd.DataFrame(rows)


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


print("PANEL A -- return, risk and cumulative growth")
hdr = (f"{'Portfolio':<28}{'Mean%':>8}{'SD%':>7}{'SR':>7}{'$1 becomes':>12}"
       f"{'maxDD%':>9}{'best%':>8}{'worst%':>9}{'%up':>7}")
print(hdr)
print("-" * len(hdr))
for _, r in out.iterrows():
    print(f"{r.portfolio:<28}{r.mean_pct:8.3f}{r.sd_pct:7.2f}{r.sharpe:7.3f}"
          f"{r.growth_of_1:11.2f}x{r.max_drawdown_pct:9.1f}"
          f"{r.best_month_pct:8.1f}{r.worst_month_pct:9.1f}"
          f"{r.pct_months_positive:7.1f}")
print("\n  Wealth compounds EXCESS returns, i.e. growth above cash, so the")
print("  long-only and self-financing portfolios stay comparable.")

print("\nPANEL B -- factor-adjusted alphas (%/month, Newey-West)")
hdr = f"{'Portfolio':<28}" + "".join(
    f"{name + ' a%':>9}{'(t)':>7}{'p':>8}" for name, _ in models)
print(hdr)
print("-" * len(hdr))
for _, r in out.iterrows():
    line = f"{r.portfolio:<28}"
    for name, _ in models:
        k = name.lower()
        line += (f"{r[f'{k}_alpha_pct']:9.3f}{r[f'{k}_t']:7.2f}"
                 f"{r[f'{k}_p']:8.4f}")
    print(line)
print("\n  *** p<0.01  ** p<0.05  * p<0.10  (" +
      ("Student-t)" if _scipy_stats is not None else "normal approximation)"))
for _, r in out.iterrows():
    marks = "  ".join(f"{name} {stars(r[f'{name.lower()}_p']) or 'ns'}"
                      for name, _ in models)
    print(f"    {r.portfolio:<28}{marks}")

if "FF6" in fitted:
    ff6_cols = dict(models)["FF6"]
    print("\nPANEL C -- FF6 factor loadings")
    print("  Whether a portfolio's alpha is genuinely new information or just")
    print("  exposure to size, value, profitability, investment or momentum.")
    hdr = f"{'Portfolio':<28}" + "".join(f"{c:>9}" for c in ff6_cols)
    print(hdr)
    print("-" * len(hdr))
    for _, r in out.iterrows():
        print(f"{r.portfolio:<28}"
              + "".join(f"{r[f'ff6_b_{c}']:9.2f}" for c in ff6_cols))

# ---------------------------------------- where does the long-short alpha come from
print("\nDECOMPOSITION -- long-short alpha by leg")
d10 = out[out.portfolio.str.contains("decile 10")].iloc[0]
d1 = out[out.portfolio.str.contains("decile 1$|decile 1\\b", regex=True)].iloc[0]
print(f"{'Model':<8}{'long leg':>11}{'short leg':>12}{'total':>10}"
      f"{'long share':>13}{'short share':>14}")
print("-" * 68)
for name, _ in models:
    k = name.lower()
    a_long, a_short = d10[f"{k}_alpha_pct"], -d1[f"{k}_alpha_pct"]
    total = a_long + a_short
    print(f"{name:<8}{a_long:11.3f}{a_short:12.3f}{total:10.3f}"
          f"{a_long / total * 100:12.1f}%{a_short / total * 100:13.1f}%")
print("\n  Avramov, Cheng & Metzker (2023) report the GKX long leg at 0.77%/month")
print("  FF6-adjusted and the short leg insignificant at -0.15%. Compare the FF6")
print("  row above: a CAPM decomposition can attribute to the short leg what is")
print("  really exposure to size and momentum.")

worst = series.long_short_10_1.idxmin()
print(f"\n  long-short's worst month: {series.YYYYMM[worst]}  "
      f"{series.long_short_10_1[worst] * 100:.1f}%   "
      f"market that month {series.mktrf[worst] * 100:+.1f}%")

corr = series["vw"].corr(series["mktrf"])
print(f"\n  corr(own value-weighted market, FF mktrf) = {corr:.4f}")
if corr < 0.90:
    print("  WARNING: below 0.90. Check that market_cap is the capitalisation")
    print("  known at t, not at t+1 -- a look-ahead weight inflates the series.")

dest = RUN / f"{PREFIX}_benchmarks.csv"
out.to_csv(dest, index=False)
series.to_csv(RUN / f"{PREFIX}_benchmark_monthly.csv", index=False)
print(f"\nsaved: {dest}")
print(f"saved: {RUN / (PREFIX + '_benchmark_monthly.csv')}")
