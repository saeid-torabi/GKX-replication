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
FF = Path("data_csv/ff3.csv")
PREFIX = "nn1"


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
ff = pd.read_csv(FF)
ff["YYYYMM"] = (pd.to_datetime(ff["dateff"]).dt.year * 100
                + pd.to_datetime(ff["dateff"]).dt.month)
series = series.merge(ff[["YYYYMM", "mktrf", "smb", "hml", "rf"]], on="YYYYMM")
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
    capm_b, capm_t, capm_p = newey_west_tstat(r, series[["mktrf"]].to_numpy())
    ff3_b, ff3_t, ff3_p = newey_west_tstat(
        r, series[["mktrf", "smb", "hml"]].to_numpy())
    rows.append({
        "portfolio": label,
        "mean_pct": r.mean() * 100,
        "sd_pct": r.std(ddof=1) * 100,
        "sharpe": annualized_sharpe_ratio(pd.Series(r)),
        "capm_alpha_pct": capm_b[0] * 100,
        "capm_t": capm_t[0],
        "capm_p": capm_p[0],
        "capm_beta": capm_b[1],
        "ff3_alpha_pct": ff3_b[0] * 100,
        "ff3_t": ff3_t[0],
        "ff3_p": ff3_p[0],
        **performance_profile(r),
    })

out = pd.DataFrame(rows)


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


print("PANEL A -- risk, return and risk-adjusted performance")
hdr = (f"{'Portfolio':<28}{'Mean%':>8}{'SD%':>7}{'SR':>7}"
       f"{'CAPM a%':>9}{'(t)':>7}{'p':>8}{'beta':>7}"
       f"{'FF3 a%':>9}{'(t)':>7}{'p':>8}")
print(hdr)
print("-" * len(hdr))
for _, r in out.iterrows():
    print(f"{r.portfolio:<28}{r.mean_pct:8.3f}{r.sd_pct:7.2f}{r.sharpe:7.3f}"
          f"{r.capm_alpha_pct:9.3f}{r.capm_t:7.2f}{r.capm_p:8.4f}{r.capm_beta:7.2f}"
          f"{r.ff3_alpha_pct:9.3f}{r.ff3_t:7.2f}{r.ff3_p:8.4f}")
print(f"\n  significance of the CAPM alpha:  "
      + "   ".join(f"{r.portfolio.split(',')[0]}{stars(r.capm_p)}"
                   for _, r in out.iterrows() if stars(r.capm_p)))
print("  *** p<0.01  ** p<0.05  * p<0.10   (Newey-West, "
      + ("Student-t)" if _scipy_stats is not None else "normal approximation)"))

print("\nPANEL B -- cumulative growth and tail behaviour")
hdr2 = (f"{'Portfolio':<28}{'$1 becomes':>12}{'max DD%':>9}"
        f"{'best mo%':>10}{'worst mo%':>11}{'% months up':>13}")
print(hdr2)
print("-" * len(hdr2))
for _, r in out.iterrows():
    print(f"{r.portfolio:<28}{r.growth_of_1:11.2f}x{r.max_drawdown_pct:9.1f}"
          f"{r.best_month_pct:10.1f}{r.worst_month_pct:11.1f}"
          f"{r.pct_months_positive:13.1f}")
print("\n  Wealth compounds EXCESS returns, i.e. growth above cash, so the")
print("  long-only and self-financing portfolios stay comparable.")

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
