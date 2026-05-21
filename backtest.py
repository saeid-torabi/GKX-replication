import numpy as np
import pandas as pd


def build_long_short_deciles(
    predictions_df,
    prediction_col="prediction",
    target_col="excess_ret",
    date_col="YYYYMM",
    weight_col=None,
):
    """
    Form 10 decile portfolios each month from model forecasts.

    This follows the paper's monthly reconstitution logic. If ``weight_col`` is
    provided, returns are value-weighted; otherwise the routine falls back to
    equal-weighted portfolio returns.
    """
    working = predictions_df.copy()
    working["decile"] = (
        working.groupby(date_col)[prediction_col]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
        .astype(int)
    )

    if weight_col is not None:
        if weight_col not in working.columns:
            raise ValueError(
                f"Weight column '{weight_col}' is missing from predictions data."
            )
        if (working[weight_col] <= 0).any():
            raise ValueError(
                f"Weight column '{weight_col}' must be strictly positive."
            )

        portfolio_rows = []
        for (month, decile), group_df in working.groupby([date_col, "decile"]):
            portfolio_rows.append(
                {
                    date_col: month,
                    "decile": decile,
                    "predicted_return": np.average(
                        group_df[prediction_col],
                        weights=group_df[weight_col],
                    ),
                    "portfolio_return": np.average(
                        group_df[target_col],
                        weights=group_df[weight_col],
                    ),
                }
            )
        portfolio_returns = pd.DataFrame(portfolio_rows)
    else:
        portfolio_returns = (
            working.groupby([date_col, "decile"])
            .agg(
                predicted_return=(prediction_col, "mean"),
                portfolio_return=(target_col, "mean"),
            )
            .reset_index()
        )

    pivoted = portfolio_returns.pivot(
        index=date_col,
        columns="decile",
        values="portfolio_return",
    ).sort_index()

    pivoted["long_short_10_1"] = pivoted[10] - pivoted[1]
    pivoted = pivoted.reset_index()

    return portfolio_returns, pivoted


def annualized_sharpe_ratio(returns, periods_per_year=12):
    returns = pd.Series(returns).dropna()
    volatility = returns.std(ddof=1)
    if returns.empty or volatility == 0 or np.isnan(volatility):
        return np.nan
    return np.sqrt(periods_per_year) * returns.mean() / volatility


def summarize_decile_performance(
    portfolio_returns,
    date_col="YYYYMM",
    return_col="portfolio_return",
    prediction_col="predicted_return",
):
    """
    Summarize GKX-style decile portfolio performance.

    Prediction, average return, and return standard deviation are reported as
    monthly percentages. Sharpe ratios are annualized using monthly returns.
    """
    returns = portfolio_returns.pivot(
        index=date_col,
        columns="decile",
        values=return_col,
    ).sort_index()
    predictions = portfolio_returns.pivot(
        index=date_col,
        columns="decile",
        values=prediction_col,
    ).sort_index()

    rows = []
    for decile in sorted(returns.columns):
        realized = returns[decile].dropna()
        forecast = predictions[decile].reindex(realized.index)
        rows.append(
            {
                "portfolio": str(decile),
                "prediction_pct_per_month": forecast.mean() * 100,
                "average_pct_per_month": realized.mean() * 100,
                "sd_pct_per_month": realized.std(ddof=1) * 100,
                "annualized_sharpe_ratio": annualized_sharpe_ratio(realized),
                "n_months": int(realized.shape[0]),
            }
        )

    if 1 in returns.columns and 10 in returns.columns:
        realized = (returns[10] - returns[1]).dropna()
        forecast = (predictions[10] - predictions[1]).reindex(realized.index)
        rows.append(
            {
                "portfolio": "10-1 H-L",
                "prediction_pct_per_month": forecast.mean() * 100,
                "average_pct_per_month": realized.mean() * 100,
                "sd_pct_per_month": realized.std(ddof=1) * 100,
                "annualized_sharpe_ratio": annualized_sharpe_ratio(realized),
                "n_months": int(realized.shape[0]),
            }
        )

    return pd.DataFrame(rows)
