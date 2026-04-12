import numpy as np
import pandas as pd


def oos_r2(df, target_col="excess_ret", prediction_col="prediction"):
    """
    GKX stock-level out-of-sample R^2 uses raw excess returns in the denominator,
    not demeaned returns.
    """
    sse = ((df[target_col] - df[prediction_col]) ** 2).sum()
    sst = (df[target_col] ** 2).sum()
    return 1.0 - (sse / sst)


def _grouped_oos_r2(df, group_col, target_col, prediction_col):
    rows = []
    for group_value, group_df in df.groupby(group_col):
        rows.append(
            {
                group_col: group_value,
                "oos_r2": oos_r2(group_df, target_col, prediction_col),
            }
        )
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def summarize_prediction_panel(df, target_col="excess_ret", prediction_col="prediction"):
    monthly = _grouped_oos_r2(
        df=df,
        group_col="YYYYMM",
        target_col=target_col,
        prediction_col=prediction_col,
    )
    monthly["year"] = monthly["YYYYMM"] // 100

    annual = _grouped_oos_r2(
        df=df.assign(year=df["YYYYMM"] // 100),
        group_col="year",
        target_col=target_col,
        prediction_col=prediction_col,
    )

    return {
        "overall_oos_r2": oos_r2(df, target_col, prediction_col),
        "monthly_oos_r2": monthly,
        "annual_oos_r2": annual,
    }


def mse(df, target_col="excess_ret", prediction_col="prediction"):
    errors = df[target_col] - df[prediction_col]
    return float(np.mean(np.square(errors)))
