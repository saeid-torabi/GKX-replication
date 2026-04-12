from dataclasses import dataclass


def month_id(year, month):
    return year * 100 + month


@dataclass(frozen=True)
class SplitSpec:
    test_year: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def generate_gkx_splits(
    test_start_year=1987,
    test_end_year=2016,
    train_start=195703,
    validation_years=12,
):
    """
    Generate the paper-style recursive windows.

    Example:
    - test year 1987
    - validation window 1975-1986
    - expanding training window 1957-1974
    """
    splits = []

    for test_year in range(test_start_year, test_end_year + 1):
        val_start_year = test_year - validation_years
        val_end_year = test_year - 1
        train_end_year = val_start_year - 1

        splits.append(
            SplitSpec(
                test_year=test_year,
                train_start=train_start,
                train_end=month_id(train_end_year, 12),
                val_start=month_id(val_start_year, 1),
                val_end=month_id(val_end_year, 12),
                test_start=month_id(test_year, 1),
                test_end=month_id(test_year, 12),
            )
        )

    return splits
