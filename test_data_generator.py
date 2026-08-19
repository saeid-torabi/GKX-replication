"""Checks on GKXDataGenerator's batch construction and sampling.

There is now a single read path -- each window's raw columns are cached in RAM
and the 920-column design matrix is rebuilt per batch -- so these verify:

  1. Feature assembly: 920 columns laid out as 94 characteristics, then the
     94x8 characteristic-macro interactions, then 74 industry dummies, with the
     interaction block equal to the outer product of its inputs.
  2. Ordered iteration returns every row exactly once, in the panel's stored
     date order (this is what prediction and the GPU path rely on).
  3. Shuffled iteration is a true permutation -- every row once per epoch, a
     different order each epoch, and each row keeping its own features.
  4. The permutation is global: one batch looks like the whole training period,
     not a slice of it. This is the property GKX specify in Internet Appendix
     B.3 and the reason the old buffered shuffle was removed.

Run:  python test_data_generator.py [path/to/gkx_base_dataset_2016.parquet]
"""
import sys

import numpy as np
import torch

from data_generator import GKXDataGenerator

DATA = sys.argv[1] if len(sys.argv) > 1 else "gkx_base_dataset_2016.parquet"
START, END = 195703, 196512
BATCH = 10000

failures = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          f"{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(label)


def collect(generator):
    xs, ys = [], []
    for batch in generator:
        xs.append(batch[0].numpy())
        ys.append(batch[1].numpy().reshape(-1))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


print(f"\ndata   : {DATA}")
print(f"window : {START}-{END}\n")

print("1. feature assembly")
gen = GKXDataGenerator(filepath=DATA, batch_size=BATCH,
                       date_start=START, date_end=END)
x, y = collect(gen)
n_char, n_macro, n_dummy = (len(gen.char_cols), len(gen.macro_cols),
                            len(gen.dummy_cols))
check("column counts are 94 / 8 / 74",
      (n_char, n_macro, n_dummy) == (94, 8, 74),
      f"got {n_char} / {n_macro} / {n_dummy}")
check("design matrix is 920 wide",
      x.shape[1] == n_char + n_char * n_macro + n_dummy,
      f"{x.shape[1]} columns")
check("no NaNs in assembled features", not np.isnan(x).any())
check("industry dummy block is binary",
      np.isin(x[:, n_char + n_char * n_macro:], (0.0, 1.0)).all())

# Compare the assembled blocks against the cached raw columns directly. An
# earlier version tried to recover the macros by dividing the interaction block
# by the first characteristic, which is numerically meaningless: characteristics
# are rank-normalised into [-1, 1] and sit at or near zero often enough that the
# division amplifies float error without bound.
k = min(200, x.shape[0])
raw_chars = gen._cache["chars"][:k]
raw_macros = gen._cache["macros"][:k]
inter = x[:k, n_char:n_char + n_char * n_macro].reshape(k, n_char, n_macro)
check("characteristic block matches the raw columns",
      np.array_equal(x[:k, :n_char], raw_chars))
check("interactions equal char x macro outer product",
      np.allclose(inter, raw_chars[:, :, None] * raw_macros[:, None, :],
                  atol=1e-6, rtol=1e-5))

print("\n2. ordered iteration")
meta_kwargs = dict(filepath=DATA, batch_size=BATCH, date_start=START,
                   date_end=END, return_metadata=True,
                   metadata_cols=["YYYYMM", "permno"])
ordered = [(b[2]["YYYYMM"].to_numpy(), b[2]["permno"].to_numpy())
           for b in GKXDataGenerator(**meta_kwargs)]
months = np.concatenate([m for m, _ in ordered])
permnos = np.concatenate([p for _, p in ordered])
check("row count matches the window", months.size == x.shape[0],
      f"{months.size:,} rows")
check("every row delivered exactly once",
      months.size == gen.total_rows, f"{gen.total_rows:,} expected")
# Prediction joins its metadata on this ordering, so it must be stable.
months_again = np.concatenate([b[2]["YYYYMM"].to_numpy()
                               for b in GKXDataGenerator(**meta_kwargs)])
check("ordered iteration is repeatable", np.array_equal(months, months_again))

# Diagnostic, not an assertion. The panel is written with
# sort_values(['permno','YYYYMM']), so consecutive rows are one stock's whole
# history rather than one month's cross-section. That layout is why a shuffle
# confined to a window of adjacent rows is harmful: it yields batches holding
# very few distinct stocks.
by_date = bool(np.all(np.diff(months) >= 0))
by_stock = bool(np.all(np.diff(permnos) >= 0))
layout = "date-major" if by_date else ("stock-major" if by_stock else "unsorted")
head = min(80_000, permnos.size)
print(f"     storage layout           : {layout}")
print(f"     distinct stocks, all rows: {np.unique(permnos).size:,}")
print(f"     distinct stocks, first {head:,}: "
      f"{np.unique(permnos[:head]).size:,}")

print("\n3. shuffled iteration is a permutation")
torch.manual_seed(0)
x_shuf, y_shuf = collect(GKXDataGenerator(filepath=DATA, batch_size=BATCH,
                                          date_start=START, date_end=END,
                                          shuffle=True))
check("same row count as ordered", x_shuf.shape == x.shape,
      f"{x_shuf.shape} vs {x.shape}")
check("order actually changed", not np.array_equal(y_shuf, y))
check("targets are a permutation, none lost or duplicated",
      np.array_equal(np.sort(y_shuf), np.sort(y)))

lookup = {}
for pos, val in enumerate(y):
    lookup.setdefault(float(val), []).append(pos)
mismatched = 0
for i in np.random.default_rng(0).choice(x_shuf.shape[0], 300, replace=False):
    if not any(np.array_equal(x[c], x_shuf[i])
               for c in lookup.get(float(y_shuf[i]), [])):
        mismatched += 1
check("sampled rows keep their own 920 features", mismatched == 0,
      f"{mismatched}/300 mismatched")

torch.manual_seed(1)
_, y_epoch2 = collect(GKXDataGenerator(filepath=DATA, batch_size=BATCH,
                                       date_start=START, date_end=END,
                                       shuffle=True))
check("a second epoch draws a different permutation",
      not np.array_equal(y_shuf, y_epoch2))

print("\n4. the permutation is global")
# Compare one batch's spread over calendar months against the population's,
# via the largest gap between their cumulative distributions (a KS statistic).
# A global permutation drives this to ~0. A shuffle confined to blocks of
# `block` consecutive rows cannot, since its first batch holds only the
# earliest months -- that reference number is reported for contrast.
torch.manual_seed(0)
batch_months = next(iter(GKXDataGenerator(shuffle=True, **meta_kwargs)))[2]["YYYYMM"].to_numpy()
uniq = np.unique(months)


def cdf(values):
    c = np.cumsum(np.array([(values == m).sum() for m in uniq], dtype=float))
    return c / c[-1]


pop_cdf = cdf(months)
ks = float(np.abs(pop_cdf - cdf(batch_months)).max())
ks_blockwise = float(np.abs(pop_cdf - cdf(months[:BATCH * 8])).max())

check("batch spans the window's whole date range",
      batch_months.min() == months.min() and batch_months.max() == months.max(),
      f"batch {batch_months.min()}-{batch_months.max()}")
check("batch covers nearly every month",
      np.unique(batch_months).size >= 0.9 * uniq.size,
      f"{np.unique(batch_months).size} of {uniq.size} months")
check("batch month distribution matches the population", ks < 0.05,
      f"KS {ks:.4f}  (a blockwise shuffle would score {ks_blockwise:.4f})")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) -> {failures}")
    sys.exit(1)
print("All checks passed.")
