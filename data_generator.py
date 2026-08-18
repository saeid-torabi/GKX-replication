import math
import numpy as np
import pandas as pd

try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised at runtime
    ds = None
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised at runtime
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


ID_COLS = {"permno", "YYYYMM", "sic2", "excess_ret", "market_cap"}
MACRO_PREFIX = "macro_"
SIC_PREFIX = "sic2_"


class GKXDataGenerator:
    """
    Stream a lean GKX base panel from Parquet and construct interactions on the fly.

    The generator accepts optional date filters so the recursive estimation routine
    can train on expanding windows, validate on rolling windows, and score a
    one-year test window without materializing the full 920-column design matrix.
    """

    def __init__(
        self,
        filepath,
        batch_size,
        macro_cols=None,
        char_cols=None,
        dummy_cols=None,
        target_col="excess_ret",
        date_col="YYYYMM",
        date_start=None,
        date_end=None,
        return_metadata=False,
        metadata_cols=None,
        shuffle=False,
        shuffle_buffer_batches=8,
        cache_window=False,
    ):
        if pq is None or ds is None:
            raise ImportError(
                "pyarrow is required to stream Parquet files. "
                "Install it with `pip install pyarrow`."
            ) from _PYARROW_IMPORT_ERROR
        if torch is None:
            raise ImportError(
                "torch is required for model training. "
                "Install it with `pip install torch`."
            ) from _TORCH_IMPORT_ERROR

        self.filepath = filepath
        self.batch_size = batch_size
        self.target_col = target_col
        self.date_col = date_col
        self.date_start = date_start
        self.date_end = date_end
        self.return_metadata = return_metadata
        self.metadata_cols = metadata_cols or [date_col, "permno"]
        self.shuffle = shuffle
        self.shuffle_buffer_batches = shuffle_buffer_batches
        if self.shuffle and self.shuffle_buffer_batches < 1:
            raise ValueError("shuffle_buffer_batches must be at least 1 when shuffle=True.")
        # When the window is cached in RAM, shuffling is a full permutation of
        # every row, so the buffer size is irrelevant and deliberately ignored.
        self.cache_window = cache_window
        self._cache = None

        # Read schema once from parquet metadata, but iterate through a dataset
        # scanner so we can push time-window filters into the file reader.
        self.parquet_file = pq.ParquetFile(filepath)
        self.column_names = self.parquet_file.schema.names
        self.dataset = ds.dataset(filepath, format="parquet")
        self.filter_expression = self._build_filter()
        self.total_rows = self.dataset.count_rows(filter=self.filter_expression)

        self.macro_cols = sorted(macro_cols or self._infer_macro_cols())
        self.char_cols = sorted(char_cols or self._infer_char_cols())
        self.dummy_cols = sorted(dummy_cols or self._infer_dummy_cols())

        self._validate_columns()
        self.num_features = (
            len(self.char_cols)
            + len(self.char_cols) * len(self.macro_cols)
            + len(self.dummy_cols)
        )

        window = f"{self.date_start or '-inf'}-{self.date_end or '+inf'}"
        if self.shuffle and self.cache_window:
            shuffle_note = ", shuffle full"
        elif self.shuffle:
            shuffle_note = f", shuffle x{self.shuffle_buffer_batches}"
        else:
            shuffle_note = ""
        if self.cache_window:
            raw_bytes = self.total_rows * (
                len(self.char_cols) * 4 + len(self.macro_cols) * 4
                + len(self.dummy_cols) + 4
            )
            shuffle_note += f", cached {raw_bytes / 1024 ** 3:.2f} GB"
        print(
            f"  [data] {window}: {self.total_rows:,} rows, "
            f"{self.num_features} features, batch {self.batch_size:,}{shuffle_note}"
        )

    def _build_filter(self):
        filters = []
        if self.date_start is not None:
            filters.append(ds.field(self.date_col) >= self.date_start)
        if self.date_end is not None:
            filters.append(ds.field(self.date_col) <= self.date_end)

        if not filters:
            return None

        expression = filters[0]
        for extra_filter in filters[1:]:
            expression = expression & extra_filter
        return expression

    def _infer_macro_cols(self):
        return [col for col in self.column_names if col.startswith(MACRO_PREFIX)]

    def _infer_dummy_cols(self):
        return [col for col in self.column_names if col.startswith(SIC_PREFIX)]

    def _infer_char_cols(self):
        return [
            col
            for col in self.column_names
            if col not in ID_COLS
            and not col.startswith(MACRO_PREFIX)
            and not col.startswith(SIC_PREFIX)
        ]

    def _validate_columns(self):
        required_cols = [
            self.target_col,
            self.date_col,
            *self.macro_cols,
            *self.char_cols,
            *self.dummy_cols,
            *self.metadata_cols,
        ]
        missing = [col for col in required_cols if col not in self.column_names]
        if missing:
            raise ValueError(f"Missing expected columns in parquet file: {missing}")

        if self.total_rows == 0:
            raise ValueError(
                "The selected time window contains zero rows. "
                "Check the split boundaries and parquet contents."
            )
        if not self.char_cols:
            raise ValueError("No firm characteristic columns were detected.")
        if not self.macro_cols:
            raise ValueError("No macro columns were detected.")
        if not self.dummy_cols:
            raise ValueError("No SIC2 dummy columns were detected.")

    def _columns_to_read(self):
        columns_to_read = (
            self.char_cols
            + self.macro_cols
            + self.dummy_cols
            + [self.target_col]
            + self.metadata_cols
        )
        return list(dict.fromkeys(columns_to_read))

    def _build_cache(self):
        """Read the window's *raw* columns into RAM once.

        The 920-column design matrix is a deterministic function of these, so it
        is rebuilt per batch by ``_assemble`` rather than stored: for the largest
        training window that is ~1.3 GB instead of ~9.7 GB. Caching the raw
        columns is what makes a full permutation of the window affordable on a
        small machine, and it also removes the repeated Parquet decode that
        otherwise happens once per epoch per network.
        """
        scanner = self.dataset.scanner(
            columns=self._columns_to_read(),
            filter=self.filter_expression,
            batch_size=self.batch_size,
        )
        chars, macros, dummies, targets, metas = [], [], [], [], []
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            df_chunk = batch.to_pandas()
            chars.append(df_chunk[self.char_cols].to_numpy(dtype=np.float32))
            macros.append(df_chunk[self.macro_cols].to_numpy(dtype=np.float32))
            # SIC dummies are 0/1, so int8 holds them exactly at a quarter the
            # cost of float32; they are widened back in _assemble.
            dummies.append(df_chunk[self.dummy_cols].to_numpy(dtype=np.int8))
            targets.append(df_chunk[self.target_col].to_numpy(dtype=np.float32))
            if self.return_metadata:
                metas.append(df_chunk[self.metadata_cols])

        self._cache = {
            "chars": np.concatenate(chars, axis=0),
            "macros": np.concatenate(macros, axis=0),
            "dummies": np.concatenate(dummies, axis=0),
            "y": np.concatenate(targets, axis=0),
            "meta": (pd.concat(metas, ignore_index=True)
                     if self.return_metadata else None),
        }

    def _assemble(self, row_index):
        """Build one batch of the 920-column design matrix from cached raw rows.

        This mirrors _iter_ordered_batches exactly, so cached and streamed runs
        produce identical feature matrices for the same rows."""
        chars_array = self._cache["chars"][row_index]
        macros_array = self._cache["macros"][row_index]
        dummies_array = self._cache["dummies"][row_index].astype(np.float32)
        targets_array = self._cache["y"][row_index]
        current_batch_size = chars_array.shape[0]

        interactions_array = (
            chars_array[:, :, None] * macros_array[:, None, :]
        ).reshape(current_batch_size, -1)
        x_final = np.concatenate(
            [chars_array, interactions_array, dummies_array], axis=1
        )

        x_tensor = torch.from_numpy(np.ascontiguousarray(x_final))
        y_tensor = torch.from_numpy(np.ascontiguousarray(targets_array)).view(-1, 1)

        if self.return_metadata:
            metadata_df = self._cache["meta"].iloc[row_index].reset_index(drop=True)
            return x_tensor, y_tensor, metadata_df
        return x_tensor, y_tensor

    def _iter_cached_batches(self, row_order):
        for start in range(0, row_order.shape[0], self.batch_size):
            batch_index = row_order[start:start + self.batch_size]
            if batch_index.shape[0] < 2:
                continue
            yield self._assemble(batch_index)

    def _iter_ordered_batches(self):
        """
        Yield one ordered batch of PyTorch tensors at a time.
        """
        if self.cache_window:
            if self._cache is None:
                self._build_cache()
            yield from self._iter_cached_batches(
                np.arange(self._cache["y"].shape[0])
            )
            return

        columns_to_read = (
            self.char_cols
            + self.macro_cols
            + self.dummy_cols
            + [self.target_col]
            + self.metadata_cols
        )
        # Preserve order and remove duplicates.
        columns_to_read = list(dict.fromkeys(columns_to_read))

        scanner = self.dataset.scanner(
            columns=columns_to_read,
            filter=self.filter_expression,
            batch_size=self.batch_size,
        )

        for batch in scanner.to_batches():
            df_chunk = batch.to_pandas()
            current_batch_size = len(df_chunk)
            if current_batch_size == 0:
                continue

            chars_array = df_chunk[self.char_cols].to_numpy(dtype=np.float32, copy=False)
            macros_array = df_chunk[self.macro_cols].to_numpy(dtype=np.float32, copy=False)
            dummies_array = df_chunk[self.dummy_cols].to_numpy(dtype=np.float32, copy=False)
            targets_array = df_chunk[self.target_col].to_numpy(dtype=np.float32, copy=False)

            # Broadcast (batch, 94, 1) against (batch, 1, 8) to produce
            # the 94 x 8 interaction block for the current chunk only.
            interactions_array = (
                chars_array[:, :, None] * macros_array[:, None, :]
            ).reshape(current_batch_size, -1)

            x_final = np.concatenate(
                [chars_array, interactions_array, dummies_array],
                axis=1,
            )

            if np.isnan(x_final).any():
                source_cols = self.char_cols + self.macro_cols + self.dummy_cols
                source_nan_counts = df_chunk[source_cols].isna().sum()
                source_nan_counts = source_nan_counts[
                    source_nan_counts > 0
                ].sort_values(ascending=False)

                raise ValueError(
                    "NaNs detected in feature matrix before tensor conversion.\n"
                    f"Source columns with NaNs:\n{source_nan_counts}"
                )

            if np.isnan(targets_array).any():
                raise ValueError(
                    f"NaNs detected in target column '{self.target_col}' "
                    "before tensor conversion."
                )

            x_tensor = torch.from_numpy(x_final)
            y_tensor = torch.from_numpy(targets_array).view(-1, 1)

            if self.return_metadata:
                metadata_df = df_chunk[self.metadata_cols].reset_index(drop=True)
                yield x_tensor, y_tensor, metadata_df
            else:
                yield x_tensor, y_tensor

    def _flush_shuffle_buffer(self, buffer):
        if self.return_metadata:
            x_tensor = torch.cat([item[0] for item in buffer], dim=0)
            y_tensor = torch.cat([item[1] for item in buffer], dim=0)
            metadata_df = pd.concat([item[2] for item in buffer], ignore_index=True)
        else:
            x_tensor = torch.cat([item[0] for item in buffer], dim=0)
            y_tensor = torch.cat([item[1] for item in buffer], dim=0)
            metadata_df = None

        permutation = torch.randperm(x_tensor.shape[0])
        for start in range(0, x_tensor.shape[0], self.batch_size):
            batch_index = permutation[start:start + self.batch_size]
            if batch_index.numel() < 2:
                continue
            if self.return_metadata:
                batch_metadata = metadata_df.iloc[batch_index.numpy()].reset_index(drop=True)
                yield x_tensor[batch_index], y_tensor[batch_index], batch_metadata
            else:
                yield x_tensor[batch_index], y_tensor[batch_index]

    def __iter__(self):
        """
        Yield one batch of PyTorch tensors at a time.
        """
        if not self.shuffle:
            yield from self._iter_ordered_batches()
            return

        if self.cache_window:
            # GKX Internet Appendix B.3: "At each step of training, a batch sent
            # to the algorithm is randomly sampled from the training dataset."
            # With the window in RAM this is a genuine permutation of every row,
            # so batches mix the whole training period instead of the handful of
            # adjacent months a streaming buffer can hold. shuffle_buffer_batches
            # plays no role here.
            if self._cache is None:
                self._build_cache()
            n_rows = self._cache["y"].shape[0]
            yield from self._iter_cached_batches(
                torch.randperm(n_rows).numpy()
            )
            return

        buffer = []
        buffered_rows = 0
        flush_threshold = self.batch_size * self.shuffle_buffer_batches
        for item in self._iter_ordered_batches():
            buffer.append(item)
            buffered_rows += item[0].shape[0]
            if buffered_rows >= flush_threshold:
                yield from self._flush_shuffle_buffer(buffer)
                buffer = []
                buffered_rows = 0

        if buffer:
            yield from self._flush_shuffle_buffer(buffer)

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return math.ceil(self.total_rows / self.batch_size)
