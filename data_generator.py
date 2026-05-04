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

        print("DataGenerator initialized:")
        print(f"  -> Connected to {filepath}")
        if self.date_start is not None or self.date_end is not None:
            print(
                f"  -> Window: {self.date_start or '-inf'} to "
                f"{self.date_end or '+inf'}"
            )
        print(f"  -> Total rows available: {self.total_rows:,}")
        print(f"  -> Batch size: {self.batch_size:,}")
        if self.shuffle:
            print(f"  -> Shuffle buffer: {self.shuffle_buffer_batches} batches")
        print(
            "  -> Feature blocks: "
            f"{len(self.char_cols)} chars, "
            f"{len(self.macro_cols)} macros, "
            f"{len(self.dummy_cols)} dummies, "
            f"{self.num_features} total features"
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

    def _iter_ordered_batches(self):
        """
        Yield one ordered batch of PyTorch tensors at a time.
        """
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
