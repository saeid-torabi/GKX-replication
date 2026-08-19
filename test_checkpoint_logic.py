"""
Logic tests for the resumable-checkpoint machinery in main.py.

Runs WITHOUT torch/pyarrow by importing main.py (whose heavy deps are import-
guarded) and, for the within-year orchestration test, swapping the torch/model/
train hooks for lightweight fakes. Focus is the NEW resume state machine:
  A) config guard + progress + idempotent per-year table writes
  B) combo/member-level resume: completed work is skipped, not recomputed
"""
import json
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Import the real module under test (its sibling imports need its dir on path).
import importlib.util

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR))
MAIN_PATH = THESIS_DIR / "main.py"
spec = importlib.util.spec_from_file_location("gkx_main", MAIN_PATH)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ---------------------------------------------------------------------------
# A) config guard + progress + idempotency
# ---------------------------------------------------------------------------
def test_config_and_progress():
    print("\n[A] config guard + progress + idempotency")
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoints"

        args = SimpleNamespace(
            model="NN1", data_path="x.parquet", epochs=5, batch_size=10000,
            learning_rate=1e-3, l1_lambda=1e-5, tune_hyperparameters=True,
            full_ensemble_grid=False, batchnorm_before_relu=False, parallel_nets=1,
            ensemble_size=3, seed=42, test_start_year=1987, test_end_year=2016,
            validation_years=12, early_stopping_patience=5,
            early_stopping_min_delta=0.0,
            max_test_years=None, decile_weight_col="market_cap",
        )
        tune_lrs = [0.001, 0.01]
        tune_l1s = [1e-5, 1e-4]
        identity = main._config_identity(args, tune_lrs, tune_l1s)

        # write + read round trip, incl input_features
        main._write_run_config(ckpt, identity, input_features=920)
        saved = main._read_run_config(ckpt)
        check("run_config round-trips input_features", saved.get("input_features") == 920)

        # identity survives a json round trip with no spurious diffs
        check("identical config -> no diffs",
              main._config_differences(saved, identity) == [])

        # a changed hyperparameter is detected
        args2 = SimpleNamespace(**{**vars(args), "learning_rate": 5e-4})
        identity2 = main._config_identity(args2, tune_lrs, tune_l1s)
        diffs = main._config_differences(saved, identity2)
        check("changed learning_rate -> diff detected",
              any(k == "learning_rate" for k, _, _ in diffs))

        # a changed tuning grid is detected
        identity3 = main._config_identity(args, tune_lrs, [1e-5, 1e-4, 1e-3])
        diffs3 = main._config_differences(saved, identity3)
        check("changed tune grid -> diff detected",
              any(k == "tune_l1_lambdas" for k, _, _ in diffs3))

        # Settings that change how batches are drawn or how many networks are
        # trained must invalidate a checkpoint: resuming across them would
        # silently blend two training schemes inside one ensemble.
        for flag, value in (("full_ensemble_grid", True),
                            ("batchnorm_before_relu", True),
                            ("batch_size", 5000)):
            flipped = SimpleNamespace(**{**vars(args), flag: value})
            diffs_flag = main._config_differences(
                saved, main._config_identity(flipped, tune_lrs, tune_l1s)
            )
            check(f"changed {flag} -> diff detected",
                  any(k == flag for k, _, _ in diffs_flag))

        # progress round trip + completed-years detection
        main._write_progress(ckpt, {1987, 1988, 1989}, total_years=30)
        prog = main._read_progress(ckpt)
        check("progress records completed years",
              prog["completed_test_years"] == [1987, 1988, 1989])
        check("progress computes next year", prog["next_test_year"] == 1990)
        check("progress not marked complete", prog["is_complete"] is False)
        completed = main._completed_years_from_checkpoints(ckpt, "NN1")
        check("completed years read back from progress", completed == {1987, 1988, 1989})

        # idempotent table append: writing the same year twice must not duplicate
        table = main._checkpoint_table_path(ckpt, "NN1", "split_results")
        row_1987 = {"test_year": 1987, "test_oos_r2": 0.01}
        main._remove_year_rows(table, 1987)
        main._append_checkpoint_rows(table, [row_1987])
        # simulate a re-save of the SAME year (as happens after a crash-resume)
        main._remove_year_rows(table, 1987)
        main._append_checkpoint_rows(table, [{"test_year": 1987, "test_oos_r2": 0.02}])
        df = main._read_checkpoint_table(ckpt, "NN1", "split_results")
        check("re-saving a year does not duplicate rows", (df["test_year"] == 1987).sum() == 1)
        check("re-saving a year keeps the newest value",
              float(df.loc[df["test_year"] == 1987, "test_oos_r2"].iloc[0]) == 0.02)

        # a second, different year appends alongside
        main._remove_year_rows(table, 1988)
        main._append_checkpoint_rows(table, [{"test_year": 1988, "test_oos_r2": 0.03}])
        df = main._read_checkpoint_table(ckpt, "NN1", "split_results")
        check("distinct years coexist", set(df["test_year"]) == {1987, 1988})

        # is_complete marks the run done and clears next_year
        main._write_progress(ckpt, {1987, 1988}, total_years=30, is_complete=True)
        prog = main._read_progress(ckpt)
        check("complete run has no next year", prog["next_test_year"] is None
              and prog["is_complete"] is True)


# ---------------------------------------------------------------------------
# B) within-year resume orchestration with fakes for torch/model/train
# ---------------------------------------------------------------------------
class FakeModel:
    def __init__(self, tag=0.0):
        self.tag = tag
    def state_dict(self):
        return {"tag": self.tag}
    def load_state_dict(self, d):
        self.tag = d["tag"]


class _FakeCuda:
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def manual_seed_all(seed):
        pass
    @staticmethod
    def empty_cache():
        pass


class FakeTorch:
    """Minimal stand-in: save/load via pickle so member .pt files round-trip."""
    cuda = _FakeCuda()

    @staticmethod
    def manual_seed(seed):
        pass

    @staticmethod
    def save(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path, map_location=None, weights_only=False):
        with open(path, "rb") as f:
            return pickle.load(f)


def install_fakes(train_counter):
    main.torch = FakeTorch()

    def fake_build(architecture, input_features, batchnorm_after_relu=True):
        return FakeModel()

    def fake_train_model(model, train_generator, val_generator, epochs,
                         learning_rate, l1_lambda, **kwargs):
        # Count only ACTUAL trainings so we can prove resumes skip work.
        train_counter["n"] += 1
        # deterministic "loss": lower l1 -> lower loss so best combo is known
        best = float(l1_lambda) * 1000 + 0.001
        m = FakeModel(tag=best)
        return {
            "model": m,
            "history": [{"epoch": 1, "train_loss": 1.0, "train_objective": 1.0,
                         "l1_penalty": 0.0, "val_loss": best, "selection_metric": best,
                         "best_metric": best, "best_epoch": 1, "improved": True,
                         "patience_counter": 0}],
            "best_metric": best,
            "best_epoch": 1,
            "epochs_trained": 1,
            "early_stopped": False,
        }

    def fake_ensemble_val_loss(member_results, val_generator,
                               max_val_batches=None, device=None):
        vals = [r["best_metric"] for r in member_results]
        return sum(vals) / len(vals)

    main.build_neural_net = fake_build
    main.train_model = fake_train_model
    main._ensemble_validation_loss = fake_ensemble_val_loss


def make_args(full_ensemble_grid=False):
    """Fake CLI args for the orchestration tests.

    This must carry every attribute the functions under test read off ``args``.
    A missing one surfaces as AttributeError mid-test rather than as a clear
    failure, so ``test_args_fixture_is_complete`` below checks the coverage
    directly instead of waiting for a run to trip over it.
    """
    return SimpleNamespace(
        model="NN1", epochs=1, learning_rate=1e-3, l1_lambda=1e-5,
        batch_size=10000,
        ensemble_size=2, seed=42,
        early_stopping_patience=5, early_stopping_min_delta=0.0,
        tune_hyperparameters=True,
        full_ensemble_grid=full_ensemble_grid, batchnorm_before_relu=False,
        parallel_nets=1,
    )


def test_args_fixture_is_complete():
    """Catch fixture drift the moment a new flag is read but not faked.

    Adding a CLI flag to main.py and wiring it into the training path without
    adding it here would otherwise fail deep inside an unrelated test.
    """
    print("\n[D] args fixtures cover what main.py reads")
    import re

    source = Path(main.__file__).read_text()
    watched = ("_run_year_resumable", "_train_ensemble_resumable",
               "_train_ensemble_parallel")
    needed = set()
    for chunk in re.split(r"\ndef ", source):
        if chunk.split("(")[0] in watched:
            needed |= set(re.findall(r"args\.(\w+)", chunk))

    provided = set(vars(make_args()))
    missing = sorted(needed - provided)
    label = (f"make_args covers all {len(needed)} args attributes read by the "
             f"{len(watched)} training functions")
    if missing:
        label += f"  -- missing: {missing}"
    check(label, not missing)


def test_within_year_resume():
    print("\n[B] within-year resume (default: tune-then-ensemble)")
    train_counter = {"n": 0}
    install_fakes(train_counter)

    split = SimpleNamespace(test_year=1990)
    tune_lrs = [0.001]
    tune_l1s = [1e-5, 1e-4]  # 2 combos; combo 0 (l1=1e-5) is the better one
    args = make_args()  # ensemble_size=2, default tune-then-ensemble

    def run(ckpt, a=args):
        return main._run_year_resumable(
            a, split, ckpt, input_features=920,
            tune_learning_rates=tune_lrs, tune_l1_lambdas=tune_l1s,
            train_generator=object(), val_generator=object(),
        )

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoints"
        # grid trains 1 net per combo (2), then tops up the winner 1->2 (+1) = 3
        mr, mh, best, tr = run(ckpt)
        check("default clean run trains grid(2)+topup(1) = 3 nets",
              train_counter["n"] == 3)
        check("best combo is the low-l1 one", best["l1_lambda"] == 1e-5)
        check("returns ensemble_size members", len(mr) == 2)
        check("one tuning row per combo", len(tr) == 2)

        # full resume: everything already on disk -> nothing retrains
        train_counter["n"] = 0
        mr2, _, best2, tr2 = run(ckpt)
        check("default full resume retrains nothing", train_counter["n"] == 0)
        check("resume selects same best combo", best2["l1_lambda"] == 1e-5)
        check("resume loads full ensemble", len(mr2) == 2)
        check("resume reconstructs tuning rows", len(tr2) == 2)

    # partial crash (default mode): fault after 2 real trainings, then resume
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoints"
        train_counter["n"] = 0
        real_train = main.train_model
        crash_state = {"n": 0}

        def crashing(*a, **k):
            r = real_train(*a, **k)
            crash_state["n"] += 1
            if crash_state["n"] >= 2:
                raise RuntimeError("injected crash")
            return r

        main.train_model = crashing
        try:
            run(ckpt)
        except RuntimeError:
            pass
        main.train_model = real_train
        check("crashed after 2 trainings", crash_state["n"] == 2)

        # combo0 saved+marked; combo1's net was lost to the crash (before save).
        # Resume retrains combo1's net (1) then tops up the winner combo0 (+1) = 2.
        train_counter["n"] = 0
        mr3, _, best3, tr3 = run(ckpt)
        check("partial resume trains only what's missing (2 nets)",
              train_counter["n"] == 2)
        check("partial resume selects best combo", best3["l1_lambda"] == 1e-5)
        check("partial resume returns full ensemble", len(mr3) == 2)

    print("\n[C] within-year resume (--full_ensemble_grid restores old behavior)")
    train_counter["n"] = 0
    args_full = make_args(full_ensemble_grid=True)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoints"
        mr, mh, best, tr = main._run_year_resumable(
            args_full, split, ckpt, input_features=920,
            tune_learning_rates=tune_lrs, tune_l1_lambdas=tune_l1s,
            train_generator=object(), val_generator=object(),
        )
        check("full-grid trains ensemble_size x n_combos = 4 nets",
              train_counter["n"] == 4)
        check("full-grid best is the low-l1 one", best["l1_lambda"] == 1e-5)
        check("full-grid returns ensemble", len(mr) == 2)


if __name__ == "__main__":
    test_config_and_progress()
    test_within_year_resume()
    test_args_fixture_is_complete()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
