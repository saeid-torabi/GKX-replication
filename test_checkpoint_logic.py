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
            ensemble_size=3, seed=42, test_start_year=1987, test_end_year=2016,
            validation_years=12, early_stopping_patience=5,
            early_stopping_min_delta=0.0, no_shuffle_train=False,
            shuffle_buffer_batches=8, max_train_batches=None, max_val_batches=None,
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

    def fake_build(architecture, input_features):
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

    def fake_ensemble_val_loss(member_results, val_generator, max_val_batches=None):
        vals = [r["best_metric"] for r in member_results]
        return sum(vals) / len(vals)

    main.build_neural_net = fake_build
    main.train_model = fake_train_model
    main._ensemble_validation_loss = fake_ensemble_val_loss


def make_args():
    return SimpleNamespace(
        model="NN1", epochs=1, learning_rate=1e-3, l1_lambda=1e-5,
        ensemble_size=2, seed=42, max_train_batches=None, max_val_batches=None,
        early_stopping_patience=5, early_stopping_min_delta=0.0,
        log_diagnostics=False, tune_hyperparameters=True,
    )


def test_within_year_resume():
    print("\n[B] within-year combo/member resume")
    train_counter = {"n": 0}
    install_fakes(train_counter)

    split = SimpleNamespace(test_year=1990)
    tune_lrs = [0.001]
    tune_l1s = [1e-5, 1e-4]  # 2 combos; combo 0 (l1=1e-5) is the better one
    args = make_args()

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoints"

        # --- full clean run: 2 combos x 2 members = 4 trainings ---
        mr, mh, best, tuning_rows = main._run_year_resumable(
            args, split, ckpt, input_features=920,
            tune_learning_rates=tune_lrs, tune_l1_lambdas=tune_l1s,
            train_generator=object(), val_generator=object(),
        )
        check("clean run trains ensemble_size * n_combos nets", train_counter["n"] == 4)
        check("best combo is the low-l1 one", best["l1_lambda"] == 1e-5)
        check("returns ensemble_size member results", len(mr) == 2)
        check("tuning rows recorded per combo", len(tuning_rows) == 2)

        # --- resume with everything already done: 0 new trainings ---
        train_counter["n"] = 0
        mr2, mh2, best2, tuning_rows2 = main._run_year_resumable(
            args, split, ckpt, input_features=920,
            tune_learning_rates=tune_lrs, tune_l1_lambdas=tune_l1s,
            train_generator=object(), val_generator=object(),
        )
        check("full resume retrains nothing", train_counter["n"] == 0)
        check("resume still selects same best combo", best2["l1_lambda"] == 1e-5)
        check("resume reloads all members", len(mr2) == 2)
        check("resume reconstructs tuning rows", len(tuning_rows2) == 2)

    # --- partial resume: crash after 3 of 4 members ---
    train_counter["n"] = 0
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "checkpoints"

        # Manually run combo 0 fully (2 members), then combo 1 with ONE member,
        # leaving combo 1 incomplete (no combo.json) -> mimics a mid-combo crash.
        combo0 = main._combo_dir(ckpt, split.test_year, 0)
        main._train_ensemble_resumable(args, combo0, object(), object(), 920, 0.001, 1e-5)
        # mark combo 0 done as the real code would
        main._combo_done_path(combo0).write_text(json.dumps({
            "combo_idx": 0, "learning_rate": 0.001, "l1_lambda": 1e-5,
            "ensemble_val_loss": 0.01, "tuning_row": {"test_year": 1990,
            "learning_rate": 0.001, "l1_lambda": 1e-5}}))
        # combo 1: train only member 0, then "crash"
        combo1 = main._combo_dir(ckpt, split.test_year, 1)
        member0_path = main._member_checkpoint_path(combo1, 0)
        # train just member 0 by calling the ensemble routine but faking a 1-member run
        saved_ens = args.ensemble_size
        args.ensemble_size = 1
        main._train_ensemble_resumable(args, combo1, object(), object(), 920, 0.001, 1e-4)
        args.ensemble_size = saved_ens
        check("pre-crash trained 3 members total", train_counter["n"] == 3)
        check("combo0 marked done", main._combo_done_path(combo0).exists())
        check("combo1 member0 checkpoint exists", member0_path.exists())
        check("combo1 not marked done", not main._combo_done_path(combo1).exists())

        # --- now resume the full year ---
        train_counter["n"] = 0
        mr3, mh3, best3, tuning3 = main._run_year_resumable(
            args, split, ckpt, input_features=920,
            tune_learning_rates=tune_lrs, tune_l1_lambdas=tune_l1s,
            train_generator=object(), val_generator=object(),
        )
        # combo0 fully cached (0), combo1 member0 cached, only member1 trains -> 1
        check("resume trains ONLY the missing member (1 net)", train_counter["n"] == 1)
        check("resume selects best combo after partial", best3["l1_lambda"] == 1e-5)
        check("resume returns full ensemble", len(mr3) == 2)


if __name__ == "__main__":
    test_config_and_progress()
    test_within_year_resume()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
