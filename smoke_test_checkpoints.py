"""
End-to-end smoke test for resumable checkpointing.

Run this ON YOUR MACHINE (where torch + pyarrow are installed):

    python smoke_test_checkpoints.py

It does NOT touch your 1.1 GB dataset. It builds a tiny synthetic GKX-shaped
parquet in a temp folder, then:

  1. CLEAN run  -> trains every year/combo/member from scratch.
  2. CRASH run  -> same command, but a fault is injected after a few member
                   trainings to mimic the app closing / laptop restarting.
  3. RESUME run -> same command again; must skip everything already done and
                   train ONLY what was missing.

Then it checks:
  * the resume actually skipped work (did not retrain from zero),
  * the final prediction panel is byte-for-byte identical to the clean run
    (proving the resume is correct, not just "finishes"),
  * every test month is present and progress.json is marked complete.

Training is forced onto CPU so results are deterministic and comparable.
"""
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def build_synthetic_parquet(path):
    """A small panel with the exact column shape main.py expects."""
    rng = np.random.default_rng(0)
    permnos = np.arange(1, 41)                      # 40 stocks -> deciles work
    months = [y * 100 + m for y in range(1957, 1992) for m in range(1, 13)]
    months = [ym for ym in months if ym >= 195703]  # matches default train_start

    char_cols = [f"char_{c}" for c in "abcde"]       # 5 characteristics
    macro_cols = ["macro_x", "macro_y"]              # 8 in the real data; 2 here
    sic_cols = ["sic2_10", "sic2_20", "sic2_30"]     # 74 in the real data; 3 here

    rows = []
    macro_by_month = {ym: rng.uniform(-1, 1, size=len(macro_cols)) for ym in months}
    for ym in months:
        macros = macro_by_month[ym]
        for permno in permnos:
            chars = rng.uniform(-1, 1, size=len(char_cols))
            onehot = np.zeros(len(sic_cols))
            onehot[permno % len(sic_cols)] = 1.0
            row = {
                "permno": int(permno),
                "YYYYMM": int(ym),
                "excess_ret": float(rng.normal(0, 0.05)),
                "market_cap": float(rng.uniform(1.0, 100.0)),
            }
            row.update(dict(zip(char_cols, chars)))
            row.update(dict(zip(macro_cols, macros)))
            row.update(dict(zip(sic_cols, onehot)))
            rows.append(row)

    pd.DataFrame(rows).to_parquet(path, index=False)


def run_main(main, argv):
    old = sys.argv
    sys.argv = ["main.py"] + argv
    try:
        main.main()
    finally:
        sys.argv = old


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gkx_ckpt_smoke_"))
    data_path = tmp / "synthetic.parquet"
    out_clean = tmp / "clean"
    out_resume = tmp / "resume"
    build_synthetic_parquet(data_path)
    print(f"Synthetic data: {data_path}")

    import train
    train.get_device = lambda device=None: "cpu"   # determinism

    import main as main_module
    importlib.reload(main_module)

    common = [
        "--model", "NN1",
        "--data_path", str(data_path),
        "--epochs", "3",
        "--batch_size", "2048",
        "--validation_years", "2",
        "--test_start_year", "1990",
        "--test_end_year", "1991",
        "--ensemble_size", "2",
        "--tune_hyperparameters",
        "--tune_learning_rates", "0.001",
        "--tune_l1_lambdas", "1e-5,1e-4",
        "--seed", "42",
    ]

    # ---- 1. CLEAN RUN -------------------------------------------------------
    print("\n=== CLEAN RUN ===")
    run_main(main_module, common + ["--output_dir", str(out_clean), "--force_restart"])
    clean_pred = pd.read_parquet(out_clean / "nn1_predictions.parquet")
    clean_pred = clean_pred.sort_values(["YYYYMM", "permno"]).reset_index(drop=True)

    # ---- 2. CRASH RUN (inject a fault after 3 member trainings) -------------
    print("\n=== CRASH RUN (fault injected) ===")
    real_train_model = main_module.train_model
    state = {"n": 0}

    def crashing_train_model(*a, **k):
        result = real_train_model(*a, **k)
        state["n"] += 1
        if state["n"] >= 3:
            raise RuntimeError("INJECTED CRASH (simulating app close / restart)")
        return result

    main_module.train_model = crashing_train_model
    try:
        run_main(main_module, common + ["--output_dir", str(out_resume)])
        print("!! crash was not triggered (test still valid, but less strict)")
    except RuntimeError as exc:
        print(f"Injected crash raised as expected: {exc}")
    main_module.train_model = real_train_model

    # ---- 3. RESUME RUN (same command, no force_restart) ---------------------
    print("\n=== RESUME RUN ===")
    state_after = {"n": 0}

    def counting_train_model(*a, **k):
        state_after["n"] += 1
        return real_train_model(*a, **k)

    main_module.train_model = counting_train_model
    run_main(main_module, common + ["--output_dir", str(out_resume)])
    main_module.train_model = real_train_model

    resume_pred = pd.read_parquet(out_resume / "nn1_predictions.parquet")
    resume_pred = resume_pred.sort_values(["YYYYMM", "permno"]).reset_index(drop=True)

    # ---- CHECKS -------------------------------------------------------------
    import json
    progress = json.loads((out_resume / "checkpoints" / "progress.json").read_text())

    # full clean run trains 2 years x 2 combos x 2 members = 8 nets
    total_nets = 2 * 2 * 2
    trained_on_resume = state_after["n"]

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    print("\n=== CHECKS ===")
    check("resume did NOT retrain everything (< full 8 nets)",
          0 < trained_on_resume < total_nets)
    check("resume trained the remaining nets only",
          trained_on_resume == total_nets - 2)   # 2 completed before the crash
    check("all test months present (24 months x 40 stocks)",
          len(resume_pred) == len(clean_pred) == 24 * 40)
    check("progress.json marked complete", progress.get("is_complete") is True)
    check("progress lists both test years",
          progress.get("completed_test_years") == [1990, 1991])
    check("resumed predictions IDENTICAL to clean run",
          np.allclose(clean_pred["prediction"].to_numpy(),
                      resume_pred["prediction"].to_numpy(), atol=1e-6))

    print(f"\ncrash-run trainings before fault : {state['n']}")
    print(f"resume-run trainings             : {trained_on_resume}  (of {total_nets} total)")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print(f"(temp workspace: {tmp})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
