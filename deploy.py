import argparse
import os
import shutil
import tempfile

import pandas as pd

from guild import ipy as guild

def main():
    args = _init_args()
    runs = _compare_runs(args)
    _save_runs(runs, args)
    best_run = _find_best_run(runs)
    _deploy(best_run, args)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label")
    p.add_argument("--output", default=".")
    return p.parse_args()

def _compare_runs(args):
    labels = [args.label] if args.label else None
    runs = guild.runs(
        labels=labels,
        operations=["train"],
        completed=True,
        terminated=True)
    if runs.empty:
        match_desc = " match '%s'" % labels[0] if labels else " to compare"
        raise SystemExit("no training runs%s" % match_desc)
    return runs.compare()

def _save_runs(runs, args):
    runs.to_csv(os.path.join(args.output, "runs.csv"))

def _find_best_run(runs):
    best_val = 0.0
    best_run = None
    for row in runs.itertuples():
        if row.roc_auc > best_val:
            best_val = row.roc_auc
            best_run = row.run.run
    assert best_run, runs
    print("Best run is %s (roc_auc=%s)" % (best_run.id, best_val))
    return best_run

def _deploy(run, args):
    output = os.path.abspath(args.output)
    print("Deploying %s to %s" % (run.id, output))
    shutil.copytree(run.path, os.path.join(output, run.id))
    os.symlink(run.id, os.path.join(output, "best-run"))

if __name__ == "__main__":
    main()
