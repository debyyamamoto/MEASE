"""Frozen, resumable repetitions of candidates selected from a shell Random Search."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_random_search import CONTEXT, METRICS, RECORDED, parse_config, rank_panel, read_results

PARAMETERS = ("score_metric threshold generations population restart_gen restart_pop "
              "restart_pct alpha redundancy_penalty redundancy_similarity_threshold "
              "ksize min_rule_support max_rule_support").split()
DEFAULT_OUTPUT = "experiments/repeated_random_search"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


@contextmanager
def lock(directory):
    path = directory / ".repetitions.lock"
    try:
        path.mkdir()
    except FileExistsError:
        raise ValueError(f"Experiment locked: {path}. Remove only after checking no process is running.")
    try:
        yield
    finally:
        path.rmdir()


def read_source_protocol(source):
    fields, datasets, hashes = {}, [], {}
    for line in (source / "protocol.txt").read_text().splitlines():
        if line.startswith("dataset="):
            name, path, time_col, event_col = line[8:].split("|")
            datasets.append(dict(name=name, path=path, time_col=time_col, event_col=event_col))
        elif len(line) > 66 and line[64:66] == "  ":
            hashes[line[66:]] = line[:64]
        else:
            key, value = line.split("=", 1)
            fields[key] = value
    if fields.get("protocol") != "mease-random-search-shell-v1" or not datasets:
        raise ValueError("Expected a completed article/random_search.sh protocol")
    required = {"main.py", "uv.lock", "pyproject.toml", *[str(p.relative_to(ROOT)) for p in (ROOT / "easd").glob("*.py")],
                *[d["path"] for d in datasets]}
    if not required.issubset(hashes):
        raise ValueError("Incomplete source code/data fingerprints")
    for name, digest in hashes.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Code/data differ from Random Search: {name}. Use matching versions; do not mix implementations.")
    return fields, datasets, hashes


def verify_shell_run(folder):
    """Check all original outputs and command, not just three Info.csv fields."""
    marker = folder / ".complete"
    expected_names = {"command.sh"}
    expected_names.update(p.name for p in folder.glob("*_0_*_*.csv") if any(
        p.name.endswith(f"_{suffix}.csv") for suffix in ("Info", "DetailedRules", "RulesMetricsResult")))
    if len(expected_names) != 4:
        raise ValueError(f"Incomplete Random Search run: {folder}")
    recorded = {}
    for line in marker.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name:
            raise ValueError(f"Unexpected checksum path: {name}")
        recorded[name] = digest
    if set(recorded) != expected_names or any(sha256(folder / n) != h for n, h in recorded.items()):
        raise ValueError(f"Invalid Random Search completion checksums: {folder}")


def prepare_plan(args):
    source = args.source.resolve()
    fields, datasets, original_hashes = read_source_protocol(source)
    configs, runs = read_results(source, fields["baseline"])
    if len(configs) != int(fields["num_configs"]):
        raise ValueError("Configuration count differs from source protocol")
    expected = pd.MultiIndex.from_product([range(len(configs)), [d["name"] for d in datasets], [0]],
                                         names=["config_id", "dataset", "run"])
    actual = pd.MultiIndex.from_frame(runs[["config_id", "dataset", "run"]])
    if len(actual) != len(expected) or len(expected.difference(actual)) or not runs.metadata_matches.all():
        raise ValueError("Random Search must have a complete, consistent panel of all configurations and datasets")
    context = dict(archive_selection=fields["archive_selection"], rate_policy=fields["rate_policy"],
                   km_time_bins=int(fields["km_time_bins"]))
    for key, value in context.items():
        if not runs[key].eq(value).all():
            raise ValueError(f"Inconsistent source context: {key}")
    for path in runs.source:
        folder = Path(path).parent
        verify_shell_run(folder)
        cid = int(folder.parents[1].name.removeprefix("config_"))
        dataset = next(d for d in datasets if d["name"] == folder.parent.name)
        # Check the actual saved invocation against ALL parameters, not only Info.csv.
        lines = (folder / "command.sh").read_text().splitlines()
        tokens = shlex.split(lines[-1])
        expected_flags = {**configs[cid], **context, "seed": fields["run_seed"], "executions": "1",
                          "comparacao": fields["baseline"], "dataset_name": dataset["name"],
                          "time_col": dataset["time_col"], "event_col": dataset["event_col"]}
        for key, value in expected_flags.items():
            flag = "--" + key
            if tokens.count(flag) != 1 or tokens.index(flag) == len(tokens) - 1:
                raise ValueError(f"Missing/duplicate argument in source invocation: {folder}: {flag}")
            actual_value = tokens[tokens.index(flag) + 1]
            try:
                agrees = float(actual_value) == float(value)
            except (ValueError, TypeError):
                agrees = actual_value == str(value)
            if not agrees:
                raise ValueError(f"Source invocation differs from manifest: {folder}: {flag}")
    cells = runs.groupby(["config_id", "dataset"], as_index=False)[[*METRICS, "#sg"]].mean()
    _, ranking = rank_panel(cells)
    if args.config_ids is not None:
        ids = args.config_ids
        if len(set(ids)) != len(ids) or not set(ids).issubset(range(len(configs))):
            raise ValueError("--config-ids must be distinct IDs from the source manifest")
    else:
        if not 3 <= args.top_k <= len(configs):
            raise ValueError("--top-k must be between 3 and the number of source configurations")
        ids = ranking.config_id.head(args.top_k).astype(int).tolist()
    if len(ids) < 3:
        raise ValueError("Friedman protocol requires at least three configurations")
    seeds = list(range(args.seed_start, args.seed_start + args.repetitions))
    if args.repetitions < 2 or args.seed_start < 0 or seeds[-1] >= 2**32:
        raise ValueError("Use r >= 2 and seeds in [0, 2**32-1]")
    if int(fields["run_seed"]) in seeds:
        raise ValueError("Repetition seeds must exclude the seed used to select candidates")
    selected = {str(cid): configs[cid] for cid in ids}
    if any(set(c) != set(PARAMETERS) for c in selected.values()):
        raise ValueError("Unexpected configuration parameters")
    snapshots = [source / "protocol.txt", source / "configs.txt", source / "configs.tsv"]
    snapshots += [Path(p) for p in runs.source]
    source_hashes = {str(p.relative_to(source)): sha256(p) for p in snapshots}
    # Both entrypoints, planner and selection policy are part of the execution protocol.
    code = ["experiments/repeat_random_search.py", "experiments/analyze_random_search.py",
            "experiments/article/repeat_random_search.sh"]
    hashes = {**original_hashes, **{name: sha256(ROOT / name) for name in code}}
    jobs = [[cid, d["name"], seed] for seed in seeds for d in datasets for cid in ids]
    random.Random(args.order_seed).shuffle(jobs)
    return dict(protocol="mease-repetitions-v1", source=str(source), source_hashes=source_hashes,
                hashes=hashes, baseline=fields["baseline"], fixed=context, plot_rank=args.plot_rank,
                selection="explicit IDs" if args.config_ids else "equal metric/dataset mean ranks; ties ordered by ID",
                configs=selected, config_ids=ids, datasets=datasets, seeds=seeds,
                repetitions=args.repetitions, order_seed=args.order_seed, jobs=jobs,
                python_hash_seed="0", environment=dict(python=platform.python_version(), platform=platform.platform())), ranking


def command_for(plan, output, cid, dataset, seed):
    spec = next(d for d in plan["datasets"] if d["name"] == dataset)
    base = output / f"config_{cid}" / f"seed_{seed}"
    command = [sys.executable, str(ROOT / "main.py"), str(ROOT / spec["path"]),
               "--dataset_name", dataset, "--time_col", spec["time_col"], "--event_col", spec["event_col"],
               "--output_dir", str(base), "--seed", str(seed), "--executions", "1",
               "--comparacao", plan["baseline"], "--plt_rank", str(plan["plot_rank"]), "--debug_performance", "off"]
    for key, value in {**plan["fixed"], **plan["configs"][str(cid)]}.items():
        command.extend(["--" + key, str(value)])
    folder = base / dataset / plan["baseline"]
    return command, folder


def outputs_for(folder, dataset, baseline):
    prefix = f"{dataset}_0_{baseline}_"
    return [folder / (prefix + suffix + ".csv") for suffix in ("Info", "RulesMetricsResult", "DetailedRules")]


def validate_metrics(values):
    array = np.array([values[key] for key in METRICS], dtype=float)
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError("Metrics must be finite and in [0, 1]")
    if not np.isfinite(values["#sg"]) or values["#sg"] < 0:
        raise ValueError("Invalid rule count")


def precise_metrics(plan, folder, dataset, cache):
    """Re-evaluate exported rules to avoid the CLI's two-decimal metric rounding."""
    from easd.dataset import Dataset
    from easd.metrics import compute_run_metrics, covered_indices
    spec = next(d for d in plan["datasets"] if d["name"] == dataset)
    if dataset not in cache:
        cache[dataset] = Dataset(pd.read_parquet(ROOT / spec["path"]), spec["time_col"], spec["event_col"])
    obj = cache[dataset]
    info_path, saved_path, detail_path = outputs_for(folder, dataset, plan["baseline"])
    detailed = pd.read_csv(detail_path)
    rules = [parse_config("{'rule': " + text + "}")["rule"] for text in detailed.Rule_Obj]
    # compute_run_metrics catches coverage errors internally; reject malformed exported rules explicitly.
    for rule in rules:
        covered_indices(rule, obj.get_data(), dataset_obj=obj)
    values = compute_run_metrics(obj.get_data(), rules, spec["time_col"], spec["event_col"],
                                 dataset_obj=obj, baseline=plan["baseline"]).as_dict()
    saved, info = pd.read_csv(saved_path), pd.read_csv(info_path)
    if len(saved) != 1 or len(info) != 1:
        raise ValueError(f"Expected one run per CSV: {folder}")
    validate_metrics(values)
    validate_metrics(saved.iloc[0])
    for key in [*METRICS, "#sg"]:
        if abs(values[key] - saved.iloc[0][key]) > .00501:
            raise ValueError(f"Recomputed metric disagrees with CLI: {folder}: {key}")
    return values


def completed(folder, required):
    marker = folder / ".complete.json"
    if not marker.exists():
        return False
    try:
        hashes = json.loads(marker.read_text())
        return set(hashes) == {p.name for p in required} and all(
            p.is_file() and sha256(p) == hashes[p.name] for p in required)
    except (ValueError, OSError):
        return False


def run_plan(plan, output, dry_run=False, max_runs=None):
    cache, executed = {}, 0
    for cid, dataset, seed in plan["jobs"]:
        command, folder = command_for(plan, output, cid, dataset, seed)
        if dry_run:
            print("PYTHONHASHSEED=0 " + shlex.join(command))
            continue
        folder.mkdir(parents=True, exist_ok=True)
        command_text = "#!/usr/bin/env bash\nset -euo pipefail\ncd -- " + shlex.quote(str(ROOT)) + "\n"
        command_text += "export PYTHONHASHSEED=0\n" + shlex.join(command) + "\n"
        command_file = folder / "command.sh"
        if command_file.exists() and command_file.read_text() != command_text:
            raise ValueError(f"Changed run command: {folder}")
        command_file.write_text(command_text)
        required = [command_file, *outputs_for(folder, dataset, plan["baseline"]), folder / "metrics_full.json"]
        if completed(folder, required):
            continue
        if max_runs is not None and executed >= max_runs:
            print(f"Stopped after {executed} new runs (--max-runs); repeat the command to resume.")
            break
        # Do not leave a previous completion marker around after a failed retry.
        (folder / ".complete.json").unlink(missing_ok=True)
        print(f"Running config_{cid} / {dataset} / seed {seed}", flush=True)
        with (folder / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONHASHSEED": "0"})
        if result.returncode:
            raise RuntimeError(f"MEASE exited {result.returncode}; see {folder / 'run.log'}")
        for path in required[1:-1]:
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing output: {path}")
        info = pd.read_csv(required[1]).iloc[0]
        for key in RECORDED + CONTEXT:
            expected = {**plan["fixed"], **plan["configs"][str(cid)]}[key]
            if info[key] != expected:
                raise ValueError(f"Info.csv disagrees with frozen parameters: {key}")
        write_json(folder / "metrics_full.json", precise_metrics(plan, folder, dataset, cache))
        write_json(folder / ".complete.json", {p.name: sha256(p) for p in required})
        executed += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/random_search_shell"))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top-k", type=int, default=10)
    group.add_argument("--config-ids", type=int, nargs="+")
    parser.add_argument("--repetitions", "-r", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=2026)
    parser.add_argument("--plot-rank", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, help="Stop after this many new runs; useful for a pilot/resume")
    args = parser.parse_args()
    if args.plot_rank < 0 or (args.max_runs is not None and args.max_runs < 1):
        parser.error("--plot-rank >= 0 and --max-runs >= 1 required")
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with lock(output):
        plan, ranking = prepare_plan(args)
        manifest = output / "plan.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != plan:
                raise ValueError("Frozen plan differs (selection, parameters, seeds, code, data or environment). Use a new directory.")
        else:
            if any(p.name != ".repetitions.lock" for p in output.iterdir()):
                raise ValueError("Nonempty directory without plan.json; choose a new output directory")
            write_json(manifest, plan)
        ranking.to_csv(output / "selection_ranking.csv", index=False)
        selected = ranking.set_index("config_id").loc[plan["config_ids"]].reset_index()
        selected.merge(pd.DataFrame([dict(config_id=int(k), **v) for k, v in plan["configs"].items()]),
                       on="config_id").to_csv(output / "selected_configs.csv", index=False)
        print(f"Configs {plan['config_ids']}; {len(plan['datasets'])} datasets; r={plan['repetitions']}; "
              f"{len(plan['jobs'])} runs; seeds {plan['seeds'][0]}..{plan['seeds'][-1]}", flush=True)
        run_plan(plan, output, args.dry_run, args.max_runs)
    print(f"Experiment: {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
