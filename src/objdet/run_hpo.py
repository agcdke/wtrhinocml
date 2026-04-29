"""
Hyperparameter Optimization for YOLO26 WildTrack Rhino Spoor Detection.
Uses ClearML HyperParameterOptimizer with NVIDIA CUDA GPUs.

Prerequisites:
    1. Run train_yolo_rhino_objdet.py at least ONCE so the base task exists in ClearML.
    2. Install: pip install clearml optuna

Usage:
    # Auto-resolve the most recent training task as the base:
    python run_hpo.py --config hpo_config.yaml

    # Specify a known base task ID:
    python ./src/objdet/run_hpo.py --config ./src/config/hpo_config.yaml --base-task-id <TASK_ID>

How it works:
    1. This script creates a ClearML "optimizer" task.
    2. It clones the base training task with different hyperparameters.
    3. Each clone is enqueued to the execution queue (where ClearML agents run).
    4. Results are collected, and the best hyperparameters are reported.

    The parameter names (e.g., "General/lr0") must match exactly what
    train_yolo_objdet.py connects via task.connect(cfg_dict, name="General").
"""
import argparse
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict

from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    UniformParameterRange,
    UniformIntegerParameterRange,
    DiscreteParameterRange,
    LogUniformParameterRange,
)

from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.hpbandster import OptimizerBOHB
from clearml.automation import RandomSearch
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.hpbandster import OptimizerBOHB
from clearml.automation import GridSearch
from clearml.automation import RandomSearch

# ── Try to import Optuna, fall back gracefully ──
try:
    DEFAULT_STRATEGY = OptimizerOptuna
except ImportError:
    try:
        DEFAULT_STRATEGY = OptimizerBOHB
    except ImportError:
        DEFAULT_STRATEGY = RandomSearch
        logging.warning("Install optuna for better HPO: pip install optuna")


# ══════════════════════════════════════════════════════════════
#  HPO Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class HPOConfig:
    """All HPO settings — loaded from YAML, no hard-coded values."""

    # ── Base task identification ──
    base_task_project:          str   = "rhino-detection"
    base_task_name:             str   = "rhino-yolo26m-training-augmented"
    base_task_id:               str   = ""
    repository:                 str   = ""
    branch:                     str   = ""
    entry_point:                str   = ""
    working_dir:                str   = "."

    # ── HPO task metadata ──
    hpo_project_name:           str   = "HPO-rhino-detection"
    hpo_task_name:              str   = "HPO-yolo26m-wildtrack-rhino"

    # ── Objective metric ──
    objective_metric_title:     str   = "test"
    objective_metric_series:    str   = "mAP50"
    objective_metric_sign:      str   = "max"

    # ── Execution ──
    execution_queue:            str   = "default"
    max_iter_per_job:           int   = 100
    min_iter_per_job:           int   = 10
    max_concurrent_tasks:       int   = 2
    total_max_jobs:             int   = 30
    time_limit_per_job_min:     float = 120.0
    pool_period_min:            float = 1.0

    # ── Search strategy: "optuna", "bohb", "random", "grid" ──
    search_strategy:            str   = "optuna"

    # ── Top experiments to report ──
    top_k:                      int   = 3

    # ── Search spaces ──
    # type: "uniform", "log_uniform", "uniform_int", "discrete"
    # name uses ClearML section/key: "General/lr0"
    search_space: list = field(default_factory=lambda: [
        {"name": "General/lr0",           "type": "log_uniform", "min": 0.0001,  "max": 0.01},
        {"name": "General/lrf",           "type": "uniform",     "min": 0.001,   "max": 0.1,    "step": 0.005},
        {"name": "General/weight_decay",  "type": "log_uniform", "min": 0.0001,  "max": 0.005},
        {"name": "General/warmup_epochs", "type": "uniform",     "min": 1.0,     "max": 5.0,    "step": 0.5},
        {"name": "General/batch",         "type": "discrete",    "values": [8, 16, 32]},
        {"name": "General/imgsz",         "type": "discrete",    "values": [480, 640, 800]},
        {"name": "General/mosaic",        "type": "uniform",     "min": 0.0,     "max": 1.0,    "step": 0.1},
        {"name": "General/mixup",         "type": "uniform",     "min": 0.0,     "max": 0.3,    "step": 0.05},
        {"name": "General/copy_paste",    "type": "uniform",     "min": 0.0,     "max": 0.3,    "step": 0.05},
        {"name": "General/dropout",       "type": "uniform",     "min": 0.0,     "max": 0.3,    "step": 0.05},
        {"name": "General/cls_pw",        "type": "uniform",     "min": 0.0,     "max": 1.0,    "step": 0.1},
        {"name": "General/degrees",       "type": "uniform",     "min": 0.0,     "max": 20.0,   "step": 2.0},
        {"name": "General/translate",     "type": "uniform",     "min": 0.0,     "max": 0.3,    "step": 0.05},
        {"name": "General/scale",         "type": "uniform",     "min": 0.2,     "max": 0.8,    "step": 0.1},
    ])


def load_hpo_config(yaml_path: str) -> HPOConfig:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"HPO config not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    fields = {f.name for f in HPOConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in fields}
    return HPOConfig(**kwargs)


# ══════════════════════════════════════════════════════════════
#  Build ClearML Parameter Ranges
# ══════════════════════════════════════════════════════════════

def build_search_space(search_space_list):
    ranges = []
    for entry in search_space_list:
        name  = entry["name"]
        ptype = entry["type"]

        if ptype == "uniform":
            ranges.append(UniformParameterRange(
                name, min_value=entry["min"], max_value=entry["max"],
                step_size=entry.get("step", 0.01)))

        elif ptype == "log_uniform":
            ranges.append(LogUniformParameterRange(
                name, min_value=entry["min"], max_value=entry["max"]))

        elif ptype == "uniform_int":
            ranges.append(UniformIntegerParameterRange(
                name, min_value=int(entry["min"]), max_value=int(entry["max"]),
                step_size=int(entry.get("step", 1))))

        elif ptype == "discrete":
            ranges.append(DiscreteParameterRange(name, values=entry["values"]))

        else:
            print(f"[HPO WARN] Unknown type '{ptype}' for {name}")
            continue

        print(f"  {name:30s}  {ptype:12s}  "
              f"{entry.get('min', entry.get('values', ''))}"
              f"{'–' + str(entry.get('max', '')) if 'max' in entry else ''}")

    return ranges


# ══════════════════════════════════════════════════════════════
#  Resolve Base Task
# ══════════════════════════════════════════════════════════════

def resolve_base_task_id(cfg: HPOConfig, cli_task_id: str = None) -> str:
    if cli_task_id:
        return cli_task_id
    if cfg.base_task_id:
        return cfg.base_task_id

    print(f"[HPO] Resolving base task: project='{cfg.base_task_project}' "
          f"name='{cfg.base_task_name}'")

    tasks = Task.get_tasks(
        project_name=cfg.base_task_project,
        task_name=cfg.base_task_name,
        task_filter={"status": ["completed", "published"]},
    )
    if not tasks:
        tasks = Task.get_tasks(
            project_name=cfg.base_task_project,
            task_name=cfg.base_task_name,
        )
    if not tasks:
        raise RuntimeError(
            f"No task found: project='{cfg.base_task_project}' "
            f"name='{cfg.base_task_name}'.\n"
            f"Run train_yolo_objdet.py first, or pass --base-task-id.")

    base = sorted(tasks, key=lambda t: t.data.created or "", reverse=True)[0]
    print(f"[HPO] Found base task: {base.id} (status={base.status})")
    return base.id


# ══════════════════════════════════════════════════════════════
#  Search Strategy Selector
# ══════════════════════════════════════════════════════════════

def get_search_strategy(name: str):
    name = name.lower()
    if name == "optuna":
        try:
            return OptimizerOptuna
        except ImportError:
            pass
    elif name == "bohb":
        try:
            return OptimizerBOHB
        except ImportError:
            pass
    elif name == "grid":
        
        return GridSearch
    return RandomSearch


# ══════════════════════════════════════════════════════════════
#  Callback
# ══════════════════════════════════════════════════════════════

def on_job_complete(job_id, objective_value, objective_iteration,
                    job_parameters, top_performance_job_id):
    print(f"\n[HPO] Trial done: {job_id}  objective={objective_value:.4f}  "
          f"best={top_performance_job_id}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HPO for YOLO26 — NVIDIA CUDA + ClearML + Optuna")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to HPO YAML config")
    parser.add_argument("--base-task-id", type=str, default=None,
                        help="ClearML base training task ID (overrides config)")
    args = parser.parse_args()

    cfg = load_hpo_config(args.config)

    # ── Create HPO controller task ──
    task = Task.init(
        project_name=cfg.hpo_project_name,
        task_name=cfg.hpo_task_name,
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )
    task.set_script(
        repository=cfg.repository,
        branch=cfg.branch,
        entry_point = cfg.entry_point,
        working_dir = cfg.working_dir,
    )
    hpo_params = {k: v for k, v in asdict(cfg).items() if k != "search_space"}
    task.connect(hpo_params, name="HPO")

    # ── Resolve base task ──
    base_task_id = resolve_base_task_id(cfg, args.base_task_id)

    # ── Build search space ──
    print(f"\n[HPO] Search space ({len(cfg.search_space)} parameters):")
    hyper_parameters = build_search_space(cfg.search_space)
    if not hyper_parameters:
        raise ValueError("No valid parameters in search_space!")

    # ── Strategy ──
    strategy_cls = get_search_strategy(cfg.search_strategy)
    print(f"\n[HPO] Strategy: {strategy_cls.__name__}")

    # ── Create optimizer ──
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=hyper_parameters,
        objective_metric_title=cfg.objective_metric_title,
        objective_metric_series=cfg.objective_metric_series,
        objective_metric_sign=cfg.objective_metric_sign,
        optimizer_class=strategy_cls,
        max_iteration_per_job=cfg.max_iter_per_job,
        min_iteration_per_job=cfg.min_iter_per_job,
        execution_queue=cfg.execution_queue,
        max_number_of_concurrent_tasks=cfg.max_concurrent_tasks,
        total_max_jobs=cfg.total_max_jobs,
        time_limit_per_job=cfg.time_limit_per_job_min,
        pool_period_min=cfg.pool_period_min,
    )

    print(f"\n{'=' * 64}")
    print(f"  HPO: {cfg.objective_metric_sign} {cfg.objective_metric_title}")
    print(f"  Base task  : {base_task_id}")
    print(f"  Strategy   : {strategy_cls.__name__}")
    print(f"  Max Iteration per Job      : {cfg.max_iter_per_job}")
    print(f"  Min Iteration per Job      : {cfg.min_iter_per_job}")
    print(f"  Trials     : {cfg.total_max_jobs} (max {cfg.max_concurrent_tasks} concurrent)")
    print(f"  Queue      : {cfg.execution_queue}")
    print(f"{'=' * 64}\n")

    # ── Run ──
    optimizer.set_report_period(cfg.pool_period_min)
    optimizer.start(job_complete_callback=on_job_complete)
    print(f"[HPO] Running — monitor at: {task.get_output_log_web_page()}\n")
    optimizer.wait()

    # ── Report top experiments ──
    print(f"\n{'=' * 64}")
    print(f"  HPO COMPLETE — TOP {cfg.top_k}")
    print(f"{'=' * 64}")

    top_exp = optimizer.get_top_experiments(top_k=cfg.top_k)
    tuned_keys = [s["name"] for s in cfg.search_space]

    for i, exp in enumerate(top_exp):
        print(f"\n  #{i+1}  {exp.id}  (status={exp.status})")
        metrics = exp.get_last_scalar_metrics()
        title, series = cfg.objective_metric_title, cfg.objective_metric_series
        if title in metrics and series in metrics[title]:
            print(f"       {title}: {metrics[title][series].get('last', 'N/A')}")
        params = exp.get_parameters()
        for k in tuned_keys:
            if k in params:
                print(f"       {k:30s}: {params[k]}")

    # ── Save best as artifact ──
    if top_exp:
        best = top_exp[0]
        best_params = best.get_parameters()
        best_tuned = {s["name"]: best_params.get(s["name"], "N/A") for s in cfg.search_space}
        best_tuned["_task_id"] = best.id
        task.upload_artifact(name="best_hyperparameters", artifact_object=best_tuned)
        print(f"\n[HPO] Best task: {best.id}")

    optimizer.stop()
    task.close()
    print(f"\n[DONE] {task.get_output_log_web_page()}")


if __name__ == "__main__":
    main()