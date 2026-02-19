"""
Three-phase YOLO training for object detection.

Phase 1 – freeze backbone (first N layers), train detection head only.
Phase 2 – unfreeze neck + head.
Phase 3 – full fine-tune.

Each phase starts from the best checkpoint of the previous phase.

Usage
-----
    # Run all three phases
    python train_yolo.py --config ../configs/config.yaml

    # Run a subset of phases (e.g. resume from phase 2)
    python train_yolo.py --phases 2 3 --start-weights ./yolo_runs/phase1_head/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_phase(
    model_path: str,
    data_yaml: str,
    phase_cfg: dict,
    training_cfg: dict,
    aug_cfg: dict,
    yolo_cfg: dict,
    project_dir: str,
) -> Path:
    """
    Execute one YOLO training phase.

    Parameters
    ----------
    model_path  : Path to weights file (.pt) or model name (e.g. 'yolov8n.pt').
    data_yaml   : Path to dataset.yaml produced by prepare_data.py.
    phase_cfg   : Phase-specific config  {name, epochs, lr0, freeze}.
    training_cfg: Shared config          {batch_size, patience, workers}.
    aug_cfg     : Augmentation config    {fliplr, degrees, scale, ...}.
    yolo_cfg    : Top-level YOLO config  {imgsz, ...}.
    project_dir : Base directory for run artefacts.

    Returns
    -------
    Path to best.pt from this phase.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.  Run: pip install ultralytics")
        sys.exit(1)

    name = phase_cfg["name"]
    print(f"\n{'=' * 60}")
    print(f"  Phase : {name}")
    print(f"  Weights : {model_path}")
    print(f"  Epochs  : {phase_cfg['epochs']}  LR: {phase_cfg['lr0']}  "
          f"Freeze: {phase_cfg['freeze']}")
    print(f"{'=' * 60}\n")

    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=phase_cfg["epochs"],
        imgsz=yolo_cfg["imgsz"],
        batch=training_cfg["batch_size"],
        lr0=phase_cfg["lr0"],
        lrf=0.01,
        freeze=phase_cfg["freeze"],
        patience=training_cfg["patience"],
        workers=training_cfg.get("workers", 4),
        project=project_dir,
        name=name,
        exist_ok=True,
        # Augmentation — conservative to preserve coordinate quality
        fliplr=aug_cfg.get("fliplr", 0.5),
        degrees=aug_cfg.get("degrees", 12.0),
        scale=aug_cfg.get("scale", 0.2),
        translate=aug_cfg.get("translate", 0.1),
        hsv_h=aug_cfg.get("hsv_h", 0.015),
        hsv_s=aug_cfg.get("hsv_s", 0.3),
        hsv_v=aug_cfg.get("hsv_v", 0.2),
        perspective=aug_cfg.get("perspective", 0.0),
        mosaic=aug_cfg.get("mosaic", 0.0),
        # General
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    best = Path(project_dir) / name / "weights" / "best.pt"
    if not best.exists():
        best = Path(project_dir) / name / "weights" / "last.pt"

    print(f"\nPhase '{name}' complete.  Best weights: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Three-phase YOLO training")
    _default_cfg = str(Path(__file__).parent.parent / "configs" / "config.yaml")
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument(
        "--phases", nargs="+", default=["1", "2", "3"],
        choices=["1", "2", "3"],
        help="Which phases to run (default: all three)",
    )
    parser.add_argument(
        "--start-weights", default=None,
        help="Override starting weights for the first requested phase.",
    )
    args = parser.parse_args()

    cfg          = load_config(args.config)
    yolo_cfg     = cfg["yolo"]
    training_cfg = yolo_cfg["training"]
    aug_cfg      = yolo_cfg["augmentation"]
    data_cfg     = cfg["data"]

    data_yaml   = str(Path(data_cfg["yolo_data_dir"]) / "dataset.yaml")
    project_dir = yolo_cfg["project_dir"]

    phase_map = {
        "1": training_cfg["phase1"],
        "2": training_cfg["phase2"],
        "3": training_cfg["phase3"],
    }

    current_weights = args.start_weights or yolo_cfg["model_weights"]

    for key in args.phases:
        current_weights = run_phase(
            model_path=str(current_weights),
            data_yaml=data_yaml,
            phase_cfg=phase_map[key],
            training_cfg=training_cfg,
            aug_cfg=aug_cfg,
            yolo_cfg=yolo_cfg,
            project_dir=project_dir,
        )

    print(f"\nAll phases complete.  Final weights: {current_weights}")


if __name__ == "__main__":
    main()
