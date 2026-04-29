"""
YOLO Object Detection with ClearML and Albumentations.
Trains YOLO26 on the WildTrack Rhino YOLO dataset using NVIDIA CUDA GPUs.

All hyperparameters are loaded from YAML config.

Run:
    python ./src/objdet/train_yolo_rhino_objdet.py --config ./src/config/objdet_rhino_config.yaml
"""
import os
import sys
import argparse
import atexit
import gc
import random
import warnings

import cv2
import torch
import numpy as np
import yaml
import psutil
from pathlib import Path
from dataclasses import dataclass, field, asdict

from clearml import Task, Dataset, OutputModel, Logger
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import albumentations as A
import ultralytics.data.dataset as ds_mod
import ultralytics.data.build   as bld_mod

# ══════════════════════════════════════════════════════════════
#  Configuration Dataclasses
# ══════════════════════════════════════════════════════════════

@dataclass
class AugConfig:
    """Albumentations augmentation pipeline configuration."""
    pipeline_p:         float = 0.8

    enable_flip:        bool  = True
    flip_lr_p:          float = 0.5
    flip_ud_p:          float = 0.0

    enable_rotate:      bool  = True
    rotate_limit:       float = 15.0
    rotate_p:           float = 0.4

    enable_affine:      bool  = True
    shear_limit:        float = 5.0
    affine_p:           float = 0.3

    enable_perspective: bool  = True
    perspective_scale:  float = 0.05
    perspective_p:      float = 0.3

    enable_crop:        bool  = True
    random_crop_p:      float = 0.2
    crop_min_area:      float = 0.7

    enable_distortion:  bool  = True
    elastic_p:          float = 0.1
    grid_distort_p:     float = 0.1

    enable_color_jitter:bool  = True
    brightness_limit:   float = 0.3
    contrast_limit:     float = 0.3
    color_jitter_p:     float = 0.5

    enable_hsv:         bool  = True
    hue_shift_limit:    int   = 15
    sat_shift_limit:    int   = 30
    val_shift_limit:    int   = 20
    hsv_p:              float = 0.4

    enable_grayscale:   bool  = True
    grayscale_p:        float = 0.05

    enable_blur:        bool  = True
    gaussian_blur_p:    float = 0.2
    motion_blur_p:      float = 0.15
    median_blur_p:      float = 0.1

    enable_noise:       bool  = True
    gaussian_noise_p:   float = 0.2
    iso_noise_p:        float = 0.1

    enable_sharpen:     bool  = True
    sharpen_p:          float = 0.2

    enable_clahe:       bool  = True
    clahe_p:            float = 0.2

    enable_dropout:     bool  = True
    coarse_dropout_p:   float = 0.15
    dropout_max_holes:  int   = 8
    dropout_max_height: int   = 32
    dropout_max_width:  int   = 32

    enable_shadow:      bool  = True
    random_shadow_p:    float = 0.3

    enable_fog:         bool  = True
    random_fog_p:       float = 0.05

    enable_rain:        bool  = True
    random_rain_p:      float = 0.05

    enable_sunflare:    bool  = True
    sun_flare_p:        float = 0.05


@dataclass
class DatasetConfig:
    """Dataset identity and versioning configuration."""
    name:               str   = "WildTrack-Rhino-YOLO"
    description:        str   = "WildTrack Rhino YOLO (train/val/test)"
    classes:            list  = field(default_factory=lambda: [
        "buffalo", "elephant", "rhino", "zebra",
    ])
    register_enabled:   bool  = True


@dataclass
class CUDAConfig:
    """NVIDIA CUDA GPU configuration."""
    # GPU device IDs: "0" single GPU, "0,1" multi-GPU, "cpu" for CPU-only
    visible_devices:   str   = "0"
    # Device passed to YOLO .train() and .val()
    device:            str   = "0"
    # Automatic mixed precision (FP16) — requires NVIDIA GPU with Tensor Cores
    amp:               bool  = True
    # cuDNN benchmark mode — faster convolutions when input sizes are constant
    cudnn_benchmark:   bool  = True
    # cuDNN deterministic mode — reproducible results (slightly slower)
    cudnn_deterministic: bool = False
    # TF32 on Ampere+ GPUs (A100, RTX 30xx, RTX 40xx) — faster matmuls
    allow_tf32:        bool  = True


@dataclass
class TrainConfig:
    """Complete training configuration — every value comes from YAML."""
    # ── Model ──
    model_variant:     str   = "yolo26m.pt"
    pretrained:        bool  = True
    task:              str = "detect"

    # ── Data ──
    data_yaml:         str   = ""
    imgsz:             int   = 640

    # ── Training schedule ──
    epochs:            int   = 100
    batch:             int   = 16
    workers:           int   = 8
    patience:          int   = 50
    save_period:       int   = 10

    # ── Optimizer ──
    optimizer:         str   = "AdamW"
    lr0:               float = 0.001
    lrf:               float = 0.01
    momentum:          float = 0.937
    weight_decay:      float = 0.0005
    warmup_epochs:     float = 3.0
    warmup_bias_lr:    float = 0.1

    # ── YOLO built-in augmentation ──
    hsv_h:             float = 0.015
    hsv_s:             float = 0.7
    hsv_v:             float = 0.4
    degrees:           float = 0.0
    translate:         float = 0.1
    scale:             float = 0.5
    shear:             float = 0.0
    flipud:            float = 0.0
    fliplr:            float = 0.0
    mosaic:            float = 1.0
    mixup:             float = 0.1
    copy_paste:        float = 0.1
    close_mosaic:      int   = 10

    # ── Regularization & imbalance ──
    multi_scale:       bool  = True
    dropout:           float = 0.0
    cls_pw:            float = 0.0

    # ── Validation ──
    val:               bool  = True
    test_split:        str   = "test"

    # ── ClearML project ──
    project_name:      str   = "rhino-detection"
    task_name:         str   = "rhino-yolo26m-training-augmented"
    output_dir:        str   = "runs/detect"

    # ── Preview grid settings ──
    preview_samples:   int   = 4
    preview_augs:      int   = 4
    preview_thumb:     int   = 200

    # ── Nested configs ──
    aug:     AugConfig     = field(default_factory=AugConfig)
    cuda:    CUDAConfig    = field(default_factory=CUDAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# ══════════════════════════════════════════════════════════════
#  NVIDIA CUDA Setup
# ══════════════════════════════════════════════════════════════

def setup_cuda(cuda_cfg: CUDAConfig):
    """
    Configure NVIDIA CUDA environment before any torch import.
    Must be called before model creation or training.
    """

    # Set visible GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_cfg.visible_devices
    print(f"[CUDA] CUDA_VISIBLE_DEVICES = {cuda_cfg.visible_devices}")

    if not torch.cuda.is_available():
        print("[CUDA] WARNING: No NVIDIA GPU detected — falling back to CPU")
        print("[CUDA]   Check: nvidia-smi, CUDA toolkit, PyTorch CUDA build")
        return

    # Report GPU info
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / (1024 ** 3)
        print(f"[CUDA] GPU {i}: {props.name}  |  {vram:.1f} GB VRAM  |  "
              f"Compute {props.major}.{props.minor}")

    # cuDNN settings
    torch.backends.cudnn.benchmark     = cuda_cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cuda_cfg.cudnn_deterministic
    print(f"[CUDA] cuDNN benchmark={cuda_cfg.cudnn_benchmark}  "
          f"deterministic={cuda_cfg.cudnn_deterministic}")

    # TF32 on Ampere+ GPUs (compute capability >= 8.0)
    if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = cuda_cfg.allow_tf32
        torch.backends.cudnn.allow_tf32       = cuda_cfg.allow_tf32
        print(f"[CUDA] TF32 = {cuda_cfg.allow_tf32}")

    print(f"[CUDA] AMP (FP16) = {cuda_cfg.amp}")
    print(f"[CUDA] PyTorch {torch.__version__}  |  "
          f"CUDA {torch.version.cuda}  |  "
          f"cuDNN {torch.backends.cudnn.version()}")


# ══════════════════════════════════════════════════════════════
#  YAML Config Loader
# ══════════════════════════════════════════════════════════════

def _build_nested_dataclass(cls, raw: dict):
    """Safely build a dataclass from a raw dict, casting types."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in raw.items():
        if k in fields:
            default = getattr(cls, k, v)
            try:
                if not isinstance(default, list):
                    kwargs[k] = type(default)(v)
                else:
                    kwargs[k] = v
            except (TypeError, ValueError):
                kwargs[k] = v
        else:
            print(f"[CONFIG WARN] Unknown {cls.__name__} key: {k}")
    return cls(**kwargs)


def load_config(yaml_path: str) -> TrainConfig:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {yaml_path}\n"
            f"Create one from the template or pass --config <path>")

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    print(f"[CONFIG] Loaded {yaml_path}")

    aug_cfg     = _build_nested_dataclass(AugConfig,     raw.pop("augmentation", {}) or {})
    cuda_cfg    = _build_nested_dataclass(CUDAConfig,    raw.pop("cuda", {}) or {})
    dataset_cfg = _build_nested_dataclass(DatasetConfig, raw.pop("dataset", {}) or {})

    nested_names = {"aug", "cuda", "dataset"}
    train_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()
                    if f.name not in nested_names}
    train_kwargs = {"aug": aug_cfg, "cuda": cuda_cfg, "dataset": dataset_cfg}
    for k, v in raw.items():
        if k in train_fields:
            try:
                train_kwargs[k] = type(getattr(TrainConfig, k, v))(v)
            except (TypeError, ValueError):
                train_kwargs[k] = v
        else:
            print(f"[CONFIG WARN] Unknown key: {k}")

    return TrainConfig(**train_kwargs)


# ══════════════════════════════════════════════════════════════
#  Albumentations Pipeline Builder
# ══════════════════════════════════════════════════════════════

def _safe_transform(cls, **kwargs):
    try:
        return cls(**kwargs)
    except Exception as exc:
        print(f"[AUG WARN] {cls.__name__} skipped -- {exc}")
        return A.NoOp()


def build_augmentation_pipeline(aug: AugConfig, imgsz: int) -> A.Compose:
    bbox_params = A.BboxParams(
        format="yolo", label_fields=["class_labels"],
        min_area=0.01, min_visibility=0.3)

    t = []

    if aug.enable_flip:
        t.append(_safe_transform(A.HorizontalFlip, p=aug.flip_lr_p))
        t.append(_safe_transform(A.VerticalFlip,   p=aug.flip_ud_p))

    if aug.enable_rotate:
        t.append(_safe_transform(A.Rotate,
            limit=aug.rotate_limit,
            border_mode=cv2.BORDER_REFLECT_101, p=aug.rotate_p))

    if aug.enable_affine:
        t.append(_safe_transform(A.Affine,
            shear=(-aug.shear_limit, aug.shear_limit),
            border_mode=cv2.BORDER_REFLECT_101, p=aug.affine_p))

    if aug.enable_perspective:
        t.append(_safe_transform(A.Perspective,
            scale=(0.01, aug.perspective_scale), p=aug.perspective_p))

    if aug.enable_crop:
        t.append(_safe_transform(A.RandomResizedCrop,
            size=(imgsz, imgsz),
            scale=(aug.crop_min_area, 1.0),
            ratio=(0.75, 1.33), p=aug.random_crop_p))

    if aug.enable_distortion:
        t.append(_safe_transform(A.ElasticTransform, alpha=1, sigma=10, p=aug.elastic_p))
        t.append(_safe_transform(A.GridDistortion, num_steps=5, distort_limit=0.2, p=aug.grid_distort_p))

    if aug.enable_color_jitter:
        t.append(_safe_transform(A.RandomBrightnessContrast,
            brightness_limit=aug.brightness_limit,
            contrast_limit=aug.contrast_limit, p=aug.color_jitter_p))

    if aug.enable_hsv:
        t.append(_safe_transform(A.HueSaturationValue,
            hue_shift_limit=aug.hue_shift_limit,
            sat_shift_limit=aug.sat_shift_limit,
            val_shift_limit=aug.val_shift_limit, p=aug.hsv_p))

    if aug.enable_grayscale:
        t.append(_safe_transform(A.ToGray, p=aug.grayscale_p))

    if aug.enable_blur:
        t.append(_safe_transform(A.GaussianBlur, blur_limit=(3, 7), p=aug.gaussian_blur_p))
        t.append(_safe_transform(A.MotionBlur,   blur_limit=(3, 9), p=aug.motion_blur_p))
        t.append(_safe_transform(A.MedianBlur,   blur_limit=5,      p=aug.median_blur_p))

    if aug.enable_noise:
        t.append(_safe_transform(A.GaussNoise, std_range=(0.0124, 0.0277), p=aug.gaussian_noise_p))
        t.append(_safe_transform(A.ISONoise, color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=aug.iso_noise_p))

    if aug.enable_sharpen:
        t.append(_safe_transform(A.Sharpen, alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=aug.sharpen_p))

    if aug.enable_clahe:
        t.append(_safe_transform(A.CLAHE, clip_limit=4.0, tile_grid_size=(8, 8), p=aug.clahe_p))

    if aug.enable_dropout:
        t.append(_safe_transform(A.CoarseDropout,
            num_holes_range=(1, aug.dropout_max_holes),
            hole_height_range=(8, aug.dropout_max_height),
            hole_width_range=(8, aug.dropout_max_width),
            fill=0, p=aug.coarse_dropout_p))

    if aug.enable_shadow:
        t.append(_safe_transform(A.RandomShadow,
            shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2),
            shadow_dimension=5, p=aug.random_shadow_p))

    if aug.enable_fog:
        t.append(_safe_transform(A.RandomFog, fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=aug.random_fog_p))

    if aug.enable_rain:
        t.append(_safe_transform(A.RandomRain,
            slant_range=(-10, 10), drop_length=15, drop_width=1,
            drop_color=(200, 200, 200), blur_value=3,
            brightness_coefficient=0.9, rain_type="drizzle", p=aug.random_rain_p))

    if aug.enable_sunflare:
        t.append(_safe_transform(A.RandomSunFlare,
            flare_roi=(0, 0, 1, 0.5), angle_range=(0, 1),
            num_flare_circles_range=(3, 6), src_radius=200,
            src_color=(255, 255, 255), p=aug.sun_flare_p))

    pipeline = A.Compose(t, bbox_params=bbox_params)
    n_active  = sum(1 for x in pipeline.transforms if not isinstance(x, A.NoOp))
    n_skipped = sum(1 for x in pipeline.transforms if isinstance(x, A.NoOp))
    print(f"[AUG] Pipeline: {n_active} active, {n_skipped} skipped")
    return pipeline


# ══════════════════════════════════════════════════════════════
#  Augmented YOLO Dataset (monkey-patched into Ultralytics)
# ══════════════════════════════════════════════════════════════

class AugmentedYOLODataset(YOLODataset):
    def __init__(self, *args, albu_pipeline=None, pipeline_p=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.albu_pipeline = albu_pipeline
        self.pipeline_p    = pipeline_p

    def __getitem__(self, index):
        item = super().__getitem__(index)

        if self.albu_pipeline is None or random.random() > self.pipeline_p:
            return item

        try:
            img = item["img"].permute(1, 2, 0).numpy()
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            orig_bboxes = item["bboxes"].numpy().tolist()
            orig_labels = item["cls"].numpy().flatten().tolist()
            if len(orig_bboxes) == 0:
                return item

            orig_bboxes = [[max(0.001, min(0.999, v)) for v in b] for b in orig_bboxes]
            result = self.albu_pipeline(image=img, bboxes=orig_bboxes, class_labels=orig_labels)

            aug_img, aug_bboxes, aug_labels = result["image"], result["bboxes"], result["class_labels"]
            if len(aug_bboxes) == 0:
                return item

            oh, ow = img.shape[:2]
            ah, aw = aug_img.shape[:2]
            if (ah, aw) != (oh, ow):
                aug_img = cv2.resize(aug_img, (ow, oh), interpolation=cv2.INTER_LINEAR)

            aug_arr = np.ascontiguousarray(aug_img)
            if aug_arr.dtype != np.uint8:
                aug_arr = aug_arr.astype(np.uint8)

            item["img"]    = torch.from_numpy(aug_arr.transpose(2, 0, 1)).contiguous()
            item["bboxes"] = torch.tensor(aug_bboxes, dtype=item["bboxes"].dtype)
            item["cls"]    = torch.tensor([[c] for c in aug_labels], dtype=item["cls"].dtype)
            if "batch_idx" in item:
                item["batch_idx"] = torch.zeros(len(aug_bboxes), dtype=item["batch_idx"].dtype)

        except Exception as exc:
            print(f"[AUG WARN] index={index}: {exc}")

        return item


def patch_yolo_dataset(albu_pipeline, pipeline_p):
    
    class Patched(AugmentedYOLODataset):
        def __init__(self, *a, **kw):
            super().__init__(*a, albu_pipeline=albu_pipeline, pipeline_p=pipeline_p, **kw)

    ds_mod.YOLODataset  = Patched
    bld_mod.YOLODataset = Patched
    print(f"[AUG] YOLO Dataset patched (pipeline_p={pipeline_p:.0%})")


# ══════════════════════════════════════════════════════════════
#  Augmentation Preview Grid
# ══════════════════════════════════════════════════════════════

def save_augmentation_preview(pipeline, image_dir, task, cfg):
    image_dir = Path(image_dir)
    img_files = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))
    if not img_files:
        return

    n_s, n_a, th = cfg.preview_samples, cfg.preview_augs, cfg.preview_thumb
    tw = th
    output_path = "/tmp/augmented_preview.jpg"

    samples = random.sample(img_files, min(n_s, len(img_files)))
    cols, rows = 1 + n_a, len(samples)
    grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for r, p in enumerate(samples):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grid[r*th:(r+1)*th, 0:tw] = cv2.resize(img, (tw, th))
        for c in range(1, cols):
            try:
                a = pipeline(image=img, bboxes=[], class_labels=[])["image"]
            except Exception:
                a = img
            grid[r*th:(r+1)*th, c*tw:(c+1)*tw] = cv2.resize(a, (tw, th))

    hdr = np.ones((24, cols * tw, 3), dtype=np.uint8) * 40
    for c, lbl in enumerate(["Original"] + [f"Aug {i}" for i in range(1, cols)]):
        cv2.putText(hdr, lbl, (c*tw+8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    grid = np.vstack([hdr, grid])
    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    task.upload_artifact(name="augmentation_preview", artifact_object=output_path)
    task.get_logger().report_image(title="Augmentation Preview", series="samples",
                                   local_path=output_path, iteration=0)


# ══════════════════════════════════════════════════════════════
#  ClearML Task Init
# ══════════════════════════════════════════════════════════════

def init_clearml_task(cfg):
    task = Task.init(
        project_name=cfg.project_name,
        task_name=cfg.task_name,
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": True, "tensorboard": True, "matplotlib": True},
    )

    # Connect hyperparameters by section name.
    # ClearML HPO references these as "General/lr0", "General/batch", etc.
    nested_names = {"aug", "cuda", "dataset"}
    cfg_dict = {k: v for k, v in asdict(cfg).items() if k not in nested_names}
    params = task.connect(cfg_dict, name="General")
    for k, v in params.items():
        if hasattr(cfg, k) and k not in nested_names:
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception:
                pass

    aug_params = task.connect(asdict(cfg.aug), name="Augmentation")
    for k, v in aug_params.items():
        if hasattr(cfg.aug, k):
            try:
                setattr(cfg.aug, k, type(getattr(cfg.aug, k))(v))
            except Exception:
                pass

    cuda_params = task.connect(asdict(cfg.cuda), name="CUDA")
    for k, v in cuda_params.items():
        if hasattr(cfg.cuda, k):
            try:
                setattr(cfg.cuda, k, type(getattr(cfg.cuda, k))(v))
            except Exception:
                pass

    ds_dict = {k: v for k, v in asdict(cfg.dataset).items() if k != "classes"}
    ds_dict["classes"] = ",".join(cfg.dataset.classes)
    task.connect(ds_dict, name="Dataset")

    # Log environment
    gpu_info = "N/A"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = f"{props.name} ({props.total_memory / (1024**3):.1f} GB)"
    task.get_logger().report_text(
        f"GPU: {gpu_info}\n"
        f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}\n"
        f"CPU: {psutil.cpu_count()} cores  |  "
        f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    print(f"[ClearML] Task ID  : {task.id}")
    print(f"[ClearML] Task URL : {task.get_output_log_web_page()}")
    return task


# ══════════════════════════════════════════════════════════════
#  ClearML Dataset Versioning
# ══════════════════════════════════════════════════════════════

def _resolve_data_root(cfg):
    yaml_path = Path(cfg.data_yaml)
    if not yaml_path.is_absolute():
        cwd_path    = Path.cwd()           / yaml_path
        script_path = Path(__file__).parent / yaml_path
        yaml_path   = cwd_path.resolve() if cwd_path.exists() \
                      else (script_path.resolve() if script_path.exists()
                            else cwd_path.resolve())
    else:
        yaml_path = yaml_path.resolve()

    data_root = yaml_path.parent
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {yaml_path}")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (data_root / sub).exists():
            print(f"[DATA WARN] Missing: {data_root / sub}")
    return yaml_path, data_root


def register_dataset(cfg, task):
    _, data_root = _resolve_data_root(cfg)
    data_root_str = str(data_root)

    try:
        all_ds = Dataset.list_datasets(dataset_project=cfg.project_name) or []
    except TypeError:
        all_ds = Dataset.list_datasets() or []

    existing = [d for d in all_ds
                if d.get("name") == cfg.dataset.name and d.get("status") == "completed"]

    if not cfg.dataset.register_enabled:
        return data_root_str

    if existing:
        ds = Dataset.get(dataset_id=sorted(
            existing, key=lambda d: d.get("created", ""), reverse=True)[0]["id"])
    else:
        ds = Dataset.create(dataset_name=cfg.dataset.name,
                            dataset_project=cfg.project_name,
                            description=cfg.dataset.description)
        ds.add_files(path=data_root_str, recursive=True)
        ds.upload(show_progress=True)
        ds.finalize()

    task.connect_configuration(
        {"dataset_id": ds.id, "dataset_name": cfg.dataset.name}, name="DatasetInfo")
    return data_root_str


# ══════════════════════════════════════════════════════════════
#  ClearML Callback
# ══════════════════════════════════════════════════════════════

class ClearMLCallback:
    def __init__(self, task, class_names):
        self.logger      = task.get_logger()
        self.task        = task
        self.class_names = class_names

    def on_train_epoch_end(self, trainer):
        e = trainer.epoch
        for name, val in trainer.metrics.items():
            g, _, k = name.partition("/")
            self.logger.report_scalar(g or "metrics", k or name, value=float(val), iteration=e)
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            for i, pg in enumerate(trainer.optimizer.param_groups):
                self.logger.report_scalar("lr", f"pg{i}", value=pg.get("lr", 0.0), iteration=e)

    def on_fit_epoch_end(self, trainer):
        e = trainer.epoch
        for name, val in trainer.metrics.items():
            g, _, k = name.partition("/")
            self.logger.report_scalar(g or "metrics", k or name, value=float(val), iteration=e)

    def on_train_end(self, trainer):
        sd = Path(trainer.save_dir)
        for aname, glob in [("best_weights", "weights/best.pt"),
                             ("last_weights", "weights/last.pt"),
                             ("results_csv",  "results.csv")]:
            p = sd / glob
            if p.exists():
                self.task.upload_artifact(name=aname, artifact_object=str(p))
        for cm in sd.glob("confusion_matrix*.png"):
            self.task.upload_artifact(name=cm.stem, artifact_object=str(cm))


# ══════════════════════════════════════════════════════════════
#  Model Training
# ══════════════════════════════════════════════════════════════

def train(cfg, task):
    model = YOLO(cfg.model_variant if cfg.pretrained
                 else cfg.model_variant.replace(".pt", ".yaml"))

    cb = ClearMLCallback(task, cfg.dataset.classes)
    model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end",   cb.on_fit_epoch_end)
    model.add_callback("on_train_end",       cb.on_train_end)

    print("\n" + "=" * 64)
    print(f"  Model     : {cfg.model_variant}  | Imgsz: {cfg.imgsz}")
    print(f"  Epochs    : {cfg.epochs}  | Batch: {cfg.batch}")
    print(f"  LR0       : {cfg.lr0}  | LRF: {cfg.lrf}  | WD: {cfg.weight_decay}")
    print(f"  Mosaic    : {cfg.mosaic} | Mixup: {cfg.mixup} | CopyPaste: {cfg.copy_paste}")
    print(f"  Dropout   : {cfg.dropout} | cls_pw: {cfg.cls_pw}")
    print(f"  Degrees   : {cfg.degrees} | Translate: {cfg.translate} | Scale: {cfg.scale}")
    print(f"  Device    : {cfg.cuda.device}  | AMP: {cfg.cuda.amp}")
    print(f"  TF32      : {cfg.cuda.allow_tf32}  | cuDNN bench: {cfg.cuda.cudnn_benchmark}")
    print("=" * 64 + "\n")

    train_kwargs = dict(
        data=cfg.data_yaml, imgsz=cfg.imgsz,
        epochs=cfg.epochs, batch=cfg.batch, workers=cfg.workers,
        optimizer=cfg.optimizer, lr0=cfg.lr0, lrf=cfg.lrf,
        momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs, warmup_bias_lr=cfg.warmup_bias_lr,
        hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
        degrees=cfg.degrees, translate=cfg.translate, scale=cfg.scale,
        shear=cfg.shear, flipud=cfg.flipud, fliplr=cfg.fliplr,
        mosaic=cfg.mosaic, mixup=cfg.mixup, copy_paste=cfg.copy_paste,
        close_mosaic=cfg.close_mosaic,
        multi_scale=cfg.multi_scale, dropout=cfg.dropout,
        device=cfg.cuda.device, amp=cfg.cuda.amp, val=cfg.val,
        save_period=cfg.save_period, patience=cfg.patience,
        project=cfg.output_dir, name=cfg.task_name, exist_ok=True,
    )

    if cfg.cls_pw > 0:
        train_kwargs["cls_pw"] = cfg.cls_pw

    results = model.train(**train_kwargs)
    return model, results


# ══════════════════════════════════════════════════════════════
#  Model Registry + Validation
# ══════════════════════════════════════════════════════════════

def publish_model(cfg, task, model):
    best_pt = Path(cfg.task) / cfg.output_dir / cfg.task_name / "weights" / "best.pt"
    if not best_pt.exists():
        print("[ClearML] best.pt not found -- skipping registry")
        return

    label_enum = {name: i for i, name in enumerate(cfg.dataset.classes)}
    out = OutputModel(task=task, name=f"{cfg.project_name}/{cfg.task_name}",
                      framework="PyTorch", label_enumeration=label_enum)
    out.update_weights(weights_filename=str(best_pt), auto_delete_file=False)
    out.update_design(config_dict={
        "model_variant": cfg.model_variant, "imgsz": cfg.imgsz,
        "num_classes": len(cfg.dataset.classes), "classes": cfg.dataset.classes,
    })
    print(f"[ClearML] Model published: {out.id}")


def validate(cfg, task, model):
    print(f"\n[INFO] Validating on '{cfg.test_split}' split ...")
    logger = task.get_logger()
    val_results = model.val(data=cfg.data_yaml, imgsz=cfg.imgsz,
                            split=cfg.test_split, device=cfg.cuda.device)

    for name, val in [("mAP50", val_results.box.map50), ("mAP50-95", val_results.box.map),
                      ("Precision", val_results.box.mp), ("Recall", val_results.box.mr)]:
        logger.report_single_value(name=f"test/{name}", value=float(val))
        print(f"  {name:<12}: {val:.4f}")

    for cls_name, ap50 in zip(cfg.dataset.classes, val_results.box.ap50):
        logger.report_single_value(name=f"test/AP50_{cls_name}", value=float(ap50))
    return val_results


# ══════════════════════════════════════════════════════════════
#  Cleanup + Main
# ══════════════════════════════════════════════════════════════

def _cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()

atexit.register(_cleanup)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 Object Detection — NVIDIA CUDA + ClearML + Albumentations")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Configure NVIDIA CUDA before anything else
    setup_cuda(cfg.cuda)

    task = init_clearml_task(cfg)

    try:
        data_root = register_dataset(cfg, task)
        albu = build_augmentation_pipeline(cfg.aug, imgsz=cfg.imgsz)

        train_img_dir = Path(data_root) / "images" / "train"
        if train_img_dir.exists():
            save_augmentation_preview(albu, str(train_img_dir), task, cfg)

        if cfg.aug.pipeline_p > 0:
            patch_yolo_dataset(albu, cfg.aug.pipeline_p)

        model, results = train(cfg, task)
        validate(cfg, task, model)
        publish_model(cfg, task, model)
    finally:
        task.close()

    print(f"\n[DONE] {task.get_output_log_web_page()}")


if __name__ == "__main__":
    main()