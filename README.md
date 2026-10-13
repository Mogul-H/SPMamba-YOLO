# SPMamba-YOLO

An object detection project built on a customized Ultralytics YOLO codebase. The repository includes a Mamba-YOLO-style model configuration, training and validation scripts, and a selective-scan implementation.

> **Note:** This repository contains the implementation and configuration files. Dataset paths and checkpoint paths in the supplied scripts are machine-specific and must be updated before running on another computer.

## Contents

- `ultralytics/` — customized Ultralytics codebase.
- `ultralytics/cfg/models/mamba-yolo/SPMamba-YOLO.yaml` — SPMamba-YOLO model configuration.
- `mbyolo_train.py` — command-line training entry point.
- `val.py` — validation example; edit the checkpoint and dataset paths before use.
- `selective_scan/` — selective-scan implementation and tests.
- `asserts/` — project figures and visual assets.
- `yolov8n.pt` — YOLOv8n checkpoint included in the repository.

## Model configuration

The model definition is located at:

```text
ultralytics/cfg/models/mamba-yolo/SPMamba-YOLO.yaml
```

The YAML configuration specifies a Mamba-YOLO-style backbone using modules such as `SimpleStem`, `VSSBlock`, `VisionClueMerge`, `SPPELAN`, and `PSA`. Its feature-fusion head uses `XSSBlock` modules and a detection head operating on three feature scales (P3, P4, and P5).

The provided configuration sets `nc: 80` and defines a `T` scale. If your dataset has a different number of classes, configure `nc` and the dataset YAML consistently with your task.

## Requirements

The project uses Python and PyTorch through the customized Ultralytics package. Exact compatible versions may depend on your CUDA, GPU driver, and selective-scan implementation.

Recommended preparation:

- Python 3.8 or later, consistent with `pyproject.toml`.
- A compatible PyTorch and CUDA installation if training on an NVIDIA GPU.
- The Python dependencies declared by this repository's `pyproject.toml`.
- Any additional dependencies required by the selective-scan modules in your environment.

Install the project in editable mode from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a PyTorch build compatible with your system before training if the default dependency resolution does not match your CUDA environment. If importing a custom scan operator fails, check the `selective_scan/` setup instructions and the compatibility of its dependencies with your installed PyTorch/CUDA versions.

## Prepare your dataset

Create a dataset YAML file describing the training and validation image paths and class names. For example:

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val

names:
  0: class_a
  1: class_b
```

Replace the example paths and class names with those of your dataset. Make sure the number of class names matches the `nc` value in the model configuration.

## Training

The training script is `mbyolo_train.py`. It accepts arguments for the dataset YAML, model configuration, batch size, image size, epochs, optimizer, device, and output directory.

Example:

```bash
python mbyolo_train.py \
  --data /path/to/your/dataset/data.yaml \
  --config ultralytics/cfg/models/mamba-yolo/SPMamba-YOLO.yaml \
  --batch_size 4 \
  --imgsz 640 \
  --epochs 100 \
  --device 0 \
  --project output_dir \
  --name spmamba-yolo
```

On Windows, the same command can be entered on one line:

```bat
python mbyolo_train.py --data "D:\path\to\dataset\data.yaml" --config "ultralytics\cfg\models\mamba-yolo\SPMamba-YOLO.yaml" --batch_size 4 --imgsz 640 --epochs 100 --device 0 --project output_dir --name spmamba-yolo
```

**Important:** The current `mbyolo_train.py` defines a machine-specific default dataset path. Always pass `--data` explicitly when running on another machine. The script's default values are batch size `4`, image size `640`, epochs `100`, optimizer `SGD`, workers `1`, and device `0`.

The script saves training outputs under the directory specified by `--project` and `--name`.

## Validation

The supplied `val.py` is an example, not a portable out-of-the-box command: its checkpoint and dataset paths are hard-coded to local Windows paths. Update them before use, or validate through the Ultralytics Python API:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
metrics = model.val(
    data="path/to/data.yaml",
    split="val",
    imgsz=640,
    batch=16,
)
```

Use the checkpoint produced by your training run and the same dataset definition used for training.

## Selective scan

The `selective_scan/` directory contains a selective-scan implementation and related tests. Refer to `selective_scan/README.md` and `selective_scan/setup.py` for implementation-specific details and setup requirements. Test commands and hardware requirements can vary with the selected implementation and installed dependencies.

## Reproducibility notes

- Update all dataset and checkpoint paths before running the scripts.
- Record the Python, PyTorch, CUDA, GPU, and dependency versions used for each experiment.
- Keep the dataset YAML, training arguments, and checkpoint associated with each run.
- Results depend on the dataset, random seeds, software versions, and hardware configuration; no benchmark results are claimed by this README.

## License

The repository includes a `LICENSE` file. Review it before using, modifying, or redistributing this code. The customized codebase includes Ultralytics components; check the applicable license terms and any licenses for third-party components used in your deployment.

## 中文说明

SPMamba-YOLO 是一个基于定制版 Ultralytics YOLO 代码库的目标检测项目。仓库包含 SPMamba-YOLO 模型配置、训练脚本、验证示例以及 selective scan 相关实现。

1. **修改数据集路径：** 训练时通过 `--data` 指定你自己的数据集 YAML 文件。
2. **确认类别数：** 数据集类别数量应与模型配置中的 `nc` 保持一致。
3. **安装依赖：** 在仓库根目录执行 `python -m pip install -e .`；PyTorch/CUDA 版本应与你的硬件环境兼容。
4. **开始训练：** 参考上方 Training 部分的命令。
5. **运行验证：** 使用你训练得到的权重，并将数据集 YAML 路径替换为实际路径。

当前训练和验证脚本中存在本机路径，因此不能直接假定它们在其他计算机上无需修改即可运行。请在正式实验前检查路径、依赖版本及许可证要求。
