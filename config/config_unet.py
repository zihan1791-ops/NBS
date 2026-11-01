"""U-Net 配置：与 BiSeNet/ScConv 共用相同的 NEU 固定划分清单。"""
from pathlib import Path

# 数据集管线的公共键（IMAGE_DIR_NAME / TRAIN_SPLIT_NAME / COLOR_MAP_BGR /
# SALT_PEPPER_THRESHOLD / AUGMENT_TRAINING / PIN_MEMORY / EXTERNAL_TEST_RATIO …）
# 都由 config.py 提供。U-Net 现在也走 scripts/dataset.py 的 build_loaders，
# 因此必须共享这份定义，而不是自己再抄一遍——否则两条训练路径的划分与增强会
# 悄悄分叉，主表就不可比了。
from config import *  # noqa: F401,F403

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_DIR / "data" / "NEU_Seg-main"
DATASET_PATHS = {
    "neu": str(DATA_ROOT),
    "sd": str(PROJECT_DIR / "data" / "SD-saliency"),
    "magnetic": str(PROJECT_DIR / "data" / "Magnetic-tile"),
    "severstal": str(PROJECT_DIR / "data" / "Severstal Steel Dataset"),
}
DATASET_DEFAULTS = {
    "neu": dict(num_classes=4,image_size=(200,200),batch_size=16,
        train_images_subdir="images/train",test_images_subdir="images/test",
        train_masks_subdir="masks/train",test_masks_subdir="masks/test"),
    "sd": dict(num_classes=2,image_size=(200,200),batch_size=32),
    "magnetic": dict(num_classes=6,image_size=(256,256),batch_size=8),
    "severstal": dict(num_classes=4,image_size=(1600,256),batch_size=4),
}
EPOCHS=300
INITIAL_LR=1e-3
WEIGHT_DECAY=1e-4
LR_FACTOR=0.5
LR_PATIENCE=6
MIN_LR=1e-6
MAX_GRAD_NORM=5.0
EARLY_STOPPING_PATIENCE=30
SEED=42
NUM_WORKERS=4
SP_THRESHOLD=0.02
NO_AUGMENT=False
VIS_ENABLED=True
VIS_MAX_SAMPLES=20

# ---- 训练协议开关（默认值即历史行为：plateau + 无 warmup + 不做确定性）----
# 论文主表要求 U-Net 与 BiSeNetV2 各臂跑同一套 recipe，此时由派生配置
# （如 configUnetP200）把这些键覆盖成 cosine / 5 / True。
SCHEDULER_NAME="plateau"          # plateau | cosine
WARMUP_EPOCHS=0                   # 仅 cosine 生效
DETERMINISTIC_ALGORITHMS=False    # True = torch.use_deterministic_algorithms(True)
SAVE_DIR=str(PROJECT_DIR / "results2中断" / "unet")
LOG_FILE=None
DATASET_COLOR_MAP={
 "neu":{0:[0,0,0],1:[255,0,0],2:[0,255,0],3:[0,0,255]},
 "sd":{0:[0,0,0],1:[255,0,0],2:[0,255,0],3:[0,0,255]},
 "magnetic":{0:[0,0,0],1:[255,0,255],2:[0,255,255],3:[255,165,0],4:[165,0,255],5:[0,255,165]},
 "severstal":{0:[0,0,0],1:[255,0,0],2:[0,255,0],3:[0,0,255]},
}
DATASET_CLASS_NAMES={
 "neu":["背景","夹杂","斑块","划痕"],
 "sd":["背景","Inclusion","Patch","Scratch"],
 "magnetic":["背景","气孔","破损","裂纹","磨损","不均匀"],
 "severstal":["背景","缺陷1","缺陷2","缺陷3"],
}

# U-Net 与 BiSeNet/ScConv 共用绘图程序时使用的图表参数。
CLASS_NAMES = tuple(DATASET_CLASS_NAMES["neu"])
PLOT_CLASS_NAMES = ("Background", "Inclusion", "Patch", "Scratch")
DATASET_PLOT_CLASS_NAMES = {
 "neu": ("Background", "Inclusion", "Patch", "Scratch"),
 "sd": ("Background", "Defect"),
 "magnetic": ("Background", "Blowhole", "Break", "Crack", "Fray", "Uneven"),
 "severstal": ("Background", "Defect 1", "Defect 2", "Defect 3"),
}
PLOT_COLORS_BGR = ((255,128,0),(0,150,0),(0,0,230),(180,0,180),(0,180,255),(180,80,0))
PLOT_BACKGROUND_BGR = (250,250,250)
PLOT_GRID_BGR = (220,220,220)
PLOT_AXIS_BGR = (70,70,70)
BEST_MARKER_BGR = (0,110,255)
RESULTS_PLOT_WIDTH = 1800
RESULTS_PLOT_HEIGHT = 1200
RESULTS_HEADER_HEIGHT = 145
CURVE_PLOT_WIDTH = 1200
CURVE_PLOT_HEIGHT = 850
PR_CURVE_PLOT_NAME = "PR_curve.png"
ROC_CURVE_PLOT_NAME = "ROC_curve.png"
F1_CURVE_PLOT_NAME = "F1_curve.png"
CONFUSION_MATRIX_PLOT_NAME = "confusion_matrix.png"
PROBABILITY_HISTOGRAM_BINS = 1000
