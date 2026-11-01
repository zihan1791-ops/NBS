"""G1 臂：修掉 SRU 门控的尺度塌缩（ScConv 的"新应用方式"之一）。

诊断证据（`scripts/diagnose_scconv.py --with_data`，对 4 个已训练检查点）：
官方 SRU 用 γ̂ = γ/Σγ 构造门控，而 γ̂ ≈ 1/C。当 C=128 时 γ̂ ≈ 0.0078，
乘到零均值单位方差的 GroupNorm 输出上，sigmoid 的输入幅度只有 ~0.008，
实测 reweight 的 std 仅 0.002、99.7% 的元素落在阈值 0.5 的 ±0.01 内。
也就是说"通过/不通过"的门控几乎是常数，SRU 无法区分 informative 与冗余特征，
只剩一个固定的通道对半切分 + 交叉相加（近似固定线性混合）。

本臂把 γ/Σγ 换回 γ 本身（SC_CONV_GAMMA_NORMALISE=False），让门控重新工作在
可用的动态范围里：阈值 0.5 附近才有真正的空间/通道选择性。

这是"应用方式"层面的修正，不改 ScConv 的结构定义，只改它的工作点。
"""
from pathlib import Path

from configScConvAblation import *  # noqa: F401,F403

OUTPUT_DIR = PROJECT_DIR / "results2中断" / "bisenetv2_ablation_G1_gate_fix"
EXPERIMENT_NAME = "G1-gate-fix"
EXPERIMENT_NOTE = "支线 G1：SRU 门控去掉 γ/Σγ 归一化，其余同 S0。"

SC_CONV_GAMMA_NORMALISE = False


def to_dict():
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in globals().copy().items() if name.isupper()}
