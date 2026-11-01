"""Architecture definitions for every model in the NBS project.

Old experiments:  U-Net, BiSeNetV2, BiSeNetV2+ScConv
New ablation  :   B0(=BiSeNetV2), S0(=BiSeNetV2+ScConv), D1, D2, P1, P2, G1, R1, R2, R1D2
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from vsdxgen import Diagram, Node, write_vsdx  # noqa: E402

TEMPLATE = Path(r"G:\NBS\docs\architecture\ChatGpt\BiSeNetV2.vsdx")
OUT_DIR = Path(r"G:\NBS\docs\architecture")

# ------------------------------------------------------------------ palette ---
C_DETAIL = dict(fill="#FDF0D5", line="#A86B00", text_color="#4A2D00")
C_SEM = dict(fill="#E8F5E9", line="#2E7D32", text_color="#1B3D1F")
C_BGA = dict(fill="#F3E5F5", line="#6A1B9A", text_color="#3A0D4A")
C_HEAD = dict(fill="#FFE9E9", line="#B3261E", text_color="#5C0F0B")
C_IO = dict(fill="#DFE9FF", line="#26418F", text_color="#0B1F4D")
C_SCC = dict(fill="#E0F7FA", line="#00695C", text_color="#003D33")
C_IDLE = dict(fill="#F2F2F2", line="#9E9E9E", text_color="#555555")


# ------------------------------------------------------------------- U-Net ---
def build_unet() -> Diagram:
    d = Diagram(
        "U-Net 网络架构",
        "输入 [B,3,200,200] → 输出 [B,4,200,200] ｜ 实线 = 主数据流，紫色虚线 = 跳跃连接 ｜ "
        "每框含层类型、关键超参与输出张量形状",
        page_title="U-Net 架构", filename="U-Net",
    )
    enc = dict(C_DETAIL)
    dec = dict(fill="#E3F2FD", line="#1565C0", text_color="#0A2A4A")
    d.add(
        Node("inc", "inc: DoubleConv(3→32)\nConv 3×3, s1, p1, bias=False\n+ BatchNorm2d + ReLU ×2\n[B,32,200,200]",
             0, 0, **enc),
        Node("down1", "down1: MaxPool2d(2) + DoubleConv(32→64)\n最大池化 200×200 → 100×100\n[B,64,100,100]", 1, 0, **enc),
        Node("down2", "down2: MaxPool2d(2) + DoubleConv(64→128)\n100×100 → 50×50\n[B,128,50,50]", 2, 0, **enc),
        Node("down3", "down3: MaxPool2d(2) + DoubleConv(128→256)\n50×50 → 25×25\n[B,256,25,25]", 3, 0, **enc),
        Node("down4", "down4: MaxPool2d(2) + DoubleConv(256→512)\n25×25 → 12×12（瓶颈）\n[B,512,12,12]", 4, 0, **enc),
        Node("up1", "up1: Upsample 双线性 → 25×25\nConcat(skip1) → DoubleConv 768→256\n[B,256,25,25]", 4, 1, **dec),
        Node("up2", "up2: Upsample 双线性 → 50×50\nConcat(skip2) → DoubleConv 384→128\n[B,128,50,50]", 3, 1, **dec),
        Node("up3", "up3: Upsample 双线性 → 100×100\nConcat(skip3) → DoubleConv 192→64\n[B,64,100,100]", 2, 1, **dec),
        Node("up4", "up4: Upsample 双线性 → 200×200\nConcat(skip4) → DoubleConv 96→32\n[B,32,200,200]", 1, 1, **dec),
        Node("outc", "outc: Conv2d 32→4\n1×1, s1, p0 分类头\n[B,4,200,200]", 0, 1, **C_IO),
    )
    for a, b in [("inc", "down1"), ("down1", "down2"), ("down2", "down3"),
                 ("down3", "down4"), ("down4", "up1"), ("up1", "up2"),
                 ("up2", "up3"), ("up3", "up4"), ("up4", "outc")]:
        d.link(a, b)
    d.link("down3", "up1", label="skip1 [B,256,25,25]", route="skip", color="#6A1B9A")
    d.link("down2", "up2", label="skip2 [B,128,50,50]", route="skip", color="#6A1B9A")
    d.link("down1", "up3", label="skip3 [B,64,100,100]", route="skip", color="#6A1B9A")
    d.link("inc", "up4", label="skip4 [B,32,200,200]", route="skip", color="#6A1B9A")
    return d


# --------------------------------------------------------------- BiSeNetV2 ---
def build_bisenet(
    *,
    name: str,
    title: str,
    subtitle: str,
    fused: str | None = None,          # None | "residual0.1" | "residual1.0" | "direct"
    placement: tuple[str, ...] = (),   # subset of {"detail","semantic"}
    gamma_fix: bool = False,
    replace_detail: bool = False,
    replace_semantic: bool = False,
) -> Diagram:
    d = Diagram(title, subtitle, page_title=name, filename=name)

    det_note = ""
    if replace_detail:
        det_note = "\n※ stage2/stage3 各 2 个重复 3×3 卷积 → ScConv+ReLU（共 4 处）"
    sem_note = ""
    if replace_semantic:
        sem_note = "\n※ 各 GELayerS1.conv1 → ScConv+ReLU（stage3×1, stage4×1, stage5×3，共 5 处）"

    d.add(
        Node("d1", "Detail Stage1\nConvBNReLU 3→64, 3×3, s2\nConvBNReLU 64→64, 3×3 ×1\n200×200 → 100×100  [B,64,100,100]",
             1, 0, **C_DETAIL),
        Node("d2", "Detail Stage2\nConvBNReLU 64→64, 3×3, s2\nConvBNReLU 64→64, 3×3 ×2" + det_note +
             "\n100×100 → 50×50  [B,64,50,50]", 3, 0, **C_DETAIL),
        Node("d3", "Detail Stage3\nConvBNReLU 64→128, 3×3, s2\nConvBNReLU 128→128, 3×3 ×2" + det_note +
             "\n50×50 → 25×25  [B,128,25,25]", 5, 0, **C_DETAIL),

        Node("s0", "Semantic Stem\nConvBNReLU 3→16, 3×3, s2 → 100×100\n双路 1×1+3×3/s2 ‖ MaxPool 3×3 s2\nConcat 32 → Conv 32→16\n200×200 → 50×50  [B,16,50,50]",
             1, 1, **C_SEM),
        Node("s3", "Semantic stage3\nGELayerS2 16→32 + GELayerS1 ×1\nGE 扩展比 6（DW 3×3 → 1×1）" + sem_note +
             "\n50×50 → 25×25  [B,32,25,25]", 2, 1, **C_SEM),
        Node("s4", "Semantic stage4\nGELayerS2 32→64 + GELayerS1 ×1" + sem_note +
             "\n25×25 → 13×13  [B,64,13,13]", 3, 1, **C_SEM),
        Node("s5", "Semantic stage5\nGELayerS2 64→128 + GELayerS1 ×3" + sem_note +
             "\n13×13 → 7×7  [B,128,7,7]", 4, 1, **C_SEM),
        Node("ctx", "ContextEmbeddingBlock\nGAP 1×1 → ConvBNReLU 128, 1×1\n广播相加 → ConvBNReLU 3×3\n[B,128,7,7]",
             5, 1, **C_SEM),
    )

    if "detail" in placement:
        d.add(Node("sd", "detail_scconv = ResidualScConv(128)\nSRU → CRU，scale = 0.1\n[B,128,25,25]", 6, 0, **C_SCC))
        d.link("d3", "sd")
        d.link("sd", "bga", route="elbow_l")
    else:
        d.link("d3", "bga", route="elbow_l")

    if "semantic" in placement:
        d.add(Node("ss", "semantic_scconv = ResidualScConv(128)\nSRU → CRU，scale = 0.1\n[B,128,7,7]", 6, 1, **C_SCC))
        d.link("ctx", "ss")
        d.link("ss", "bga")
    else:
        d.link("ctx", "bga")

    bga_label = "BGA 双边引导聚合\n" \
                "detail_main: DWConv 3×3 + Conv 1×1\n" \
                "detail_gate: Conv 3×3 s2 + AvgPool 3×3 s2 → 7×7\n" \
                "semantic_gate: Conv 3×3 ｜ semantic_main: DWConv+Conv 1×1\n" \
                "σ 门控双向融合 → ConvBNReLU 128→128\n" \
                "[B,128,25,25]"
    d.add(Node("bga", bga_label, 7, -1, **C_BGA))

    if fused:
        if fused.startswith("residual"):
            scale = fused.replace("residual", "")
            gate = "γ/Σγ 归一化（原版）" if not gamma_fix else "γ 直接归一化（G1 修正）"
            txt = f"fused_scconv = ResidualScConv(128)\nx = x + {scale} × ScConv(x)\n{gate}\n[B,128,25,25]"
        else:
            gate = "γ/Σγ 归一化（原版）" if not gamma_fix else "γ 直接归一化（G1 修正）"
            txt = f"fused_scconv = ScConv(128)\nx = ScConv(x)（无残差旁路）\n{gate}\n[B,128,25,25]"
        d.add(Node("sf", txt, 8, -1, **C_SCC))
        d.link("bga", "sf")
        d.link("sf", "head")
    else:
        d.link("bga", "head")

    d.add(
        Node("head", "SegmentationHead\nConvBNReLU 128→1024, 3×3, s1, p1\nDropout2d p = 0.1\nConv2d 1024→4, 1×1\n[B,4,25,25]",
             9, -1, **C_HEAD),
        Node("out", "Upsample 双线性 → 200×200\n主输出 logits\n[B,4,200,200]", 10, -1, **C_IO),
        Node("aux2", "Aux2: feat2 → 128 → 4\n[B,4,200,200]", 1, 2, **C_HEAD),
        Node("aux3", "Aux3: feat3 → 128 → 4\n[B,4,200,200]", 2, 2, **C_HEAD),
        Node("aux4", "Aux4: feat4 → 128 → 4\n[B,4,200,200]", 3, 2, **C_HEAD),
        Node("aux5", "Aux5: feat5 → 128 → 4\n[B,4,200,200]", 4, 2, **C_HEAD),
    )
    d.link("head", "out")

    # main flow inside branches (declared after nodes exist)
    d.link("d1", "d2")
    d.link("d2", "d3")
    d.link("s0", "s3")
    d.link("s3", "s4")
    d.link("s4", "s5")
    d.link("s5", "ctx")

    # auxiliary supervision (one head per semantic feature map)
    d.link("s0", "aux2", label="feat2", dashed=True, color="#B3261E", route="h", arrow=False)
    d.link("s3", "aux3", label="feat3", dashed=True, color="#B3261E", route="h", arrow=False)
    d.link("s4", "aux4", label="feat4", dashed=True, color="#B3261E", route="h", arrow=False)
    d.link("s5", "aux5", label="feat5", dashed=True, color="#B3261E", route="h", arrow=False)
    return d


def build_b0() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2", title="BiSeNetV2 网络架构",
        subtitle="输入 [B,3,200,200] → 输出 [B,4,200,200] ｜ 双分支 + 双边引导聚合（BGA） ｜ "
                 "实线 = 主数据流，红色虚线 = 辅助监督 ｜ 无 ScConv（对应消融臂 B0）",
    )


def build_s0() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv", title="BiSeNetV2 + ScConv 网络架构",
        subtitle="输入 [B,3,200,200] → 输出 [B,4,200,200] ｜ ScConv 挂在 BGA 之后（fused），带残差 scale=0.1 ｜ "
                 "对应消融臂 S0，与旧实验 bisenetv2_scconv 等价",
        fused="residual0.1",
    )


def build_d1() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_D1", title="BiSeNetV2 + ScConv（D1：残差尺度 1.0）",
        subtitle="消融臂 D1 ｜ 唯一变量：SC_CONV_RESIDUAL_SCALE 0.1 → 1.0（模块完全激活） ｜ "
                 "SRU → CRU 结构不变，位置仍为 fused",
        fused="residual1.0",
    )


def build_d2() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_D2", title="BiSeNetV2 + ScConv（D2：无残差直连）",
        subtitle="消融臂 D2 ｜ 唯一变量：去掉残差旁路，ScConv 直接承担 BGA 输出 ｜ "
                 "3 种子确认 test mIoU +0.50%，是唯一有正收益的附加式配置",
        fused="direct",
    )


def build_p1() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_P1", title="BiSeNetV2 + ScConv（P1：只作用于细节分支）",
        subtitle="消融臂 P1 ｜ 唯一变量：SC_CONV_PLACEMENT=('detail',)，ScConv 只细化 Detail 分支输出 25×25×128 ｜ "
                 "带残差 scale=0.1",
        placement=("detail",),
    )


def build_p2() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_P2", title="BiSeNetV2 + ScConv（P2：只作用于语义分支）",
        subtitle="消融臂 P2 ｜ 唯一变量：SC_CONV_PLACEMENT=('semantic',)，ScConv 只细化 Context 输出 7×7×128 ｜ "
                 "带残差 scale=0.1",
        placement=("semantic",),
    )


def build_g1() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_G1", title="BiSeNetV2 + ScConv（G1：修门控尺度塌缩）",
        subtitle="消融臂 G1 ｜ 唯一变量：SC_CONV_GAMMA_NORMALISE=False，SRU 门控用 γ 而非 γ/Σγ，"
                 "动态范围由 ~0.008 恢复到可用区间 ｜ 位置 fused，带残差 0.1",
        fused="residual0.1", gamma_fix=True,
    )


def build_r1() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_R1", title="BiSeNetV2 + ScConv（R1：替换式 · 细节分支）",
        subtitle="消融臂 R1 ｜ 替换式：Detail stage2/stage3 的 4 个重复 3×3 卷积 → ScConv+ReLU，"
                 "通道与分辨率不变；不含附加式 ScConv ｜ 参数量 −294K，GFLOPs −0.59",
        replace_detail=True,
    )


def build_r2() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_R2", title="BiSeNetV2 + ScConv（R2：替换式 · 语义分支）",
        subtitle="消融臂 R2 ｜ 替换式：Semantic 5 个 GELayerS1.conv1 → ScConv+ReLU（stride 8/16/32 三尺度）；"
                 "不含附加式 ScConv ｜ 参数量 −390K，GFLOPs −0.05",
        replace_semantic=True,
    )


def build_r1d2() -> Diagram:
    return build_bisenet(
        name="BiSeNetV2_ScConv_R1D2", title="BiSeNetV2 + ScConv（R1 + D2 组合）",
        subtitle="消融臂 R1D2 ｜ Detail 替换式（同 R1）+ BGA 之后无残差 ScConv（同 D2） ｜ "
                 "检验两个位置的收益是否可叠加",
        fused="direct", replace_detail=True,
    )


BUILDERS = [
    build_unet,
    build_b0,
    build_s0,
    build_d1,
    build_d2,
    build_p1,
    build_p2,
    build_g1,
    build_r1,
    build_r2,
    build_r1d2,
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for builder in BUILDERS:
        d = builder()
        out = OUT_DIR / f"{d.filename}.vsdx"
        write_vsdx(d, out, TEMPLATE)
        pw, ph = d.page_size()
        print(f"{out.name:36s} {out.stat().st_size:7d} B  page {pw:5.2f}×{ph:5.2f} in  "
              f"nodes={len(d.nodes):2d} edges={len(d.edges):2d}")


if __name__ == "__main__":
    main()
