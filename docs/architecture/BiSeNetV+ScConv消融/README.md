# NBS 模型架构图（Visio .vsdx）

本目录收录项目全部 11 个模型架构的 Visio 图，均为 `.vsdx`（Visio 2013+ 格式），可直接双击用
Microsoft Visio 打开、编辑、导出 PDF/PNG。图形为矢量形状 + 动态连接线，文字、配色、框体都可继续修改。

## 文件清单

| 文件 | 对应模型 | 关键结构 / 差异 | 页面尺寸 (in) |
| --- | --- | --- | --- |
| `U-Net.vsdx` | 旧实验 U-Net | 编码 32→512、4 次池化 + 4 次双线性上采样、4 条跳跃连接、1×1 输出 4 类 | 11.0 × 11.7 |
| `BiSeNetV2.vsdx` | 旧实验 BiSeNetV2 / 消融臂 **B0** | 双分支 + BGA，无 ScConv，4 个辅助头 | 13.4 × 18.8 |
| `BiSeNetV2_ScConv.vsdx` | 旧实验 BiSeNetV2+ScConv / 消融臂 **S0** | ScConv 挂 BGA 之后（fused），残差 scale=0.1 | 13.4 × 20.7 |
| `BiSeNetV2_ScConv_D1.vsdx` | 消融臂 D1 | 唯一变量：残差尺度 0.1 → **1.0** | 13.4 × 20.7 |
| `BiSeNetV2_ScConv_D2.vsdx` | 消融臂 D2 | 唯一变量：**去掉残差**，ScConv 直连 BGA 输出 | 13.4 × 20.7 |
| `BiSeNetV2_ScConv_P1.vsdx` | 消融臂 P1 | 唯一变量：ScConv 只作用于**细节分支**输出 | 13.6 × 20.5 |
| `BiSeNetV2_ScConv_P2.vsdx` | 消融臂 P2 | 唯一变量：ScConv 只作用于**语义分支** Context 输出 | 13.7 × 20.5 |
| `BiSeNetV2_ScConv_G1.vsdx` | 消融臂 G1 | 唯一变量：SRU 门控用 **γ** 而非 γ/Σγ（修尺度塌缩） | 13.4 × 20.7 |
| `BiSeNetV2_ScConv_R1.vsdx` | 消融臂 R1 | **替换式**：Detail stage2/3 的 4 个重复 3×3 → ScConv+ReLU | 15.6 × 19.3 |
| `BiSeNetV2_ScConv_R2.vsdx` | 消融臂 R2 | **替换式**：Semantic 5 个 GELayerS1.conv1 → ScConv+ReLU | 16.6 × 19.3 |
| `BiSeNetV2_ScConv_R1D2.vsdx` | 消融臂 R1D2 | R1 替换式 + D2 无残差（组合臂） | 15.6 × 21.2 |

## 消融臂与文件映射

`../../../results[归档]/results2中断` 中的 10 个 arm 全部覆盖：

| arm | 架构 | 文件 |
| --- | --- | --- |
| B0 | 原版 BiSeNetV2 | `BiSeNetV2.vsdx` |
| S0 | ScConv 挂 fused + 残差 0.1 | `BiSeNetV2_ScConv.vsdx` |
| D1 | 残差尺度 1.0 | `BiSeNetV2_ScConv_D1.vsdx` |
| D2 | 无残差直连 | `BiSeNetV2_ScConv_D2.vsdx` |
| P1 | 只作用于 detail | `BiSeNetV2_ScConv_P1.vsdx` |
| P2 | 只作用于 semantic | `BiSeNetV2_ScConv_P2.vsdx` |
| G1 | 门控去 γ/Σγ | `BiSeNetV2_ScConv_G1.vsdx` |
| R1 | detail 替换式 | `BiSeNetV2_ScConv_R1.vsdx` |
| R2 | semantic 替换式 | `BiSeNetV2_ScConv_R2.vsdx` |
| R1D2 | R1 + D2 | `BiSeNetV2_ScConv_R1D2.vsdx` |

B0 与旧实验 `bisenetv2`、S0 与旧实验 `bisenetv2_scconv` 的**架构完全相同**（差别只在训练协议：
确定性算法、cosine + warmup、90 轮），因此共用同一张架构图，未另存重复文件。若需要按 arm 命名的
独立副本，改 `_build/models_defs.py` 中的 `filename` 后重跑即可。

## 图例

- **实线箭头**：主数据流；**紫色虚线**：跳跃连接（U-Net skip）；**红色虚线**：辅助分割头监督（仅训练）。
- 每个框内文字依次为：层/模块名 → 关键超参（通道、卷积核、步幅、填充、扩展比等）→ 输出张量形状。
- 连线上标注该张量的形状，格式 `[B, C, H, W]`；U-Net 中 `[B, ...]` 的 B 为 batch。
- 配色：黄 = 细节分支 / 编码器，绿 = 语义分支，紫 = BGA 融合，青 = ScConv，红 = 分割头，蓝 = 输入输出。

## 统一设定

- 输入 `[B, 3, 200, 200]`（NEU-Seg，200×200），输出 `[B, 4, 200, 200]`（背景 / 夹杂 / 斑块 / 划痕）。
- BiSeNetV2 两侧特征在 BGA 前对齐到 25×25×128；辅助头分别接 feat2/3/4/5（16/32/64/128 通道）。
- U-Net 使用 `base_channels=32`、`num_classes=4`（与 `scripts/train_unet.py` 调用一致）。

## 重新生成

图由脚本生成，改结构或改文案后可直接重跑：

```powershell
& G:\NBS\.venv\Scripts\python.exe G:\NBS\docs\architecture\_build\models_defs.py
```

- `_build/vsdxgen.py`：VSDX 写入器（复用 `ChatGpt/BiSeNetV2.vsdx` 的 OPC 骨架，重绘页面）。
- `_build/models_defs.py`：11 个模型的节点/连线定义与文案。
- `preview/`：由 vsdx 渲染的 PNG 速览（Visio 导出），仅便于快速查看或插图；正式编辑请用 `.vsdx`。
- `ChatGpt/`：早期版本的同名图，保留未动，可对照。
