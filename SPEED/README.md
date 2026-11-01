# NBS 所有训练模型推理速度评价

脚本默认递归扫描 `G:\NBS\results` 下所有 `best.pth`，从对应 `config.json` 还原每个 run 的模型结构和 ScConv 设置，逐 checkpoint 测速。当前结果目录有多个种子和旧版实验，因此报告会逐 run 列出，不会只取三个代表模型。

在 `G:\NBS` 执行：

```powershell
.\.venv\Scripts\python.exe .\SPEED\speed_benchmark.py
```

默认 CUDA 自动可用则使用 CUDA，输入 `1×3×200×200`、batch=1、FP32；预热 20 次，延迟与吞吐各 50/100 次每轮，重复 3 轮。快速检查可用：

```powershell
.\.venv\Scripts\python.exe .\SPEED\speed_benchmark.py --warmup 1 --latency-iters 2 --throughput-iters 3 --repeats 1
```

可通过 `--device cpu|cuda`、`--height`、`--width`、`--batch-size`、`--amp fp16|bf16`、`--results-dir`、`--output-dir` 调整参数。`models.json` 是手动指定模型时的样例，默认扫描不受它限制。已有测速数据只需重建表图（例如补充配置说明）时，可传 `--refresh-existing .\\SPEED\\RES\\speed_results.json`，不会重新跑模型。

报告和图表输出到 `SPEED\RES`：`推理速度对比.md`、`latency_comparison.png`、`throughput_comparison.png`、`speed_results.csv`、`speed_results.json`。测速只包括模型前向，不含图像预处理和端到端数据传输；报告会列出波动提示及未成功加载的权重。

