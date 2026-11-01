"""扫描 NBS/results2中断 下所有 best.pth，逐 checkpoint 测模型推理延迟、吞吐和显存。"""
from __future__ import annotations
import argparse, csv, importlib, json, statistics, sys, time
from contextlib import nullcontext
from pathlib import Path, PosixPath, WindowsPath
from typing import Any
import cv2
import numpy as np
import torch

SPEED_DIR=Path(__file__).resolve().parent
PROJECT_DIR=SPEED_DIR.parent
DEFAULT_RESULTS=PROJECT_DIR/"results2中断"
DEFAULT_OUTPUT=SPEED_DIR/"RES"
for folder in (PROJECT_DIR/"config",PROJECT_DIR/"models",PROJECT_DIR/"scripts"):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
ARM_NAMES={
 "bisenetv2_ablation_B0_baseline":"B0 基线", "bisenetv2_ablation_S0_scconv_fused":"S0 fused 残差",
 "bisenetv2_ablation_D1_scale1":"D1 残差 scale1", "bisenetv2_ablation_D2_direct":"D2 fused 直连",
 "bisenetv2_ablation_G1_gate_fix":"G1 门控修正", "bisenetv2_ablation_P1_detail":"P1 detail 插入",
 "bisenetv2_ablation_P2_semantic":"P2 semantic 插入", "bisenetv2_ablation_P3_detail_semantic":"P3 双分支插入",
 "bisenetv2_repl_R1_detail":"R1 detail 替换", "bisenetv2_repl_R2_semantic":"R2 semantic 替换",
 "bisenetv2_repl_R1_plus_D2":"R1+D2 组合", "bisenetv2_control_C2a_conv1x1":"C2a Conv1x1 对照",
 "bisenetv2_control_C2b_conv3x3":"C2b Conv3x3 对照", "bisenetv2_control_C2c_relu":"C2c ReLU 对照",
 "bisenetv2_p200_B0_baseline":"B0 P200", "bisenetv2_p200_D2_direct":"D2 P200",
 "bisenetv2":"BiSeNetV2", "bisenetv2_config1":"BiSeNetV2 config1",
 "bisenetv2_scconv":"旧版 ScConv", "bisenetv2_scconv_config1":"旧版 ScConv config1",
 "bisenetv2_scconv_config2":"旧版 ScConv config2", "bisenetv2_scconv_config3":"旧版 ScConv config3",
 "unet":"U-Net", "unet_p200":"U-Net P200",
}
CHART_NAMES={
 "bisenetv2_ablation_B0_baseline":"B0", "bisenetv2_ablation_S0_scconv_fused":"S0",
 "bisenetv2_ablation_D1_scale1":"D1", "bisenetv2_ablation_D2_direct":"D2",
 "bisenetv2_ablation_G1_gate_fix":"G1", "bisenetv2_ablation_P1_detail":"P1-detail",
 "bisenetv2_ablation_P2_semantic":"P2-semantic", "bisenetv2_ablation_P3_detail_semantic":"P3",
 "bisenetv2_repl_R1_detail":"R1", "bisenetv2_repl_R2_semantic":"R2",
 "bisenetv2_repl_R1_plus_D2":"R1+D2", "bisenetv2_control_C2a_conv1x1":"C2a",
 "bisenetv2_control_C2b_conv3x3":"C2b", "bisenetv2_control_C2c_relu":"C2c",
 "bisenetv2_p200_B0_baseline":"B0-P200", "bisenetv2_p200_D2_direct":"D2-P200",
 "bisenetv2":"Old-BiSeNet", "bisenetv2_config1":"Old-BiSeNet-C1",
 "bisenetv2_scconv":"Old-ScConv", "bisenetv2_scconv_config1":"Old-ScConv-C1",
 "bisenetv2_scconv_config2":"Old-ScConv-C2", "bisenetv2_scconv_config3":"Old-ScConv-C3",
 "unet":"U-Net", "unet_p200":"U-Net-P200",
}


def discover_checkpoints(results_dir:Path)->list[dict[str,Any]]:
    specs=[]
    for checkpoint in sorted(results_dir.rglob("best.pth")):
        run_dir=checkpoint.parent
        cfg_path=run_dir/"config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"checkpoint 缺少 config.json，无法安全重建模型：{checkpoint}")
        cfg=json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        settings=cfg.get("settings",{})
        effective=cfg.get("effective_arguments",cfg)
        model_label=str(cfg.get("model", "" )).lower()
        if "unet" in model_label or run_dir.parts[-2].lower().startswith("unet"):
            kind="unet"
        elif "USE_SC_CONV" in settings:
            kind="scconv"
        else:
            kind="bisenet"
        run_rel=run_dir.relative_to(results_dir).as_posix()
        parts=run_rel.split("/")
        arm=parts[1] if parts[0].lower()=="第一轮scconv应用" and len(parts)>2 else parts[0]
        display=ARM_NAMES.get(arm,arm)
        chart_name=CHART_NAMES.get(arm,arm.replace("bisenetv2_",""))
        seed=effective.get("seed",effective.get("RANDOM_SEED",effective.get("SEED")))
        if seed is None:
            # 旧配置也会把种子保存在 settings 或目录里。
            seed=settings.get("RANDOM_SEED",settings.get("SEED"))
        display_name=f"{display} s{seed}" if seed is not None else f"{display} {run_dir.name}"
        metrics_path=run_dir/"metrics.json"
        metrics=json.loads(metrics_path.read_text(encoding="utf-8-sig")) if metrics_path.is_file() else {}
        chart_label=f"{chart_name} s{seed}" if seed is not None else f"{chart_name} {run_dir.name}"
        specs.append({"name":display_name,"plot_label":chart_label,"arm":display,"type":kind,"settings":settings,
            "effective_arguments":effective,"checkpoint":checkpoint.resolve(),"run_path":run_rel,
            "seed":seed,"saved_model_variant":metrics.get("model_variant",""),
            "trained_epochs":metrics.get("epochs_trained",effective.get("epochs")),
            "num_classes":effective.get("num_classes",settings.get("NUM_CLASSES",cfg.get("num_classes",4))),
            "in_channels":settings.get("INPUT_CHANNELS",3)})
    if not specs: raise FileNotFoundError(f"在 {results_dir} 下没有找到 best.pth")
    return specs


def settings_for_manual_spec(spec):
    if spec.get("settings") is not None: return spec["settings"]
    cfg=importlib.import_module(spec["config"])
    return {k:v for k,v in vars(cfg).items() if k.isupper()}


def build_model(spec:dict[str,Any])->torch.nn.Module:
    kind=spec["type"].lower()
    settings=settings_for_manual_spec(spec)
    nclass=int(spec.get("num_classes",4)); inch=int(spec.get("in_channels",3))
    if kind=="unet":
        from UNet import build_unet
        return build_unet(in_channels=inch,base_channels=int(settings.get("BASE_CHANNELS",spec.get("base_channels",32))),num_classes=nclass)
    # All BiSeNet-family models are reconstructed from the saved per-run config snapshot,
    # so legacy ablations and current configs use their own channel/ScConv settings.
    base=importlib.import_module("config")
    sys.modules["config"]=base
    common=dict(num_classes=nclass,in_channels=inch,
        auxiliary=bool(settings.get("USE_AUXILIARY_HEADS",True)),
        main_head_channels=int(settings.get("MAIN_HEAD_CHANNELS",1024)),
        detail_channels=settings.get("DETAIL_CHANNELS",base.DETAIL_CHANNELS),
        detail_stage_repeats=settings.get("DETAIL_STAGE_REPEATS",base.DETAIL_STAGE_REPEATS),
        semantic_channels=settings.get("SEMANTIC_CHANNELS",base.SEMANTIC_CHANNELS),
        expansion=int(settings.get("GE_EXPANSION",base.GE_EXPANSION)),
        stage_repeats=settings.get("STAGE_REPEATS",base.STAGE_REPEATS),
        auxiliary_head_channels=int(settings.get("AUXILIARY_HEAD_CHANNELS",128)),
        dropout=float(settings.get("HEAD_DROPOUT",0.1)))
    if kind=="bisenet":
        sys.modules.pop("bisenetv2",None)
        from bisenetv2 import build_bisenetv2
        return build_bisenetv2(**common)
    if kind=="scconv":
        # Import base ScConv config as safe defaults; saved snapshot values take precedence.
        sys.modules["configScConv"]=importlib.import_module("configScConv")
        for name in ("bisenetv2_scconv","bisenetv2"): sys.modules.pop(name,None)
        from bisenetv2_scconv import build_bisenetv2_scconv
        common.update(scconv_enabled=bool(settings.get("USE_SC_CONV",True)),
            scconv_group_num=int(settings.get("SC_CONV_GROUP_NUM",4)),
            scconv_gate_threshold=float(settings.get("SC_CONV_GATE_THRESHOLD",.5)),
            scconv_alpha=float(settings.get("SC_CONV_ALPHA",.5)),
            scconv_squeeze_ratio=int(settings.get("SC_CONV_SQUEEZE_RATIO",2)),
            scconv_group_size=int(settings.get("SC_CONV_GROUP_SIZE",2)),
            scconv_group_kernel_size=int(settings.get("SC_CONV_GROUP_KERNEL_SIZE",3)),
            scconv_gamma_normalise=bool(settings.get("SC_CONV_GAMMA_NORMALISE",True)),
            scconv_placement=settings.get("SC_CONV_PLACEMENT",("fused",)),
            scconv_residual=bool(settings.get("SC_CONV_RESIDUAL",True)),
            scconv_residual_scale=float(settings.get("SC_CONV_RESIDUAL_SCALE",.1)),
            scconv_replace=settings.get("SC_CONV_REPLACE",()),
            scconv_replace_keep_relu=bool(settings.get("SC_CONV_REPLACE_KEEP_RELU",True)),
            scconv_fused_block=settings.get("SC_CONV_FUSED_BLOCK","scconv"))
        return build_bisenetv2_scconv(**common)
    raise ValueError(f"不支持模型类型：{kind}")


def load_model(spec,device):
    model=build_model(spec)
    with torch.serialization.safe_globals([WindowsPath,PosixPath]):
        ckpt=torch.load(spec["checkpoint"],map_location="cpu",weights_only=True)
    state=ckpt.get("model_state",ckpt) if isinstance(ckpt,dict) else ckpt
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return model,ckpt if isinstance(ckpt,dict) else {}


def sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)


def amp_context(device,amp):
    if device.type!="cuda" or amp=="off": return nullcontext()
    dtype=torch.float16 if amp=="fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda",dtype=dtype)


def describe_mode(spec):
    if spec["type"]=="unet": return "U-Net 标准结构"
    if spec["type"]=="bisenet": return "BiSeNetV2（未启用 ScConv）"
    s=spec.get("settings",{})
    if not bool(s.get("USE_SC_CONV",True)): return "ScConv 关闭（BiSeNetV2 基线）"
    placement=s.get("SC_CONV_PLACEMENT",["fused"])
    replace=s.get("SC_CONV_REPLACE",[])
    if isinstance(placement,str): placement=[placement]
    if isinstance(replace,str): replace=[replace]
    parts=[]
    if placement:
        block=str(s.get("SC_CONV_FUSED_BLOCK","scconv"))
        where="/".join(placement)
        if "fused" in placement and block!="scconv": where += f"（fused={block}）"
        if bool(s.get("SC_CONV_RESIDUAL",True)):
            where += f"，残差 scale={float(s.get('SC_CONV_RESIDUAL_SCALE',.1)):g}"
        else: where += "，无残差"
        parts.append("附加 " + where)
    if replace:
        parts.append("替换 " + ",".join(replace) + ("（保留 ReLU）" if s.get("SC_CONV_REPLACE_KEEP_RELU",True) else ""))
    if not parts: parts.append("ScConv 未配置插入/替换位置")
    if not bool(s.get("SC_CONV_GAMMA_NORMALISE",True)): parts.append("SRU gamma 归一化关闭")
    return "；".join(parts)


def benchmark(spec,device,args):
    model,ckpt=load_model(spec,device)
    batch=args.batch_size
    x=torch.randn(batch,int(spec.get("in_channels",3)),args.height,args.width,device=device)
    if device.type=="cuda":
        x=x.contiguous(memory_format=torch.channels_last); model=model.to(memory_format=torch.channels_last)
    def forward():
        with torch.inference_mode(),amp_context(device,args.amp):
            y=model(x)
            if isinstance(y,dict): y=y.get("out",next(iter(y.values())))
            if not torch.is_tensor(y) or y.ndim!=4: raise RuntimeError("输出不是 [N,C,H,W] logits")
        return y
    for _ in range(args.warmup): y=forward()
    sync(device); lat=[]
    if device.type=="cuda":
        for _ in range(args.repeats):
            events=[]
            for _ in range(args.latency_iters):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); y=forward(); end.record(); events.append((start,end))
            sync(device); lat.extend(a.elapsed_time(b)/batch for a,b in events)
    else:
        for _ in range(args.repeats):
            for _ in range(args.latency_iters):
                t=time.perf_counter(); y=forward(); lat.append((time.perf_counter()-t)*1000/batch)
    through=[]
    for _ in range(args.repeats):
        sync(device); t=time.perf_counter()
        for _ in range(args.throughput_iters): y=forward()
        sync(device); through.append(batch*args.throughput_iters/(time.perf_counter()-t))
    peak=None
    if device.type=="cuda":
        del y; sync(device); baseline=torch.cuda.memory_allocated(device); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(min(20,max(1,args.latency_iters))): y=forward()
        sync(device); peak=max(0,torch.cuda.max_memory_allocated(device)-baseline)/(1024**2); del y
    row={"name":spec["name"],"plot_label":spec.get("plot_label",spec["name"]),"arm":spec["arm"],"application_mode":describe_mode(spec),"type":spec["type"],"run_path":spec["run_path"],
        "seed":spec.get("seed"),"trained_epochs":spec.get("trained_epochs"),"saved_model_variant":spec.get("saved_model_variant",""),
        "config":spec.get("config","snapshot"),"checkpoint":str(spec["checkpoint"]),"checkpoint_epoch":ckpt.get("epoch"),
        "input_shape":[batch,int(spec.get("in_channels",3)),args.height,args.width],"output_shape":list(y.shape) if device.type!="cuda" else None,
        "params":sum(p.numel() for p in model.parameters()),"latency_ms_mean":statistics.mean(lat),
        "latency_ms_p50":float(np.percentile(lat,50)),"latency_ms_p95":float(np.percentile(lat,95)),
        "throughput_images_s":statistics.mean(through),"throughput_std_images_s":statistics.stdev(through) if len(through)>1 else 0.0,
        "peak_extra_gpu_memory_mb":peak,"amp":args.amp}
    if row["output_shape"] is None:
        # y was released after memory measurement; infer expected shape without another device allocation.
        row["output_shape"]=[batch,int(spec.get("num_classes",4)),args.height,args.width]
    del model,x
    if device.type=="cuda": torch.cuda.empty_cache()
    return row


def draw_chart(rows,key,title,unit,path,higher=False):
    width,height=1500,max(700,150+len(rows)*72)
    image=np.full((height,width,3),(250,250,250),np.uint8)
    cv2.putText(image,title,(45,55),cv2.FONT_HERSHEY_SIMPLEX,1.0,(35,40,45),2,cv2.LINE_AA)
    left,right,top,bottom=380,width-90,100,height-60
    values=[float(r[key]) for r in rows]; maximum=max(values) if values else 1.0; axis=maximum*1.2 if maximum>0 else 1
    for tick in range(6):
        xx=left+round((right-left)*tick/5); value=axis*tick/5
        cv2.line(image,(xx,top),(xx,bottom),(220,222,225),1,cv2.LINE_AA)
        cv2.putText(image,f"{value:.1f}",(xx-18,bottom+25),cv2.FONT_HERSHEY_SIMPLEX,.48,(90,95,100),1,cv2.LINE_AA)
    colors=[(205,120,35),(45,155,85),(190,90,150),(65,145,210),(100,100,195)]
    step=(bottom-top)/max(1,len(rows))
    for i,(row,value) in enumerate(zip(rows,values)):
        ym=int(top+(i+.5)*step); end=left+int((right-left)*value/axis)
        cv2.putText(image,row.get("plot_label",row["name"])[:40],(left-325,ym+6),cv2.FONT_HERSHEY_SIMPLEX,.47,(40,45,50),1,cv2.LINE_AA)
        cv2.rectangle(image,(left,int(ym-step*.23)),(end,int(ym+step*.23)),colors[i%len(colors)],-1)
        label=f"{value:.3f}" if key=="latency_ms_mean" else f"{value:.1f}"
        cv2.putText(image,label,(min(end+8,width-90),ym+6),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,35,40),1,cv2.LINE_AA)
    cv2.putText(image,unit,(left,height-15),cv2.FONT_HERSHEY_SIMPLEX,.52,(70,75,80),1,cv2.LINE_AA)
    cv2.putText(image,"Higher is better" if higher else "Lower is better",(width-275,52),cv2.FONT_HERSHEY_SIMPLEX,.48,(90,95,100),1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),image): raise OSError(f"无法写图：{path}")


def save_outputs(rows,failures,discovered,device,args):
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    csv_path=out/"speed_results.csv"; json_path=out/"speed_results.json"; md_path=out/"推理速度对比.md"
    fields=["name","plot_label","arm","application_mode","type","run_path","seed","trained_epochs","saved_model_variant","checkpoint_epoch","input_shape","latency_ms_mean","latency_ms_p50","latency_ms_p95","throughput_images_s","throughput_std_images_s","params","peak_extra_gpu_memory_mb","amp","checkpoint"]
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],list) else r[k] for k in fields})
    meta={"created_at":time.strftime("%Y-%m-%d %H:%M:%S"),"device":str(device),"gpu_name":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "pytorch":torch.__version__,"cuda":torch.version.cuda,"results_root":str(args.results_dir),"discovered_checkpoints":discovered,
        "completed_checkpoints":len(rows),"failed_checkpoints":failures,"height":args.height,"width":args.width,"batch_size":args.batch_size,
        "warmup":args.warmup,"latency_iters_per_repeat":args.latency_iters,"throughput_iters_per_repeat":args.throughput_iters,
        "repeats":args.repeats,"amp":args.amp,"results2中断":rows}
    json_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    draw_chart(rows,"latency_ms_mean","Inference latency (all checkpoints)","milliseconds / image",out/"latency_comparison.png")
    draw_chart(rows,"throughput_images_s","Inference throughput (all checkpoints)","images / second",out/"throughput_comparison.png",True)
    unstable=[r for r in rows if (r["throughput_images_s"]>0 and r["throughput_std_images_s"]/r["throughput_images_s"]>.10) or (r["latency_ms_p50"]>0 and r["latency_ms_p95"]/r["latency_ms_p50"]>1.5)]
    lines=["# 所有已训练模型推理速度对比","",f"> 生成时间：{meta['created_at']}  ",f"> 扫描到 {discovered} 个 checkpoint，成功测速 {len(rows)} 个；设备：`{device}`{' / '+torch.cuda.get_device_name(device) if device.type=='cuda' else ''}；PyTorch：`{torch.__version__}`；CUDA：`{torch.version.cuda or '不可用'}`。  ",f"> 输入：`{args.batch_size}×3×{args.height}×{args.width}`；AMP：`{args.amp}`；报告按每个 run 的 `best.pth` 分别列出。","","## 逐模型结果","","| 实验臂/模型 | ScConv应用方式 | 种子 | run 目录 | best epoch | 平均延迟 (ms/张) | P50 | P95 | 吞吐 (张/秒，均值±SD) | 参数量 | 峰值额外显存 (MiB) |","|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        mem="—" if r["peak_extra_gpu_memory_mb"] is None else f"{r['peak_extra_gpu_memory_mb']:.1f}"
        lines.append(f"| {r['arm']} | {r['application_mode']} | {r['seed'] if r['seed'] is not None else '—'} | `{r['run_path']}` | {r['checkpoint_epoch'] if r['checkpoint_epoch'] is not None else '—'} | {r['latency_ms_mean']:.3f} | {r['latency_ms_p50']:.3f} | {r['latency_ms_p95']:.3f} | {r['throughput_images_s']:.2f} ± {r['throughput_std_images_s']:.2f} | {r['params']:,} | {mem} |")
    lines += ["","![所有 checkpoint 延迟对比](latency_comparison.png)","","![所有 checkpoint 吞吐对比](throughput_comparison.png)","","## 测量设置","",f"- 对 `results2中断` 下每个 `best.pth` 单独重建网络并严格加载其对应权重；结构参数优先读取同目录 `config.json` 快照，因此 ScConv 的插入位置、残差、门控和替换方式均按该次训练配置还原。",
        f"- 输入固定为 {args.batch_size}×3×{args.height}×{args.width}，预热 {args.warmup} 次。延迟每轮 {args.latency_iters} 次、共 {args.repeats} 轮；吞吐每轮 {args.throughput_iters} 次、共 {args.repeats} 轮。",
        "- CUDA 延迟使用 CUDA Event，CPU 延迟使用高精度计时；吞吐使用同步后的批量墙钟时间。GPU 默认 channels-last，所有模型设置一致。",
        "- 显存为模型加载后推理前向期间的额外峰值分配量，不是整个进程或设备总占用；CPU 模式不提供显存值。",
        "- 只测设备驻留输入上的网络前向，不含图像读取、预处理、CPU-GPU 传输、argmax/后处理。速度受设备负载、温度、功耗策略和其他进程影响。","",
        "## 结果稳定性",""]
    if unstable:
        lines.append("重复波动超过提示阈值（吞吐 CV >10% 或 P95/P50 >1.5）的模型，建议空闲 GPU 后复测，不要直接作为正式论文速度结论：")
        for r in unstable:
            cv=100*r["throughput_std_images_s"]/max(r["throughput_images_s"],1e-12); ratio=r["latency_ms_p95"]/max(r["latency_ms_p50"],1e-12)
            lines.append(f"- `{r['run_path']}`：吞吐 CV={cv:.1f}%，P95/P50={ratio:.2f}")
    else: lines.append("所有 run 均未触发波动提示阈值；正式部署仍应在目标设备复测。")
    if failures:
        lines += ["","## 未能加载/测速的 checkpoint",""]
        lines.extend([f"- `{f['run_path']}`：{f['error']}" for f in failures])
    lines += ["","## 输出文件","","- `speed_results.csv`：每个 checkpoint 一行的测速表。","- `speed_results.json`：结果、配置来源及环境参数。","- `latency_comparison.png`、`throughput_comparison.png`：本报告引用的逐 checkpoint 对比图。","","> 不同种子 checkpoint 结构相同，速度理论上应接近；逐 run 测量主要用于完整覆盖结果目录和发现异常。正式比较时应在同一设备、空闲负载下重复测量。数据为模型前向速度，不等于端到端应用延迟。"]
    md_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Markdown: {md_path}\nCSV: {csv_path}\nCharts: {out/'latency_comparison.png'}, {out/'throughput_comparison.png'}")


def main():
    p=argparse.ArgumentParser(description="逐个评价 results2中断 下所有训练完成的模型 checkpoint")
    p.add_argument("--results2中断-dir",type=Path,default=DEFAULT_RESULTS,help="结果目录，递归发现 best.pth")
    p.add_argument("--specs",type=Path,default=None,help="可选：手动模型清单 JSON；不传则自动扫描全部 best.pth")
    p.add_argument("--refresh-existing",type=Path,default=None,help="保留已有测速数值，仅按当前配置快照重新生成表格/图表")
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--height",type=int,default=200); p.add_argument("--width",type=int,default=200); p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--warmup",type=int,default=20); p.add_argument("--latency-iters",type=int,default=50); p.add_argument("--throughput-iters",type=int,default=100); p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--amp",choices=("off","fp16","bf16"),default="off"); a=p.parse_args()
    if min(a.height,a.width,a.batch_size,a.latency_iters,a.throughput_iters,a.repeats)<1 or a.warmup<0: p.error("尺寸、batch、迭代数/重复次数须为正，warmup 不得为负")
    if a.device=="cuda" or (a.device=="auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available(): p.error("当前 PyTorch 不支持 CUDA")
        device=torch.device("cuda")
    else: device=torch.device("cpu")
    if device.type=="cpu": torch.set_num_threads(max(1,min(torch.get_num_threads(),8)))
    if a.refresh_existing:
        meta=json.loads(a.refresh_existing.read_text(encoding="utf-8-sig"))
        specs=discover_checkpoints(a.results_dir)
        by_run={s["run_path"]:s for s in specs}
        rows=meta["results2中断"]
        for row in rows:
            if row["run_path"] in by_run:
                row["application_mode"]=describe_mode(by_run[row["run_path"]])
                row["plot_label"]=by_run[row["run_path"]]["plot_label"]
        a.height=meta["height"]; a.width=meta["width"]; a.batch_size=meta["batch_size"]
        a.warmup=meta["warmup"]; a.latency_iters=meta["latency_iters_per_repeat"]
        a.throughput_iters=meta["throughput_iters_per_repeat"]; a.repeats=meta["repeats"]; a.amp=meta["amp"]
        device=torch.device(meta["device"])
        save_outputs(rows,meta.get("failed_checkpoints",[]),meta.get("discovered_checkpoints",len(specs)),device,a)
        return
    if a.specs:
        data=json.loads(a.specs.read_text(encoding="utf-8-sig")); specs=data.get("models",data) if isinstance(data,dict) else data
    else: specs=discover_checkpoints(a.results_dir)
    for index,spec in enumerate(specs,1):
        spec.setdefault("arm",spec.get("name","自定义模型"))
        spec.setdefault("run_path",str(spec.get("checkpoint","custom")))
        spec.setdefault("plot_label",spec.get("chart_label",f"Model {index}"))
        checkpoint=Path(spec["checkpoint"])
        if not checkpoint.is_absolute(): spec["checkpoint"]=(PROJECT_DIR/checkpoint).resolve()
        spec["settings"]=settings_for_manual_spec(spec)
    rows=[]; failures=[]
    for i,spec in enumerate(specs,1):
        try:
            print(f"[{i}/{len(specs)}] {spec['name']} | {spec['checkpoint']}",flush=True)
            row=benchmark(spec,device,a); rows.append(row)
            print(f"    {row['latency_ms_mean']:.3f} ms/张；{row['throughput_images_s']:.2f} 张/秒",flush=True)
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            failures.append({"run_path":spec.get("run_path",str(spec.get("checkpoint",""))),"error":error})
            print(f"    失败：{error}",file=sys.stderr,flush=True)
    if not rows: raise RuntimeError("没有任何 checkpoint 成功完成测速")
    save_outputs(rows,failures,len(specs),device,a)
    if failures: print(f"注意：{len(failures)}/{len(specs)} 个 checkpoint 失败，详见报告。",file=sys.stderr)

if __name__=="__main__": main()
