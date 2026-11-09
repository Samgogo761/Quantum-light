# A0 层 A：M88-full 探索 / 生产（2026-08-02）

## 用户修订的生产目标

- 探索：`M88-full @ k20`，GH 3×3，扫描 \(r\in\{0.5,1.5,2.5\}\)、\(\theta_s\in\{0,90,180,270\}\)，±N 共享节点
- 生产：`M88-full @ k50`（含 H10；**不用** M88-c20@k40）
- 谐波主结果：H2/H5/H7/H9/H10

## 服务器路径

```text
REPO=/public/home/wangjs/project/New_SBEs/Quantum-light
TB+ = .../CrI3_TB/wannier/models/M88_allI_2p/CrI3_M88_tb.dat
TB- = .../CrI3_M88_tb_mN_Pconj.dat
分区 = part_1
```

## 提交探索

```bash
cd /public/home/wangjs/project/New_SBEs/Quantum-light
python3 deploy/a0_layerA_m88full/make_gh_nodes.py \
  --outdir deploy/a0_layerA_m88full/nodes_gh3 \
  --gh-order 3 --I-bar 1.0e11
mkdir -p runs/a0_m88full_k20
cd runs/a0_m88full_k20
WORKDIR=$PWD NODES_DIR=$PWD/../../deploy/a0_layerA_m88full/nodes_gh3 \
  sbatch ../../deploy/a0_layerA_m88full/sbatch_explore_k20.sh
```

完成后用 `tools/analysis/combine_pmN_nodes.py` 对每个 \((r,\theta)\) 配对 ±N。

## 本地 hBN 验收

```bash
py -3 examples/hBN_model_tb/a0_e2e/run_a0_e2e_hbn.py
# 见 examples/hBN_model_tb/a0_e2e/a0_e2e_report.md
```
