# P0–P2 归档说明

- 所有结果文件均在 `QL-TIP-1D/results/bias_v1/`；完整复数态和大幅原始数组在 `QL-TIP-1D/data/raw/bias_v1/`。`local_artifact_hashes.csv` 逐文件记录大小和 SHA-256，含未入 Git 的 NPZ。
- `spectra_origin.csv` 约 11 MB，超过单文件 2 MiB 默认上限，保留本地、不登记 Git；可由 `results/bias_v1/<run_id>/spectrum_*.csv` 通过 `scripts/export_bias_data.py` 重建。
- 每 run 的 `summary.json`、`config.json`、`provenance.json`、`trace.csv` 和该 run 的谱 CSV 已登记；`data/raw` 的 NPZ 不入 Git，哈希见上表。
- 运行源码哈希记录在每个 `provenance.json`；`configs/generated_bias_v1/` 保存实际输入配置，包括失败 r01 的历史配置。
- 未归档的原始态可用配置、源码哈希和本说明定位；不得用重新运行覆盖已有 run_id。
