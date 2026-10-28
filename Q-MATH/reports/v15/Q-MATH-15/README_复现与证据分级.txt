Q-MATH-15 — 2026-10-22

核心研究文件：Q-MATH-15_数学框架纠错_经典光最优界_共享k验证_研究报告.md

需要 Python 3.10+ / numpy / scipy / mpmath / matplotlib

复现已经验证的数学证书（无需大型矩阵传播）：
    python q15_audit_certificate.py
    python q15_figures.py
    python q15_source_norm_bound.py

复现两个 k 共享同一量子光场的基准（计算耗时较长，存在非收敛警告）：
    python q15_shared_lowfock.py
    python q15_single_pair_lowfock.py
    python q15_null_coupling.py

使用相同的原始 M0 model.py / static_audit / SCF，在 QMATH13_M0_H2H4/deps 下。
绝对不可将当前共享双 k 的 5/7 泵浦 Fock 结果当作收敛量。

主要新证书针对已存储的 23 维、固定单 k、弱辐射源算符，经典正 P 输入使用统一的 4π 相位扫描协议。
实际远场、BZ 积分、无限 Fock K_02 尚未严格认证。代码里 full-Fock 范数界已用生产 M0 几何重算。

扩展审计：
    python q15_duhamel_residual_probe.py
    python q15_duhamel_tail_certificate.py

注意：Duhamel 残差外的额外 Fock 尾部已另用指数级数界估计；实际 23 维 |2> 残差积分~0.046，未达到充分的无限维收敛证书。
