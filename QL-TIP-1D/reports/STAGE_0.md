# Stage 0 本地环境与范围审计

日期：2026-11-03。继承用户当日任务书与两段讨论；独立于 Q-MATH 的物理模型和既有研究结果。决定先开展单一有限阱的 CPU 基线，按关卡进入少量800 nm相干光验证。

已读取并索引任务书、两段讨论、两篇 Nature Physics PDF（13页与10页）及背景 Word。Word 含公式，提取时加入 OMML 数学文本；原件未修改。外部资料 SHA-256 见 references/source_inventory.json；论文参数证据与补充材料限制见 docs/literature_audit.md。

实测环境：Windows11；Intel Core i5-14600K，20逻辑线程；可见物理内存16,860,057,600 bytes（约15.70 GiB）。GPU 是 NVIDIA GeForce RTX5060 Ti，nvidia-smi 报告16311 MiB显存、驱动591.86、计算能力12.0。现场 GPU 架构与 [NVIDIA 官方表](https://developer.nvidia.com/cuda/gpus)相符。

默认 Python3.14 没有 NumPy/SciPy。已建立项目 `.venv`，使用 Python3.12.14，安装 NumPy2.5.4、SciPy1.18.1、pandas3.1.0、pytest9.2.0、PyYAML6.0.3；完整锁定清单见 requirements-lock.txt。首次 venv 建立被沙箱临时目录权限阻止，获准后重试成功；没有改动其他科研环境。

注册表检测到 Wolfram15.0.0 和 Origin2019b（9.65.169）；WolframScript 可执行路径存在。未启动 Wolfram 内核，故没有声称许可证或 Mathematica 数值验证通过；本轮选择任务书允许的独立解析方阱检查。Origin 未执行导入测试，交付标准 CSV 供用户导入。

CuPy未安装，PATH中没有nvcc。本轮未执行GPU核函数或FFT，驱动识别正常不等于CUDA/CuPy兼容性已经验证。[CuPy14.2官方安装页](https://docs.cupy.dev/en/stable/install.html)确有 Windows CUDA12.x/13.x wheel；是否安装留待 CPU 实际成本与可验证收益支持。qmsolve仅查阅算法与许可证说明，没有安装或复制其求解代码。

实际命令包括 inspect_inputs.py、audit_environment.py、nvidia-smi 和项目环境安装。原始参数没有为匹配目标能量而调整。科学状态由 Stage1和Stage2报告分别判定，环境安装不算科学验证。
