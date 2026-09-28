# 交接说明（2026-09-28，旧机器 RTX 3060 → 新机器 RTX 5080）

新会话先读这份，细节查 `ALGORITHM.md`（算法链路与全部开关）。用中文和用户交流。

## 用户的工作习惯（必须遵守）
- **测量/实验前先讲方案**：要回答的问题、协议、能/不能回答什么、成本，等确认再跑。
- **没有明确的"跑吧/起吧/排上"不要启动实验**；规划语境里的"下面可以跑 X"不是授权。
- 长任务用 `setsid nohup ... &`（会话结束不被杀）；同时最多开几条由机器 CPU/内存决定。
- 不要再提议 LN / QK-norm 这类对 attention 的归一化（实测会放大 attention 的有效学习率，run 13/14 都更差）；
  用户原则："让 attention 慢慢进入策略"——用门控 `attn_gate_init`。
- 不加 μ 上界 `cfm_velocity_bound`（用户判断：flow 会一直把 μ 推到上界然后训崩）。
- 用户关心 token 消耗：回答简洁，别重复读大文件。

## 新机器环境搭建（这些都还没做）
脚本里写死了路径 `/root/.cache/conda/epymarl-main` 和 `/venv/MPE/bin/python`。最省事：
```bash
cd <克隆目录> && git checkout fpo-vpred-bound            # main 分支没有这些代码
mkdir -p /root/.cache/conda && ln -s "$(pwd)" /root/.cache/conda/epymarl-main
uv venv --python 3.10 /venv/MPE                            # 旧机器是 Python 3.10
uv pip install --python /venv/MPE/bin/python -r scripts/env/requirements_mpe_freeze.txt \
   --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
SP=$(/venv/MPE/bin/python -c "import site;print(site.getsitepackages()[0])")
patch -p1 -d "$SP" < scripts/env/gymnasium_robotics_mujoco_multi_getpid.patch   # 必须打，否则多 worker 的
#   ManySegmentSwimmer/Ant 会互删临时 XML（ParseXML: Error opening file）
/venv/MPE/bin/wandb login          # 用户自己粘 key；项目 MAFPO_V0，team adolf9555-university-of-technology-sydney
```
检查：`nvidia-smi`、`nproc`、`cat /sys/fs/cgroup/cpu.max`（Vast 只分一半 CPU/内存）、`free -g`。
每条 run 约占 5~6 GB 内存、约 2 个 CPU 线程（主进程单线程占满 1 核 + 8 个环境 worker）、1~2 GB 显存；
瓶颈是 CPU 和内存，不是 GPU。冒烟：跑一个 `t_max=24000 use_wandb=False` 的短 run，跑完删掉
`results/sacred/.../<n>` 和 `results/models/<name>_*`。换了 GPU/CPU，结果不会和旧机器逐位一致，属正常。

## 算法（一句话）
MAFPO-Gauss：μ = flow 终点（endpoint 参数化，K=5 步 Euler，最后一步系数为 1 所以 μ = g_4），u = μ + n，
a = sigmoid(u)；ratio 是条件在存下的 eps 上的精确高斯比（注意：NCDPO 2025 已有同样做法，不能当贡献）。
**CommFlow** = 在每步积分里做 agent 间 attention（`flow_attention=True`），用门控 `z + α⊙m` 慢慢引入。
MAPPO = `gauss_mu_source=mlp`（同一个 learner）；MAPPO+attention = `mu_attention=True`（在 h 上通信的基线）。

## 主线配置（命令模板，都在 `--config=mafpo_gauss` 下）
```
公共：gauss_sigma_mode=ppo entropy_coef=0.0 sigma_param=exp sigma_init=1.0 seed=<s> save_model=True
FLOW：gauss_mu_source=flow flow_param=endpoint cfm_rollout_steps=5 endpoint_zero_init=False
      endpoint_init_scale=0.01 eps_per_episode=False eps_rho=0.0
CommFlow：FLOW + flow_attention=True flow_attention_heads=4 attn_gate_init=0.01 attn_out_init_scale=1.0
MAFPO：FLOW + flow_attention=False
MAPPO：gauss_mu_source=mlp        MAPPO+attn：gauss_mu_source=mlp mu_attention=True flow_attention_heads=4
      attn_gate_init=0.01 attn_out_init_scale=1.0
Ant：--env-config=mamujoco with env_args.key=mamujoco-Ant-4x2 ...（test_eps_mode=zero）
dispersion：--env-config=vmas（N=4、40 步、共享奖励）+ obs_agent_id=False；flow 类加 test_eps_mode=sample
续训：checkpoint_path="results/models/<目录>" t_max=<新总步数>（结束时必定存一次 checkpoint）
```
回报日志是 N 个 agent 求和：Ant ÷4、HalfCheetah-6x1 ÷6、dispersion ÷4 = 吃到的食物数。

## 已有结果（全是单 seed 或 2 seed）
- **Ant-4x2（lr 1e-3，÷4，最后 1M 均值）**：CommFlow s0 10M 2979 / 15M 4476 / 20M 4303；MAPPO s0 1966 / 1828 / 1600；
  CommFlow s42 10M 1566 / 15M 3382；MAPPO s42 2459 / 2949。→ 15M 两个 seed 都领先，10M 一胜一负。
  CommFlow 1e-3 会先在 1000~1500 **停滞很久**（s0 到约 5M、s42 到约 9M，期间常摔倒、KL 中位数约 0.22）再起飞。
  MAPPO 最好的学习率是 3e-4（s0 10M 2641，只跑到 10M）；CommFlow 3e-4 稳定但 10M 只有 2376。
  门控 α 在训练中被策略压低（约 0.006）。MAFPO 无 attention s0 10M 2737（≈ CommFlow）→ attention 的功劳未证实。
  16M 后续训那条摔倒变多、回报每次测试 ±500 抖动，嫌疑是 reward 标准化减均值抹掉了存活激励（未验证）。
- **HalfCheetah-6x1**：attention 基本空转，带不带持平；只用来证明"不吃亏"。
- **dispersion 试跑（3M）**：所有方法都没学会（没有一局吃完 4 个，低于"抱团依次吃"启发式的 2.35 个）。
  **原因之一是评估方式**：测试时 eps=0、n=0、无 id、agent 不碰撞 → 4 个 agent 永远重叠，结构上只能抱团。
  训练回报（带噪声）：CommFlow 1.12 ≈ MAPPO+attn 1.12 > CommFlow epsEp 0.82 > MAFPO 0.57 > MAPPO 0.32。
  CommFlow（eps 每步）的门控被压到 0.001、attention 实际被关掉，却远好于 MAFPO → 单 seed 差异不可信。

## 正在旧机器上跑（旧机器地址 122.59.250.166，ssh2.vast.ai:22040）
`disp5m_commflow_n4`（test_eps_mode=sample）、`disp5m_mappo_n4`，5M，2026-09-28 16:40 左右启动，约 4~5 小时；
结果在旧机器 `results/sacred/mafpo_gauss/vmas-dispersion/6`、`7`，wandb `DISP5M-*`。

## 下一步（用户还没最终拍板，启动前要确认）
1. **Ant 补齐对照，都到 15M**：MAPPO 3e-4 s0/s42；**MAPPO+attention 1e-3 s0/s42（最关键：CommFlow 必须赢它，
   否则审稿人会说"这不就是 CommNet"）**；MAFPO 无 attention 1e-3 s42；CommFlow 1e-3 第三个 seed。
2. **dispersion**：看 5M 结果再定。事先定好的判读：全都学不会 → 先 N=2 / 加长（唯一可弃用该环境的理由是调整后仍无法学会）；
   大家都能分开 → 收紧（时限 30、N=6）；CommFlow ≈ MAPPO+attn → 收益来自通信本身；**高斯基线行、flow 类不行** →
   用户说这是写文章要攻克的难点，关键对照是 K=1 的 CommFlow（`cfm_rollout_steps=1`）。都要多 seed。
3. CommFlow 1e-3 的停滞期：候选"按 KL 自适应学习率"或 5e-4（未实现）。
4. 其他候选改动（都未实现、都要先问）：`x += α(g−x)` 让各轮都进入 μ（α=0.2 时 eps 残留 0.33）、reward 只除标准差不减均值、
   actor 输入的 obs 归一化按 agent 分开统计。
