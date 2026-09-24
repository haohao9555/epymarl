# MAFPO-Gauss / CommFlow —— 算法链路与开关清单

> 单一事实来源。2026-09-24 整理，所有数字都是本仓库实测，不是引用。
> 改动算法时同步更新这份文档。

---

## 1. 文件

```
src/config/algs/mafpo_gauss.yaml          84 行   全部开关
src/modules/agents/mafpo_gauss_actor.py  274 行   GRU 编码 + flow head + sigma
src/controllers/mafpo_gauss_mac.py       163 行   rollout：采 eps、积分、加噪、squash
src/learners/mafpo_gauss_learner.py      491 行   精确高斯 ratio + PPO + critic + ADER
src/modules/critics/mafpo_gauss_critic.py  32 行   V(s)
src/modules/critics/mafpo_gauss_q_critic.py 33 行  Q(s,a)，仅 ADER 用
src/envs/mamujoco_wrapper.py                     动作 (0,1) → [low,high]，obs/action padding
src/components/obs_normalizer.py                 MAC 与 critic 共用的运行归一化
```

一句话概括：

```
MAPPO    mu = MLP(h)
MAFPO    mu = ODE_K(h, eps)        <- 唯一的区别
```

`u = mu + n`、`a = sigmoid(u)`、PPO clip 全都和 MAPPO 一模一样。
`gauss_mu_source=mlp` 就精确退化成 MAPPO，两者共用同一个 learner/critic/runner。

---

## 2. Rollout（每个环境时间步）

```
obs 归一化 + agent one-hot ──→ fc1 ──→ GRU ──→ h_t          [B, N, 128]
                                                 │
eps [B,N,A] ──────────────── x_0 ────────────────┤
                                                 ↓
        ┌──────── K 步循环，h_t 每步重复喂入 ────────┐
        │  inp = [ h_t , x_i , embed_t(i/K) ]       │  128 + A + 8
        │  z   = relu( vel_fc1(inp) )               │  128
        │  z   = z + MultiHeadAttn(z,z,z)           │  <- agent 间协商（可选）
        │  g   = vel_fc2(z)                         │  A
        │  x_{i+1} = x_i + (g - x_i)/(K - i)        │
        └───────────────────────────────────────────┘
                                                 ↓
                                        x_1 = x_K   [B, N, A]
                       u = x_1 + n,  n ~ N(0, sigma_i^2)
                       a = sigmoid(u) ──仿射──→ [low, high]
```

**形状**：`eps / x_i / g / x_1 / n / u / a` 全是 `[B,N,A]`；只有 `h` 和 `z` 是
`[B,N,128]`，`sigma` 是 `[N,A]`（广播到 B）。

**B 是多少**：rollout 时 B = `batch_size_run` = 8（并行环境数），恒定。
训练时 B = 攒够 `fpo_rollout_timesteps`(2048) 条 transition 所需的 episode 数，
随 ep_len 变：HalfCheetah(ep=1000) B=8；Hopper 早期(ep=22) B=96。
minibatch 再从 `[B*T, N, *]` 里抽 `fpo_minibatch_size`=512 行。

**存进 buffer 的四样**：`z=eps`、`action_raw=x_1`、`action_noise=n`、`actions=a`。
**K 个 g 不存**，训练时用同一个 eps 重跑一遍循环重现。

---

## 3. Flow 内部：三个已验证的结构性事实

### 3.1 x_1 是 ODE 终点，同时是高斯均值

端点参数化下网络输出的 `g` **是位置不是速度**（`vel_*` 这些名字是 velocity 模式的
历史遗留）。速度由 `v = (g - x)/(1 - t)` 推出，Euler 更新 `x += dt*v` 化简成
`x += (g - x)/(K - i)`。两条路实测等价（差异 3e-8）。

### 3.2 末步系数为 1，前 K-1 步的残差贡献**精确为零**

```
x_1 = x_{K-1} + (g_{K-1} - x_{K-1})/1 = g_{K-1}
```

解析：残差通路上 g_i 的系数 = 1/(K-i) * prod_{j>i}(1 - 1/(K-j))，对 i<K-1 全是 0。
实测 |dx_1/dg_i|（K=5）：

```
g_0  0.004073    g_1  0.006873    g_2  0.014037    g_3  0.043882    g_4  1.000000
```

前面几步**只能通过"把 x 递推到某处、作为最后一次 head 调用的输入"**影响 x_1，
而 head 对 x 输入的敏感度实测只有 0.088。

**后果**：
- 监督拟合里 K=1/5/10 几乎无差别（MSE 0.0048 / 0.0053 / 0.0050）
- "K 步 = K 轮协商"站不住：前 4 轮 attention 的输出只有最后一轮的 0.4%~4% 影响
- velocity 参数化没有这个问题（每步系数都是 1/K，梯度均摊）

### 3.3 t 只取 K 个离散值，Fourier 嵌入退化成查找表

K 固定 ⇒ t ∈ {0, 1/K, ..., (K-1)/K}。8 维 `embed_t` 实际上是一张 K 行的固定表。
**我们的 flow 不是连续时间 ODE，是 K 层权重共享、每层带固定身份嵌入的残差网络。**
训练和测试用同一个 K，真 ODE 的性质（改 K 提精度）一次都没用上、也没验证过。

---

## 4. 训练（攒够 2048 条 transition 触发一次）

```
1. critic：GAE(gamma=0.99, lambda=0.95) -> advantages          每个 epoch 重算
2. 重建 h 序列：encode_sequence(_build_inputs_all(batch))，h0=0
                与 rollout 逐位一致（已验证 diff 6e-7）
3. 用 buffer 里存的同一个 eps 重新积分 K 步 -> mu_new
4. u = mu_old + n                                （都是存的，不重算）
5. log r = sum_d [ -0.5((u-mu_new)/s_new)^2 + 0.5((u-mu_old)/s_old)^2 - log(s_new/s_old) ]
   r = exp(clamp(log r, ±cfm_rho_clip))
   loss = -min( r*A , clip(r, 1±0.2)*A )
6. sigma 走 gauss_sigma_mode
7. obs_normalizer 在 train() 结束后才更新（保证 h_new / h_old 同一套统计量）
```

epochs=4，每 epoch `ceil(有效行数/512)` 个 minibatch。

**为什么必须有终端噪声 n**：没有 n 时策略是确定性映射 `(h,eps) -> a`，动作密度
是 N(0,I) 经 flow 推前的分布，要算它得求 K 步网络的雅可比行列式（而且 head 不可逆）。
加了 n 之后，**条件在 eps 上**策略就是精确的 `N(x_1, sigma^2)`，密度闭式。
把 eps 存下来就能条件在它上面 ⇒ ratio 是精确的高斯密度比。
theta 未更新时 ratio 严格 = 1（实测 log_ratio = 0.000000）。

**这个设计的代价**：优化的是 `pi(a|h,eps)` 而不是 `pi(a|h)`。PPO 从不把 flow 当
输运映射看，只问"给定这个 eps，x_1 摆对了吗"。梯度
`E[A*(u-mu)/sigma^2 * dmu/dtheta]` 在固定 eps 下把 mu 拉向"该 eps 下高 advantage
的 u"，**但拿到哪个 u 由 n 决定、n 与 eps 独立** ⇒ 对 n 求平均就是 mode averaging。
想让 mu 依赖 eps，advantage 必须与 eps 相关；想让 advantage 与 eps 相关，mu 必须
先依赖 eps。**先有鸡还是先有蛋**——这是 flow 惰性的根本原因，比任何初始化问题都底层。
`flow_attention` 和 `eps_rho>0` 都是在试图制造 advantage-eps 相关性来打破这个僵局。

---

## 5. 全部开关

| flag | 默认 | 主线实验用 | 作用与证据 |
|---|---|---|---|
| `gauss_mu_source` | `flow` | flow / **mlp** | mlp = MAPPO，共用同一份 learner |
| `flow_param` | `velocity` | **endpoint** | endpoint：网络输出终点 g；velocity：输出速度 v |
| `endpoint_zero_init` | `True` | **False** | 见 §6.1，True 会把 vel_fc1 和 attention 的梯度打成精确 0 |
| `cfm_rollout_steps` (K) | 10 | **5** | 见 §3.2，K 几乎不影响表达能力 |
| `flow_attention` | `False` | **True** | ODE 每步一轮 agent 间 self-attention；执行需要通信，非严格 CTDE |
| `flow_attention_heads` | 4 | 4 | |
| `eps_rho` | `0.0` | 见 §6.2 | eps 的 AR(1) 时间相关。0=每步独立，1=整局固定 |
| `eps_per_episode` | `False` | — | `eps_rho=1.0` 的别名 |
| `test_eps_mode` | `"zero"` | **sample** | zero：test 时 eps=0（评估 x_1(h,0)，轨迹上的任意一点）；sample：eps~N(0,I) 抽一次、n=0 |
| `mu_attention` | `False` | 未用 | mlp 路径上的 agent 间 attention = "会通信的高斯"基线，对照 CommNet/TarMAC |
| `sigma_param` | `sigmoid` | **exp** | exp 是教科书参数化；sigmoid 有界版本来自 macflow，会削弱基线 |
| `sigma_init` | 0.3 | **1.0** | |
| `gauss_sigma_mode` | `ader` | **ppo** | ppo：sigma 进 PPO loss；ader：总熵守恒下重分配；fixed：冻结 |
| `entropy_coef` | 0.0 | 0.0 | 仅 ppo 模式生效。熵项是**求和**，所以每个 log sigma 拿到的梯度恒为 -entropy_coef，与 agent 数无关 |
| `cfm_velocity_bound` | 0.0 | **0（应改）** | >0 时对 head 输出做 tanh 限幅。见 §6.3 |
| `cfm_rho_clip` | 3.0 | 3.0 | log_ratio 的 clamp |
| `use_delta_v_ratio` | False | False | PolicyFlow 式 ratio，已弃用 |
| `fpo_rollout_timesteps` | 2048 | 2048 | 按**步数**攒 batch（不是按 episode 数） |
| `fpo_minibatch_size` | 512 | 512 | |

ADER 相关（`gauss_sigma_mode=ader` 时才构造 Q critic）：
`ader_estimator` / `ader_lr` / `ader_entropy_budget` 等，见 yaml。
注意 **`sigma_mode != ader` 时 metrics 里的 `q_loss/q_mean` 恒为 0**，
那是字典无条件初始化造成的日志噪声，不是计算。

---

## 6. 本轮（2026-09-23/24）的改动与依据

### 6.1 `endpoint_zero_init` 会冻住梯度

同一个回归 loss 上逐层量梯度范数（初始时刻，真 actor）：

```
mu 网络              总梯度      分层
MLP (MAPPO)        2.25e-01    mu_fc1 =1.05e-01   mu_fc2 =1.90e-01
K5 + zero_init     1.11e-01    vel_fc1=0.00e+00   vel_fc2=1.08e-01   attn=0.00e+00
K5 无 zero_init    2.48e-01    vel_fc1=1.14e-01   vel_fc2=2.08e-01
```

机制：`vel_fc2.weight=0` ⇒ 反传乘这个零 ⇒ **vel_fc1 和 attention 梯度恒为 0**
（attention 自己的 out_proj 也是零初始化，双重零）。初始状态下 flow head 退化成
"冻结随机特征上的一个线性层"。

**收益是重复的、代价是独有的**：zero_init 当初为解决 sigmoid 饱和引入，但端点参数化
本身就给 mu≈0（初始 std(mu|h)：velocity 1.0057 / endpoint+zero 0.0000 /
endpoint 无zero **0.0135**）。所以关掉它同时拿到"快启动"和"正常梯度"，不是权衡。

监督拟合验证（同目标、同 lr、同步数）：

```
mlp          MSE 0.00173
K5_zero      MSE 0.00507
K5_nozero    MSE 0.00245     <- 误差减半
```

### 6.2 `eps_rho`：eps 的时间尺度

```
eps_t = rho * eps_{t-1} + sqrt(1 - rho^2) * xi_t,   xi ~ N(0, I)
```

边际**精确保持 N(0,I)**，所以 `pi(a|h,eps)` 和 ratio 一字不动，只改 eps 序列的
时间相关结构。相关时间 tau ≈ 1/(1-rho)。实测自相关严格按 rho 走。

解决的 tradeoff：

| | rho=0（每步独立） | rho=1（整局固定） |
|---|---|---|
| 角色持续性 | 无，t 时刻的分工 t+1 作废 | 整局稳定 |
| 独立 eps 样本 / 次更新 | 8000（HalfCheetah） | **8** |
| 后果 | 目标函数奖励"忽略 eps"的退化解 | eps 梯度标准误放大 ~32 倍 |

`rho=0.99`（tau≈100 步）是中间点：~80 个有效样本 + 100 步角色记忆。**未验证**。

### 6.3 `cfm_velocity_bound` 应该打开（未做）

`mafpo_v0_fixed_hc6x1_10M` 在 8.41M **数值爆炸**：

```
Msteps   ret/6   sigma   mu_max        KL    gradN   mu_shift
  8.30    6161   0.234     7.56    0.0216    1.42     0.0328
  8.41     267   0.236    40.96   34.2084    2.34     0.5953
  8.52     -37   0.260   198.14  14396.02    4.57    18.3448
```

`grad_norm_clip=10` 没救（爆炸时 gradN 只有 4.57，没触发）。问题是 **mu 无上界**：
`cfm_velocity_bound=0` 关掉了 head 输出的 tanh 限幅。正常运行时 mu_abs_max 只有
6~8（Humanoid 最高 15.8），**bound=20 即可，对表达能力无损**。

同期 Humanoid 没爆炸，差别在 sigma：爆炸时 HalfCheetah sigma=0.234（1/sigma^2=18.3），
Humanoid sigma=0.785（1/sigma^2=1.6），**mu 梯度的放大差 11 倍**。

### 6.4 其它

- `mu_attention`：mlp 路径加 agent 间 attention，零初始化 out_proj 所以初始逐位
  等于 MAPPO（实测 max|diff| = 0.00e+00）。这是对照 CommNet/TarMAC 的关键基线，**还没跑过**。
- `gymnasium_robotics` 本地 patch：`mujoco_multi.py` 里 ManySegmentAnt/Swimmer 的
  `.auto.xml` 用固定文件名、用完即删，8 个并行 worker 会互相删对方的文件
  （`ParseXML: Error opening file`）。已给文件名加 `os.getpid()`，备份在
  `mujoco_multi.py.orig`。**重装该包会丢失。**

---

## 7. 已知的实验事实（全部单 seed，除非注明）

```
env                alg     配置                          best   tail   final
HalfCheetah-6x1    MAPPO                                 7986   7220   7961
                   MAPPO   sigma 冻结 0.3                7464   7273   7464
                   MAFPO   end K5 attn                   7646   6887   6590
                   MAFPO   end K5 attn nozero epsEp      6334   2716   4299   <- 8.41M 爆炸
Humanoid-9|8       MAPPO                                 7773   7520   7674
                   MAFPO   end K5 attn nozero epsEp      7263   6968   7123
Hopper-3x1         MAFPO   vel K10 sigma 冻结             981    842    916
                   MAFPO   vel K10 sigma 可学             920    786    898
                   MAFPO   end K5 attn                    608    577    606
                   MAPPO   seed 0                         831     10     10   <- 5.02M 崩，不恢复
                   MAPPO   seed 1, entropy 1e-3           553    194    200   <- 1.84M 掉，卡住
Ant-4x2            MAPPO                                 2896   2475   2794
```

（return 全部已除以 agent 数。本仓库 `common_reward=True, reward_scalarisation=sum`，
所以日志里的原始 return = N × 单智能体口径。）

**Hopper 上 MAPPO 出问题 2/2，MAFPO 0/3**——同环境、同代码、同超参。样本太小，
但这是目前最有可能成为论文卖点的方向（"flow 的 mu 参数化更不容易崩"）。

**flow 活性**（std(mu|h) 对 eps 的 spread / sigma，7.5M checkpoint 实测）：

```
Hopper  velocity K10（旧）            24.4%    |dmu_i/deps_i| 0.181
Hopper  endpoint+zero+attn            0.4%    |dmu_i/deps_i| 0.018
HC6x1   endpoint+zero+attn            0.5%    cross/self 0.561
HC6x1   endpoint+zero 无attn          0.7%    cross/self -0.000（精确 0）
```

attention 的跨 agent 通道是真的（无 attention 对照精确为 0），但整条 flow 在
0.4~0.7% 的量级运作，所以通道传递的信息量约等于零。

---

## 8. 未解决 / 未验证

1. **flow 惰性**（§4 最后一段的鸡生蛋问题）——`eps_rho` 和 attention 是两个尝试，都没验证有效
2. **`cfm_velocity_bound` 没开**，HalfCheetah 已经因此爆炸过一次
3. **K 的意义**——§3.2 显示前 K-1 步几乎不参与，"K 轮协商"的说法目前站不住
4. **连续时间性质没验证**——训练 K=5、推理 K=20 是否一致，没测过
5. **全是单 seed**
6. **缺 `mu_attention` 基线**（Gaussian + h 上通信），这是审稿必问的对照
7. **Hopper 上 MAPPO 为什么崩**——已排除刀尖效应、sigma 导致的梯度失控、熵不足；
   唯一剩下的线索是崩溃瞬间 `critic_grad_norm` 0.055 → 0.837（15 倍），未追查
8. **toy 不可靠**：flow+attention 8 seed 只有 2 个成功（0.851 / 0.949），
   其余卡在 0.242。能力是有的（监督拟合分工正确率 1.000），可靠性没有。
   `figs/toy_coordination.png/json` 是修 RNG bug 之前的产物，四条曲线重合，**勿用**
