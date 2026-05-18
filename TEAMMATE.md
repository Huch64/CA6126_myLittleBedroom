# 队友说明

设计已定型，100K pilot 验证有效（total=4.87）。下面是当前版本的核心思路、上手命令和可选调整方向。

**预期评分参考**（基于 20 分总分）

| 维度 | 满分 | 估计 | 备注 |
|------|------|------|------|
| 🆕 Novelty | 5 | 4-5 | 卧室任务 + reward 4 因子 + factored heads + hard/soft 分离都是亮点 |
| 📐 Formalism | 2 | 2 | spec 完整，MDP 描述清楚 |
| 💻 Environment | 3 | 3 | env 干净，mask + render + verify 全套 |
| 🎬 Showcase | 5 | 4-5 | 取决于视频质量（random vs trained 对比鲜明）|
| 📈 Training process | 5 | 4-5 | reward 设计迭代 + curve + ablation 都是讲料 |
| **总计** | **20** | **17-20** | 多做点 ablation 和报告打磨能稳到 19+ |

---

## 当前版本核心

**Reward**：`R = A × privacy × light × efficiency`

- **硬约束放在 action mask**：必须有床、step 0 强制放床、不越界、功能区合理、DONE 在床放过后才解锁
- **软偏好放在 reward**（连续梯度）：
  - `privacy` = `1 − 0.7 × pillow_ratio` ∈ [0.3, 1]，枕头中心偏离门口的角度
  - `light` = `1 − 0.7 × window_ratio` ∈ [0.3, 1]，窗户被高家具挡的比例
  - `efficiency` = `1 − waste_ratio` ∈ [0, 1]，**不设 floor**，强化空间利用激励

**网络**：Factored output heads（fid/x/y/ori/done 分头预测，合成 41185 联合 logit + mask + softmax）；MLP backbone 128-128；LR linear decay。

**超参**（pilot 验证有效的设置）：`--ent-coef 0.05 --n-steps 1024`

完整细节见 `my_little_bedroom_spec.md`。

---

## 上手

```bash
git pull && pip install -r requirements.txt
python sanity_check.py
```

跑训练（前台，建议 100K 先 pilot 一下）：

```bash
python train.py --name <run_name> --timesteps 100000 --ent-coef 0.05 --n-steps 1024
```

终端会自动打印滚动进度：
```
[live] step= 50000  ep= 9000  total=3.20  priv=0.75  light=0.88  eff=0.60  n_pl=4.5  bed=100%
```

跑完出视频和图：

```bash
python render.py --episodes 5 --seed 0 --save videos/random.mp4
python render.py --episodes 5 --seed 0 --model runs/<run_name>/best/best_model.zip --save videos/trained.mp4
python plot_training.py --run <run_name>
```

---

## 健康指标参考

| 指标 | 终态期望 |
|------|---------|
| `bed` | 100% |
| `total` | 6 ~ 8 |
| `priv` | 0.85+ |
| `light` | 0.90+ |
| `eff` | 0.65+ |
| `n_pl` | 4-5 |

**reward 上限基线**（用 random search + greedy 探索 30 个房间得出）：

| baseline | total mean | 说明 |
|----------|-----------|------|
| 单次 random | 3.2 | 不学习的下限 |
| 100K pilot (我们当前) | 4.9 | 训练 100K 步后 |
| best-of-500 random | **9.9** | 同房间试 500 次取最好（理论近似上限） |
| best-of-3000 random | ~10-11 | 更多采样的上限估计 |

所以训练好的 agent 目标是**逼近 best-of-N**，不是单 random。500K 训练应该能爬到 6-8，多 seed + 调参可能到 8+。

---

## 可调方向

如果对当前结果不满意，或者想自己实验，建议从这些开始：

| 想改 | 改法 | 适用场景 |
|------|------|---------|
| 探索强度 | `--ent-coef 0.03` ~ `0.08` | total 卡住不涨 / 训练剧烈震荡 |
| 学习率 | `--lr 1e-4` / `5e-4` | 默认 3e-4 太激进或太慢 |
| Rollout 长度 | `--n-steps 256` / `2048` | 想更频繁 update / 想更稳定 |
| 训练时长 | `--timesteps 500000` | 想拿更高 final reward |
| Reward 权重 | 改 `env.py` 顶部常量 | 想测试不同设计 |

建议一次只动一个变量，方便归因。

---

## 排错

**训练卡住不涨**：可能 ent_coef 不够，试 0.08。

**`[live]` 不出来**：命令里有 `nohup ... > file 2>&1 &`，输出去了文件。要么直接前台跑（不要 nohup），要么 `tail -f` 那个 file 看。

**停下重跑**：
```bash
pkill -f "train.py.*<run_name>"
rm -rf runs/<run_name>
```

**从 checkpoint 恢复**（万一中途崩了）：

```bash
# 看现有 checkpoint
ls runs/<run_name>/checkpoints/

# 用某个 checkpoint 继续训练（custom 脚本，参考 SB3 API）
python -c "
from sb3_contrib import MaskablePPO
from train import FactoredMaskablePolicy   # 必须加载自定义 policy
model = MaskablePPO.load('runs/<run_name>/checkpoints/model_300000_steps.zip')
# ... 创建 env, 继续 learn
"
```

默认每 50K 步存一份 (`--checkpoint-freq` 调整)。设 `--checkpoint-freq 0` 关掉。

---

## 已知限制（写报告时可以提，也是潜在改进方向）

**1. 床的功能区不受 mask 保护**

```python
# action_masks() 里:
if CATALOG[p.fid].cat == "bed":
    continue   # 床的 zone 故意跳过，允许 nightstand 贴床头
```

后果：其他家具（desk / wardrobe / cabinet）**可以放进床的 zone 里**，理论上可能出现"衣柜挡在床前"这种不合理布局。当前 pilot 实测影响很小（agent 学到了合理布局），但严格说是个设计漏洞。

**2. 床自己的合法性条件过于宽松**

```python
# 当前: partial=True 意思是 3 个 zone 加起来 ≥ 1 个 cell 可用即可
# 更严的版本会要求"至少一个 zone 完整可用"
```

意味着可能允许"床三面顶墙、只剩 1 格空地"这种极端配置。pilot 没看到明显问题，但放任 RL 学贴墙时可能撞到这个。

**3. 部分超参是"魔法数字"**

- `FACTOR_FLOOR = 0.3`（privacy/light 的下限）
- `0.7`（线性 remap 的系数，即 1 − FACTOR_FLOOR）
- `D_PAIR = 4`（nightstand 配对距离衰减）
- `NIGHTSTAND_PAIR_BONUS = 1.0`

都没系统调过，跑通就用了。

**4. 单 seed 训练**

pilot 只跑了 seed=0。如果想报告里加 mean ± std，需要 3-5 个 seed 重跑。

**5. Efficiency 学习慢**

100K pilot 里 efficiency 从 0.58 涨到 0.64，提升不大。500K 可能涨到 0.70+，但贴墙的 emergent 行为需要更多步数。

**6. 跟"近似上限"还有差距**

用 random search 估计上限（同房间试 500 次取最好）：mean ≈ 10。我们 100K pilot 是 4.9——**离上限还有 2 倍 gap**。500K 训练或许能爬到 6-8，但是否能到 9-10 不确定。这是值得探索的方向（更多 training + 调超参）。

---

## 进一步实验方向（可选）

如果想给报告加分，从上面几个限制入手都行。具体推荐：

- **Ablation study**（性价比最高）：去掉 bed-first 或某个 reward 因子，对比 eval reward → 直接证明 mask / factor 的必要性
- **多 seed 实验**：跑 3 个 seed 看 reward 方差，报告里加 error bar
- **修床 zone 保护**：让 desk/wardrobe/cabinet 不能进床 zone（保留 nightstand 例外），看 reward 是否提升
- **Reward landscape audit**：`python reward_audit.py --n 2000 --save plots/audit.png` → 生成 reward 分布图加报告
- **训练 500K**：当前 100K 已经 work，500K 应该能到 5.5-6.0 区间

都不是必须，看你时间。

---

## 资源

- `my_little_bedroom_spec.md`：完整 MDP / reward 形式化
- `my_little_bedroom.html`：浏览器打开，交互式 reward 预览
- `README.md`：项目总览
