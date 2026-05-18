# 队友说明

设计已定型，100K pilot 验证有效（total=4.87）。下面是当前版本的核心思路、上手命令和可选调整方向。

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
| `total` | 4.5 ~ 5.5+ |
| `priv` | 0.85+ |
| `light` | 0.90+ |
| `eff` | 0.60+ |
| `n_pl` | 4-5 |

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

---

## 进一步实验方向（可选）

如果想给报告加分，几个值得做的：

- **Ablation study**：去掉 bed-first 或某个 reward 因子，对比 eval reward → 证明 mask / factor 的必要性
- **多 seed 实验**：跑 3 个 seed 看 reward 方差
- **Reward landscape audit**：`python reward_audit.py --n 2000 --save plots/audit.png`

都不是必须，看你时间。

---

## 资源

- `my_little_bedroom_spec.md`：完整 MDP / reward 形式化
- `my_little_bedroom.html`：浏览器打开，交互式 reward 预览
- `README.md`：项目总览
