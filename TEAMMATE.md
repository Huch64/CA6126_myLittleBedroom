# RL 训练任务

## 装依赖
```bash
git pull
pip install -r requirements.txt
```

## 跑训练（~3h）
```bash
python train.py --name <你的run名>
```

## 想试不同方向，改这些参数

| 想看 | 改 | 试试 |
|---|---|---|
| 探索强度 | `--ent-coef` | 0.01（默认）/ 0.03 / 0.05 |
| dense reward 兜底 | `--placement-bonus` | 0.1 ~ 0.3 |
| 学习率 | `--lr` | 1e-4（稳）/ 5e-4（快） |
| 训练时长 | `--timesteps` | 200_000 / 1_000_000 |
| seed 方差 | `--seed` | 42 / 7 |

更激进：改 `env.py` 顶部的 reward 常量（`PILLOW_W` / `D_TAU` / `AVAIL_FACTOR` 等）。建议每次只动一个变量，方便归因。

## 看进度
```bash
tensorboard --logdir runs/
```
关注 `eval/mean_reward`，理想是单调爬升。

## 跑完
```bash
python plot_training.py --run <你的run名>
python render.py --episodes 5 --seed 0 \
    --model runs/<你的run名>/best/best_model.zip \
    --save videos/trained.mp4
```

录视频用 seed 0，方便互相对照。

---

## reward 关键改动（context）

旧的 `R = A − D − W`（加减）让 agent 学会"啥也不放 = 0 分"比"乱摆 = 负分"安全，**收敛到不放任何家具**。

改成 **`R = A × comfort × waste_eff`**（乘法 + 比例）：

- 惩罚是"受影响格子的比例"（pillow 被看见 %、窗户被挡 %、空地不可达 %）
- 跨房间大小自动归一
- 数学上 A=0 才 R=0，**只要放东西 R 就 > 0**

完整 spec 见 `my_little_bedroom_spec.md`。验证 reward 健康度可以跑 `python reward_audit.py`。
