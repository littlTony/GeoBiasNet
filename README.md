# GeoBiasNet 原始 reshape 复现版

本项目用于重新验证原实现的问题数据流，训练代码只保留最终模型、H3.6M 训练、验证、测试及必需依赖。精简代码包不包含消融模型、递归记忆实验、数据或训练产物；GitHub 仓库原有的发布权重和可视化资源保留为历史材料。

## 有意保留的原始行为

模型类来自 [公开源码提交 52db282](https://github.com/littlTony491/GeoBiasNet/tree/52db2828d00f97419c086cd4a0fe5489ba07d291)。模型类和实际调用的基础层，其计算代码保持原样；仅删除不需要的导入、演示入口和未使用的类。`provenance.json` 记录了来源和 AST 校验摘要。

下面这行完整保留，**没有改成 view 或 permute**：

```python
gt_spatial = gt.reshape(bs, self.in_features // self.node_n, self.node_n, -1)
```

训练、验证、测试均将完整 `[B, input_n + output_n, 66]` 传入模型；没有在入口截断未来帧。因此，原先从错误布局读取未来 GT 的路径仍然存在。测试中“未来输入梯度非零”是预期结果，表示复现版没有意外修复问题。

同时保留公开模型的 DCT 系数与 GT 相减、未经保护的融合权重归一化，以及原 GraphConvolution 未使用的 bias。这份代码用于复核原行为。

## 来源边界

- 最终模型类、七个实际使用的基础层定义与公开版 AST 一致。
- H3.6M 数据集类、读取/采样、骨架和坐标转换函数，从整理前的归档按定义提取，计算代码未修改。
- 公开仓库没有包含其调用的 `main_h36m_ddp_long.py`。本目录训练入口按本地归档 `V2/v2_long_co_h36m.py` 的五输出监督逻辑适配；模型调用从历史的额外参数形式改为公开模型实际接受的 `model(full_input)`。并非逐字恢复缺失的 DDP 文件。
- 公开仓库和归档中均未找到与发布权重对应的原始四维 MLP 源文件。本目录复用上一版按调用方式和权重形状恢复的 XYZ/时间 MLP。发布权重可严格加载，但这不能证明缺失文件的所有历史细节均已复原。

训练目标保持为最终输出和四级输出的五个 MPJPE 之和，每项覆盖历史和未来的完整序列；没有改用修正版的“仅最终输出、仅未来帧”损失。

## 使用环境和数据

从仓库克隆后，在项目根目录运行。本机可复用已有 `barnet-train` 环境：

```powershell
git clone https://github.com/littlTony491/GeoBiasNet.git
cd GeoBiasNet
conda activate barnet-train
```

独立安装可使用 `environment.yml` 和 `requirements.txt`；运行依赖只有 PyTorch、NumPy。已验证 Python 3.10.22、PyTorch 2.11.0+cu128、NumPy 2.2.6。

```powershell
conda env create -f environment.yml
conda activate geobiasnet-original-reshape
```

数据优先读取本目录 `dataset/h3.6m`，本机也会自动找到相邻 `BARNet/dataset/h3.6m`。无需复制现有数据；其他机器使用 `--data-dir` 指定包含 S1/S5/S6/S7/S8/S9/S11 的 H3.6M 目录。

## 训练与测试

默认配置对应公开长时预测设置：10→50 帧，DCT 长度 60，宽度 16，批量 16，50 轮，初始学习率 0.005。学习率按归档的 `0.1 ** (1 / epochs)` 在每轮开始前衰减，包括第 1 轮；梯度裁剪阈值保留为 10000。

```powershell
python train.py --device cuda:0 --output-n 50 --output-dir checkpoints/h36m_original_reshape_10in50out
python test.py --device cuda:0 --output-n 50 --checkpoint checkpoints/h36m_original_reshape_10in50out/best.pt --output-dir results/h36m_original_reshape_10in50out
```

使用仓库保留的原发布权重进行测试：

```powershell
python test.py --device cuda:0 --output-n 50 --checkpoint checkpoint/ckpt_best.pth.tar --output-dir results/published_checkpoint
```

本项目的有效入口为 `train.py` 和 `test.py`，它们替代旧的 `main_h36m_test.py` 与指向缺失 DDP 文件的 `h36m_test_long.sh`。

需要与当前 10→25 帧任务核对时，两条命令均改用 `--output-n 25`，并使用新的输出目录。模型始终是同一个最终模型，没有消融选择开关。

训练每轮执行训练、验证和按动作测试，并写入 `history.jsonl`、逐时刻指标、`last.pt`、`best.pt` 和编号检查点；独立测试生成 `test_metrics.json` 和 `test_metrics.csv`。检查点带有复现版标记，防止与修正版的权重混用。测试入口也接受原公开权重；恢复训练要求使用本复现版生成的完整检查点。

恢复同一任务时保持原参数和计划轮数，例如：

```powershell
python train.py --device cuda:0 --output-n 50 --epochs 50 --resume latest --output-dir checkpoints/h36m_original_reshape_10in50out
```

`--max-batches 1`、`--actions walking`、`--skip-rate 1000` 仅用于短流程检查，正式验证保持默认的完整数据和全部动作。原采样函数也保持原样，建议使用默认 `--test-sample-num -1` 或历史使用的 256。

## 原训练划分与指标口径

- 训练：S1/S6/S7/S8/S9。
- 默认验证：`--validation-split 2`，即 S5，保留归档训练脚本的选择；验证数据打乱顺序。
- 测试：S5，按 15 个动作分别计算再等权平均。验证和测试主体相同也是此原始流程的行为。需要明确使用独立 S11 验证时，可指定 `--validation-split 1`，它不会修复 reshape。
- 指标：未来帧的 32 关节 MPJPE，单位毫米。保留公开脚本的有效关节写回、重复关节补全，以及其余坐标沿用 GT 的处理。训练/验证是样本平均，测试是动作平均。

当前修正版使用不同的损失、验证划分和 22 关节统计口径，因此两份程序的数值差异不能全部归因于 reshape。若要做单变量对照，需要另外统一这些条件。

## 已执行的验证

```powershell
python -m unittest discover -s tests -v
```

8 项测试通过，覆盖原计算代码不变、完整序列输入、五输出监督、原 32 关节指标、未来 GT 梯度路径、原未使用偏置，以及检查点与随机状态恢复。

此外已验证：

- 在大小为 2、3、1 的混合批次上，相同参数、随机状态和输入下，整理版与归档五输出训练步更新后的所有模型状态张量完全相等，最大差值 0，训练指标也完全相等；训练时仍跳过单样本尾批。
- 同一组混合批次的测试逐时刻 MPJPE 与公开测试函数完全一致，最大差值 0，包含跨批次的 NumPy 累加口径。
- 公开 10→50 权重 `strict=True` 加载通过；扰动未来 GT 后输出最大变化约 2537.90，未来输入梯度绝对值之和约 14318.66，说明原问题路径仍存在。这是随机输入上的诊断数值，不是测试集成绩。
- RTX 5070 Ti 上，10→25 和 10→50 的原模型前向、反向通过。
- 真实 walking 数据上的一批训练、验证、测试和断点恢复通过；未在本目录启动新的完整长期训练。

详细验证摘要在 `verification.json`。本目录与正在训练的 `BARNet` 修正版分开存放。
