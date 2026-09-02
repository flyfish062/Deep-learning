<div align="center">

# 🧠 Deep-Learning

### 从自动求导到 Transformer 的学习记录

**纯 Python · PyTorch · From Scratch · Autograd · Attention**

</div>

---

## 📌 项目简介

这是一个准大二本科生的深度学习学习仓库，主要记录我把一些基础组件亲手实现一遍的过程。

仓库按照由浅入深的顺序包含三个 notebook：

1. 用纯 Python 演示标量自动求导；
2. 用 PyTorch 手写一个字符级 GPT 的主要结构；
3. 参考 Transformer 论文实现一个简化的 Encoder-Decoder，并用小任务测试。

这里的“从零”主要指**没有调用 `nn.Transformer`、`nn.MultiheadAttention` 等完整封装**，而是自己展开注意力、mask、残差和训练流程。GPT 和 Transformer 仍然使用 PyTorch 提供的张量、基础网络层、优化器和自动求导。

项目目前以学习和理解为主，不是生产级框架，也不是论文级复现。代码保留了较多中文注释，重点说明关键张量形状和设计原因。

---

## 🗺 项目总览

| 文件 | 内容 | 验证方式 |
|---|---|---|
| [`micrograd.ipynb`](./micrograd.ipynb) | 标量计算图与自动求导 | 单神经元梯度、一次梯度下降 |
| [`gpt.ipynb`](./gpt.ipynb) | 字符级 Decoder-only GPT | Tiny Shakespeare 训练与生成 |
| [`transformer_ZeroToDone.ipynb`](./transformer_ZeroToDone.ipynb) | 简化 Encoder-Decoder Transformer | 数字序列 copy task |

学习路径可以概括为：

```text
计算图与梯度  →  自注意力与 GPT  →  Encoder-Decoder 与交叉注意力
```

---

## ⚙️ 环境与运行

建议环境：

- Python 3.10+（Notebook 保存时使用 Python 3.10.20）
- PyTorch 2.0+
- Jupyter Notebook 或 JupyterLab

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Jupyter 后，建议按以下顺序运行：

1. `micrograd.ipynb`：直接运行即可；
2. `gpt.ipynb`：需要从仓库根目录运行，并保证 `input.txt` 在同一目录；
3. `transformer_ZeroToDone.ipynb`：运行最后的 copy task。

GPT notebook 会在 CUDA 可用时使用 GPU；Transformer notebook 当前默认使用 CPU。CPU 也可以运行，但训练速度会慢一些。

如果使用命令行执行 notebook：

```bash
jupyter nbconvert --to notebook --execute --inplace micrograd.ipynb
jupyter nbconvert --to notebook --execute --inplace gpt.ipynb
jupyter nbconvert --to notebook --execute --inplace transformer_ZeroToDone.ipynb
```

> `--inplace` 会覆盖 notebook 中原有的输出。如果想保留当前输出，建议先复制文件再执行。

---

## ① micrograd：标量自动求导

`micrograd.ipynb` 只使用 Python 标准库中的 `math`，实现了一个简化的 `Value` 类。

每个节点保存自己的数值、梯度、父节点和局部反向传播规则。代码支持：

- 加减乘除和幂运算；
- `relu`、`tanh`；
- 数字在左侧的反向运算，例如 `2 + a`。

`backward()` 会先构造拓扑排序，再从 loss 开始逆序应用链式法则。最后的 demo 用一个简单神经元检查梯度，并手动更新一次参数。

这个 notebook 主要用于理解自动求导的基本机制，不涉及张量、广播或完整的神经网络框架。

---

## ② GPT：字符级 Decoder-only Transformer

`gpt.ipynb` 在 Tiny Shakespeare 字符语料上训练一个小型语言模型。主要结构由代码手动组合：

- 字符编码和 batch 采样；
- 单头因果自注意力与多头注意力；
- token embedding、位置 embedding；
- Pre-LN、残差连接和前馈网络；
- 训练、评估和自回归生成。

当前配置为 6 层、6 个注意力头、384 维嵌入、256 的上下文长度，约 10.79M 参数，训练 3000 步。

Notebook 中保存过一次运行结果：

| step | train loss | val loss |
|---:|---:|---:|
| 0 | 4.2221 | 4.2306 |
| 1000 | 1.3913 | 1.6010 |
| 2000 | 1.1891 | 1.5078 |
| 2999 | 1.0702 | 1.4839 |

生成部分加入了 `temperature` 和 `top-k` 参数。输出只是字符级模型在这份语料上学习到的续写片段，可能有拼写和语法问题；它不是 ChatGPT，也不代表通用的语言理解能力。

---

## ③ Transformer：简化的 Encoder-Decoder

`transformer_ZeroToDone.ipynb` 参考《Attention Is All You Need》的基本结构，手动实现了：

- 缩放点积注意力和多头注意力；
- 正弦位置编码；
- Encoder、Decoder 和交叉注意力；
- padding mask、因果 mask 和 batch 封装；
- Pre-LN 残差结构；
- Label Smoothing、Noam 学习率公式和贪心解码。

这里是一个教学性质的简化实现，采用了 Pre-LN 结构，并不等同于论文原始实现的所有细节。

最后的 demo 使用随机生成的数字序列进行 copy task：

```text
输入： [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
输出： [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

这个结果只说明模型在当前很小的合成任务上完成了训练和贪心解码流程，不代表它能够进行真实机器翻译。当前 demo 中 `smoothing=0.0`，因此没有实际启用标签平滑效果。

---

## 💡 我在学习中重点关注的内容

- 计算图如何保存运算关系，反向传播为什么要按拓扑序进行；
- 同一个变量经过多条路径时，梯度为什么需要累加；
- 因果 mask 如何让语言模型不能看到未来字符；
- self-attention 和 cross-attention 的区别；
- 残差连接、LayerNorm 和位置编码分别解决什么问题；
- 训练阶段和自回归推理阶段的数据流有什么不同。

---

## 📂 目录结构

```text
Deep-Learning/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
├── input.txt
├── micrograd.ipynb
├── gpt.ipynb
└── transformer_ZeroToDone.ipynb
```

---

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)
- Andrej Karpathy 的 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- 李沐等，[动手学深度学习](https://zh.d2l.ai/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

感谢这些论文、教程和开源项目提供的学习材料。本仓库的实现主要用于学习和整理。

---

## 📄 License

本项目使用 [MIT License](./LICENSE)。

---

*从梯度、注意力到 Encoder-Decoder，这个仓库记录了我目前对深度学习基础组件的学习过程。*
