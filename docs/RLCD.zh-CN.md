# RLCD：面向校准决策的强化学习

[English](RLCD.md) · [简体中文](RLCD.zh-CN.md)

GS1 训练的是它真正用于行动的概率分布：原生决策头为当前候选项打分，强化学习根据采样决策的实际概率计算奖励，应用再读取训练后的同一分布。决策质量、不确定性与部署行为由此通过同一个策略连接起来。

**RLCD**（Reinforcement Learning for Calibrated Decisions）描述面向校准决策的强化学习目标，这一术语由 [TypeSafe 随 Jev 提出](https://typesafe.ai/blog/introducing-system-one-models-and-jev)。GS1 的具体方法是 **CRPO**（Calibrated Reward Policy Optimization，校准奖励策略优化）：将 [RLCR](https://arxiv.org/abs/2507.16806) 的“正确性加校准”奖励用于原生离散决策，并结合 [GRPO](https://arxiv.org/abs/2402.03300) 提出的组相对、带裁剪的策略优化。

核心包括概率感知的 CRPO、学习显式拒答的受约束单步变体，以及面向序列控制提出的多步扩展。训练完成后，还可以通过置信度阈值调节部署行为。

## 统一决策分布，以监督适配作为起点

输入 $x_i=(s_i,u_i,\mathcal A_i)$ 包含状态、问题与当前候选集合。对于 $K_i$ 个候选项，模型输出

$$
q_{ik}=\pi_\theta(k\mid x_i),\qquad
\hat y_i=\arg\max_k q_{ik}.
$$

目标分布 $\tau_i$ 满足 $\tau_{ik}\geq0$、$\sum_k\tau_{ik}=1$。硬标签 $y_i^*$ 对应 one-hot 目标；软标签保留不确定性或标注分歧。强化学习中的采样候选项可能不同于贪心答案，其奖励使用**该候选项自身的概率** $q_{ik}$。

监督适配先以负对数似然（NLL）对齐决策头、深度路由器与结构 token 偏移，再以 NLL 加期望多分类 Brier 损失联合适配文本骨干和决策模块：

$$
\begin{aligned}
\ell_{\mathrm{NLL},i}&=-\sum_k\tau_{ik}\log q_{ik},\\
\ell_{\mathrm{Brier},i}
&=\mathbb E_{Y\sim\tau_i}\sum_k(q_{ik}-\mathbf1_{\{k=Y\}})^2\\
&=\sum_k(q_{ik}-\tau_{ik})^2+1-\sum_k\tau_{ik}^2,\\
\mathcal L_{\mathrm{SFT}}
&=\frac1N\sum_i(\ell_{\mathrm{NLL},i}+\ell_{\mathrm{Brier},i}).
\end{aligned}
$$

每个逻辑问题权重相同。两种损失都属于适当评分规则（proper scoring rules），允许将真实歧义保留为分布。结构偏移合并到导出的 embedding 中，视觉编码器予以保留。后续强化学习以适配后的决策策略为基础。

## CRPO：奖励真正用于选择动作的概率

### 正确性加动作自身的 Brier 奖励

每轮 rollout 开始时，冻结行为策略 $\pi_{\theta_{\mathrm{old}}}$，保存其分布 $q_i^{\mathrm{old}}$。对每个问题，从该分布中独立、**有放回地**采样 $G\geq2$ 个候选项索引。

对于硬标签，记 $I_{ik}=\mathbf1_{\{k=y_i^*\}}$，奖励为

$$
R_{ik}^{\mathrm{old}}
=I_{ik}-(q_{ik}^{\mathrm{old}}-I_{ik})^2
=\begin{cases}
2q_{ik}^{\mathrm{old}}-(q_{ik}^{\mathrm{old}})^2,&k=y_i^*,\\
-(q_{ik}^{\mathrm{old}})^2,&k\neq y_i^*.
\end{cases}
$$

第一项奖励答对，第二项针对“采样动作是否正确”这一事件施加二元 Brier 惩罚：

| 采样动作的概率 | 答对时的奖励 | 答错时的奖励 |
|---|---:|---:|
| 0.2 | 0.36 | −0.04 |
| 0.8 | 0.96 | −0.64 |

表中数值仅用于说明奖励计算。高置信错误受到更强惩罚。由于这里的置信来自原生策略，仅修改文字中的置信度、保持动作分布不变，无法改变奖励。计算时也不能用最大候选概率或 API 派生的 `confidence` 替代采样动作自身的概率。

对于软标签，应对参考答案取期望：

$$
\begin{aligned}
R_{ik}^{\mathrm{old}}
&=\mathbb E_{Y\sim\tau_i}
\left[\mathbf1_{\{k=Y\}}-(q_{ik}^{\mathrm{old}}-\mathbf1_{\{k=Y\}})^2\right]\\
&=2\tau_{ik}q_{ik}^{\mathrm{old}}-(q_{ik}^{\mathrm{old}})^2\\
&=\tau_{ik}-(q_{ik}^{\mathrm{old}}-\tau_{ik})^2-\tau_{ik}(1-\tau_{ik}).
\end{aligned}
$$

结果方差项 $\tau_{ik}(1-\tau_{ik})$ 不可省略：直接把平方误差中的硬标签指示量替换为 $\tau_{ik}$，会得到不同的奖励。

### 无需 critic 的组相对信用分配

对样本 $k_{ig}\sim q_i^{\mathrm{old}}$，记 $r_{ig}=R_{i,k_{ig}}^{\mathrm{old}}$，定义

$$
\begin{aligned}
\mu_i&=\frac1G\sum_g r_{ig},\qquad
\sigma_i^2=\frac1G\sum_g(r_{ig}-\mu_i)^2,\\
A_{ig}&=\operatorname{sg}\left[
\frac{r_{ig}-\mu_i}{\sigma_i+\epsilon_{\mathrm{adv}}}\right].
\end{aligned}
$$

其中 $\operatorname{sg}$ 表示停止梯度，$\epsilon_{\mathrm{adv}}>0$ 用于稳定分母。允许重复采到同一候选项；组内奖励完全相同时，优势为零。基线不需要单独训练 critic。一次 rollout 前向即可为一个问题的全部候选项打分，之后从固定分布多采样几个索引，无需额外模型前向；策略更新仍需重新计算当前分布。

### 带裁剪的策略更新与可选参考 KL

使用采样动作的概率比与裁剪代理项：

$$
\begin{aligned}
\rho_{ig}(\theta)&=\frac{q_{i,k_{ig}}}{q_{i,k_{ig}}^{\mathrm{old}}},\\
\ell_{\mathrm{clip}}(\rho,A)
&=\min\{\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\},\\
\mathcal J_{\mathrm{CRPO}}(\theta)
&=\frac1{NG}\sum_{i,g}\ell_{\mathrm{clip}}(\rho_{ig},A_{ig})
-\frac\beta N\sum_i D_{\mathrm{KL}}(q_i\|q_i^{\mathrm{ref}}).
\end{aligned}
$$

最大化 $\mathcal J_{\mathrm{CRPO}}$。其中 $N$ 是问题组数，$G$ 是每组采样数，$\epsilon$ 控制裁剪范围。参考策略保持固定，行为策略快照则在每轮 rollout 时刷新。设置 $\beta=0$ 即移除 KL 惩罚。候选项可枚举，因此可以精确计算离散 KL：$\sum_k q_{ik}\log(q_{ik}/q_{ik}^{\mathrm{ref}})$。

**行为概率、奖励和优势在整个更新周期内均停止梯度。** 梯度通过当前策略的概率比和参考 KL 传播，不穿过已保存的 Brier 奖励。CRPO 是策略梯度代理目标，与监督阶段的 Brier 回归不同。奖励包含概率信息，并不自动证明最终策略已经校准。

## 单步强化学习：学习何时拒答

### 同时衡量准确率与非幻觉率

对于硬标签任务，在实质答案之外加入显式拒答动作 $\bot$。随机策略下答对、拒答和答错的期望比例分别为

$$
c_\theta=\frac1N\sum_i\pi_\theta(y_i^*\mid x_i),\qquad
a_\theta=\frac1N\sum_i\pi_\theta(\bot\mid x_i),\qquad
w_\theta=1-c_\theta-a_\theta.
$$

准确率为以**全部问题**为分母的 $c_\theta$，拒答同样计入分母。沿用 [AA-Omniscience](https://arxiv.org/abs/2511.13029) 对知识可靠性的关注，定义非幻觉率（non-hallucination rate，NHR）为

$$
\operatorname{NHR}(\theta)=\frac{a_\theta}{a_\theta+w_\theta}.
$$

NHR 衡量未答对的结果中有多少选择了拒答。始终拒答会得到 NHR $=1$、准确率 $=0$，因此不能单独用 NHR 衡量决策质量。当 $a_\theta+w_\theta=0$ 时，比值未定义；仓库评测约定将这一完全答对情形的 NHR 记为 $1$。

### 在给定 NHR 约束下最大化准确率

对目标水平 $h_0\in[0,1]$，优化

$$
\max_\theta c_\theta
\quad\text{subject to}\quad
g(\theta)=(1-h_0)a_\theta-h_0w_\theta\geq0.
$$

当 NHR 分母为正时，该约束等价于 NHR $\geq h_0$。线性形式也能处理完全答对的情况，避免除零。对应的拉格朗日函数与动作奖励为

$$
\begin{aligned}
\mathcal L_{\mathrm{NHR}}(\theta,\lambda)
&=c_\theta+\lambda g(\theta),\qquad\lambda\geq0,\\
R_\lambda(k,y_i^*)&=\begin{cases}
1,&k=y_i^*,\\
-\lambda h_0,&k\notin\{y_i^*,\bot\},\\
\lambda(1-h_0),&k=\bot.
\end{cases}
\end{aligned}
$$

动作奖励的期望恰好等于该拉格朗日函数。乘子 $\lambda$ 根据目标 NHR 调节错误的代价与拒答的价值。

### 保留约束奖励的尺度

这一变体用 $R_\lambda$ **替换** CRPO 的 Brier 形状奖励，并使用不做标准差归一化的留一法优势：

$$
A^\lambda_{ig}=\operatorname{sg}\left[
r^\lambda_{ig}-\frac1{G-1}\sum_{g'\neq g}r^\lambda_{ig'}\right].
$$

逐组除以标准差会改变拉格朗日奖励的尺度。保留裁剪策略更新，以 $A^\lambda$ 替代 $A$，并对上述约束问题设置 $\beta=0$。在 rollout 和策略更新期间固定 $\lambda$，之后按下式更新：

$$
\lambda_{t+1}=\max\{0,\lambda_t-\eta_\lambda\widehat g_t\},
\qquad\eta_\lambda>0.
$$

从原始 rollout 结果估计 $\widehat g_t=(1-h_0)\widehat a_t-h_0\widehat w_t$，不使用经过优势归一化或裁剪的量。约束被违反时增大 $\lambda$，存在余量时减小 $\lambda$。这一约束针对随机策略的训练比例；包含拒答在内的最终贪心部署策略仍需单独验证。

## 多步强化学习：将 CRPO 扩展到长时序决策

多步形式是 CRPO 的一项**扩展方案**。训练材料列出的环境包括《魔兽争霸 III》、StarCraft II 和计算机操作任务。历史 environment-GRPO 运行属于基线，不能作为多步 CRPO 的实测结果。

### 以完整交互作为 rollout 单位

环境适配器提供公开证据与可用动作，将经过校验的决策转换为后端控制，并返回下一观测、奖励及回合边界。reset 开始新回合，终止或截断结束采集。隐藏游戏信息和辅助服务元数据不进入策略输入。

对任务 $i$，采集 $G\geq2$ 条轨迹 $\xi_{ig}$，长度为 $T_{ig}\geq1$。同组轨迹共享任务配置和环境种子，但使用独立环境实例及策略采样随机流。在第 $t$ 步，

$$
\begin{aligned}
x_{igt}&=(s_{igt},u_{igt},\mathcal A_{igt}),\\
q_{igtk}^{\mathrm{old}}&=\pi_{\theta_{\mathrm{old}}}(k\mid x_{igt}),\qquad
k_{igt}\sim q_{igt}^{\mathrm{old}},\\
p_{igt}^{\mathrm{old}}&=q_{igt,k_{igt}}^{\mathrm{old}}.
\end{aligned}
$$

状态包含当前公开观测以及保留的交互历史，可以表示为对话历史，也可以是附带执行反馈的结构化快照。每条轨迹自己的动作决定后续输入与候选集合。每次决策都要保存输入、候选项、采样动作及其行为概率。随着环境演化，后续步骤需要新的前向计算。

### 用终局反馈为轨迹评分

记 $Z_{ig}\in\{0,1\}$ 为任务是否成功，将它作为该轨迹各次采样决策的共享反馈：

$$
\begin{aligned}
r_{igt}^{\mathrm{old}}
&=Z_{ig}-(p_{igt}^{\mathrm{old}}-Z_{ig})^2,\\
R_{ig}^{\mathrm{old}}
&=\frac1{T_{ig}}\sum_{t=0}^{T_{ig}-1}r_{igt}^{\mathrm{old}}\\
&=Z_{ig}-\frac1{T_{ig}}\sum_{t=0}^{T_{ig}-1}(p_{igt}^{\mathrm{old}}-Z_{ig})^2.
\end{aligned}
$$

时间平均避免得分随回合长度线性累积。成功轨迹得分位于 $[0,1]$，失败轨迹位于 $[-1,0]$。只有一步、且任务成功等同于答对时，该奖励精确退化为硬标签 CRPO。环境原生回报仍作为独立诊断指标保留。

终局反馈提供的是**粗粒度的跨步信用分配**，并未为每个中间动作提供正确性标签。保存的 $p_{igt}^{\mathrm{old}}$ 是动作选择概率，并非最终任务成功的预测概率。若定义

$$
\eta_{igt}^{\mathrm{old}}
=\Pr(Z_{ig}=1\mid x_{igt},k_{igt};\pi_{\theta_{\mathrm{old}}}),
$$

则有

$$
\mathbb E[r_{igt}^{\mathrm{old}}\mid x_{igt},k_{igt}]
=2\eta_{igt}^{\mathrm{old}}p_{igt}^{\mathrm{old}}-(p_{igt}^{\mathrm{old}})^2.
$$

该期望包含结果方差。与离散目标分布 $\tau_i$ 不同，各动作的条件成功概率 $\eta$ 之和不必为一。训练直接使用观测到的终局结果，无需估计 $\eta$；这一奖励本身也不构成任务成功概率已经校准的证明。

### 将轨迹相对优势分配给整条轨迹

根据**轨迹得分**计算组均值与标准差：

$$
\begin{aligned}
\mu_i&=\frac1G\sum_g R_{ig}^{\mathrm{old}},\qquad
\sigma_i^2=\frac1G\sum_g(R_{ig}^{\mathrm{old}}-\mu_i)^2,\\
A_{ig}&=\operatorname{sg}\left[
\frac{R_{ig}^{\mathrm{old}}-\mu_i}{\sigma_i+\epsilon_{\mathrm{adv}}}\right].
\end{aligned}
$$

每条轨迹对组统计贡献相同，其所有采样决策共享 $A_{ig}$。组内得分完全相同时，优势为零。终局结果相同的轨迹仍可能因保存的动作概率不同而得到不同分数。

在保存的输入上重新计算 $q_{igt}=\pi_\theta(\cdot\mid x_{igt})$，最大化

$$
\begin{aligned}
\rho_{igt}(\theta)&=\frac{q_{igt,k_{igt}}}{p_{igt}^{\mathrm{old}}},\\
\mathcal J_{\mathrm{MCRPO}}(\theta)
&=\frac1{NG}\sum_{i,g}\frac1{T_{ig}}\sum_{t=0}^{T_{ig}-1}
\left[\ell_{\mathrm{clip}}(\rho_{igt},A_{ig})
-\beta D_{\mathrm{KL}}(q_{igt}\|q_{igt}^{\mathrm{ref}})\right].
\end{aligned}
$$

其中 $N$ 是任务组数。在**每条轨迹内部**平均代理项与 KL，使不同长度的轨迹拥有相同权重；这与历史基线中所有 transition 等权的归一化方式不同。只有一步时，奖励、组统计和目标均退化为前述硬标签 CRPO。

行为策略与参考策略的角色沿用单步 CRPO。保存的概率、结果、奖励和优势均停止梯度；梯度通过当前策略概率比与精确参考 KL 传播。若一个环境动作需要多次条件候选选择，则在该环境步内平均各次选择的得分和代理项。每次选择使用自身的条件概率，不使用整条轨迹的概率乘积。

## Rollout 与更新流程

以下伪代码概括方法，不是仓库提供的训练入口。

```text
从监督适配后的决策策略出发；若使用 KL，则固定参考策略。
对每轮 rollout：
    冻结当前策略的行为快照。
    按组采集 G >= 2 个决策或完整轨迹。
    保存访问过的输入、候选集合、采样动作及行为概率。
    选择本轮的奖励和优势：
        CRPO：动作自身的 Brier 奖励；标准化的组相对优势。
        NHR 变体：R_lambda；不做标准差归一化的留一法优势；beta = 0。
        多步 CRPO：对终局反馈奖励取时间平均；
                  计算标准化的轨迹优势，并在轨迹内各步共享。
    在本次更新周期内固定 rollout 概率、奖励与优势。
    在已保存输入上重算当前概率，最大化裁剪目标。
    若使用 NHR 约束，从原始 rollout 结果比例更新 lambda。
    下一轮 rollout 刷新行为策略快照。
在独立留出数据上验证实际部署规则。
```

## 训练后的置信度阈值

学习拒答会改变策略；置信度阈值用于训练后选择运行点。对于 $K>1$ 的 Choice，GS1 沿用 [Jev 的置信度读出](https://docs.typesafe.ai/confidence)：

$$
c^{\mathrm{Choice}}(q)=\frac{K\max_kq_k-1}{K-1}.
$$

均匀分布映射为零，确定分布映射为一；只有一个候选项时，Choice 置信度为一。阈值 $\gamma$ 对应规则

$$
\hat y_\gamma(x)=\begin{cases}
\arg\max_k\pi_\theta(k\mid x),&c^{\mathrm{Choice}}(q)\geq\gamma,\\
\bot,&\text{otherwise}.
\end{cases}
$$

如果拒答本身已经是候选项，即使达到阈值，选中它也仍然是拒答。应用还可以将低置信样本转向补充证据或更昂贵的模型。固定 $K$ 时，该置信度只是最大概率的单调变换，不提供独立证据；它衡量分布集中程度，不是另行估计的正确概率。应在目标任务的验证分布上选择阈值。[API 语义](api.zh-CN.md#置信度)

## 评测实际部署的策略

| 要回答的问题 | 测量内容 |
|---|---|
| 概率是否有用？ | 留出问题上的期望多分类 Brier、top-label ECE 和可靠性分箱 |
| 拒答是否改善权衡？ | 全部问题上的准确率、NHR、覆盖率，以及答对／拒答／答错的原始计数 |
| 应选择哪个运行点？ | 最终贪心规则与所选阈值对应的准确率—NHR、风险—覆盖率曲线 |
| 序列控制是否成功？ | 任务成功率、环境原生回报、回合长度和任务专属结果，与 CRPO 塑形得分分别记录 |

覆盖率为 $1-a$；当覆盖率为正时，已作答问题上的准确率为 $c/(1-a)$。两者均不能替代全部问题上的准确率。Top-label ECE 使用最大候选概率，不使用归一化后的 Choice `confidence`。在验证集上选择运行设置，并在独立评测集上报告结果。

当前仓库提供模型架构、推理、服务与评测。本文说明训练目标，不提供完整训练器、训练语料或超参数配方。多步形式作为扩展方案的状态，应与历史 environment-GRPO 结果明确区分。进一步参见[模型架构](model.zh-CN.md)、[评测协议](../benchmark/README.zh-CN.md)、[AA 评测约定](../benchmark/aadi/README.zh-CN.md)与[公开结果（英文）](../benchmark/results.md)。
