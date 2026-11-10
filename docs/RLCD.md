# RLCD: Reinforcement Learning for Calibrated Decisions

[English](RLCD.md) · [简体中文](RLCD.zh-CN.md)

GS1 trains the probability distribution it uses to act. Its native decision head scores the available candidates, reinforcement learning assigns rewards to sampled decisions using their actual probabilities, and applications receive the resulting distribution. This connects decision quality, uncertainty and deployment behavior through the same policy.

**RLCD** describes the goal of reinforcement learning for calibrated decisions, a term introduced by [TypeSafe with Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). GS1's concrete method is **Calibrated Reward Policy Optimization (CRPO)**. It adapts the correctness-plus-calibration reward of [RLCR](https://arxiv.org/abs/2507.16806) to native categorical decisions and combines it with the group-relative, clipped policy optimization introduced in [GRPO](https://arxiv.org/abs/2402.03300).

The main components are probability-aware CRPO, a constrained single-step variant that learns explicit abstention, and a proposed multi-step extension for sequential control. Confidence thresholds provide an additional deployment control after training.

## A shared distribution, prepared by supervised adaptation

An input $x_i=(s_i,u_i,\mathcal A_i)$ contains the state, question and available alternatives. For $K_i$ candidates, the model produces

$$
q_{ik}=\pi_\theta(k\mid x_i),\qquad
\hat y_i=\arg\max_k q_{ik}.
$$

A target distribution $\tau_i$ satisfies $\tau_{ik}\geq0$ and $\sum_k\tau_{ik}=1$. A hard answer $y_i^*$ corresponds to a one-hot target; soft targets retain uncertainty or disagreement. During RL, a sampled candidate can differ from the greedy answer. Its reward uses **that candidate's probability** $q_{ik}$.

Supervised adaptation first aligns the decision head, depth routers and structural-token offsets under negative log-likelihood (NLL), then jointly adapts the text backbone and decision modules with NLL plus expected multiclass Brier loss:

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

Each logical question receives equal weight. Both losses are proper scoring rules, so ambiguity can remain distributional. Structural offsets are merged into exported embeddings, and the vision encoder is retained. The RL stages below build on this adapted decision policy.

## CRPO: reward the probability used to choose an action

### Correctness plus an action-specific Brier reward

At the start of a rollout round, freeze a behavior policy $\pi_{\theta_{\mathrm{old}}}$ and save its distribution $q_i^{\mathrm{old}}$. For each question, independently sample $G\geq2$ candidate indices **with replacement** from this distribution.

For a hard reference answer, let $I_{ik}=\mathbf1_{\{k=y_i^*\}}$. The reward is

$$
R_{ik}^{\mathrm{old}}
=I_{ik}-(q_{ik}^{\mathrm{old}}-I_{ik})^2
=\begin{cases}
2q_{ik}^{\mathrm{old}}-(q_{ik}^{\mathrm{old}})^2,&k=y_i^*,\\
-(q_{ik}^{\mathrm{old}})^2,&k\neq y_i^*.
\end{cases}
$$

The first term rewards correctness. The second is a binary Brier penalty for the event that the sampled action is correct:

| Sampled action probability | Reward when correct | Reward when wrong |
|---|---:|---:|
| 0.2 | 0.36 | −0.04 |
| 0.8 | 0.96 | −0.64 |

These are illustrative reward values. Confident errors receive a larger penalty. Because confidence comes from the native policy, changing a verbal confidence string cannot change the reward while leaving the action distribution fixed. Neither the maximum candidate probability nor the API's derived `confidence` replaces the sampled action probability here.

For a soft target, take the expectation over the reference answer:

$$
\begin{aligned}
R_{ik}^{\mathrm{old}}
&=\mathbb E_{Y\sim\tau_i}
\left[\mathbf1_{\{k=Y\}}-(q_{ik}^{\mathrm{old}}-\mathbf1_{\{k=Y\}})^2\right]\\
&=2\tau_{ik}q_{ik}^{\mathrm{old}}-(q_{ik}^{\mathrm{old}})^2\\
&=\tau_{ik}-(q_{ik}^{\mathrm{old}}-\tau_{ik})^2-\tau_{ik}(1-\tau_{ik}).
\end{aligned}
$$

The outcome-variance term $\tau_{ik}(1-\tau_{ik})$ matters: simply replacing the hard indicator with $\tau_{ik}$ inside the squared error gives a different reward.

### Group-relative credit without a critic

For samples $k_{ig}\sim q_i^{\mathrm{old}}$, write $r_{ig}=R_{i,k_{ig}}^{\mathrm{old}}$. Define

$$
\begin{aligned}
\mu_i&=\frac1G\sum_g r_{ig},\qquad
\sigma_i^2=\frac1G\sum_g(r_{ig}-\mu_i)^2,\\
A_{ig}&=\operatorname{sg}\left[
\frac{r_{ig}-\mu_i}{\sigma_i+\epsilon_{\mathrm{adv}}}\right].
\end{aligned}
$$

Here $\operatorname{sg}$ stops gradients and $\epsilon_{\mathrm{adv}}>0$ stabilizes the denominator. Repeated candidates are valid; a constant-reward group has zero advantage. The baseline requires no learned critic. One rollout forward pass scores all candidates for a question, so drawing additional indices from that fixed distribution requires no additional model forwards. Policy updates still recompute the current distribution.

### A clipped policy update with optional reference KL

Use the sampled action's probability ratio and a clipped surrogate:

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

Maximize $\mathcal J_{\mathrm{CRPO}}$. Here $N$ counts question groups, $G$ is the number of samples per group, and $\epsilon$ controls clipping. The reference policy is fixed; the behavior snapshot is refreshed each rollout round. Setting $\beta=0$ removes the KL penalty. Enumerable candidates allow exact categorical KL, $\sum_k q_{ik}\log(q_{ik}/q_{ik}^{\mathrm{ref}})$.

**Behavior probabilities, rewards and advantages stay detached throughout an update cycle.** Gradients flow through current-policy ratios and the reference KL, not through the saved Brier reward. CRPO is a policy-gradient surrogate, distinct from supervised Brier regression. A probability-aware reward alone does not establish that the resulting policy is calibrated.

## Single-step RL: learn when to abstain

### Accuracy and non-hallucination must be measured together

For hard-label tasks, add an explicit abstention action $\bot$ alongside the substantive answers. Let the stochastic policy's expected correct, abstention and wrong-answer fractions be

$$
c_\theta=\frac1N\sum_i\pi_\theta(y_i^*\mid x_i),\qquad
a_\theta=\frac1N\sum_i\pi_\theta(\bot\mid x_i),\qquad
w_\theta=1-c_\theta-a_\theta.
$$

Accuracy is $c_\theta$ over **all questions**, including abstentions in the denominator. Following the knowledge-reliability motivation of [AA-Omniscience](https://arxiv.org/abs/2511.13029), define the non-hallucination rate (NHR) as

$$
\operatorname{NHR}(\theta)=\frac{a_\theta}{a_\theta+w_\theta}.
$$

NHR measures how often a non-correct outcome is an abstention. Always abstaining gives NHR $=1$ and accuracy $=0$, so NHR alone cannot measure decision quality. When $a_\theta+w_\theta=0$, the ratio is undefined; the repository's evaluation convention reports an NHR of $1$ for this perfectly correct case.

### Maximize accuracy subject to an NHR target

For a requested level $h_0\in[0,1]$, optimize

$$
\max_\theta c_\theta
\quad\text{subject to}\quad
g(\theta)=(1-h_0)a_\theta-h_0w_\theta\geq0.
$$

For a positive NHR denominator, this is equivalent to NHR $\geq h_0$. The linear constraint also handles the perfectly correct case without division. Its Lagrangian and per-action reward are

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

The expected action reward is exactly this Lagrangian. The multiplier $\lambda$ sets the cost of an error and the value of abstention according to the requested NHR.

### Preserve the constraint reward scale

This variant **replaces** the Brier-shaped CRPO reward with $R_\lambda$. It uses an unnormalized leave-one-out advantage:

$$
A^\lambda_{ig}=\operatorname{sg}\left[
r^\lambda_{ig}-\frac1{G-1}\sum_{g'\neq g}r^\lambda_{ig'}\right].
$$

Dividing by each group's standard deviation would change the Lagrangian reward scale. Retain the clipped policy update, use $A^\lambda$ in place of $A$, and set $\beta=0$ for the stated constraint problem. Hold $\lambda$ fixed during rollout and policy updates, then update it by

$$
\lambda_{t+1}=\max\{0,\lambda_t-\eta_\lambda\widehat g_t\},
\qquad\eta_\lambda>0.
$$

Estimate $\widehat g_t=(1-h_0)\widehat a_t-h_0\widehat w_t$ from raw rollout outcomes, before advantage normalization or clipping. A constraint violation increases $\lambda$; slack decreases $\lambda$. The constraint concerns stochastic training rates: the final greedy deployment policy, including abstentions, requires its own validation.

## Multi-step RL: extend CRPO to long-horizon decisions

The multi-step formulation is a **proposed extension** of CRPO. The training notes name Warcraft III, StarCraft II and computer-use tasks as environments. Historical environment-GRPO runs are baselines, not measurements of multi-step CRPO.

### Roll out complete interactions

An environment adapter exposes public evidence and available actions, translates validated decisions into backend controls, and returns the next observation, reward and episode boundaries. Reset starts a fresh episode; termination or truncation ends collection. Hidden game information and auxiliary service metadata remain outside policy inputs.

For task $i$, collect $G\geq2$ trajectories $\xi_{ig}$ of lengths $T_{ig}\geq1$. Group members share the task configuration and seed, with independent environment instances and policy-sampling streams. At step $t$,

$$
\begin{aligned}
x_{igt}&=(s_{igt},u_{igt},\mathcal A_{igt}),\\
q_{igtk}^{\mathrm{old}}&=\pi_{\theta_{\mathrm{old}}}(k\mid x_{igt}),\qquad
k_{igt}\sim q_{igt}^{\mathrm{old}},\\
p_{igt}^{\mathrm{old}}&=q_{igt,k_{igt}}^{\mathrm{old}}.
\end{aligned}
$$

The state contains the current public observation and any retained interaction history. Its representation can be dialogue history or a structured snapshot with execution feedback. Each trajectory's own actions determine subsequent inputs and candidate sets. Save the input, candidates, sampled action and its behavior probability at each decision. Later steps require new forward passes as the environment evolves.

### Use terminal feedback to score a trajectory

Let $Z_{ig}\in\{0,1\}$ denote task success. Use it as shared feedback for the trajectory's sampled decisions:

$$
\begin{aligned}
r_{igt}^{\mathrm{old}}
&=Z_{ig}-(p_{igt}^{\mathrm{old}}-Z_{ig})^2,\\
R_{ig}^{\mathrm{old}}
&=\frac1{T_{ig}}\sum_{t=0}^{T_{ig}-1}r_{igt}^{\mathrm{old}}\\
&=Z_{ig}-\frac1{T_{ig}}\sum_{t=0}^{T_{ig}-1}(p_{igt}^{\mathrm{old}}-Z_{ig})^2.
\end{aligned}
$$

Temporal averaging prevents the score from accumulating linearly with episode length. Successful trajectories score in $[0,1]$; unsuccessful ones score in $[-1,0]$. With one decision and success equal to answer correctness, the reward reduces exactly to hard-label CRPO. Keep the environment's native return as a separate diagnostic.

Terminal feedback gives **coarse temporal credit**, not a correctness label for each intermediate action. The saved $p_{igt}^{\mathrm{old}}$ is an action-selection probability, not a predicted probability of eventual success. If

$$
\eta_{igt}^{\mathrm{old}}
=\Pr(Z_{ig}=1\mid x_{igt},k_{igt};\pi_{\theta_{\mathrm{old}}}),
$$

then

$$
\mathbb E[r_{igt}^{\mathrm{old}}\mid x_{igt},k_{igt}]
=2\eta_{igt}^{\mathrm{old}}p_{igt}^{\mathrm{old}}-(p_{igt}^{\mathrm{old}})^2.
$$

This expectation includes outcome variance. Unlike a categorical target $\tau_i$, conditional success probabilities $\eta$ need not sum to one across actions. Training uses observed terminal outcomes without estimating $\eta$; the reward does not by itself establish calibration of task-success probabilities.

### Share trajectory-relative advantage across its decisions

Compute the group mean and standard deviation from the **trajectory scores**:

$$
\begin{aligned}
\mu_i&=\frac1G\sum_g R_{ig}^{\mathrm{old}},\qquad
\sigma_i^2=\frac1G\sum_g(R_{ig}^{\mathrm{old}}-\mu_i)^2,\\
A_{ig}&=\operatorname{sg}\left[
\frac{R_{ig}^{\mathrm{old}}-\mu_i}{\sigma_i+\epsilon_{\mathrm{adv}}}\right].
\end{aligned}
$$

Each trajectory contributes equally to these statistics, and all its sampled decisions share $A_{ig}$. A constant-score group has zero advantage. Identical terminal outcomes can still produce different scores because the saved action probabilities differ.

At each saved input, recompute $q_{igt}=\pi_\theta(\cdot\mid x_{igt})$ and maximize

$$
\begin{aligned}
\rho_{igt}(\theta)&=\frac{q_{igt,k_{igt}}}{p_{igt}^{\mathrm{old}}},\\
\mathcal J_{\mathrm{MCRPO}}(\theta)
&=\frac1{NG}\sum_{i,g}\frac1{T_{ig}}\sum_{t=0}^{T_{ig}-1}
\left[\ell_{\mathrm{clip}}(\rho_{igt},A_{ig})
-\beta D_{\mathrm{KL}}(q_{igt}\|q_{igt}^{\mathrm{ref}})\right].
\end{aligned}
$$

Here $N$ counts task groups. Averaging the surrogate and KL **within each trajectory** gives equal weight to trajectories of different lengths. This differs from weighting all transitions equally, as in the historical baseline. For a single decision, the reward, group statistics and objective reduce to the preceding hard-label CRPO formulation.

The behavior policy and reference policy have the same roles as in single-step CRPO. Saved probabilities, outcomes, rewards and advantages stay detached; gradients flow through current-policy ratios and exact reference KL. If an environment action requires several conditional candidate selections, average their individual scores and surrogate terms within that environment step. Each selection uses its own conditional probability, not a product over the trajectory.

## The rollout and update cycle

The following pseudocode summarizes the methodology; it is not a training entry point supplied by this repository.

```text
Start from the supervised-adapted decision policy; fix a reference if using KL.
For each rollout round:
    Freeze a behavior snapshot of the current policy.
    Collect groups of G >= 2 decisions or complete trajectories.
    Save visited inputs, candidate sets, sampled actions and behavior probabilities.
    Choose the reward and advantage for this round:
        CRPO: action-specific Brier reward; standardized group advantage.
        NHR variant: R_lambda; unnormalized leave-one-out advantage; beta = 0.
        Multi-step CRPO: terminal-feedback scores averaged over time;
                        standardized trajectory advantage shared across its steps.
    Freeze rollout probabilities, rewards and advantages for the update cycle.
    Recompute current probabilities on saved inputs and maximize the clipped objective.
    If using the NHR constraint, update lambda from raw rollout outcome fractions.
    Refresh the behavior snapshot for the next rollout round.
Validate the actual deployment rule on held-out data.
```

## Confidence thresholds after training

Learned abstention changes the policy. A confidence threshold selects an operating point after training. For a Choice with $K>1$, GS1 follows [Jev's confidence readout](https://docs.typesafe.ai/confidence):

$$
c^{\mathrm{Choice}}(q)=\frac{K\max_kq_k-1}{K-1}.
$$

A uniform distribution maps to zero and a deterministic distribution to one; singleton Choice confidence is one. A threshold $\gamma$ gives

$$
\hat y_\gamma(x)=\begin{cases}
\arg\max_k\pi_\theta(k\mid x),&c^{\mathrm{Choice}}(q)\geq\gamma,\\
\bot,&\text{otherwise}.
\end{cases}
$$

When abstention is already a candidate, selecting it still means abstaining, regardless of the threshold. Applications can also route low-confidence cases to additional evidence or a more expensive model. For fixed $K$, this confidence is a monotone transformation of the maximum probability and supplies no independent evidence. It measures concentration, not a separately estimated probability of correctness. Choose thresholds on the intended validation distribution. [API semantics](api.md#confidence)

## Evaluate the policy that will be deployed

| Question | Measurements |
|---|---|
| Are the probabilities useful? | Expected multiclass Brier, top-label ECE and reliability bins on held-out questions |
| Does abstention improve the trade-off? | Accuracy over all questions, NHR, coverage and raw correct/abstained/wrong counts |
| Which operating point should be used? | Accuracy–NHR and risk–coverage curves for the final greedy rule and selected thresholds |
| Does sequential control succeed? | Task success, native environment return, episode length and task-specific outcomes, separately from the shaped CRPO score |

Coverage is $1-a$; accuracy among answered questions is $c/(1-a)$ when coverage is positive. Neither replaces accuracy over all questions. Top-label ECE uses the maximum candidate probability, not the normalized Choice `confidence`. Select operating settings on validation data and report results on a separate evaluation split.

The current repository provides model architecture, inference, serving and evaluation. This document explains the training objectives; it does not supply a complete trainer, training corpus or hyperparameter recipe. The multi-step formulation's proposed status must remain separate from historical environment-GRPO results. See the [model architecture](model.md), [evaluation protocol](../benchmark/README.md), [AA evaluation conventions](../benchmark/aadi/README.md) and [published results](../benchmark/results.md).
