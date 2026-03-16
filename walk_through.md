⏺ 好，我会从理论第一原理出发，逐步连接到代码实现。用中文讲，数学符号保留。                                                                                                                   
                                                                                                                                                                                             
  ---
  第一部分：问题设定（为什么需要这个方法）                                                                                                                                                   
                                                            
  标准因果推断的困难                                                                                                                                                                         
                                                            
  我们想估计的东西是平均因果效应（ATE）：

  $$\beta(a) = E[Y(a)]$$

  即把所有人的治疗强制设为 $a$ 时，结局的期望值。

  问题：有未观测混淆变量 $U$（需求 DGP 里的 demand），$U$ 同时影响 $A$（price）和 $Y$（sales），所以 $E[Y|A=a] \neq E[Y(a)]$，不能直接回归。

  PCI 的思路：用代理变量

  Proximal Causal Inference 引入两组代理变量：

  ┌──────┬─────────────────┬─────────────────┬──────────────────────────────────────┐
  │ 符号 │      名称       │      代码       │               DGP 含义               │
  ├──────┼─────────────────┼─────────────────┼──────────────────────────────────────┤
  │ $Z$  │ Treatment proxy │ treatment_proxy │ cost1, cost2（只影响 $A$，通过 $U$） │
  ├──────┼─────────────────┼─────────────────┼──────────────────────────────────────┤
  │ $W$  │ Outcome proxy   │ outcome_proxy   │ views（由 $U$ 驱动，不直接影响 $Y$） │
  └──────┴─────────────────┴─────────────────┴──────────────────────────────────────┘

  关键假设：$Z \perp Y(a) \mid U$，$W \perp A \mid U$。两组代理变量从两侧"夹住"混淆 $U$。

  桥函数（Bridge Function）

  PCI 证明，如果存在函数 $h(W, A)$ 满足以下条件矩约束（CMR）：

  $$E[Y - h(W, A) \mid A, Z] = 0 \quad \text{（对几乎所有 } (A,Z) \text{）}$$

  那么 $\beta(a) = E[h(W, a)]$。

  这就是我们要估计的对象——用数据学出 $h_\theta$，使得残差 $r_i(\theta) = Y_i - h_\theta(W_i, A_i)$ 在给定 $(A_i, Z_i)$ 下条件期望为零。

  ---
  第二部分：DeepGMM 的 Min-Max 游戏（原始 + PCI）

  为什么用 min-max？

  CMR $E[r \mid A, Z]=0$ 等价于：对任意函数 $f$，

  $$E[f(A, Z) \cdot r_i(\theta)] = 0$$

  这是一个无限维的矩条件。DeepGMM 的思路：把 $f$ 也参数化为神经网络 $f_\tau$（critic），然后解这个 min-max 游戏：

  $$\min_\theta \max_\tau ; U(\theta, \tau)$$

  critic $f_\tau$ 的任务是"找到最难满足的方向"，generator $h_\theta$ 的任务是"把所有方向的矩都拉回零"。

  具体目标函数

  原始 DeepGMM（OptimalMomentObjective，simple_moment_objective.py:88）：

  $$U(\theta, \tau) = \underbrace{\frac{1}{n}\sum_i f_\tau(L_i) \cdot \varepsilon_i(\theta)}{\text{moment}} - \underbrace{\frac{\lambda}{n}\sum_i f\tau(L_i)^2 \cdot
  \varepsilon_i(\bar\theta)^2}_{\text{freg}}$$

  其中 $L_i = (A_i, Z_i)$，$\varepsilon_i = h(W_i,A_i) - Y_i$（注意原始代码用 $h-Y$，PCI 用 $Y-h$，符号相反但均衡等价）。

  第二项的作用：最优 critic 的闭合解是 $f^*(L) \propto \frac{E[r \mid L]}{E[r^2 \mid L]}$，f_reg 项就是为了让 critic 找到这个方向。

  ---
  第三部分：MAR 缺失问题

  问题

  $W$（views）部分缺失。缺失机制是 MAR（Missing at Random）：

  $$P(\delta_W = 1 \mid W, A, Z, Y, U) = P(\delta_W = 1 \mid A, Z, Y) =: \pi(L^+)$$

  其中 $L^+ = (A, Z, Y)$。关键：给定 $L^+$ 后，缺失与 $W$ 无关（MAR 条件）。

  代码实现（demand_scenario.py:57）：

  l_plus = np.concatenate([treatment, treatment_proxy, outcome], axis=1)  # (A, Z, Y)
  l_plus = (l_plus - mean) / std   # 标准化
  alpha = [1.6, 0.8, -0.8, 1.2]   # Y 的系数 1.2 ≠ 0 → 不是 MCAR
  score = l_plus @ alpha
  probs = sigmoid(score + intercept)   # P(δ=1 | L+)
  delta_w = Bernoulli(probs)

  如果直接用 complete-case（naive 方法）

  只用 $\delta_W=1$ 的样本：

  $$\hat U_{\text{naive}} = \frac{1}{\sum \delta_i} \sum_{\delta_i=1} f(L_i) \cdot r_i(\theta)$$

  问题：这实际上在估计 $E[f(L) \cdot r(\theta) \mid \delta_W=1]$，而非 $E[f(L) \cdot r(\theta)]$。因为 $\delta_W$ 依赖 $Y$（进而依赖 $r$），所以 $E[r \cdot f \mid \delta=1] \neq 0$
  即使真正的 $h^*$ 使得 $E[r \cdot f] = 0$。Naive 方法有偏。

  修正方法：imputed residuals（MAR 插补残差）

  利用 MAR 条件，可以构造一个无偏插补残差：

  $$\tilde{r}i(\theta) = \delta_i \cdot r_i(\theta) + (1 - \delta_i) \cdot \hat{m}\theta(L^+_i)$$

  其中 $\hat{m}_\theta(L^+) = \widehat{E}[r(\theta) \mid L^+, \delta=1]$ 是用 complete cases 拟合的插补模型。

  为什么这样做是无偏的？ 由 MAR 条件：

  $$E[r_i(\theta) \mid L^+_i] = E[r_i(\theta) \mid L^+_i, \delta_i=1]$$

  所以：

  $$E[\tilde{r}_i(\theta)] = E[\delta_i \cdot r_i + (1-\delta_i) \cdot E[r_i \mid L^+_i]] = E[r_i(\theta)]$$

  插补项"填补"了缺失单元的期望残差。

  同样，对 $\bar\theta$-residual 的平方：

  $$\tilde{s}i(\bar\theta) = \delta_i \cdot r_i(\bar\theta)^2 + (1-\delta_i) \cdot \hat{v}{\bar\theta}(L^+_i)$$

  其中 $\hat{v}_{\bar\theta}(L^+) = \widehat{E}[r(\bar\theta)^2 \mid L^+, \delta=1]$。

  代码实现（pci_moment_objective.py:42）：

  # 对有 W 的样本：直接算真实残差
  residual[obs_mask] = y[obs_mask] - h(concat(W[obs_mask], A[obs_mask]))

  # 对缺失 W 的样本：用插补模型填入 m̂_θ(L+)
  residual[~obs_mask] = m_theta[~obs_mask]   # m_theta 由学习循环传入

  # 同理构造 s̃
  s_tilde[obs_mask] = residual_obs.pow(2)
  s_tilde[~obs_mask] = v_theta_bar[~obs_mask]

  # 最终目标
  moment = (f(L) * residual).mean()          # E[f · r̃]
  f_reg  = lambda * (f(L)^2 * s_tilde).mean()  # λ · E[f² · s̃]

  ---
  第四部分：Cross-Fitting（为什么必须）

  问题

  $\hat{m}_\theta$ 是用数据训练出来的，如果用同一批数据来：
  1. 训练 $\hat{m}_\theta$
  2. 用 $\hat{m}\theta$ 来评估 $U(\theta, \tau)$ 并更新 $h\theta$

  那么 $h_\theta$ 会通过 $\hat{m}_\theta$ 过拟合同一份数据，产生 feedback loop（自举偏差）。

  解决方案：K-Fold Cross-Fitting

  把 $n$ 个样本分为 $K$ 折。对第 $k$ 折（$I_k$）的样本：
  - 用其余 $K-1$ 折（$I_{-k}$）的 complete cases 训练 $\hat{m}^{(-k)}\theta$ 和 $\hat{v}^{(-k)}{\bar\theta}$
  - 用这个 out-of-fold 模型来预测 $I_k$ 中缺失单元的插补值

  这样对每个样本 $i \in I_k$，其插补值由没有用过 $i$ 的模型给出——切断了 feedback loop。

  代码实现（learning_pci.py:70）：

  def _cross_fit_models(self, data, residual, residual_sq):
      for k, val_idx in enumerate(self.fold_indices):
          # 训练数据 = 其余所有 fold
          train_idx = cat([fold_indices[j] for j != k])
          obs_train = train_idx[delta[train_idx]]   # 只用 complete cases

          m_model = fit_ridge(features[obs_train], residual[obs_train])
          v_model = fit_ridge(features[obs_train], residual_sq[obs_train])

          # 对 val_idx（第 k 折）做 out-of-fold 预测
          m_all[val_idx] = m_model.predict(features[val_idx])
          v_all[val_idx] = clamp(v_model.predict(features[val_idx]), min=1e-8)

  插补模型：使用 岭回归（_fit_linear_model），特征是 $L^+ = (A, Z, Y)$。这是一个简单但足够的线性模型，用于估计 $E[r(\theta) \mid L^+]$。

  ---
  第五部分：$\bar\theta$ 的 EMA 更新

  目标函数里有两个不同的 $\theta$：
  - $\theta$：当前正在优化的参数（通过 $r_i(\theta)$ 进入 moment 项，有梯度）
  - $\bar\theta$：一个"慢速参考"版本（通过 $s_i(\bar\theta)$ 进入 f_reg 项，梯度截断）

  $\bar\theta$ 的作用：提供一个稳定的方差估计 $E[r^2 \mid L^+]$，避免 critic 的正则项随 $\theta$ 剧烈波动。

  更新方式（指数移动平均 EMA）：

  $$\bar\theta \leftarrow (1-\alpha)\bar\theta + \alpha\theta, \quad \alpha = 0.05$$

  即每步只把当前 $\theta$ 的 5% 掺入 $\bar\theta$，$\bar\theta$ 缓慢跟踪 $\theta$。

  代码（learning_pci.py:106，已修复后）：

  def _update_theta_bar(self):
      for key in self.theta_bar.keys():
          self.theta_bar[key] = (1 - ema_alpha) * self.theta_bar[key] \
                               + ema_alpha * current[key].detach()
          #                    ↑ 保留 95% 旧值   ↑ 掺入 5% 新值，detach() 阻断梯度

  在训练循环里：先 load $\bar\theta$ 模型，计算 $v_{\bar\theta}$（残差平方的插补），然后 backward。$v_{\bar\theta}$ 的计算结果在传入 calc_objective 之前已经 .detach()，所以没有梯度流回
  $\bar\theta$。

  ---
  第六部分：训练循环整体流程

  每个 epoch：
    ① h_bar ← load θ̄ 的模型快照
    ② 全量计算 residual = Y - h_θ(W,A)   [for complete cases]
    ③ 全量计算 residual_bar = Y - h_̄θ(W,A) [for complete cases]
    ④ cross-fit: m_all, v_all ← 用 residual, residual_bar² 训练 K 折插补模型

    ⑤ 随机打乱样本，分 batch：
       for batch in batches:
         h_loss, f_loss = calc_objective(h, f, ..., m_all[batch], v_all[batch])
         h_loss.backward()  → 更新 h_θ（minimise moment）
         f_loss.backward()  → 更新 f_τ（maximise moment）

    ⑥ theta_bar ← EMA update

  关键：步骤 ①-④ 用全量数据计算插补值（m_all, v_all 是全部 n 个样本的向量），步骤 ⑤ 按 batch 索引取子集 m_all[idx] 传入 objective。

  ---
  第七部分：ATE 估计（estimate_beta）

  训练完成后，$\hat\beta(a) = E[h_{\hat\theta}(W_i, a)]$。但 $W$ 有缺失，所以同样需要插补：

  $$\hat\beta(a) = \frac{1}{n}\sum_{i=1}^n \hat{q}_{a,\hat\theta}^{(-k(i))}(L^+_i)$$

  其中：
  - 对 $\delta_i=1$ 的单元：$\hat{q} = h_{\hat\theta}(W_i, a)$（直接算）
  - 对 $\delta_i=0$ 的单元：$\hat{q}^{(-k)}$ 是用 $I_{-k}$ 中 complete cases 拟合的线性模型，目标变量是 $h_{\hat\theta}(W_j, a)$，特征是 $L^+_j$

  这也是 cross-fitting 的：预测第 $k$ 折缺失单元用的 $\hat{q}$，是从不含第 $k$ 折的数据训练出来的。

  代码（learning_pci.py:170）：

  # 先算所有 observed 单元的 h_θ(W_i, a)
  y_hat[delta] = h(concat(W[delta], a_col[delta]))

  # 再对 missing 单元做 cross-fit q̂ 预测
  for k, val_idx in fold_indices:
      train_obs = I_{-k} ∩ {δ=1}
      q_model = fit_ridge(L+[train_obs], y_hat[train_obs])   # 以 h_θ(W,a) 为目标
      y_hat[val_missing] = q_model.predict(L+[val_missing])

  return y_hat.mean()   # = β̂(a)

  ---
  第八部分：三种实验模式连接

  ┌──────────┬─────────────────────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
  │   mode   │                  数据处理                   │        使用的目标函数        │           ATE 估计           │
  ├──────────┼─────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ oracle   │ delta_w 全设为 1（假设 W 全观测）           │ 标准 moment，无需插补        │ 直接 $\bar h(W_i, a)$        │
  ├──────────┼─────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ modified │ 保留真实 delta_w                            │ MAR-imputed residuals        │ cross-fit $\hat{q}$          │
  ├──────────┼─────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ naive    │ 过滤掉 $\delta=0$ 行，只训练 complete cases │ 标准 moment on filtered data │ 只对 $\delta=1$ 的测试集平均 │
  └──────────┴─────────────────────────────────────────────┴──────────────────────────────┴──────────────────────────────┘

  oracle 是上界（无缺失），naive 是有偏基准，modified 是我们的方法。run_pci_compare.py 用 Monte Carlo 跨重复对比三者的 ATE bias。

  ---
  一句话串联全部

  ▎ 找 $h_\theta$ 使得 $E[f_\tau(A,Z) \cdot (Y - h_\theta(W,A))] = 0$ 对所有 $f_\tau$ 成立。当 $W$ 部分缺失时，把残差替换成 MAR-插补版本 $\tilde{r}$，用 cross-fitting
  切断插补模型和优化目标之间的 feedback loop，训练完后 $\hat\beta(a) = E[\hat{q}_{a,\hat\theta}(L^+)]$ 给出无偏的 ATE 估计。