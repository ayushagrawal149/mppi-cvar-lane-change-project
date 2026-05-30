# Theory: CVaR-MPPI for Highway Lane-Change Planning

This document gives the full theory behind the project in plain language while keeping
every step mathematically exact. It is written to match the project scope set out in the
weekly reports ([`submission/`](../submission/)) — in particular the **Week 3/4 problem
formulation** and the **two-stage (offline value iteration + online RA-MPPI) method** —
and it draws *only* on the five papers in [`literature/`](../literature/):

- **[P1]** Williams, Aldrich, Theodorou, *Model Predictive Path Integral Control: From Theory to Parallel Computation*, J. Guid. Control Dyn., 2017. — the MPPI algorithm.
- **[P2]** Rockafellar, Uryasev, *Optimization of Conditional Value-at-Risk*, J. of Risk, 2000. — the CVaR definition.
- **[P3]** Chow, Ghavamzadeh, *Algorithms for CVaR Optimization in MDPs*, NeurIPS 2014. — policy gradient / actor-critic and the Lagrangian view of CVaR.
- **[P4]** Chow, Tamar, Mannor, Pavone, *Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach*, NeurIPS 2015. — the CVaR MDP, CVaR-Bellman operator, value iteration.
- **[P5]** Yin, Zhang, Tsiotras, *Risk-Aware MPPI Using Conditional Value-at-Risk*, 2022. — how CVaR and MPPI are combined (RA-MPPI).

The four questions this document answers, **in order**, are:

1. MPPI in highway lane change, with full mathematics (§2).
2. The dynamic-programming concepts involved (§3).
3. What CVaR is and how it is defined (§4).
4. How CVaR is integrated with MPPI to solve the lane-change problem (§5).

§1 first states the **problem formulation** and supplies the mathematics that supports
every idea in it; §6 ties the theory to the actual code.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $x_k\in\mathbb R^{n_x}$ | joint ego–traffic state at step $k$ |
| $u_k=v_k+\epsilon_k\in\mathbb R^{n_u}$ | applied ego control = mean control $v_k$ + injected noise $\epsilon_k$ |
| $\epsilon_k\sim\mathcal N(0,\Sigma_\epsilon)$ | MPPI exploration noise on the control |
| $w_k$ | exogenous traffic disturbance (other drivers, model error) |
| $F(\cdot)$ / $\widetilde F(\cdot)$ | nominal (deterministic) / disturbed (stochastic) dynamics |
| $K,\;k=0,\dots,K-1$ | planning horizon and step index |
| $m=1,\dots,M$ | index over sampled control sequences ("rollouts") |
| $n=1,\dots,N$ | index over disturbed traffic futures per rollout |
| $q(x),\ \phi(x_K)$ | running state cost, terminal cost (cost-to-go) |
| $\ell(\tilde x_k),\ L(\tilde x)=\sum_k\ell(\tilde x_k)$ | running **risk** cost and its trajectory sum |
| $\lambda>0$ | MPPI temperature |
| $\alpha\in(0,1)$ | CVaR confidence level; the tail is the worst $1-\alpha$ fraction |
| $C_u$ | cap on admissible tail risk |
| $V^*(x,y)$ | CVaR-MDP optimal value on augmented state $(x,y)$ |

> **Convention note (read once).** The papers use two CVaR conventions. **[P2]** and the
> reports write the confidence level (e.g. $0.9,0.95,0.99$) and average the worst
> $1-\alpha$ fraction, so the Rockafellar–Uryasev factor is $\tfrac{1}{1-\alpha}$.
> **[P3]/[P4]** let $\alpha$ be the *tail fraction* itself, with factor $\tfrac1\alpha$.
> This document uses the **confidence-level** convention (matching Week 3/4 eq. 1); the
> $\beta$ that appears in the reports' Monte-Carlo estimator is the same confidence level.

---

## 1. Problem formulation (with full supporting mathematics)

This is the exact problem from Week 3/4 of the reports. We control one ego vehicle making a
lane change on a three-lane highway among uncertain surrounding traffic. Below, **every
modeling choice is justified mathematically**, not just stated.

### 1.1 The optimization problem

Over a horizon $K$, with state cost $q$, terminal cost $\phi$, and risk cost
$L(\tilde x)=\sum_{k=0}^{K-1}\ell(\tilde x_k)$, we seek the mean control sequence
$\mathbf v=(v_0,\dots,v_{K-1})$ solving
$$
\boxed{\;
\min_{\mathbf v}\;
\mathbb E\!\left[\phi(x_K)+\sum_{k=0}^{K-1}\Big(q(x_k)+\tfrac{\lambda}{2}\,v_k^\top\Sigma_\epsilon^{-1}v_k\Big)\right]
\quad\text{s.t.}\quad
\mathrm{CVaR}_\alpha\!\big(L(\tilde x)\big)\le C_u .
\;}
\tag{1}
$$
In words: *find the most efficient lane-change plan (reach the target lane, hold speed, stay
comfortable) whose **worst-case tail risk** stays under a budget $C_u$.* The next four
subsections supply the math behind each ingredient.

### 1.2 Two dynamics, and what "unbiased estimate" means mathematically

We use **two** models (the central device of **[P5]**):

- **Nominal, deterministic** dynamics for planning:
  $x_{k+1}=F(x_k,u_k)$.
- **Disturbed, stochastic** dynamics for risk evaluation:
  $\tilde x_{k+1}=\widetilde F(\tilde x_k,u_k,w_k)$, with $w_k$ the random traffic behavior.

The reports require that the nominal model be an **unbiased estimate** of the disturbed one.
Mathematically this is the statement that the nominal next state equals the *mean* of the
disturbed next state:
$$
F(x_k,u_k)=\mathbb E_{w_k}\!\big[\widetilde F(x_k,u_k,w_k)\big].
\tag{2}
$$
Equation (2) is what makes it legitimate to plan on $F$ (cheap, deterministic) while
measuring risk on $\widetilde F$ (expensive, stochastic): the planner's expected trajectory
is the mean of the true random trajectory, so it introduces no systematic bias — only the
*spread* (which is exactly what CVaR will quantify).

### 1.3 Why the control is $u_k=v_k+\epsilon_k$ and where $\tfrac{\lambda}{2}v_k^\top\Sigma_\epsilon^{-1}v_k$ comes from

MPPI explores by **injecting Gaussian noise** into the control: $u_k=v_k+\epsilon_k$,
$\epsilon_k\sim\mathcal N(0,\Sigma_\epsilon)$. This is not arbitrary — it is the mechanism
that turns the optimal-control problem into an *expectation that can be sampled* (§2.2).
The quadratic term $\tfrac{\lambda}{2}v_k^\top\Sigma_\epsilon^{-1}v_k$ is **not** a hand-added
regularizer: it is forced by the theory. It is exactly the control-cost weighting that
appears when the path-integral / free-energy transformation is applied (§2.2, **[P1]**), and
it equals the Kullback–Leibler "price" of shifting the noise distribution by a mean control
$v_k$ (the Radon–Nikodym derivative between $\mathcal N(v_k,\Sigma_\epsilon)$ and
$\mathcal N(0,\Sigma_\epsilon)$). The temperature $\lambda$ trades exploration against
exploitation. The full derivation is §2.2.

### 1.4 The cost functions

The **state cost** $q$ encodes the task (Week 3, and `src/costs/lane_change.py`):
$$
q(x,u)=w_{\text{lane}}(Y-Y_{\text{tgt}})^2+w_{\text{speed}}(v-v_{\text{des}})^2
       +w_{\text{coll}}\,\Phi_{\text{coll}}(x,\text{traffic})+w_{\text{act}}\|u\|^2_w ,
$$
i.e. *track the target-lane center, hold desired speed, repel from other cars (a smooth
Gaussian collision potential), and penalize violent controls.*

The **risk cost** $\ell$ is a separate, safety-only quantity (Week 3): a
time-to-collision–shaped potential plus a lateral-encroachment indicator, summed to
$L(\tilde x)=\sum_k\ell(\tilde x_k)$. Crucially $L$ is treated as a **random variable**
because $\tilde x$ follows the *stochastic* dynamics — this is the object CVaR acts on.

### 1.5 Why a CVaR **constraint**, not an expected-cost term

The objective in (1) is an **expectation**, which is *risk-neutral*: it optimizes average
behavior and is provably blind to rare catastrophes. Formally, for a maneuver whose risk
$L$ is small with probability $0.98$ but huge with probability $0.02$, the mean
$\mathbb E[L]$ can be small even though the bad tail is unacceptable. The fix is to bound the
**tail** explicitly with $\mathrm{CVaR}_\alpha(L)\le C_u$, where (§4)
$$
\mathrm{CVaR}_\alpha(L)=\mathbb E\big[L\mid L\ge\mathrm{VaR}_\alpha(L)\big]
$$
is the average cost over the worst $1-\alpha$ fraction of traffic outcomes. Separating
"efficiency" (the expectation) from "safety" (the CVaR constraint) is what lets the planner
be aggressive on average yet safe in the tail — the project's core thesis.

### 1.6 Turning the constraint into something solvable

A hard constraint inside a sampling planner is awkward, so we relax it with a **Lagrangian**
(the standard CVaR device of **[P3]** eq. 4):
$$
\min_{\mathbf v}\;\max_{A\ge0}\;
J(\mathbf v)+A\big(\mathrm{CVaR}_\alpha(L)-C_u\big),
\qquad J(\mathbf v)=\mathbb E\Big[\phi(x_K)+\textstyle\sum_k\big(q+\tfrac\lambda2 v_k^\top\Sigma_\epsilon^{-1}v_k\big)\Big].
\tag{3}
$$
For a fixed penalty weight $A$, (3) becomes an **unconstrained** problem: minimize
$J(\mathbf v)+A\,(\mathrm{CVaR}_\alpha(L)-C_u)^+$. This is precisely the per-rollout penalty
RA-MPPI adds (§5.4); the indicator "only penalize when the budget is exceeded"
($\mathbf 1\{\mathrm{CVaR}_\alpha>C_u\}$) is the realization of the complementary-slackness
condition $A(\mathrm{CVaR}_\alpha-C_u)=0$. **This is the bridge from the formulation (1) to
the algorithm (§5).**

---

## 2. MPPI for the highway lane-change problem

### 2.1 The vehicle model

The ego car is a **4-state kinematic bicycle** (Week 3):
$$
\dot X=v\cos\psi,\quad \dot Y=v\sin\psi,\quad \dot\psi=\tfrac{v}{L}\tan\delta,\quad \dot v=a,
$$
state $x=(X,Y,\psi,v)$, control $u=(a,\delta)$, wheelbase $L$. The discrete map $F$ in
(2) is the forward-Euler step of this ODE. In code
([`src/dynamics/bicycle.py`](../src/dynamics/bicycle.py)) a slip-angle variant
$\beta=\arctan(\tfrac12\tan\delta)$ with lever $L/2$ is used so the planner's integrator
matches HighwayEnv exactly (removing model-mismatch artifacts); the standard form above is
recovered with lever $L$.

### 2.2 Derivation: MPPI as a path-integral solution of stochastic optimal control

This is the mathematics promised in Week 1 ("derive the MPPI update from the Bellman
principle"). Define the **state cost of a trajectory** $\tau$ under the deterministic
dynamics driven by noise:
$$
S(\tau)=\phi(x_K)+\sum_{k=0}^{K-1}q(x_k).
$$
Let $\mathbb P$ be the distribution of trajectories under the *uncontrolled* base noise
($v_k\equiv0$, $u_k=\epsilon_k$). The **free energy** (**[P1]**) is
$$
\mathcal F(S)=-\lambda\log \mathbb E_{\mathbb P}\!\Big[\exp\!\big(-\tfrac1\lambda S(\tau)\big)\Big].
$$
For *any* controlled trajectory distribution $\mathbb Q$, a Jensen / Legendre-duality
argument gives the bound
$$
\mathcal F(S)\;\le\;\mathbb E_{\mathbb Q}\big[S(\tau)\big]+\lambda\,D_{\mathrm{KL}}\!\big(\mathbb Q\,\|\,\mathbb P\big),
\tag{4}
$$
and the bound is **tight** for the optimal distribution $\mathbb Q^*$ defined by
$$
\frac{d\mathbb Q^*}{d\mathbb P}(\tau)=\frac{\exp\!\big(-\tfrac1\lambda S(\tau)\big)}{\mathbb E_{\mathbb P}\!\big[\exp(-\tfrac1\lambda S)\big]}.
\tag{5}
$$
So the best controller is the one whose induced distribution is closest (in KL) to
$\mathbb Q^*$. Parameterizing $\mathbb Q$ by the mean control $\mathbf v$ and minimizing the
KL term in (4), the right-hand-side control cost becomes exactly
$\tfrac{\lambda}{2}\sum_k v_k^\top\Sigma_\epsilon^{-1}v_k$ — **this is where the quadratic
term of (1) comes from** — and the optimal mean control is the
$\mathbb Q^*$-expectation of the applied controls:
$$
v_k^*=\mathbb E_{\mathbb Q^*}[u_k]
=\frac{\mathbb E_{\mathbb P}\big[\exp(-\tfrac1\lambda S)\,u_k\big]}{\mathbb E_{\mathbb P}\big[\exp(-\tfrac1\lambda S)\big]}.
\tag{6}
$$
Equation (6) is an **expectation**, hence estimable by Monte-Carlo sampling — that is MPPI.
(The exponential-of-value structure in (5) is the discrete-time image of the linearized
**Hamilton–Jacobi–Bellman** equation under a Cole–Hopf transform; this is the precise sense
in which "each rollout is an approximate Bellman backup," Week 1.)

### 2.3 The MPPI algorithm (what the code does)

Replacing the expectation in (6) by the sample average over $M$ rollouts gives the loop in
[`src/controllers/mppi.py`](../src/controllers/mppi.py). At state $x_0$:

**Step 1 — Sample** (`_sample_noise`). Draw $M$ control sequences $u^m=\mathbf v+\epsilon^m$, with $\epsilon^m\sim\mathcal N(0,\Sigma_\epsilon)$.

**Step 2 — Roll out & cost** (`_rollout_cost`). Propagate each $u^m$ under $F$ and sum its cost (**[P5]** eq. 3):

$$
S_m=\phi(x_K^m)+\sum_{k=0}^{K-1}\Big(q(x_k^m)+\gamma\,v_k^\top\Sigma_\epsilon^{-1}(v_k+\epsilon_k^m)\Big).
$$

**Step 3 — Weight** by softmin (**[P5]** eq. 5), where $\beta=\min_m S_m$ is subtracted for numerical stability only:

$$
\omega_m=\exp\!\Big(-\tfrac1\lambda(S_m-\beta)\Big).
$$

**Step 4 — Average** (the sampled form of (6), **[P5]** eq. 4):

$$
\mathbf v^{+}=\frac{\sum_m\omega_m\,u^m}{\sum_m\omega_m}.
$$

**Step 5 — Apply** (`step`). Execute $v_0^{+}$, shift $\mathbf v$ forward (warm start), and re-plan.

### 2.4 Why MPPI suits lane change

Other cars create sharp, non-convex, even discontinuous cost walls. MPPI never
differentiates the cost (it only evaluates $S_m$), so collision indicators and hard
constraints are admissible. It is **embarrassingly parallel** — thousands of rollouts on a
GPU — meeting the $\simeq 100$ ms control budget. The smooth Gaussian collision potential in
`step_cost` ensures even crash-free rollouts feel a gradient pulling the weighted average
away from danger.

> **Gap that motivates everything else.** Step 2 sums *expected* cost; the controller is
> risk-neutral and tail-blind (§1.5). Closing this gap with CVaR is the project.

---

## 3. The dynamic-programming concepts involved

### 3.1 The lane change as a Markov Decision Process

All five papers cast the task as an **MDP** $\mathcal M=(\mathcal X,\mathcal A,C,P,x_0,\gamma)$
(**[P4]** §2.2, **[P3]** §2): states $x$, actions $u$, cost $C(x,u)=q$, transition kernel
$P(\cdot\mid x,u)$, discount $\gamma$, start $x_0$. **Markov** = the next state depends only
on the present state and action. Once a policy $\mu$ is fixed, $P$ defines a **Markov chain**
over states; here its randomness is the uncertain traffic (the $w_k$ of §1.2).

### 3.2 Risk-neutral DP: value iteration, policy iteration, policy training

Classical DP minimizes **expected discounted cost** (**[P4]** §1):
$\min_\mu \mathbb E\big[\sum_t\gamma^t C(x_t,\mu(x_t))\mid x_0\big]$, solved by

- **Value iteration:** iterate the Bellman optimality operator
  $V_{k+1}(x)=\min_u\big[C(x,u)+\gamma\sum_{x'}P(x'\mid x,u)V_k(x')\big]$; it is a
  $\gamma$-contraction, so $V_k\to V^*$.
- **Policy iteration:** alternate policy evaluation ($V^\mu$) and greedy improvement.
- **Policy training (policy gradient):** for continuous/large spaces, parameterize the
  policy and follow the objective gradient — the RL view of **[P3]**.

The continuous-time limit of the Bellman equation is the **Hamilton–Jacobi–Bellman (HJB)**
PDE; §2.2 showed MPPI approximates its solution by sampling. MPPI itself is *receding-horizon*:
it solves the **finite-$K$** version online from $x_0$, executes one step, and re-plans —
a fast approximation to the full DP problem.

### 3.3 Risk-sensitive DP: the CVaR MDP

**[P4]** and **[P3]** replace the expectation with CVaR. The **CVaR MDP** (**[P4]** eq. 3) is
$$
\min_\mu\;\mathrm{CVaR}_\alpha\!\Big(\lim_{T\to\infty}\sum_{t=0}^T\gamma^tZ_t\,\Big|\,x_0,\mu\Big).
$$
A key result (**[P4]** Prop. 1) is that **minimizing CVaR = being robust** to worst-case
perturbations of the transition probabilities within a budget — so a CVaR-optimal
lane-change policy is automatically robust to a mis-modeled traffic distribution.

### 3.4 Dynamic programming *for* CVaR (the offline foundation of this project)

CVaR is not an expectation, so the naive backup fails. **[P4]** §3 fixes this:

1. **CVaR Decomposition Theorem** (**[P4]** Thm 2): the future CVaR decomposes recursively,
   but the next step uses a *different* confidence level — so we **augment the state** with the
   confidence variable $y\in(0,1]$.
2. On $(x,y)$ there is a genuine **CVaR-Bellman operator** (**[P4]** eq. 6; this is exactly
   Week 3/4 eq. 2):
   $$
   \mathbf T[V](x,y)=\min_{u}\Big[C(x,u)+\gamma\!\!\max_{\xi\in\mathcal U_{\mathrm{CVaR}}(y,P)}\sum_{x'}\xi(x')\,V\big(x',y\,\xi(x')\big)P(x'\mid x,u)\Big].
   \tag{7}
   $$
   The inner $\max$ over the **risk envelope** $\mathcal U_{\mathrm{CVaR}}$ is the dual form of
   CVaR (worst-case reweighting; §4.3). $\mathbf T$ is a **$\gamma$-contraction** and
   **concavity-preserving** (**[P4]** Lemma 3), so value iteration $V_{t+1}=\mathbf T[V_t]$
   converges (**[P4]** Thm 4) and a **stationary Markov policy on $(x,y)$ is optimal**
   (**[P4]** Thm 5). Because $y$ is continuous, **[P4]** Algorithm 1 runs value iteration on a
   finite grid of log-spaced $y$-points with linear interpolation.

When the model is unknown/too large, **[P3]** instead **trains a parameterized policy**: it
Lagrangianizes the CVaR-constrained objective (**[P3]** eq. 4 — the same relaxation as our
§1.6) and runs a **three-timescale policy-gradient / actor-critic** scheme on
$(\theta,\nu,\lambda)$ (policy parameters, the VaR variable, the multiplier) to a locally
risk-optimal saddle point.

> **How this project uses §3.4.** The plan (Week 3/4 §IV-A) is the **offline component**:
> discretize a coarse three-lane MDP over (lane index, gap-to-lead, ego speed) on $\sim 10^3$
> cells with $21$ log-spaced confidence points, run interpolated CVaR value iteration (7) to
> get $V^*(x,y)$, and **load it as the MPPI terminal cost** $\phi(x_K)\leftarrow V^*(\Pi x_K,y)$
> ($\Pi$ snaps the continuous terminal state to the grid). This supplies a tail-risk–shaped
> *cost-to-go beyond the rollout horizon* that a finite-$K$ MPPI rollout cannot see. Offline
> cost $\mathcal O(|\mathcal X||\mathcal Y||\mathcal A|\,C_\zeta\,n_{\text{iter}})$ is paid once;
> online lookups are microsecond interpolations. This is standard **MPC-with-terminal-value-function**:
> truncating the infinite-horizon DP at $K$ and using $\phi=V^*$ as the tail makes the
> finite-horizon objective in (1) a consistent approximation of the full CVaR-MDP problem of §3.3.

---

## 4. What is CVaR, and how is it defined?

### 4.1 Intuition

Treat a maneuver's risk cost $L$ as a **random variable** (random because traffic is
random). Two tail questions:

- **VaR (Value-at-Risk):** "what cost am I 95% sure I won't exceed?" — a *quantile*.
- **CVaR (Conditional Value-at-Risk):** "*given* I land in the worst 5%, what is my *average*
  cost there?" — the **mean of the tail beyond VaR**. CVaR therefore captures *how bad* the
  rare disasters are, not merely how often they occur (**[P2]**, **[P4]** §2.1).

### 4.2 Formal definitions

For a cost $Z$ with CDF $F(z)=\mathbb P(Z\le z)$ and confidence level $\alpha\in(0,1)$:
$$
\mathrm{VaR}_\alpha(Z)=\min\{\,z:\mathbb P(Z\le z)\ge\alpha\,\},
\qquad
\boxed{\;\mathrm{CVaR}_\alpha(Z)=\mathbb E\big[\,Z\mid Z\ge\mathrm{VaR}_\alpha(Z)\,\big]\;}
$$
(the boxed form is exactly the reports' eq. 1 / **[P4]** §2.1, valid when $Z$ has no atom at
the VaR point).

### 4.3 The Rockafellar–Uryasev variational form (the workhorse)

Computing CVaR from 4.2 needs VaR first. **[P2]** (Thm 1) removes that: CVaR is the minimum
of a simple convex function (**[P2]** eq. 4–5, **[P3]** eq. 1, **[P4]** eq. 1):
$$
\boxed{\;
\mathrm{CVaR}_\alpha(Z)=\min_{w\in\mathbb R}\Big\{\,w+\tfrac{1}{1-\alpha}\,\mathbb E\big[(Z-w)^+\big]\Big\},\qquad (t)^+=\max(t,0).
\;}
\tag{8}
$$
Two facts that make CVaR usable in optimization:

- $F_\alpha(w)=w+\tfrac{1}{1-\alpha}\mathbb E[(Z-w)^+]$ is **convex** in $w$, and the minimizer
  $w^*$ **equals $\mathrm{VaR}_\alpha(Z)$** — VaR comes free (**[P2]** Thm 1).
- CVaR is a **coherent** risk measure (convex, monotone, translation-equivariant,
  positively homogeneous; **[P2]** §1), unlike VaR.

A dual / robustness reading (**[P4]** eq. 2) underlies the inner $\max$ of the CVaR-Bellman
operator (7):
$$
\mathrm{CVaR}_\alpha(Z)=\max_{\xi\in\mathcal U_{\mathrm{CVaR}}(\alpha,\mathbb P)}\mathbb E_\xi[Z],
\qquad
\mathcal U_{\mathrm{CVaR}}=\Big\{\xi:\xi(\omega)\in\big[0,\tfrac{1}{1-\alpha}\big],\ \textstyle\int\xi\,d\mathbb P=1\Big\},
$$
i.e. CVaR is the **worst-case expectation under a reweighted distribution** — the precise
link between "risk-sensitive" and "robust" of §3.3.

### 4.4 Estimating CVaR from samples (Monte-Carlo)

We don't know $L$'s distribution in closed form, but we can sample it. With samples
$z_1,\dots,z_q$, **[P2]** eq. 9 gives the sample version of (8):
$$
\widetilde F_\alpha(w)=w+\frac{1}{q(1-\alpha)}\sum_{i=1}^q(z_i-w)^+,
\qquad \widehat{\mathrm{CVaR}}_\alpha=\min_w\widetilde F_\alpha(w).
$$
Equivalently, the **tail-average estimator** of **[P5]** eq. 12 (and reports' eq. 3): sort
the sampled costs, keep the $N_o$ above $\widehat{\mathrm{VaR}}_\alpha$, and average them. This
sampling estimate is what makes CVaR compatible with the sampling planner MPPI (§5).

---

## 5. Integrating CVaR with MPPI for highway lane change (RA-MPPI)

This is the project's method, following **[P5]** (RA-MPPI). The repo's
[`mppi.py`](../src/controllers/mppi.py) docstring states the design exactly: *the CVaR
augmentation reuses the MPPI sampler and only modifies the per-rollout cost aggregator.* The
full system is **two-stage**: §3.4's offline value iteration supplies $\phi=V^*$, and the
online RA-MPPI below enforces the tail constraint.

### 5.1 The risk-constrained control problem (restated)

This is exactly (1)/(3): minimize the MPPI objective subject to
$\mathrm{CVaR}_\alpha(L(\tilde x))\le C_u$ (**[P5]** eq. 8). §1.6 already reduced the
constraint to a penalty; §5.4 realizes it per rollout.

### 5.2 Two dynamics, again

Nominal $F$ drives the $M$ MPPI rollouts and their base costs $S_m$ (§2.3). Disturbed
$\widetilde F$ (with arbitrary, possibly non-Gaussian $w_k$ — e.g. a stochastic Intelligent
Driver Model with Gaussian acceleration noise $w_k\sim\mathcal N(0,\sigma^2)$ and a 2%
per-step lane-change probability, Week 3) is used **only** to evaluate risk. Equation (2)
guarantees $F$ is the unbiased mean of $\widetilde F$, so this split is principled.

### 5.3 Monte-Carlo CVaR for each candidate maneuver

For **each** sampled control sequence $u^m$, additionally simulate $N$ disturbed futures,
yielding risk samples $L(\tilde x^{m,n})$, $n=0,\dots,N-1$. Estimate that maneuver's tail
risk by §4.4 (**[P5]** eq. 12 / reports' eq. 3):
$$
\widehat{\mathrm{CVaR}}^{\,m}_\alpha=\frac{1}{N_o}\sum_{n:\,L(\tilde x^{m,n})\ge\widehat{\mathrm{VaR}}^{\,m}_\alpha} L(\tilde x^{m,n}).
\tag{9}
$$
Every candidate lane change now carries a number: *"if things go badly, how bad on average?"*

### 5.4 Soft trajectory filtering — the constraint becomes a weight

Realizing the Lagrangian penalty of §1.6 **per rollout** (**[P5]** eq. 14–16):
$$
J_C^m=A\cdot\widehat{\mathrm{CVaR}}^{\,m}_\alpha\cdot\mathbf 1\big\{\widehat{\mathrm{CVaR}}^{\,m}_\alpha>C_u\big\},
\qquad
\tilde S_m=J_C^m+S_m,
\tag{10}
$$
then apply the **ordinary** MPPI weighting/averaging to the modified costs (**[P5]** eq. 17, 4):
$$
\tilde\omega_m=\exp\!\Big(-\tfrac1\lambda(\tilde S_m-\min_{m}\tilde S_m)\Big),
\qquad
\mathbf v^{+}=\frac{\sum_m\tilde\omega_m\,u^m}{\sum_m\tilde\omega_m}.
\tag{11}
$$
The indicator is complementary slackness ($A(\mathrm{CVaR}-C_u)=0$): a maneuver is punished
*only if* its tail risk exceeds the budget. This is the "reuse the sampler, change only the
aggregator" plan — all of §2.3 is untouched; we just add $J_C^m$ before computing weights, so
dangerous maneuvers are down-weighted out of the blend.

### 5.5 Risk-cost sensitivity scaling (practical fix)

If $J_C^m$ is tiny next to $\tilde S_m$, the controller ignores risk; if $A$ is too large it
becomes over-conservative. **[P5]** §III-C rescales the *variance* of the risk samples while
fixing their mean (**[P5]** eq. 18–19):
$$
L_a(\tilde x^{m,n})=B\big(L(\tilde x^{m,n})-\bar L^m\big)+\bar L^m,
\qquad \bar L^m=\tfrac1N\textstyle\sum_n L(\tilde x^{m,n}),
$$
so $B>1$ increases sensitivity to risk-level changes without simply inflating the penalty.

### 5.6 The full per-step loop (offline $V^*$ + online RA-MPPI; **[P5]** Algorithm 1)

1. **(offline, once)** Compute $V^*(x,y)$ by CVaR value iteration (7); set $\phi(x_K)=V^*(\Pi x_K,y)$.
2. Read $x_0$.
3. **Parallel over $m$:** sample $u^m=\mathbf v+\epsilon^m$, roll out under $F$, accumulate $S_m$ (with terminal $\phi=V^*$).
4. **Parallel over $m$:** simulate $N$ disturbed futures, compute (variance-scaled) $\widehat{\mathrm{CVaR}}^{\,m}_\alpha$ (9).
5. Add penalty (10): $\tilde S_m=S_m+A\,\widehat{\mathrm{CVaR}}^{\,m}_\alpha\,\mathbf 1\{\cdot>C_u\}$.
6. Risk-aware weights and update (11); apply $v_0^{+}$, warm-start, repeat.

### 5.7 Why this solves the lane-change problem, and what to expect

The offline $V^*$ injects the **long-horizon** tail-risk cost-to-go (what a short rollout
misses); the online RA-MPPI injects **fine-grained, real-time** interaction with the
*current* surrounding vehicles (what the coarse grid misses). Together they realize the
constrained problem (1). In **[P5]**'s experiments, RA-MPPI cut collisions by **~50–80%**
versus plain MPPI at **about the same lap time**, under Gaussian, uniform, and impulsive
disturbances. For lane change this predicts: the CVaR planner completes the maneuver about as
quickly/smoothly as risk-neutral MPPI but with far fewer crashes in the dangerous tail —
exactly the **hard-brake** and **cut-in** scenarios in
[`src/experiments/scenarios.py`](../src/experiments/scenarios.py). Tightening $C_u$ or raising
$\alpha$ trades a little performance for more safety — a tunable risk dial grounded in §3's DP
theory and §4's CVaR definition. The evaluation plan (Week 3) sweeps
$\alpha\in[0.7,0.99]$ and $C_u$, reporting collision rate, near-miss rate (TTC < 1.5 s),
lane-change completion time, and control smoothness.

---

## 6. Map to the implementation and to the reports

| Theory element | Where it lives |
|---|---|
| Problem formulation (1)–(3) | Week 3/4 §III; this doc §1 |
| Nominal dynamics $F$ | [`src/dynamics/bicycle.py`](../src/dynamics/bicycle.py) |
| State cost $q$, collision potential | [`src/costs/lane_change.py`](../src/costs/lane_change.py) |
| Risk-neutral MPPI loop (currently implemented baseline) | [`src/controllers/mppi.py`](../src/controllers/mppi.py) |
| CVaR penalty $J_C^m$ ("reuse sampler, change aggregator") | planned augmentation, `mppi.py` docstring; §5.4 |
| Offline CVaR value iteration $V^*$ | Week 3/4 §IV-A; this doc §3.4 |
| Scenarios (hard-brake, cut-in) | [`src/experiments/scenarios.py`](../src/experiments/scenarios.py) |

### Summary: paper → topic

| Topic | Primary sources |
|---|---|
| Problem formulation & supporting math (§1) | **[P5]** §III, **[P3]** eq. 4, **[P2]** |
| MPPI formulation/derivation/algorithm (§2) | **[P1]**, **[P5]** §II |
| MDP / value iteration / policy iteration / policy training (§3) | **[P4]** §2–3, **[P3]** §2–4 |
| CVaR definition & Rockafellar–Uryasev form (§4) | **[P2]**, **[P3]** §2, **[P4]** §2.1 |
| CVaR + MPPI integration, RA-MPPI (§5) | **[P5]** §III–IV |
