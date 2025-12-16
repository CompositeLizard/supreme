---
---
## 1) Lecture 3 Overview: From Regression to Classification

Welcome to lecture 3. Today is about **classification and regression**. We’re moving from the first task we did last time—regression, where we fit a line—to a task that looks very similar at first but has a few subtle differences: **classification**.

Recall that classification is for discrete labels, like identifying the animal in a photo: cat, pig, horse, and so on. These problems are extremely common in machine learning.

The plan today is to start with a **probabilistic view of linear regression**. The reason is that we want to reinterpret what we did last time in a way that naturally extends to classification and, on Wednesday, to a richer class of models called **exponential family models**. At a high level, the theme is that these models all look the same: we want an abstraction that lets us solve them, do inference, and reason about them in a uniform way.

After that probabilistic view, we’ll talk about classification, and then introduce the workhorse model for classification: **logistic regression**. Confusingly, it has “regression” in the name even though it’s a classification algorithm—don’t blame us. This is a classical method originally developed by statisticians, and machine learning uses it heavily (sometimes under different names in deep learning, such as a linear layer plus softmax).

Finally, we’ll parallel the structure of the previous lecture and talk about how to solve the logistic regression optimization problem. In particular, we’ll introduce **Newton’s method**, which can converge very quickly when applicable, but has expensive steps and often isn’t practical for the largest-scale ML settings.

If you take one message from this lecture, it should be:
- what classification is,
- why it differs from regression,
- what logistic regression is,
- and how we can solve it.

As before, there’s an online thread for questions if you prefer asking there. Let’s get started.

## 2) Setup: Least Squares Regression (Reminder)

We begin with our familiar least squares setup. We are given a training set:

$$\{(x^{(i)}, y^{(i)})\}_{i=1}^{n}$$

Here,
- $$x^{(i)} \in \mathbb{R}^d$$ (or $$\mathbb{R}^{d+1}$$ if we use the bias convention),
- $$y^{(i)} \in \mathbb{R}$$ is real-valued (so this is regression).

The $$d+1$$ convention comes from appending a bias feature $$x_0 = 1$$ to every example. If you don’t remember it, just keep in mind that the “$$+1$$” is by convention.

Our goal in least squares regression was to find parameters $$\theta \in \mathbb{R}^{d+1}$$ that minimize the squared residuals:

$$
\theta
\in
\arg\min_{\theta}
\sum_{i=1}^{n}
\left(y^{(i)} - h_\theta(x^{(i)})\right)^2,
$$

where the hypothesis is:

$$
h_\theta(x) = \theta^T x.
$$

We wrote $$h_\theta(\cdot)$$ in a general form even though it’s just a dot product here, because we’ll reuse the same framework for other models.

## 3) The “Why” Question: Why Squared Error?

Last time, we minimized a sum of squares without much justification. Today we ask: **why that objective?** The point of answering “why” is that it tells us how to generalize the idea to classification and beyond.

This will introduce one of our favorite distributions in this course: the **Gaussian**.

## 4) A Probabilistic (Generative) View of Linear Regression

We now introduce our first true “generative model” view in the course. We assume the data were generated as:

$$
y^{(i)} = {\theta^*}^T x^{(i)} + \varepsilon^{(i)}.
$$

Here:
- $$\theta^*$$ is an unknown “true” parameter vector that generated the data.
- $$\varepsilon^{(i)}$$ is a noise (error) term for example $$i$$.

If there were no noise, the data would lie perfectly on a hyperplane. The noise represents whatever we can’t explain with our features—measurement error, unobserved factors, or simply missing information. Importantly, we do **not** observe $$\varepsilon^{(i)}$$ in the training set; it is a modeling construct that explains how $$x^{(i)}$$ and $$y^{(i)}$$ are linked.

This is called a “forward” or “generative” model because it describes how you could generate $$y$$ if you knew $$x$$, $$\theta^*$$, and how to draw $$\varepsilon$$.

## 5) Assumption 1: Noise Has Mean Zero

We typically assume the noise has mean zero:

$$
\mathbb{E}[\varepsilon^{(i)}] = 0.
$$

Intuitively, this says the noise is not systematically pushing outcomes up or down on average. It’s “unbiased.” Operationally, if there *were* a constant bias, the intercept term in $$\theta^*$$ could absorb it.

## 6) Assumption 2: Noise Terms Are Independent

We also often assume the noise terms are independent across examples. A strong form is:

$$
\mathbb{E}\!\left[\varepsilon^{(i)}\varepsilon^{(j)}\right]=
\mathbb{E}[\varepsilon^{(i)}]\;\mathbb{E}[\varepsilon^{(j)}]
\quad\text{for } i \neq j.
$$

The interpretation is: knowing the error on one training example doesn’t tell you anything about the error on another. This assumption is not always true in real data, but it lets us do clean mathematics and make progress.

At this stage, it’s useful to think of these assumptions not as “literally true,” but as “useful modeling assumptions.” In statistical modeling, the question is often less “is this exactly true?” and more “is this a useful approximation, and what do we give up by making it?”

## 7) Assumption 3: Constant Variance (Homoskedastic Noise)

We also need a way to quantify “how noisy” the data are. A standard assumption is that the noise has a constant variance:

$$
\mathrm{Var}(\varepsilon^{(i)}) = \sigma^2.
$$

Because $$\mathbb{E}[\varepsilon^{(i)}]=0$$, variance simplifies to:

$$
\mathrm{Var}(\varepsilon^{(i)})=
\mathbb{E}\!\left[(\varepsilon^{(i)})^2\right].
$$

### Question: Why does variance become $\mathbb{E}[(\varepsilon^{(i)})^2]$?

Variance is:

$$
\mathrm{Var}(\varepsilon) = \mathbb{E}\!\left[(\varepsilon - \mu)^2\right].
$$

Here $$\mu = \mathbb{E}[\varepsilon] = 0$$, so:

$$
\mathrm{Var}(\varepsilon) = \mathbb{E}[\varepsilon^2].
$$

### Question: What is $$\varepsilon^{(i)}$$ “as a value”?

It is a random scalar sampled for each example. A mental model is: for each example, the world generates a “clean” outcome ${\theta^*}^T x^{(i)}$, then adds a random perturbation $$\varepsilon^{(i)}$$, resulting in the observed label $$y^{(i)}$$.

This noise level matters. If $$\sigma^2$$ is large and your output differences are tiny (like trying to distinguish values that differ by 0.1 when the noise has variance 1), then you’re likely reading into noise. Later we’ll discuss how learning procedures scale with noise.

## 8) Why the Gaussian Appears

Here is the key point: if you assume the noise is unbiased (mean 0) and has a fixed variance $$\sigma^2$$, and you do not want to assume anything else (e.g., about higher moments), then a natural and, in a specific sense, *maximally uninformative* choice is the Gaussian distribution.

So we model:

$$
\varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2).
$$

This means:
- each $$\varepsilon^{(i)}$$ is drawn from a normal distribution,
- with mean 0,
- and variance $$\sigma^2$$.

## 9) Interpreting $$\mathcal{N}(\mu, \sigma^2)$$

A normal distribution $$\mathcal{N}(\mu, \sigma^2)$$ is centered at $$\mu$$ and has spread controlled by $$\sigma^2$$ (or equivalently $$\sigma$$, the standard deviation).

Graphically:
- the peak is at $$\mu$$,
- the distribution is symmetric around $$\mu$$,
- most probability mass lies near $$\mu$$.

A useful rule of thumb is that a large fraction of mass lies within one standard deviation:

$$
\mu \pm \sigma.
$$

(Exact percentages depend on the precise statement and convention; the main point is that the Gaussian is “peaked” around the mean and decays as you move away.)

## 10) Why Gaussians Show Up So Often

One reason Gaussians appear frequently is that when you have many small additive effects, the sum tends to look Gaussian (as suggested by the central limit theorem intuition). If noise is the accumulation of many tiny independent disturbances, then modeling it as Gaussian is often reasonable.

If the philosophical justification doesn’t resonate, that’s fine. The Gaussian is a convenient, widely-used model, and it gives us a clean path to derive the squared-error objective and generalize to classification.


## 11) Q&A and Notation Clarifications

**Question:** *Are we always going to assume a population structure in this class?*  

We will do a lot with population statistics. We won’t do much with sampling until a couple of key points, and the difference between those two won’t matter for the actual solves we’re doing right now. It’s a great question, but if that distinction doesn’t make sense yet, don’t worry.

**Question:** *Is $$\mu = 0$$ here?*  

Yes. In our earlier setting where we had $$\varepsilon^{(i)}$$ as the noise term, we assume its mean is 0. So for our noise model, $$\mu = 0$$.

## 12) The Gaussian PDF and What the Symbols Mean

We are looking at the Gaussian (normal) probability density function:

$$
p(z;\mu,\sigma^2)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(
-\frac{(z-\mu)^2}{2\sigma^2}
\right).
$$

There are a few important things to notice:

- The term $$\frac{1}{\sqrt{2\pi\sigma^2}}$$ is a **normalizing constant**. It ensures the total area under the curve is 1, which is what makes this a valid probability density function.
- The exponent contains a **quadratic term** $$-(z-\mu)^2$$, which is one reason least squares will show up later.
- The notation $$p(z;\mu,\sigma^2)$$ uses a **semicolon**. This is **not** conditioning. It means $$\mu$$ and $$\sigma^2$$ are **parameters** of the distribution. You don’t “condition on” parameters; you plug them in.

So:

- $$z$$ is the variable.
- $$\mu$$ and $$\sigma^2$$ are parameters.

We *will* use conditional probability soon, and that will be written with a vertical bar $$|$$.

## 13) Conditional Distribution for Linear Regression

Now we write something that *is* conditional: the distribution of $$y^{(i)}$$ given $$x^{(i)}$$ and parameter $$\theta$$.

From the generative model:

$$
y^{(i)} = \theta^T x^{(i)} + \varepsilon^{(i)},
\quad
\varepsilon^{(i)} \sim \mathcal{N}(0,\sigma^2),
$$

we get:

$$
p\!\left(y^{(i)} \mid x^{(i)};\theta\right)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(
-\frac{\left(y^{(i)}-\theta^T x^{(i)}\right)^2}{2\sigma^2}
\right).
$$

This answers: **given features $$x^{(i)}$$, what is the probability distribution over possible labels $$y^{(i)}$$?**

We often write this more compactly as:

$$
y^{(i)} \mid x^{(i)};\theta \sim \mathcal{N}\!\left(\theta^T x^{(i)}, \sigma^2\right).
$$

So this “compact mouthful” is the same statement as the expanded density above.

### Question: Why does conditioning on $$x$$ give a distribution over $$y$$?

Because the randomness comes from $$\varepsilon^{(i)}$$. Once we condition on $$x^{(i)}$$ (which we observe), the only remaining uncertainty in $$y^{(i)}$$ is due to the noise term, and that noise is Gaussian.

### Question: Is multiplication implied in expressions like $$\theta^T x^{(i)}$$?

Yes. It’s the usual dot product / matrix-vector multiplication.

## 14) Key Idea: Choosing $$\theta$$ Chooses a Distribution

A hidden but important point is: **picking $$\theta$$ picks a distribution**.

Once $$x^{(i)}$$ is fixed and $$\sigma^2$$ is fixed, choosing a value of $$\theta$$ determines the mean $$\theta^T x^{(i)}$$, which determines the conditional distribution of $$y^{(i)}$$. Different $$\theta$$ values lead to different distributions, and we can compare how well they “line up” with the observed data.

Intuitively: if a $$\theta$$ makes predictions close to observed $$y^{(i)}$$ values, then the observed data will look more likely under that model.

## 15) Likelihood: How Likely Is the Observed Data Under $$\theta$$?

We define the likelihood of parameters $$\theta$$ as:

$$
L(\theta)=
p\!\left(y^{(1)},\dots,y^{(n)} \mid x^{(1)},\dots,x^{(n)};\theta\right).
$$

Using the independence (iid-style) assumption, we can factor this as a product:

$$
L(\theta)=
\prod_{i=1}^{n} p\!\left(y^{(i)} \mid x^{(i)};\theta\right).
$$

Substituting the Gaussian form:

$$
L(\theta)=
\prod_{i=1}^{n}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(
-\frac{\left(y^{(i)}-\theta^T x^{(i)}\right)^2}{2\sigma^2}
\right).
$$

### Note on Parameters: What about $$\sigma^2$$?

A student pointed out correctly that the likelihood formally depends on $$\theta$$ **and** $$\sigma^2$$. For now, we treat $$\sigma^2$$ as fixed (given). You could also estimate it jointly, but that’s a slightly different model and requires a bit more work.

## 16) Log-Likelihood: Turning Products Into Sums

The product likelihood is awkward to optimize, so we take a log. Define:

$$
\ell(\theta) = \log L(\theta).
$$

Using $$\log \prod = \sum \log$$ and $$\log \exp(a) = a$$:

$$
\ell(\theta)=\sum_{i=1}^{n}
\left[
-\frac{1}{2}\log(2\pi\sigma^2)-
\frac{\left(y^{(i)}-\theta^T x^{(i)}\right)^2}{2\sigma^2}
\right].
$$

A student noted correctly that the log introduces terms like $$\log \sigma$$ (more precisely $$\log(2\pi\sigma^2)$$). That term does **not** depend on $$\theta$$, so when optimizing over $$\theta$$ it behaves like a constant.

## 17) From Maximum Likelihood to Least Squares

We want the most likely $$\theta$$, i.e. maximum likelihood:

$$
\theta_{\text{ML}} = \arg\max_{\theta} \ell(\theta).
$$

Because log is monotone, maximizing likelihood and maximizing log-likelihood are equivalent.

Dropping constants that do not depend on $$\theta$$, maximizing $$\ell(\theta)$$ is equivalent to minimizing:

$$
\sum_{i=1}^{n}
\left(y^{(i)}-\theta^T x^{(i)}\right)^2.
$$

The factor $$\frac{1}{2\sigma^2}$$ can also be ignored for the minimizer because multiplying the objective by a positive constant does not change which $$\theta$$ minimizes it. So we arrive at:

$$
\theta=
\arg\min_{\theta}
\frac{1}{2}\sum_{i=1}^{n}
\left(y^{(i)}-\theta^T x^{(i)}\right)^2,
$$

which is exactly least squares, the $$J(\theta)$$ from the previous lecture.

## 18) Why We Walked Through This

The point of going through this slowly is that we’ll reuse the same playbook repeatedly:

1. **Specify a probabilistic model** (a distribution).
2. Write down the **likelihood** of the observed data.
3. Take logs to get a **log-likelihood** (product → sum).
4. Turn maximizing likelihood into minimizing a **loss function**.
5. Solve it using the same optimization machinery (gradient descent, SGD, etc.).

This mapping from a distribution to a loss function will become nearly automatic over the next couple of lectures.

## 19) Question: What if $$\sigma^2$$ is unknown?

If $$\sigma^2$$ is not given, you can estimate it too. That becomes a model with unknown variance. The solution does not reduce as cleanly to just least squares; you do a little extra work to estimate $$\sigma^2$$, but it’s not complicated and may show up in homework. For now, we assume $$\sigma^2$$ is fixed.


# Classification and Logistic Regression

## 20) What classification is

In classification, we are still given training examples:

$$
\{(x^{(i)}, y^{(i)})\}_{i=1}^n,
$$

but now the labels are **discrete**. For binary classification we use:

$$
y^{(i)} \in \{0,1\}.
$$

(You could also encode the classes as $$\{-1,1\}$$, which can make some algebra cleaner, but in this course we’ll use $$\{0,1\}$$.)

By convention:
- $$0$$ is often called the **negative class**
- $$1$$ is often called the **positive class**

These names are just conventions. Example: “tumor present” vs “no tumor,” or “cat” vs “not cat.”

## 21) Why not just use linear regression?

A natural first thought is: “If $$y \in \{0,1\}$$, why not fit a line and then threshold?”  
Sometimes this works in practice, but it has problems:

- Regression is trying to minimize squared residuals, so a **single outlier** (a point far away) can pull the fitted line in a way that makes the induced decision boundary behave strangely.
- There is no reason a regression model’s raw outputs should naturally behave like probabilities.
- The model can produce predictions below 0 or above 1, which is awkward if we want a probabilistic interpretation.

This motivates using a model that treats the target as categorical from the start.

## 22) Logistic regression: the key trick

Logistic regression keeps the **linear score** but passes it through a nonlinear function so the output lies in $$[0,1]$$.

We define the hypothesis as:

$$
h_\theta(x) = g(\theta^T x),
$$

where the function $$g$$ is the **sigmoid**:

$$
g(z) = \frac{1}{1 + e^{-z}}.
$$

So explicitly:

$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}.
$$

### What the sigmoid does

The sigmoid “squashes” any real input into the interval $$[0,1]$$:

- If $$\theta^T x$$ is very large, $$h_\theta(x) \approx 1$$ (saturates near 1)
- If $$\theta^T x$$ is very negative, $$h_\theta(x) \approx 0$$ (saturates near 0)

This avoids the unbounded behavior linear regression can have.

### Why not use a step function?

A hard threshold (step) function would map directly to 0/1, but its derivative is zero almost everywhere, giving optimization algorithms no useful gradient signal. The sigmoid provides a **smooth transition** and supports gradient-based optimization.

### Link function terminology

The sigmoid is a popular **link function** (sometimes called an inverse link function in other literature). There are other link functions (probit, etc.), but sigmoid is the standard choice here.

## 23) Probabilistic interpretation

The core probabilistic move is:

$$
P(y=1 \mid x;\theta) = h_\theta(x),
$$

and since there are only two classes:

$$
P(y=0 \mid x;\theta) = 1 - h_\theta(x).
$$

This is testable as a calibration idea: if the model outputs 0.7 often, then roughly 70% of those cases should truly be class 1 (in a well-calibrated model).

## Likelihood for binary classification

Using conditional independence over examples, the likelihood is:

$$
L(\theta)
= \prod_{i=1}^n P(y^{(i)} \mid x^{(i)};\theta).
$$

A compact way to write the per-example probability is:

$$
P(y^{(i)} \mid x^{(i)};\theta)=
\left(h_\theta(x^{(i)})\right)^{y^{(i)}}
\left(1 - h_\theta(x^{(i)})\right)^{1-y^{(i)}}.
$$

This works because:
- If $$y^{(i)}=1$$, then the second exponent is 0 and the term reduces to $$h_\theta(x^{(i)})$$.
- If $$y^{(i)}=0$$, then the first exponent is 0 and the term reduces to $$1-h_\theta(x^{(i)})$$.

So:

$$
L(\theta)=
\prod_{i=1}^n
\left(h_\theta(x^{(i)})\right)^{y^{(i)}}
\left(1 - h_\theta(x^{(i)})\right)^{1-y^{(i)}}.
$$

## 24) Log-likelihood

We take logs to turn products into sums:

$$
\ell(\theta) = \log L(\theta)
= \sum_{i=1}^n
\left[
y^{(i)} \log h_\theta(x^{(i)})
+
(1-y^{(i)}) \log\left(1 - h_\theta(x^{(i)})\right)
\right].
$$

This is in the additive form that works well with SGD / mini-batches.

## 25) Gradient form (parallel to linear regression)

A useful and somewhat surprising simplification is that the derivative ends up looking like “error times feature,” just as in linear regression.

For the $$j$$-th coordinate:

$$
\frac{\partial}{\partial \theta_j}\,\ell(\theta)=
\sum_{i=1}^n
\left(y^{(i)} - h_\theta(x^{(i)})\right)x^{(i)}_j.
$$

So the structure matches what we saw before:
- prediction error: $$y^{(i)} - h_\theta(x^{(i)})$$
- multiplied by the feature coordinate: $$x^{(i)}_j$$
- summed over data points

### Log-likelihood vs likelihood notation

- $$L(\theta)$$ = likelihood
- $$\ell(\theta)$$ = log-likelihood

## 26) Optimization note: ascent vs descent

Since $$\ell(\theta)$$ is something we **maximize**, you can write gradient ascent:

$$
\theta^{(t+1)} = \theta^{(t)} + \alpha \nabla_\theta \ell(\theta^{(t)}).
$$

Often we instead minimize the **negative log-likelihood** (a loss), which turns it into gradient **descent**. That’s where the “minus sign” usually shows up.

---

# Newton’s Method (Optimization Tool)

Now we shift from modeling to optimization.

## 27) Root finding view

Newton’s method is typically introduced for finding roots:

$$
f(x)=0.
$$

For minimization, the connection is that a minimum of a smooth function satisfies:

$$
\nabla f(x)=0,
$$

so root finding becomes relevant.

## 28) 1D Newton update

Starting from an initial guess $$\theta^{(0)}$$, Newton’s method updates by:

$$
\theta^{(t+1)}=
\theta^{(t)} - \frac{f(\theta^{(t)})}{f'(\theta^{(t)})}.
$$

It is extremely fast near the solution (quadratic convergence under good conditions).

## 29) Multidimensional Newton update

For minimizing a scalar objective (like negative log-likelihood), the multidimensional Newton step is:

$$
\theta^{(t+1)}=
\theta^{(t)}-
H(\theta^{(t)})^{-1}\,\nabla_\theta \ell(\theta^{(t)}),
$$

where:
- $$\nabla_\theta \ell(\theta)$$ is the gradient (vector)
- $$H(\theta)$$ is the Hessian matrix:

$$
H_{jk}(\theta)=
\frac{\partial^2 \ell(\theta)}{\partial \theta_j\,\partial \theta_k}.
$$

## 30) Why Newton is great—and why it’s often impractical in ML

Newton’s method:
- needs no learning rate $$\alpha$$
- converges very quickly in terms of iterations

But the Hessian is large:
- If $$\theta \in \mathbb{R}^d$$, then $$H(\theta)\in\mathbb{R}^{d\times d}$$.
- For modern models with millions or billions of parameters, $$d^2$$ is enormous.
- Storing and inverting the Hessian can be impossible due to memory/time limits.

That’s why classical statistics tools (like R’s logistic regression) historically used Newton-style methods, but large-scale machine learning tends to favor first-order methods like SGD, or approximations to second-order methods.


# Comparing Optimization Methods in Machine Learning

## 31) Quick check: does the algorithm make sense?

The update rule we’ve been using is the core idea behind gradient-based optimization: start from an initial parameter guess, compute a direction that improves the objective, and take a step. If you recognize it, that’s great—if not, the important point is that each “iteration” means **one update of the model parameters**.

## 32) What we want to compare

We’ve now seen several methods. To put them in context, we’ll compare:

1. **Cost per iteration** (how much computation one update requires)
2. **How many iterations** it takes to reach a given error tolerance (a rough convergence comparison)

This is the fundamental tradeoff:  
**cheap steps but many of them** vs **expensive steps but few of them**.

## 33) SGD (Stochastic Gradient Descent)

In “pure” SGD, each iteration uses **one training example**.

- **Per-iteration cost:** proportional to the number of parameters/features  
  $$
  \text{cost} \propto d
  $$
  It does **not** depend on dataset size $n$, because you only touch one point per step.

- **Practical implication:** you can train without ever seeing all of your data.  
  In some real settings, people sample a large dataset and train on only part of it, because a full pass would be too slow.

- **Convergence (rough intuition):** SGD’s gradient estimate is noisy, so it often needs many updates to stabilize.

## 34) Batch Gradient Descent

Batch gradient descent computes the gradient using **all $n$ data points** every iteration.

- **Per-iteration cost:** depends on both the number of data points and the parameter dimension  
  $$
  \text{cost} \propto nd
  $$

- **Tradeoff:** the gradient is accurate, but each step can be expensive if $n$ is large.  
  Even one full iteration might be too slow for modern datasets.

## 35) Newton’s Method

Newton’s method uses **second-order information** (the Hessian), and usually also uses all data points.

- **Per-iteration cost:** expensive because of the Hessian and interactions between parameters  
  $$
  \text{cost} \propto nd^2
  $$
  The $d^2$ term comes from computing and using the Hessian, which contains second derivatives capturing interactions between every pair of parameters.

- **Convergence:** extremely fast in iterations. A typical rough statement is:
  $$
  \text{iterations} \sim \log\left(\frac{1}{\varepsilon}\right)
  $$
  So if you want a very small error tolerance like $\varepsilon=10^{-16}$, you might only need on the order of tens to hundreds of steps.

- **Why it fails at scale:** if $d$ is huge (millions, billions, or more), then storing or inverting anything like a Hessian becomes infeasible.  
  A billion squared is already enormous; a trillion squared is beyond practical.

## 36) The big picture tradeoff

- If you had hardware that could evaluate gradients on all $n$ points instantly, batch gradient descent might be attractive.
- If you had an “oracle” that could compute Hessians cheaply, Newton’s method could dominate.
- In the real world, modern ML tends to live in regimes where:
  - $n$ is huge (web-scale text, massive image collections)
  - $d$ is huge (modern models can have billions or trillions of parameters)

So we end up preferring methods like SGD that avoid $d^2$ scaling and avoid full passes over massive datasets.

## 37) Mini-batch Gradient Descent (the practical default)

Mini-batch is the “in-between” approach: instead of 1 example, you use **$B$ examples per step**, where $B \ll n$.

- **Per-iteration cost:**  
  $$
  \text{cost} \propto Bd
  $$

- **Why it’s used:** not mainly because it changes theoretical convergence rates, but because of **parallelism**.  
  On GPUs and other parallel hardware, processing a batch of size $B$ can be close to the wall-clock cost of processing 1 point (up to a point). So mini-batching is often “free” or nearly free.

- **Extra benefit:** averaging over $B$ samples reduces gradient noise somewhat, giving more stable updates than pure SGD.

## 38) Classical stats vs machine learning mindset

In many classical statistics settings:
- $d$ is small
- $n$ is moderate
- people care about extremely precise solutions

For those regimes, second-order or full-batch methods can make sense.

Machine learning tends to emphasize:
- very large $d$
- very large $n$
- solutions that are *approximate* but useful

A surprising empirical fact is that these approximate solutions are often quite robust, even when we don’t fully understand why.

## 39) Extra question: what does $\Omega(\cdot)$ mean?

The notation $\Omega(\cdot)$ means “grows at least as fast as.”  
For example, saying something costs $\Omega(d)$ means that asymptotically, you can’t do better than scaling proportional to $d$ (up to constant factors).

## 40) Other axes you could compare (beyond this chart)

- **Noise / local minima behavior:** SGD’s randomness may help it avoid getting stuck compared to methods that aggressively follow curvature, though this is not universally guaranteed and is often only provable under specific assumptions.
- **Numerical precision:** modern ML increasingly uses low-precision arithmetic (e.g., FP16, FP8, even integers). SGD-style methods tend to be more compatible with these constraints than Hessian-based methods.

## 41) Wrap-up

We’ve compared:
- SGD: cheap steps, many steps
- Batch GD: expensive steps, fewer steps than SGD (often), but costly when $n$ is large
- Newton: very expensive steps (especially due to $d^2$), very few steps

Next, the course moves toward exponential family models, where the same basic workflow continues: define a probabilistic model, derive a likelihood (or log-likelihood), and optimize it with scalable methods.

