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

$$\left\{ (x^{(i)}, y^{(i)}) \right\}_{i=1}^{n}$$

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
\mathbb{E}\!\left[\varepsilon^{(i)}\varepsilon^{(j)}\right]
=
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
\mathrm{Var}(\varepsilon^{(i)})
=
\mathbb{E}\!\left[(\varepsilon^{(i)})^2\right].
$$

### Question: Why does variance become $$\mathbb{E}[(\varepsilon^{(i)})^2]$$?

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
