## 1) Welcome and How This Three-Lecture Block Will Work

Hello—welcome to CS229. We’re starting a block of three lectures that I have the privilege of teaching, where we’ll walk through the building blocks and basics.

Before getting into the plan for the three lectures, I want to cover a couple of logistics. I posted something on Ed explaining why I’m structuring these lectures the way I am. You’re not obligated to read it, but if you’re interested, go ahead. I’m happy to take feedback and discuss it.

One thing I liked about the pandemic era was that more people asked questions during class. I think part of that was because people used Zoom’s anonymous feature, and I wish we still had that. We don’t in this class for various reasons, so instead we’ll use an Ed thread I just set up called “Lecture 2.” Feel free to post questions there. I may not take all of them—I reserve the right to skip questions—but TAs may answer some, and I’ll try to follow up on what’s there. It’s genuinely helpful when you ask questions, and I’m happy to talk about pretty much anything. Questions relevant to the class are especially helpful, but I’m open to whatever you want to discuss.

Before my lectures, I also posted a couple of downloads: a handwritten note of what I’m going to talk about (the same notes I use, with small edits), and a template in case you want to follow along. You don’t need any of that. You can sit here, watch the video later, ask questions, do whatever works for you. The goal is simply to make sure you have the material and any data I show you in front of you, so things are easy to follow and easy to copy into your own notes if you want.

For the lecture delivery, I’m going to try using the iPad. I like the whiteboard feel, and this is a good compromise because it slows me down. If I get excited, I can start talking nonsense, so this format helps keep me focused—at least we’ll see how long that lasts.

## 2) The Plan: Building Up Models From Linear Regression to Exponential Family

Over these first three lectures, we’re going to build increasingly sophisticated machine learning models. You’ll see that they’re closely related to a model you probably already know: **linear regression**. If you don’t know linear regression, don’t worry. Today’s lecture is essentially linear regression with slightly fancier notation and a bit more detail around the algorithm. Conceptually, it’s still “fit a line.”

In the next lecture, we’ll generalize from regression to **classification**, and that introduces a couple of twists. We’ll choose notation carefully, because that viewpoint lets us expand to a much larger class of models called the **exponential family**. Those models will show up throughout the course. We’ll give a precise definition that covers many statistical models under one abstraction, so you don’t have to learn each model as a completely separate object. Instead, you’ll learn a unified way to find parameters, do inference, get predictions, and understand algorithms.

As we go, I’ll try to highlight what carries over to what I’d call modern or industrial machine learning. The way we solve these optimization problems is essentially the same way we train models used in practice—from image detection to search, to natural language processing, to translation. This abstraction shows up in a surprising number of places.

The underlying workhorse algorithm we’ll see is **stochastic gradient descent**. We’ll introduce it in the simplest possible setting so you can build intuition for how it works.

So the structure for this block is: **linear regression**, then **classification**, then the generalized **exponential family**. These will have very parallel structure. When you go back to your notes later, you should be able to identify the model part, the solving part, and how they map across different problems.

After this block, Tengyu takes over and teaches neural nets, kernels, and other topics. Then I come back later for unsupervised learning, where the structure shifts again, and graphical models make an appearance.

## 3) Today’s Agenda

Today, we’ll go through:

1. **Basic definitions** (a bit pedantic, but if something isn’t clear you should ask—otherwise I haven’t done my job).
2. **Linear regression**—fitting a line, eventually in high dimensions, so we’ll start abstracting the idea.
3. **Batch gradient descent** and **stochastic gradient descent**. Terminology in this area is messy. This algorithm was called incremental gradient descent in the 1960s. These are old ideas that have been used for a long time, and they’re still what we use every day.
4. A brief discussion of the **normal equation**, mainly because it tends to show up in homeworks and it’s a good place to practice **vector derivatives**. You will need to know vector derivatives to make your life easier, since you’ll occasionally compute gradients. The normal equation is a convenient place to check your work because you have a good sense of what the answer should look like.

Normal equations are not the most important concept in the course, but they’re solid and worth knowing.

## 4) Supervised Learning Setup: The Prediction Function and the Training Set

Let’s talk about **supervised learning**. This whole block is supervised learning, and it will follow the same general schema.

We want a **prediction function**, which we’ll consistently denote by %h5. It maps from some set $\mathcal{X}$ to some set $\mathcal{Y}$:

$h: \mathcal{X} \to \mathcal{Y}.$

Before getting formal, here are examples of what (\mathcal{X}) and (\mathcal{Y}) could be:

* $\mathcal{X}$: images, and $\mathcal{Y}$: labels like “cat,” “dog,” and so on. “Does this image contain a cat?” is the simplest yes/no version.
* $\mathcal{X}$: text, and $\mathcal{Y}$: labels like “hate speech” vs. “not hate speech.” This is an example where we’d like machine learning to do better than it currently does in practice.
* We’ll also use house data, which is a classic statistics and machine learning task.

To make this supervised, we also need a **training set**. Formally, it’s just a set of pairs:


${(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)}.$

Each $x_i \in \mathcal{X}$ is an encoding of the input. For images, $x_i$ could be bits, RGB values, or some representation of pixel intensities. For text, it might be ASCII/Unicode or some other representation. Later, we’ll almost always abstract this away and work in a vector space, and we’ll talk about where those vectors come from. Each $y_i \in \mathcal{Y}$ is the label.

Given the training set, our job is to find a good hypothesis $h$. We call it (h) partly because it’s a “hypothesis.”

## 5) What “Good” Means and Why Generalization Matters

The word “good” will occupy a lot of our attention. Intuitively, a good $h$ should do better than random guessing on the examples—if you show me images with cats, I should label most of the cats correctly. Of course, in real machine learning, we don’t get it right all the time, and the models are still useful, so we’ll develop statistical notions like doing well “on average.”

There are also more advanced notions of goodness. You might care about how performance varies across groups. You might want the worst-group performance to be as strong as possible, not just average performance. That kind of change typically corresponds to choosing a different loss function, and the math framework can accommodate it with relatively straightforward modifications. For now, we’ll stick to the simplest notion—basic accuracy or error on the task—but keep in mind the framework is broader.

Also, the space of all functions from $\mathcal{X}) to (\mathcal{Y}$ is enormous. We can’t search over all possible functions, so we’ll restrict the class of functions we consider. That restriction is a big part of model design.

A key point—sometimes counterintuitive at first—is that we don’t just care about predicting the labels of the training examples we already have. We care about predicting well on **new** inputs we haven’t seen before. If I take a new picture on my phone, I don’t care whether the model memorized my last picture; I care whether it works on the new one. The new image will be similar in some sense, but it will look different.

That idea forces an assumption: we think of the training examples as drawn from a larger population, and we want to do well on future samples from that same population. If you train on pictures of my daughter and test on cars, you’re out of luck—the training distribution needs to match the prediction setting. At the same time, modern trends complicate this: large models trained on broad web data seem to perform decently across many domains, which creates a strange and evolving notion of what “good” means.

## 6) Terminology: Classification vs. Regression

This is just terminology:

* If $\mathcal{Y}$ is **discrete**, we call it **classification**. The simplest case is binary classification (yes/no). You can also have many classes (car, plane, truck, specific car model, etc.).
* If $\mathcal{Y}$ is **continuous**, we call it **regression**.

Today we’ll focus on regression using house prices. In lecture three, we’ll switch to classification, which has subtle differences.

## 7) Looking at Real House Price Data

Now let’s look at real data: the **Ames housing dataset**. Historically, there’s also the famous Boston housing dataset, but we’ll use Ames here. You can download the Ames data and try tasks like predicting sale prices—there’s even a Kaggle competition using it.

These are real houses and real sales. The dataset has on the order of 90+ columns (around 93 features). I’m showing only a subset: sale price, lot area (a square-footage-like feature), and a few other fields. The first thing you should do with any new dataset—and I cannot emphasize this enough—is **look at it**. Plot it. Inspect it. People often rush to run fancy methods without even checking what the data looks like. In practice, staring at data for a while is one of the most valuable things you can do, especially for projects.

When you plot square feet on the x-axis and price on the y-axis, you see a general upward trend: bigger houses tend to cost more. Of course, it’s not the whole story. Neighborhood desirability, condition, and many other factors matter. Those additional columns become additional features later. But as a first model, this is useful.

You can also look at another feature like number of bedrooms and plot price against it. You’ll see spread within each bedroom count—three-bedroom homes vary in price, four-bedroom homes vary in price—which reinforces the idea that no single feature fully explains the outcome.

## 8) Our Goal: Learn a Hypothesis (h)

Going back to the abstraction: we want a hypothesis $h$ that maps from lot area (or square footage) to sale price. That is, you show me the input $x$, and I output a predicted price $h(x)$.

There are infinitely many possible functions that could do this—scramble, memorize, or consult an oracle. So we need a representation for $h$ that restricts the class of functions we consider.

## 9) The Linear Model: A First Restricted Hypothesis Class

We’ll start with a class of models called “linear.” If you’re a stickler, you’ll notice this is technically an affine function; I’m going to allow that mild abuse of language for now.

With one feature $x_1$ (say square footage), the model is:

$h(x) = \theta_0 + \theta_1 x_1.$

Here $\theta_0$ and $\theta_1$ are parameters (weights) of the model. Geometrically, this is a line.

* $\theta_0$ is the intercept: the predicted value at $x_1 = 0$.
* $\theta_1$ is the slope.

When you predict for a particular example, you take its observed $x_1$ value, locate it on the x-axis, and read off the corresponding y-value on the line. That value is your predicted price. The difference between the predicted and actual price is the residual, and soon we’ll define an objective that tries to make those residuals small in a principled way.

At this scale, the linear model doesn’t look unreasonable: there’s a clear trend with some error, and it gives us a first predictive model that is simple, familiar, and easy to analyze.

That’s the setup. Next, we’ll make “good” precise by defining an error measure and then showing how gradient descent (batch and stochastic) finds parameters $\theta$ that fit the data.


## 10) Generalizing to Multiple Features

Now I want to generalize the model. Imagine we have a dataset with many features. Using the house-price example, we might have features like **size**, **bedrooms**, and **lot size**, and the target **price**. Here, **price** is our $y$, and the features are our $x$'s. I’ll write some example numbers just to keep the notation concrete—the specific values don’t matter. What matters is how we index the first data point, the second data point, and so on, and how those map into feature coordinates like $x^{(1)}_1$, $x^{(2)}_1$, etc.

## 11) Turning an Affine Model into a Linear One

I called the model “linear,” but if you’re being precise, the form $$\theta_0 + \theta_1 x_1$$ is actually **affine** because of the intercept term $$\theta_0$$. The standard trick to make the notation genuinely linear is to introduce a new feature $$x_0$$ that is always equal to 1.

We adopt the convention:

$x_0 \equiv 1.$

With that convention, the intercept becomes $\theta_0 x_0$, so the whole expression is a linear function of the feature vector. This is purely a notational convenience—don’t get stuck on it.

## 12) The Linear Model in $$d$$ Dimensions

With $$d$$ (original) features, the model becomes:

$h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \cdots + \theta_d x_d = \sum_{j=0}^{d} \theta_j x_j,$

with the reminder:

$x_0 = 1.$

This now supports high-dimensional inputs. High-dimensional spaces behave differently than low-dimensional ones, and they’re often where machine learning gets interesting. In many modern settings, we embed data into a vector space (sometimes with hundreds of dimensions) and then train a linear model on top. In this course, for now, the features will be human-interpretable and come directly from the table: given a row $$x^{(i)}$$, we fill in values like size, bedrooms, lot size, and so on, and then compute a prediction.

## 13) Vector Notation for Features and Parameters

To simplify notation, we use column vectors. For the $$i$$-th example, define the feature vector:

$x^{(i)} = \begin{bmatrix}x^{(i)}_0 \\x^{(i)}_1 \\\vdots \\x^{(i)}_d\end{bmatrix},$

where $$x^{(i)}_0 = 1$$ and, for example, $$x^{(i)}_1$$ might be the size feature (like 2104 square feet), $$x^{(i)}_2$$ might be bedrooms, and so on.

The parameter vector is:

$$
\theta =
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_d
\end{bmatrix}.
$$

The pedantry here is intentional: we’ll reuse this exact mapping between “raw data row” and “feature vector” in many different settings, even when the meaning of $$\theta$$ changes across models.

The label for example $$i$$ is $$y^{(i)}$$, which is the price in our running example. Each pair $$\left(x^{(i)}, y^{(i)}\right)$$ is a **training example**.

## 14) Writing the Whole Training Set as a Matrix

Now we package all training inputs into a matrix $X$ with one row per example. If there are $$n$$ examples, then $$X$$ has $$n$$ rows. Because we introduced $$x_0 = 1$$, each example has $d+1$ coordinates, so $$X \in \mathbb{R}^{n \times (d+1)}$$:

$X \in \mathbb{R}^{n \times (d+1)}.$

That “$+1$” comes entirely from the extra constant feature $$x_0$$. It’s a common place people get tripped up, so it’s worth emphasizing.

At this point, we have many equivalent ways to write the dataset—pairs, vectors, matrices—but we still haven’t answered the core question: **how do we choose a good model?**

## 15) Choosing a Good Model: Residuals and Squared Error

Why do we think a fitted line is good? Because it makes small errors. If every point lay exactly on the line, the error for each point would be 0. More generally, we want to minimize the residuals (the prediction errors).

For historical and computational reasons, we look at **squared residuals**. You could use absolute values instead; you just don’t want signed errors because a negative error shouldn’t “cancel out” a positive one. The intuition is: errors should be penalized, and larger errors should be penalized more.

## 16) Defining the Hypothesis and the Cost Function

We will write the hypothesis explicitly as a function parameterized by $$\theta$$:

$$
h_\theta(x) = \sum_{j=0}^{d} \theta_j x_j.
$$

To pick a good model, we want $$h_\theta(x^{(i)})$$ to be close to $$y^{(i)}$$ for paired training examples. We formalize this by defining a **cost function** (also called the least-squares objective):

$$
J(\theta)=
\frac{1}{2}
\sum_{i=1}^{n}
\left(h_\theta\!\left(x^{(i)}\right) - y^{(i)}\right)^2.
$$

- $$h_\theta\!\left(x^{(i)}\right)$$ is the prediction on the $$i$$-th input.
- $$y^{(i)}$$ is the corresponding label.
- The square penalizes large errors more heavily.

The factor of $$\tfrac{1}{2}$$ is just a convenience: when we take derivatives, the 2 cancels. It does **not** change which $$\theta$$ minimizes the cost.

## 17) What We Actually Solve

Our model is completely determined by the parameters $$\theta_0, \theta_1, \ldots, \theta_d$$. By restricting ourselves to this class of linear models, we reduce the search from “all possible functions” to “functions representable by a weight vector $$\theta$$.”

A “good model” now means: choose the parameters that minimize the cost function:

$$
\min_{\theta} \; J(\theta).
$$

This optimization problem is what we mean by “training” the linear regression model. For linear regression, what’s nice is that we can solve it cleanly, and later we’ll see algorithms (like gradient descent) that generalize to more complex models.

## 18) Question: Why Include the $\tfrac{1}{2}$?

A student asked: why include the $$\tfrac{1}{2}$$ in the least-squares cost?

The key point is that we only care about the **minimizer**, not the absolute scale of the cost. Multiplying the cost by a constant doesn’t change which $$\theta$$ minimizes it. The $$\tfrac{1}{2}$$ is included because it makes derivatives cleaner: when differentiating $$\left(\cdot\right)^2$$, a factor of 2 appears and cancels the $$\tfrac{1}{2}$$.

## 19) Why This Setup Matters

We are setting up a pattern that will repeat throughout the course:

1. Choose a hypothesis class (parameterized by $$\theta$$).
2. Define a cost function that penalizes bad predictions.
3. Solve $$\min_\theta J(\theta)$$ using optimization methods.

We’ll replace $$J(\theta)$$ with different cost functions for classification and other statistical models, but much of what comes next—how we optimize, how we compute gradients, how we interpret the solution—will carry over.


## 20) Solving the Optimization Problem: Why We Use Gradient Descent

How are we going to solve this? There are many ways. If you’ve taken linear algebra, you might think: “Just compute the normal equations and you’re done.” If you’ve used MATLAB or NumPy, you might think: “Just call a least-squares solver.” We *will* briefly talk about the normal equations later, but we’re going to solve it in a way that sets us up for the rest of machine learning.

Machine learning often deals with objective functions that are not as nice as linear regression. Historically, a lot of the early focus was on **convex** (bowl-shaped) problems, where we cared deeply about whether we got the “right” $$\theta$$ and whether there was a unique solution. Modern machine learning often doesn’t have that luxury. We may not even know whether we reached the best solution, or how long to train. In fact, there are still papers saying things like “you should run your models longer,” and people realize we didn’t even notice. That’s the world we live in.

So we’ll use an approach that generalizes: **gradient-based optimization**.

## 21) Intuition: The Shape of the Cost Function

Imagine $$J(\theta)$$ is our cost function. As an aside, least-squares for linear regression is convex, so it really does have a nice bowl shape. But when I draw a function with bumps and valleys, I’m illustrating what can happen in more general machine learning problems.

If a function is non-convex, it can have **local minima**—points that are lower than nearby points—but not the best overall. In convex problems, every local minimum is also a global minimum. We’ll return to that later; for now, just keep the intuition: gradient descent can get stuck in bad places for general objectives, but for least squares we’re in good shape.

## 22) Gradient Descent: Start With a Guess and Walk Downhill

We start with an initial guess $$\theta^{(0)}$$. How do we pick it? Sometimes randomly, sometimes by a heuristic. Entire research papers are written about initialization. For least squares (and some other models we’ll study), initialization won’t matter much because we can still reach the right solution.

Now suppose we’re at some parameter value $$\theta^{(t)}$$ and we can “look around.” If we can go downhill from where we are, the greedy idea is:

1. compute the gradient $$\nabla_\theta J(\theta^{(t)})$$
2. move in the opposite direction of the gradient (downhill)

The update rule is:

$$
\theta^{(t+1)}=
\theta^{(t)} - \alpha \,\nabla_\theta J(\theta^{(t)}).
$$

Here:
- $\nabla_\theta J(\theta^{(t)})$ is the gradient of the cost with respect to $$\theta$$
- $\alpha$ is the **learning rate** (step size)

This is easiest to picture in one dimension: the derivative tells you which direction the function increases, so going the opposite way decreases the function.

### 23) How Far Do We Step?

That’s what $$\alpha$$ controls.

- If $\alpha$ is too large, you can overshoot and bounce around or diverge.
- If $\alpha$ is too small, you make tiny progress and the algorithm is very slow.

In practice, you often pick $\alpha$ and adjust it with trial and error. There’s theory for simple convex settings, and modern deep learning often uses “adaptive” optimizers that tune step sizes automatically, but the basic idea is still the same.

After stepping, you get a new point $\theta^{(1)}$, then you repeat: compute the gradient again, step again, and so on.

## 24) Questions About Notation and High Dimensions

### What does $$\nabla_\theta$$ mean?

A student asked about the “denominator” / notation: $$\nabla_\theta$$ just means “gradient with respect to $$\theta$$.” In one dimension it’s just a derivative. In multiple dimensions, it’s the vector of partial derivatives (one per coordinate).

### How do we visualize the gradient in higher dimensions?

One way to think about it: each component of the gradient is the slope along one coordinate direction. In 2D, you can imagine looking along the $$\theta_1$$ axis to get a 1D slice, compute its slope, then looking along $$\theta_2$$ to get the other slope. The gradient stacks those slopes into a vector that points in the direction of steepest ascent; stepping opposite it moves you downhill.

High dimensions can be strange: objectives can have **saddles** (saddle points), where the surface curves up in one direction and down in another. We won’t worry about convergence details right now, but this is one reason optimization can be tricky in modern ML.

## 25) When Do We Stop?

In practice, you stop when the update becomes small:

$$
\|\theta^{(t+1)} - \theta^{(t)}\|
\ \text{is small},
$$

or when the gradient norm $\|\nabla_\theta J(\theta^{(t)})\|$ is small, or after a fixed number of steps. You set a tolerance (sometimes near machine precision like $$10^{-6}$$ or $$10^{-16}$$, or larger if you just want a fast approximate solution).

For general non-convex problems, gradient descent can converge to a poor local minimum. In a convex bowl-shaped objective, that issue disappears: every local minimum is global. Least squares is convex, which is why it behaves well here.

## 26) Computing the Gradient for Least Squares

Now let’s compute the derivatives, and connect back to the fact that $$J(\theta)$$ is a sum over data points.

Recall the least-squares objective:

$$
J(\theta)=
\frac{1}{2}\sum_{i=1}^{n}
\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2.
$$

We want the partial derivative with respect to the $$j$$-th parameter $$\theta_j$$:

$$
\frac{\partial}{\partial \theta_j} J(\theta)=
\frac{\partial}{\partial \theta_j}
\left(
\frac{1}{2}\sum_{i=1}^{n}
\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
\right).
$$

Because differentiation is linear, we can push the derivative inside the sum:

$$
\frac{\partial}{\partial \theta_j} J(\theta)=
\sum_{i=1}^{n}
\frac{\partial}{\partial \theta_j}
\left(
\frac{1}{2}
\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
\right).
$$

Differentiate the square term. The factor $$\tfrac{1}{2}$$ cancels the 2 from differentiating $$\left(\cdot\right)^2$$:

$$
\frac{\partial}{\partial \theta_j} J(\theta)=
\sum_{i=1}^{n}
\left(h_\theta(x^{(i)}) - y^{(i)}\right)
\cdot
\frac{\partial}{\partial \theta_j} h_\theta(x^{(i)}).
$$

This expression has a nice interpretation:

- $\left(h_\theta(x^{(i)}) - y^{(i)}\right)$ is the **signed error** (misprediction).
- $\frac{\partial}{\partial \theta_j} h_\theta(x^{(i)})$ tells you how the model’s prediction changes as you change parameter $$\theta_j$$.

This “error times derivative” structure is a basic template that will reappear constantly (it resembles a simple chain-rule pattern).

### 27) Specializing to Linear Regression

For linear regression:

$h_\theta(x) = \sum_{k=0}^{d} \theta_k x_k.$

Then:

$\frac{\partial}{\partial \theta_j} h_\theta(x) = x_j,$

and at a specific training example:

$\frac{\partial}{\partial \theta_j} h_\theta(x^{(i)}) = x^{(i)}_j.$

So the gradient component becomes:

$\frac{\partial}{\partial \theta_j} J(\theta)=\sum_{i=1}^{n}\left(h_\theta(x^{(i)}) - y^{(i)}\right) x^{(i)}_j.$

## 28) Batch Gradient Descent Update (Coordinate Form)

Plugging this into the gradient descent update, the $$j$$-th coordinate update is:

$\theta_j^{(t+1)}=\theta_j^{(t)}-\alpha\sum_{i=1}^{n}\left(h_{\theta^{(t)}}(x^{(i)}) - y^{(i)}\right)x^{(i)}_j.$

At this stage, we are summing over *all* training examples each step, which is why this is called **batch gradient descent**. We’ll contrast this with stochastic gradient descent soon.

The indices matter:
- $j$ indexes the parameter/feature coordinate.
- $i$ indexes the training example.

## 29) Vector Notation (All Coordinates at Once)

Using vector notation, we can write the same idea more compactly as a single vector update:

$$
\theta^{(t+1)}=\theta^{(t)}-
\alpha
\sum_{i=1}^{n}
\left(h_{\theta^{(t)}}(x^{(i)}) - y^{(i)}\right)
x^{(i)}.
$$

This “loops over all $$j$$ at once” by treating $$\theta$$ and $$x^{(i)}$$ as vectors.

## 30) Question: Is the Same Learning Rate Used for Every $\theta_j$?

A student asked whether $\alpha$ is the same for every coordinate.

In the basic gradient descent rule shown here, yes: $\alpha$ is a single step size shared across all coordinates, typically fixed per iteration (though in practice it may change over time, e.g., decay with $t$).

More sophisticated methods can use different effective step sizes per coordinate (adaptive optimizers), but for now you should think of $$\alpha$$ as one small constant chosen so steps are not too large.

