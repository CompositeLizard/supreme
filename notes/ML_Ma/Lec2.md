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

That’s the setup. Next, we’ll make “good” precise by defining an error measure and then showing how gradient descent (batch and stochastic) finds parameters (\theta) that fit the data.

