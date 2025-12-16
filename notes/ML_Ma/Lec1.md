---
---
## 1) Course Welcome and Instructor Introductions

My name is Tony Ma. This quarter, the course will be taught by two instructors: me and Chris. I work on machine learning and machine learning theory, including theory across different areas of machine learning—reinforcement learning, representation learning, supervised learning, and related topics.

I’d like Chris to introduce himself as well.

Chris: I’m Chris, also in the machine learning group. I’m especially interested in how the systems we build are changing because of machine learning. It has been a fascinating time over the last 10 years. I started out thinking a lot about optimization and how we scale up large models, back when machine learning had relatively few applications in everyday life. Over the last couple of years, we’ve built things that, hopefully, some of you have used. My students have contributed to products like Search, Gmail, Assistants, and other areas. More recently, I’ve been focused on how to make these models robust. We’ll also have a great new lecture from Tengyu on what are called foundation models—large, self-supervised models that have been getting a lot of attention.

Percy, Tatsu, and I co-taught a course about foundation models last term. This course is exciting because it gives you the foundational layer of machine learning that all of those developments are built on. It’s a great time to study machine learning because it’s no longer abstract: you use machine learning products every day. The goal is that you’ll gain insight into how they work and why there is still so much research left to do. I’m really looking forward to lecturing you all.

## 2) How the Teaching Will Alternate

You’ll see Chris and me alternate every few weeks. Next lecture will be Chris, and then after two or three weeks you’ll see me again.

## 3) Teaching Team Structure

Let me introduce the teaching team. We will have 12 TAs, plus one head TA and a course coordinator. The head TA will also serve as the course coordinator. They will handle much of the behind-the-scenes work, so you may not need to interact with them often, but they will be organizing the whole TA team.

We currently have 12 TAs, and we may add more if enrollment increases. I didn’t ask the TAs to attend the first lecture, partly because they would also need to wear masks, and the pictures on the slide serve a similar purpose. You will see them often in office hours and other course settings.

## 4) What Today’s Lecture Covers

In this lecture, I’ll spend the first part on logistics and the overall course structure. Then I’ll introduce, at a high level, the topics covered in the course.

## 5) Where to Find Course Information Online

We worked hard to make the course information available online in a single place. The course website links to several Google Docs: one for logistics, another for the syllabus, and also links to lecture notes and final project guidelines. In principle, everything I’m presenting today is a subset of what’s on the website—and it’s a small subset. I encourage you to read through those documents, especially when you have questions. A good first step is to check whether the documents already answer what you’re wondering about.

After that, feel free to ask questions.

## 6) Prerequisites and Background Expectations

The first topic is prerequisites. You’ll hear different opinions about difficulty: some students say the course is challenging, others say it’s easier, often depending on their background. That is why I want to emphasize prerequisites early. Having the right background will help you meet your goals in the course.

**Probability:** The most important prerequisite is probability at the level of CS109 or Stats. You should at least have seen terms like dispersion, random variables, expectation, conditional probability, variance, and density. You do not need to recall everything instantly, but you should recognize these concepts from a previous course.

**Linear algebra:** You also need linear algebra, including matrix multiplication and eigenvectors. Linear algebra is covered in courses like Math 104, 113, and 205. The logistics document lists additional relevant courses. The key skills we rely on most are matrix multiplication and eigenvectors.

**Programming:** We require basic programming knowledge, especially in Python and NumPy. If you know Python but not NumPy, that is usually fine—NumPy is mostly basic numerical operations. If you do not know Python or NumPy but you know another language like C++, you can likely transition fairly easily since a lot of the work is adapting syntax. If you have no programming background, the course will be difficult, because homeworks typically have both a math component and a programming component.

One of the most challenging situations on homeworks is when your code goes wrong—which happens all the time, even for instructors—and you’re not sure whether the issue is your math or your code. Sometimes you might think you derived the wrong equations, when the real problem is that you used NumPy incorrectly. We will review Python and NumPy in some TA lectures to refresh your skills, or to help you learn the basics, but you should come in with some programming foundation.

## 7) TA Review Lectures and Course Intensity

We will provide materials and TA lectures to review backgrounds. There will be three lectures each on programming, linear algebra, and probability.

This is a mathematically intense course for many students, depending on background, so consider this a heads-up. It helps if you know at least two out of the three areas (programming, linear algebra, probability) relatively well. That way you are less likely to get stuck on “entangled” issues where it is unclear whether you are missing the math or the implementation.

That challenge is also part of what makes the course exciting and rewarding.

## 8) Course Goal and What “Foundations” Means

The goal of the course is to give you the foundations of machine learning. This is the foundational layer, and it is also an introductory course. You do not need to have taken a machine learning course before taking this one.

At the same time, we hope that after completing the course, you will feel comfortable enough with the basics to apply machine learning to applications. If you want to become an expert in specific areas like NLP or vision, you’ll likely need additional courses, but this course is designed to set up the machine learning foundations that support broader AI and AI-related applications.

Because of that, the course covers a diverse set of topics and does involve mathematics. We will do very little in the way of formal proofs, but we will do many mathematical derivations. You will do derivations in homeworks, and we will also do derivations in lecture.

If you have questions during lecture, feel free to stop me. Also, yes—lectures are recorded, and you can find recordings on Canvas.

## 9) Academic Integrity and Honor Code Expectations

The second important thing I want to discuss is the honor code. It may feel awkward to mention this so early, but in the past there have been honor code violations, and it is genuinely unfortunate. It’s difficult and sad to have to report students, and I don’t want to see that happen again. That is why I want to be direct about expectations from the beginning.

If you are not intentionally violating the honor code, you generally do not need to worry. Still, here are key points (and these are also on the course website in more detail):

We encourage study groups. You may discuss homework problems with others, but you must write your solutions independently, and you must list the names of the people you discussed the homework with.

It is an honor code violation to copy, refer to, or look at written or coding solutions from a previous year. This includes (but is not limited to) official solutions from previous years, solutions posted online, solutions you or someone else wrote in previous years, or solutions for related problems. If you apply common sense and avoid intentionally doing anything inappropriate, you should be fine.

We do check code using software tools, and the course staff and TAs handle academic integrity issues when they occur. I’m not trying to stress you out, but I do want to put the policy up front.

## 10) Course Components: Project, Homework, Midterm, and Support Sections

Beyond homework, another major component is the course project. We encourage you to form groups of one to three people. The evaluation criteria are the same whether you work alone or in a group of two or three. More information is on the course website.

Typically, you’ll apply machine learning to an application or topic you are interested in. This is one of my favorite parts of the course. Each quarter we receive around 100 project submissions, covering a wide variety of topics and applications of machine learning. You are welcome to pursue other topics as well. You can also focus on core algorithms, which is also great, but many students explore applications in areas like music and finance.

The course also has four homeworks and a midterm. There is no final exam. The main graded components are the midterm, the course project, and the homeworks.

Another component is TA lectures, which are optional. If you find them useful, you should attend; if not, you don’t have to.

There are two sets of TA-led sessions:

**Friday TA lectures (Friday section):** We will likely have six to seven weeks of these. The first three weeks review foundational material, especially concepts related to machine learning. Later weeks cover more advanced topics that are not required but may be interesting.

**Discussion sections:** These are meant to be interactive. Since the course is large, it can be harder to make lectures interactive for every person. Discussion sections are smaller sessions led by TAs, designed to feel more like a traditional classroom. They help bridge the gap between lectures and homework. TAs will work through problems similar to homework (sometimes simpler), and the sessions will often involve live problem-solving, students presenting solutions, and discussion among students. You can find the exact time and format in the logistics Google Doc.

## 11) Platforms, Logistics, and Deadline Policies

There is more detailed information on the course website and the logistics Google Doc, which is quite comprehensive. Recordings are on Canvas. Canvas also has a course calendar. The syllabus page links to lecture notes.

For Q&A, we will use Ed. We strongly encourage you to use Ed for communicating with us in almost all situations. You can make private posts or anonymous posts depending on what you need. If you do not have access to Ed, you may need to email us; you can email the head TA to get access.

Homework submission will be through Gradescope. The logistics document also explains late-day policies. One important heads-up: we do not allow late days for the final project. The main reason is that, especially in spring quarter, the grading deadline is very tight—only a few days after finals week—and some students need final grades quickly due to graduation requirements. We also do not want to make the project deadline too early, because it would conflict with homework deadlines.

The final project deadline is on Monday of finals week, though you should double-check the exact date in the documents. We try to set it as late as possible while still meeting grading constraints, which is why late days aren’t allowed for the project. Additional FAQs are also in the Google Doc.

## 12) Student Question About Discussion Sections

Student: For discussion, will we be assigned to a specific session, or do we get to choose which discussion sessions we go to?

Answer: Currently, we have two TAs offering two discussion sessions. We will try to keep the materials in both sessions essentially the same. The times are not fully set yet, but you should be able to choose whichever session you want. It’s probably best to consistently attend one session so the TA gets to know you, but you don’t have to. Discussion sections are optional, and you should attend based on your needs.

## 13) Transition to the Scientific Content of the Course

If there are no other questions, we’ll move on to the more scientific part of the course. As I said, the main goal is to build your foundations in machine learning. We will cover a diverse set of topics, and we will approach them in a mathematical way.

## 14) What Is Machine Learning?

Let me start with some definitions of machine learning. What is machine learning? Since it’s such an active and widely researched topic, there isn’t a single definition that perfectly fits everything. Instead, I want to share a couple of historical definitions that describe the field well.

In 1959, the phrase “machine learning” was introduced by Arthur Samuel. He described machine learning as the field of study that gives computers the ability to learn without being explicitly programmed. The idea of “without being explicitly programmed” is important. For example, Samuel discussed this in a paper titled *“Some Studies in Machine Learning Using the Game of Checkers—Recent Progress.”* The point is that if you write a piece of code that plays checkers by explicitly hard-coding a fixed strategy—first do this, then do that, using branching logic—then that does not really count as machine learning. Machine learning requires relying on the computer to improve through learning, rather than simply following an explicitly specified strategy.

## 15) Learning From Experience: Mitchell’s Definition

That raises the question: how do you give a computer the ability to learn?

Tom Mitchell’s definition adds more context. He says that a computer program is said to learn from experience **E**, with respect to some class of tasks **T** and a performance measure **P**, if its performance at tasks in **T**, as measured by **P**, improves with experience **E**. The wording is almost rhythmic, and it highlights a few key concepts.

**Experience (E):** Using the checkers example again, experience could mean data. That data might come from games the program plays against itself, games played by humans in the past, or other sources of information you collect. In many machine learning settings, “experience” largely means “data.”

**Performance measure (P):** The performance measure is also essential, and there isn’t one universal choice. Different tasks need different metrics. In checkers, the performance measure might be winning rate. It could also incorporate additional goals, such as winning in fewer moves. In other tasks, performance might mean predicting something accurately. In fact, some machine learning research focuses on defining the right performance measure and formulating the right problem, while other research assumes the measure is fixed and asks how to maximize performance under it.

**Improves with experience:** The phrase “improves with experience E” means that if you have more experience—more data—your algorithm should perform better. That improvement is evidence that the program is learning from experience. If performance does not improve as you get more data, then it might indicate that you are not really learning, and instead you may just be running an explicitly programmed strategy that does not adapt.

**Tasks (T):** Finally, the “tasks” are what you are trying to do—in this example, playing checkers well and winning. We will see many other kinds of tasks in machine learning, including predicting labels from inputs, discovering structure in data, and making sequential decisions like gameplay.

This lecture is meant to be high-level, so feel free to stop me and ask questions at any point.

## 16) A High-Level Taxonomy of Machine Learning

With tasks in mind, how do we build a taxonomy of machine learning? Not everyone agrees with any single taxonomy 100%, but a common high-level baseline is:

* **Supervised learning**
* **Unsupervised learning**
* **Reinforcement learning**

These categories are not completely separated. A more realistic picture includes overlaps. Reinforcement learning often uses supervised learning as a component, and many real applications combine ideas across categories. In practice, these categories can be viewed not only as “tasks,” but also as tools or methods for formulating and solving problems. Still, as a first approximation, it’s useful to think of them as three roughly distinct types of tasks.

I’ll start with supervised learning.

## 17) Supervised Learning: House Price Prediction as a Running Example

In supervised learning, we’ll use house price prediction as a running example. I’ll introduce it in an abstract way, and you can think of house price prediction as the concrete application.

You are given a dataset with **n samples**, where each sample is a pair ((x, y)). Here (x) can be a number or a vector, and (y) is a number. If we draw the data as a scatterplot, each point corresponds to one ((x, y)) pair.

In this example, (x) is the **square footage** of the house, and (y) is the **price**. The task is to use square footage (x) to predict price (y). Using Mitchell’s terminology, the dataset is the experience; in everyday language, we just call it the data.

The goal is to learn from the dataset how to predict the price given the square footage. If (x = 800), what is (y)? That value of (x) might not appear in the dataset. If it did appear, you could just read off the corresponding price. The challenge is when you need to predict for an input that wasn’t seen in the data.

One standard approach is **linear regression**: fit a line to the data, and then predict by reading off the value on that line at (x = 800). You could also fit other models, such as a quadratic curve. In fact, in the artificial example shown here, a quadratic curve would fit the data better. In lectures 2 and 3, our goal will be to discuss how to fit a linear model and how to fit a quadratic model for prediction.

House price prediction is just one example. Many applications share the same structure: you are given ((x, y)) pairs and want to predict (y) from (x).

## 18) Features, Inputs, Labels, Outputs, and Notation

We can make the house price prediction problem more realistic. Instead of using only size, you might also know the **lot size**, and other information. Then the goal becomes predicting price using both size and lot size.

When the input has multiple components, we call these components **features**. Size is one feature; lot size is another. Now the input has two features, so (x) is two-dimensional. If you plot the dataset in this case, you would have three dimensions: (x_1), (x_2), and (y).

In supervised learning, we often call the input the **features** or the **input**, and we call (y) the **label** or the **output**. Another heads-up: in machine learning, many concepts have multiple names. Some people say “features,” others say “inputs.” We will try to be clear about the naming, but in lectures we will mostly use “input” and “output” because they tend to be less ambiguous.

The mathematical goal is to find a function that maps input to output.

**Notation note:** We will use consistent notation throughout the course. A superscript typically indicates which example you are talking about (the sample index). A subscript indicates the coordinate of the vector. For example, (x^{(i)}) is the input vector for example (i), and (x^{(i)}_j) is the (j)-th coordinate of that vector.

Sometimes the labels (y) are also called **supervision** (or the set of labels is called the supervision). This is why it’s called **supervised** learning: you observe labels in the dataset. The dataset itself may also be called the **training set** or **training examples**.

## 19) High-Dimensional and Infinite-Dimensional Features

Inputs can be high-dimensional. For a house listing online, you might have many features: number of floors, condition, zip code, and so on. Then (x) is a (d)-dimensional vector, and you use it to predict (y).

In lectures 6 and 7, we will even talk about **infinite-dimensional features**. One way this can happen is by constructing many new features from existing ones. For instance, instead of using just (x_1) and (x_2), you might add (x_1 x_2) as a new feature. You might also include powers like (x_d^k) for different integers (k). This creates a potentially infinite collection of features.

We will also discuss **feature selection**: not all features are necessarily useful. If you include too many features, you might **overfit**, which is a concept we will introduce and return to later. The idea is that some features matter a lot for prediction, while others may not help and can even hurt generalization.

## 20) Two Main Types of Supervised Learning: Regression and Classification

A common distinction in supervised learning depends on what kind of labels you have.

**Regression:** The label (y) is a real number. House price prediction is a regression problem because price is continuous.

**Classification:** The label is discrete. For example, instead of predicting a price, you could predict the type of residence: “house” or “townhouse.” That is a classification problem because the output is one of a finite set of categories. You can make classification more complex by allowing more than two categories.

One way to visualize a classification dataset is to plot the inputs (for example, size and lot size) and use different symbols for different labels. A triangle might represent “house” and a circle might represent “townhouse.” The prediction task becomes: given a new input ((\text{size}, \text{lot size})), determine whether it is a house or a townhouse.

You can also fit a classifier, such as a linear classifier, that separates the two types of points. Then, given a new point, you classify it based on which side of the separating line it falls on.

## 21) Broader Supervised Learning Applications: Vision and Language

Next, I want to mention broader applications of machine learning that we won’t fully cover in detail.

**Image classification:** You are given images, and each image has a label describing the main object in the image. A well-known dataset for this is ImageNet, created by a Stanford group led by Professor Fei-Fei Li’s team. It is historically important because it played a major role in deep learning taking off in the last 5 to 10 years. The task format is: (x) is the raw pixels (represented as numbers, typically a matrix), and (y) is the object category.

**Object localization and detection:** Instead of labeling the main object category, you might want to locate objects in the image using bounding boxes. In that case, the label (y) can be a set of coordinates describing the bounding box. A single box can be represented by two points (four numbers), so here (y) becomes a vector rather than a single value.

**Natural language processing (machine translation):** In translation, the input (x) is a sentence in one language (for example English), and the output (y) is a sentence in another language (for example Chinese). Here the output space is discrete, but it is much more complex than a simple “house vs. townhouse” classification, because there are enormously many possible sentences.

We will talk a bit about language applications and large language models in one of the new lectures added this year. However, the course mainly focuses on foundational techniques of supervised learning, rather than the full details of how to solve each application as effectively as possible. For deep dives into applications like vision or NLP, you would typically take more specialized courses.

## 22) Q&A: Translation as Regression or Classification

Question: In the translation case, is it a regression problem or a classification problem?

Answer: I would call it a classification problem, because the set of possible outputs is still technically discrete. You can think of the set of all possible Chinese sentences as finite, even though it is extremely large. That said, it’s not a “vanilla” classification problem, because the label space is so huge that you have to treat it differently than ordinary classification with a small number of classes.

## 23) Q&A: When and Why Use Infinite-Dimensional Features?

Question: When do you use infinite-dimensional features?

Answer: The answer depends on more context that we’ll build later in the course, but the general idea is this. First, how do you create infinitely many features? You build them from your original input features. Suppose you start with hundreds of raw features. You can construct many additional ones by taking combinations: products of features, powers like (x_d^k) where (k) can be any integer, and other transformations. That process can generate an infinite set.

Why use them? One reason is that you don’t know ahead of time which features will be most useful. You can create a large family of possible features and let the learning algorithm determine which ones matter and how to combine them. In practice, you often don’t literally keep an infinite number of explicit features; after learning, you may discover that only some features are important.

A key point is that infinite-dimensional features do not necessarily mean infinite runtime. There are tricks that let you work with these features implicitly, so the algorithm’s runtime and memory use remain finite, and sometimes can even be quite fast.

These are excellent questions—thank you for asking them.

## 24) Transition and Course Roadmap: Unsupervised Learning Comes Next

Any other questions? If not, the second part of the course will be about **unsupervised learning**. Chris will likely give about five lectures on unsupervised learning.

## 25) What Unsupervised Learning Means

If we keep using the house dataset as an example, the key difference is that in unsupervised learning you are given a dataset **without labels**. You only see the (x)’s, not the (y)’s. In other words, you don’t know how the houses in the dataset were sold in the past, and you don’t have the “triangle versus circle” information that tells you which points correspond to which category.

In the townhouse-versus-house example, supervised learning would show triangles and circles to indicate labels. In unsupervised learning, you do not have that label information—you only see a cloud of points in the scatterplot. Even so, when you look at the plot as a human, you can often tell that something is happening: one bunch of points looks different from another bunch. You might suspect there are two types of residences represented in the data, even if you don’t know what they are called.

That is the nature of unsupervised learning. The goal is to **discover interesting structure in the data without labels**. You want to uncover the hidden structure.

## 26) Clustering as a Core Unsupervised Task

One basic approach is **clustering**: dividing points into groups so that each group has some internal similarity. In the house dataset, clustering would attempt to separate the points into groups that reflect the underlying structure. A “bad” clustering might split the points in a way a human would find unnatural. A good clustering algorithm would likely separate the points into two groups that correspond to the two underlying types of residences. The algorithm would not know the words “townhouse” or “house,” but it could still discover that two distinct types of things are present in the dataset.

In lectures 12 and 13, we will talk about several algorithms for discovering structure, including **k-means clustering** and **mixture of Gaussians**.

## 27) Unsupervised Learning Example: Gene Clustering

Unsupervised learning also shows up in domains like biology. One example comes from Daphne Koller’s group. The application is **gene clustering**: you have many individuals, and for a particular part of the genome you can group individuals’ genetic patterns into clusters. Visually, you can often see cluster-like structure in the data. In that work, the clusters corresponded to how individuals respond to a certain medicine. Once you can group people into clusters, you can potentially apply the appropriate type of treatment to each group.

## 28) Unsupervised Learning Example: Latent Semantic Analysis for Documents

Another example is **latent semantic analysis (LSA)**. The name is not important; the point is the structure of the problem. You have a collection of documents, and each document contains words. You build a table where each entry represents how frequently a word appears in a document—for example, how often the word “power” appears in Document 6.

At first glance, it can be hard to see any pattern. The structure is not obvious. But with the right algorithm, you can reorder (or regroup) the words and the documents to reveal an interesting block structure. Once you do that, the blocks can become interpretable. One block might clearly correspond to a topic like space travel, with words such as “shuttle,” “space,” “launch,” and “booster.” You learn that a certain group of words and a certain group of documents align with a shared topic.

In that sense, you can infer the topics present in the dataset—perhaps five topics overall. Each topic tends to be associated with certain words, and each document is often mostly about one topic, sometimes two.

This is widely used in social science. For example, researchers might have a huge collection of political blog posts and want to study trends. They cannot label a million posts by hand, so they use topic discovery methods to group documents, then apply statistical analysis to understand what is happening across the corpus. This approach is useful beyond politics as well—history, psychology, and other areas can use similar ideas. The core techniques have been around for decades, and they remain common in practice, though there are more advanced methods today.

## 29) More Recent Unsupervised Advances: Word Embeddings From Large Unlabeled Text

A more recent development (around 2013–2014) is learning from a very large unlabeled dataset such as **Wikipedia**. You download documents as raw text with no human labels and learn **word embeddings**: representing each word as a vector.

The motivation is that these vectors act as numerical representations of words and can capture semantic meaning. Similar words end up with similar vectors. Beyond that, relationships between words can be encoded in the *directions* between vectors.

A classic illustration is that vectors for countries (Italy, France, Germany, USA) appear near each other, and vectors for capitals (Rome, Paris, Berlin) also form a related pattern. Even more strikingly, the direction from “Italy” to “Rome” can be similar to the direction from “France” to “Paris” and “Germany” to “Berlin.” This suggests a way to infer analogies: if you take the vector for “US” and move in the “country-to-capital” direction, you might land near “DC” or “Washington.” That example also highlights ambiguity: “Washington” can refer to a state or a person, so its vector may be less cleanly defined, and words with multiple meanings can behave differently.

You can also cluster word vectors to discover groups of related scientific terms or topics. Hierarchical clustering can help when categories overlap—for example, “mathematical physics” might fall between “math” and “physics” clusters. There is a lot of structure in these embeddings that you can leverage to solve tasks, and we will return to some of these ideas later.

## 30) Foundation Models and Large Language Models

More recently, there has been a major breakthrough: **large language models**. Many people are excited about them, and Stanford has many researchers working in this area. Roughly speaking, these are language models trained on very large-scale datasets—Wikipedia, and often much larger collections of online text. Training them can be extremely costly; even training a single large model might cost on the order of $10 million.

These models are powerful because they can be used for many different purposes. A well-known example is **GPT-3**, which you will likely hear about often. We will discuss this in one lecture.

Examples include generating coherent stories from a short prompt, answering questions about a passage (including questions where the answer is contained in the text), unscrambling letters into a meaningful word, and even answering simple arithmetic prompts like “95 times 45.” What is especially notable is not that the model can do any one of these tasks, but that it can do many tasks after being trained on a gigantic unlabeled corpus without being told in advance which specific tasks it must solve. You can often specify the task in natural language, and the same model can respond appropriately across very different requests.

That flexibility is a big part of why these are called **foundation models**—a term emphasized in a Stanford-written white paper—because a single trained model can support a wide range of applications, sometimes with little additional modification.

## 31) Q&A: Do Models Memorize Math Problems? What About Incorrect Information?

A student asked about the multiplication example and whether it could simply be memorized from the training corpus. The concern is whether “95 × 45” already appeared somewhere online and the model just memorized it.

The response was that while some math problems do appear in the training data, it’s unlikely that every possible pair of two-digit multiplications appears. So there must be some generalization: the model sees many numerical patterns, formulas, and operations in the corpus and extrapolates to new instances, including longer-digit multiplication.

A follow-up question asked about “pollution” in the corpus—whether the training data contains false information. The answer was that the corpus definitely contains incorrect information, but there is usually more correct information than incorrect, and the model learns patterns that tend to align with the dominant signal. However, there is also an area called **data poisoning**, where an adversary intentionally changes a small part of the training data so the model learns something wrong. That is possible, and it creates real risk, though targeted adversarial poisoning is not easy to carry out and is not believed to happen frequently in typical settings.

## 32) Reinforcement Learning: Learning Sequential Decisions

The last major part of the course is **reinforcement learning**, which will likely be two or three lectures near the end.

Reinforcement learning is about learning to make **sequential decisions**. Compared to supervised and unsupervised learning—which often look like prediction problems—reinforcement learning focuses on decisions that have long-term consequences. If you play chess, a move affects the future. You have to think about long-term ramifications. It is also sequential: you take a sequence of steps, and each step changes what happens next.

This is why reinforcement learning is used in settings like playing Go (for example, AlphaGo) or controlling robots. For a robot, you often control many joints and must decide how to move them over time. One example shown is a humanoid robot in simulation. The goal is to get the humanoid to walk to the right as fast as possible.

The learning process is often described as trial and error. The algorithm tries some actions, sees the robot fall, and updates the strategy. It may find partial progress—perhaps the robot takes a step but loses balance—and then adjusts the strategy again. Over iterations, the behavior improves, though it may remain unnatural. The “best” strategy for the robot might not look like human walking, and even if a strategy looks odd, it could still be effective under the objective being optimized.

## 33) A Key Difference: Interactive Data Collection and the Training Loop

A major distinction from supervised and unsupervised learning is that in many reinforcement learning setups you can collect data **interactively**. In supervised learning, you are typically given a fixed dataset and cannot ask for more labeled examples. In reinforcement learning, your actions generate new experiences. You try a strategy, observe the outcome, and that becomes new data.

At a high level, reinforcement learning often has a loop between **data collection** and **training**: you act using the current policy, observe feedback, incorporate the new data, and improve the policy. Over time, the dataset grows, and the algorithm can improve as it sees more experience.

## 34) Q&A: When Does Feedback Arrive?

A student asked whether feedback happens after each step or only after an entire sequence.

The answer was that there are multiple formulations. The most typical is that you observe feedback after each decision, but in many real situations feedback can be delayed, or you may not be able to update your strategy immediately due to computation limits, physical constraints, or communication constraints.

In some settings you have a **delayed reward** problem. In others, you may only be allowed to update your strategy a small number of times—sometimes framed as limited “deployment rounds.” In high-stakes settings like controlling a nuclear plant, you would not want an algorithm to change control strategies constantly. You might instead run experiments for a longer period, gain confidence that a new strategy is better (and safe), then deploy it and continue collecting feedback.

The discussion also emphasized that many real problems involve **multiple criteria**. For a nuclear plant, safety is critical. Even in robotics, letting a robot fall too often can damage hardware. Different research communities focus on different metrics depending on the application: training time, safety, robustness, and how multipurpose a model is.

## 35) Additional Topics Between the Big Three

Beyond supervised, unsupervised, and reinforcement learning, the course includes topics that sit between these areas.

One is **deep learning basics**, covered in about two lectures. Deep learning uses neural networks as a model parameterization, and it can be used with supervised learning, unsupervised learning, reinforcement learning, and other settings. Deep learning has been central to major progress in machine learning over roughly the last seven years.

We will also discuss **learning theory** for one or two lectures. The goal is not to go deep into proofs, but to understand trade-offs behind practical decisions: how to select features, how to reduce test error, and how to think about generalization. There will also be a lecture on how to use these insights to tune machine learning models in practice—what decisions matter when you implement and train algorithms.

Finally, we will have a guest lecture on broader aspects of machine learning, especially **robustness and fairness**. Machine learning now has significant societal impact because it works in real applications and can create serious issues. James Zou will give a guest lecture on fairness and robustness.

## 36) Closing

That’s everything I want to cover for today.
