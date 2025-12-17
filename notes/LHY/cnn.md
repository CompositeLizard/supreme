## 1) Introduction to Network Architecture and Convolutional Neural Networks

In this lecture, we discuss the design of network architectures, starting with one of the most important and widely used architectures: the convolutional neural network, or CNN. CNNs are especially designed for processing images. Through this example, the goal is to help you understand not only what a network architecture looks like, but also the intuition behind its design.

A natural question to ask is: why do we design different network architectures? Can architecture design help the network perform better? To answer this, we focus on an image-related task and gradually motivate the structure of CNNs through the characteristics of image recognition problems.

---

## 2) Image Classification Setup and Output Representation

Consider the task of image classification. We give an image to a machine and ask it to decide what kinds of objects appear in the image. As discussed previously, classification is a familiar task.

In this discussion, we assume that the size of the input image is fixed. For example, we may fix the input image size to $$100 \times 100$$. All images used by the model are assumed to have the same size. This is a common assumption in modern image recognition systems. Even though images in the real world may have different sizes or rectangular shapes, we usually preprocess them by rescaling all images to the same fixed size before feeding them into the model.

Since our task is classification, the output of the model is expressed as a one-hot vector, denoted by $$\hat{y}$$. For instance, if the category is “cat,” then the dimension corresponding to “cat” has value 1, while all other dimensions are 0. The length of this vector determines how many different object categories the model can recognize. If the vector length is 2000, then the model can identify 2000 different objects. Strong image recognition systems often recognize more than 1000 objects, and sometimes even tens of thousands.

After applying a softmax layer, the model produces an output $$y'$$. We want the cross-entropy between $$y'$$ and $$\hat{y}$$ to be as small as possible. Since cross-entropy has already been covered, we will not revisit it here.

---

## 3) Images as Tensors and Vectorized Inputs

The next question is how to use an image as the input to a model. For a computer, an image is represented as a three-dimensional tensor. If you are not familiar with tensors, you can think of them as matrices with more than two dimensions. A matrix has two dimensions, while anything with more than two dimensions is called a tensor.

An image has three dimensions: width, height, and number of channels. The width and height represent the image resolution and determine the total number of pixels. The channel dimension corresponds to color information. In a standard RGB image, each pixel is composed of three color channels: red, green, and blue.

To use an image as the input to a neural network, we can straighten this three-dimensional tensor into a vector. For a $$100 \times 100$$ RGB image, this results in $$100 \times 100 \times 3$$ numbers. By arranging all these values into a single row, we obtain a very long vector that can be used as the input to a network. Each element of this vector represents the intensity of a specific color channel at a specific pixel location.

So far, we have only discussed fully connected networks. If we use this vector as input, the input dimension is $$100 \times 100 \times 3$$, which is extremely large.

---

## 4) Parameter Explosion in Fully Connected Networks

Now suppose the first layer of our model contains 1000 neurons. Each neuron has a weight corresponding to every dimension of the input vector. Therefore, the number of weights in the first layer is

$$
1000 \times 100 \times 100 \times 3 = 3 \times 10^7.
$$

This is a very large number of parameters. While increasing the number of parameters increases the flexibility and expressive power of the model, it also increases the risk of overfitting. Conceptually, the more flexible a model is, the easier it is for it to overfit the training data.

Given the characteristics of image recognition tasks, we should ask whether we really need a fully connected network with so many parameters.

---

## 5) Observation 1: Local Patterns and Receptive Fields

The first key observation is that image recognition is largely about detecting local patterns. For example, to recognize a bird, a system may look for patterns such as a beak, eyes, or claws. If these patterns are detected and combined, the system may conclude that the image contains a bird.

Interestingly, humans use a similar approach. When people recognize objects, they often focus on the most distinctive features. This suggests that detecting local patterns is a reasonable strategy for machines as well.

Importantly, to detect a local pattern like a beak, it is not necessary to look at the entire image. A small region is usually sufficient. Therefore, a neuron does not need to take the whole image as input. It only needs to look at a small part of the image.

This leads to the concept of a **receptive field**. In a CNN, each neuron is responsible for a small local region of the image, called its receptive field. For example, a neuron may focus on a $$3 \times 3 \times 3$$ region, which includes a $$3 \times 3$$ area across all three RGB channels.

The values in this receptive field are straightened into a 27-dimensional vector, which serves as the input to the neuron. The neuron then assigns one weight to each of these 27 dimensions and produces an output that is passed to the next layer.

Receptive fields can overlap with each other, and multiple neurons can be assigned to the same receptive field. This allows the network to detect different types of patterns within the same local region.

---

## 6) Designing Receptive Fields: Size, Shape, and Kernels

Receptive fields can be designed in many ways. They can have different sizes, such as $$3 \times 3$$ or $$11 \times 11$$, depending on the scale of the patterns to be detected. They can also have different shapes, including rectangular ones. In theory, even very unusual receptive fields are possible, although they are rarely useful in practice.

In standard image recognition tasks, receptive fields usually cover all channels, since patterns are not assumed to exist only in a single color channel. As a result, we typically specify only the height and width of the receptive field. This pair is called the **kernel size**. A $$3 \times 3$$ kernel is the most common choice, as it is usually sufficient and computationally efficient.

Although some patterns may be larger than $$3 \times 3$$, this issue will be addressed later. For now, it is enough to note that small kernels are standard in CNNs.

---

## 7) Stride, Overlapping, and Padding

Receptive fields are applied across the entire image by shifting them horizontally and vertically. The amount of movement each time is called the **stride**, which is a hyperparameter chosen by the designer. Common stride values are 1 or 2. Small strides are preferred because they allow receptive fields to overlap, which helps ensure that patterns near boundaries are not missed.

When a receptive field extends beyond the boundary of the image, we use **padding**. Padding means filling the missing values, most commonly with zeros. Other padding methods are also possible, such as using the average pixel value or replicating boundary values. Padding ensures that edge regions of the image are also processed by the network.

By sliding receptive fields across the image with a chosen stride and padding strategy, every part of the image is covered by some receptive field, and local patterns can be detected everywhere.

---

## 8) Observation 2: Parameter Sharing Across Locations

The second key observation is that the same pattern may appear in different locations of an image. For example, a beak may appear in the upper left corner or the center of the image. Although the locations differ, the pattern itself is the same.

If we assign separate neurons with independent parameters to detect the same pattern at different locations, we introduce many redundant parameters. This is inefficient and unnecessary.

To address this, CNNs use **parameter sharing**. Neurons that detect the same type of pattern in different receptive fields share exactly the same weights. Although their parameters are identical, their outputs differ because their inputs come from different regions of the image.

This idea is analogous to offering one large course instead of many identical courses in different departments. The content is the same; only the audience changes.

In practice, each receptive field is associated with a group of neurons, and neurons at the same relative position across receptive fields share parameters. Each shared set of parameters is called a **filter**. If there are 64 neurons per receptive field, then there are 64 filters, each applied across all locations in the image.

---

## 9) From Simplifications to Convolutional Layers

We have now introduced two major simplifications compared to fully connected networks:

1. **Receptive fields**, which restrict neurons to local regions of the image.
2. **Parameter sharing**, which forces neurons detecting the same pattern at different locations to share weights.

Each of these simplifications reduces model flexibility. A fully connected network can, in principle, learn to focus on local regions by setting some weights to zero. However, once we impose receptive fields and parameter sharing, we explicitly restrict what the network can learn.

Combining receptive fields and parameter sharing gives us a **convolutional layer**. Networks built using convolutional layers are called **convolutional neural networks**, or CNNs.

CNNs have a larger model bias, meaning they are less flexible than fully connected networks. However, this bias is well aligned with the structure of image data. As a result, CNNs perform exceptionally well on image-related tasks and are less prone to overfitting.

While this architecture works extremely well for images, it should be applied to other tasks only after carefully considering whether those tasks share similar structural properties.

This concludes our introduction to CNNs and the motivation behind their design.


## 10) A Second, More Common Story for Introducing CNN

Now we are going to talk about another way of introducing CNN. This second way is exactly the same as the first way—it is the same story told in a different way. You can decide which version you prefer. If the first version did not feel clear when you listened to it, you can use this second version as another perspective.

Here starts the second version.

---

## 11) What a Convolutional Layer Is: Filters as Pattern Detectors

A convolutional layer contains many different filters. Each filter has a shape like

$$
3 \times 3 \times \text{channel}.
$$

Here, “channel” means the number of channels in the input image. For a color image, we have three channels (RGB). For a black-and-white image, we have only one channel.

So inside a convolutional layer, we have a collection of filters, and each filter is a tensor with shape $$3 \times 3 \times \text{channel}$$. The purpose of a filter is to capture a pattern in the image. Naturally, the pattern has to fit within the range of $$3 \times 3 \times \text{channel}$$ in order to be captured directly by that filter.

The values inside a filter are the parameters of the model. In practice, these values are unknown at the beginning and are learned through training. But for explanation, we will assume the filter values have already been learned, and we will look at how filters operate on an image to detect patterns.

---

## 12) How a Filter Scans an Image: Inner Products, Stride, and Outputs

Let’s go through an example. We assume the channel count is 1, meaning the image is black and white. Suppose the image is size $$6 \times 6$$, and we already know the filter parameters.

To apply a filter, we place it on the image—starting from the upper left corner—and compute the inner product between the filter values (nine values for a $$3 \times 3$$ filter) and the corresponding $$3 \times 3$$ patch in the image. That inner product produces one output number.

Then we move the filter a little to the right. The moving distance is called **stride**. In the first version of the story, we used stride $$=2$$; in this version, we set stride $$=1$$. After shifting right by one position, we compute the inner product again to get the next output number.

We repeat this: slide to the right, compute another inner product, and keep going. When we finish scanning across one row, we move the filter down a little and continue the same process. We repeat until the filter reaches the lower right corner.

So the output of a filter is a grid of numbers—each number corresponds to how strongly that filter matches the local patch of the image at that location.

---

## 13) Example Intuition: Why the Filter “Detects” a Pattern

How does a filter detect a pattern? Look at the structure of the filter. For example, suppose the filter has ones along the diagonal. Then we can ask: what kind of patch in the image will make the inner product result as large as possible?

If the corresponding locations in the image patch contain three ones aligned with that diagonal, the output becomes large. This explains why, in the output grid, locations where the filter’s target pattern appears will produce larger values. In other words, large values in the output indicate that the pattern captured by that filter appears at those positions in the image.

That is the basic meaning of convolution: the filter slides across the image, and the output values reflect pattern matches at different locations.

---

## 14) From Filters to Feature Maps

Every filter repeats the same scanning process. For example, a second filter might detect a different pattern, such as “ones appearing in a straight line.” It scans the image in the same way—place it, compute an inner product, shift by the stride, compute again, and continue until the entire image is scanned.

Each filter produces its own grid of numbers. If we have many filters—say 64 filters—then we obtain 64 grids of numbers.

This output is called a **feature map**. You can think of it as a new “image,” except its number of channels equals the number of filters. So if the original image has 1 channel (black and white), after a convolutional layer with 64 filters, we get a new representation with 64 channels—one channel per filter.

---

## 15) Stacking Convolutional Layers and Channel Matching

A convolutional layer can be stacked multiple times. Suppose we add a second convolutional layer. The second layer also contains many filters. Each filter still has height and width $$3 \times 3$$, but the channel dimension must match the channel count of the input it receives.

After the first convolutional layer, the output “image” has 64 channels (because we used 64 filters). Therefore, the filters in the second convolutional layer must have shape

$$
3 \times 3 \times 64.
$$

More generally, the channel dimension of a filter equals the number of channels in the input feature map it processes. If the input is a color image with 3 channels, then first-layer filters have shape $$3 \times 3 \times 3$$. If the first layer outputs 64 channels, then second-layer filters have shape $$3 \times 3 \times 64$$.

---

## 16) Why Small Filters Still See Large Patterns in Deep Networks

Now we answer an important question: if we always set filter size to $$3 \times 3$$, does that mean the network cannot detect large patterns?

Actually, it can. The key is that deeper layers have a larger effective range on the original image. For example, consider a second convolutional layer using $$3 \times 3$$ filters on the feature map produced by the first layer. A single position in the first-layer feature map corresponds to a region in the original image. Therefore, a $$3 \times 3$$ region in the first-layer feature map corresponds to a larger region in the original image—such as a $$5 \times 5$$ range in this explanation.

So although each filter is only $$3 \times 3$$, stacking convolutional layers increases the effective receptive field. The deeper the network is, the larger the range it can “see,” even when using the same $$3 \times 3$$ filters. If the network is deep enough, it can still detect large patterns.

---

## 17) Connecting the Two Stories: Neurons, Weight Sharing, and Convolution

So far we have told two versions of the same story.

In the first version, we described neurons with receptive fields, and we said that neurons in different receptive fields can share parameters. In this second version, the shared parameters are exactly the **filters**.

A filter contains parameters such as $$3 \times 3 \times 3$$ numbers (for a color image). In this explanation, we ignored bias to keep the story simple, but in real implementations filters typically also include bias terms.

In the first version, “weight sharing across different receptive fields” corresponds to “a filter scanning through the entire image” in the second version. That scanning operation is what we call **convolution**, which is why this layer is called a convolutional layer.

So:

- Different receptive fields correspond to different spatial positions where the filter is applied.
- Shared parameters correspond to the filter itself.
- Sliding the filter across the image is convolution.

---

## 18) A Third Common Component: Pooling

When CNNs are used for image recognition, there is a third commonly used component called **pooling**.

Pooling comes from another observation: if we subsample a large image—such as removing certain rows and columns—the image becomes smaller, but the main content may still be recognizable after resizing. For example, a bird still looks like a bird even in a smaller version of the image.

Pooling is designed to reduce the spatial size of the feature maps. One important point is that pooling has **no parameters**. It has no weights and nothing to learn. In that sense, pooling is more like an activation function (such as sigmoid or ReLU): it is a fixed operator.

There are many kinds of pooling. Here we focus on **max pooling**.

---

## 19) Max Pooling: Grouping Values and Taking the Maximum

Max pooling works by grouping nearby values and choosing a representative value from each group. For example, we can group outputs in $$2 \times 2$$ blocks. For each $$2 \times 2$$ group, max pooling takes the maximum value as the representative.

You do not have to choose the maximum—this is a design choice. There are other pooling methods, such as min pooling (taking the minimum) or average pooling (taking the average). Similarly, grouping does not have to be $$2 \times 2$$; it could be $$3 \times 3$$, $$4 \times 4$$, or other sizes.

Pooling reduces the width and height of the feature map but keeps the number of channels unchanged. For example, if we have a $$4 \times 4$$ feature map and we pool with $$2 \times 2$$ grouping, it becomes a $$2 \times 2$$ feature map. If there are 64 channels, all 64 channels remain—only the spatial resolution shrinks.

---

## 20) Modern Practice: Pooling Is Optional and Sometimes Removed

In practice, convolution and pooling are often used together. A common design is to do convolution several times and then apply pooling once—for example, two convolution layers followed by one pooling operation.

However, pooling can slightly harm performance when you need to detect very subtle details, because subsampling can lose information. In recent years, many architectures have started to remove pooling entirely for image classification and use **fully convolutional** designs (networks built from convolutional layers without pooling). One reason is that computing power has become much stronger, and pooling was historically used mainly to reduce computation by shrinking feature maps.

So pooling is common and useful, but it is optional. Many modern designs choose not to include it.

---

## 21) From CNN Features to Classification: Flatten, Fully Connected Layers, and Softmax

After passing through several convolutional layers (and possibly pooling), we eventually need to produce a final classification output. A typical step is called **flatten**.

Flatten means taking the values that are arranged in a matrix (or tensor) form and straightening them into a single “flat” vector. After flattening, we feed this vector into one or more fully connected layers, then apply softmax, and finally obtain the classification result.

A classic image classification network often includes convolutional layers, pooling, flatten, fully connected layers, and softmax. We will practice an image classification problem in homework 3.

---

## 22) CNN Beyond Images: Using CNN for Go (AlphaGo Example)

Besides image recognition, you have probably heard of CNN in another famous application: playing Go. If we teach machine learning without mentioning AlphaGo, it might feel like something is missing, so let’s mention it briefly.

We can treat playing Go as a classification problem. The input is the current Go board position, and the output is where the next move should be.

We already know the input to a network is typically a vector. One simple representation is to view the Go board as a $$19 \times 19$$ grid. We can represent it as a $$19 \times 19$$ vector-like structure (a flattened grid). For each position:

- put 1 if there is a black stone,
- put $$-1$$ if there is a white stone,
- put 0 if it is empty.

This is only one possible representation—you could design other encodings. The key idea is that we can represent the board state numerically and feed it to a network.

Then playing Go becomes a classification problem with $$19 \times 19$$ categories, where the network outputs which position is the best next move.

A fully connected network could solve this, but CNN tends to perform better.

---

## 23) Why CNN Helps for Go: Similarities to Images

Why is CNN better for Go? You can view the Go board as an image with resolution $$19 \times 19$$. This is a very small “image” compared to typical photos, but the analogy still works: each “pixel” corresponds to a board position.

What about channels? In a normal image, channels are RGB. In AlphaGo’s original design, each board position is described using 48 channels. That means for every position (every “pixel”), they use 48 numbers to describe what is happening there—features such as whether a stone could be captured, whether neighboring stones have certain colors, and other Go-specific conditions. These features reflect expert knowledge.

Since CNN is designed for images, using CNN for Go implies that Go and images share important structural characteristics. The same two observations apply:

1. You only need to look at a small local region to detect important patterns.
2. The same pattern can appear in different locations.

In fact, AlphaGo uses a $$5 \times 5$$ filter size in its first layer, suggesting that patterns within a $$5 \times 5$$ local board region contain useful information.

---

## 24) Pooling Is Not Always Suitable: What AlphaGo Actually Does

People sometimes assume pooling is required for CNN. In image recognition, subsampling often does not destroy the meaning of the image. But with Go, subsampling is not reasonable: if you remove rows or columns, the board position changes completely and the game is no longer the same.

This leads to a common doubt: if pooling is essential, would AlphaGo have an obvious weakness? But AlphaGo is extremely strong, so it cannot rely on something with such an obvious flaw.

To resolve this, the speaker read AlphaGo’s Nature paper carefully. The paper itself is short—about five or six pages—but the network architecture details are in the appendix rather than the main text.

According to the appendix, AlphaGo’s network works roughly like this:

- Transform the board into an “image” of size $$19 \times 19 \times 48$$.
- Use **zero padding** (add zeros outside the border if a filter exceeds the image range).
- First layer: kernel size $$5 \times 5$$, with $$k=192$$ filters (they experimented with 128 and 256 and found 192 worked best).
- Stride $$=1$$.
- Apply rectifier nonlinearity, meaning **ReLU**.
- Layers 2 through 12: also use zero padding, kernel size $$3 \times 3$$, 192 filters per layer, stride $$=1$$.
- Final layer: apply softmax, since it is a classification problem.

The key thing is: **they did not use pooling**.

This is a good example showing there is no “golden rule” for architecture design. Pooling is not always good, and it is not suitable for Go. When you choose an architecture component, you must understand what it does and decide whether it matches the nature of the task.

---

## 25) CNN in Other Domains: Speech and Language Need Different Designs

CNN has also been used in speech recognition and language processing in recent years. We do not go into details here, but if you want to apply CNN in these domains, you must read the literature carefully.

The design of receptive fields and parameter sharing can be very different between image processing and speech or language tasks. Receptive fields for speech are designed based on the characteristics of language signals, so you cannot assume that CNN designed for images will work naively in other areas. You have to think carefully about the structure of the data and design receptive fields accordingly.

---

## 26) A Major Weakness of CNN: Scaling and Rotation

Now we discuss a weakness of CNN: CNN cannot naturally handle image scaling and image rotation.

For example, suppose you train a CNN on many dog images of the same size, and it learns to recognize dogs. If you zoom in on the image, the CNN may fail to recognize it. This may seem strange, because the shape looks the same to humans. But from the network’s perspective, when images are transformed into vectors, the numerical values can be very different after scaling or rotation. Even if humans see the same object, the raw input values change significantly.

Because of this, CNN is not as powerful as you might assume in terms of handling scale and rotation invariance.

That is why **data augmentation** is commonly used in image recognition. With data augmentation, you can crop parts of images and enlarge them, or apply random rotations, so that the CNN becomes familiar with different sizes and angles of the same object. This often improves performance.

You might ask whether there exists an architecture that can deal with scaling and rotations more directly. The speaker mentions that there is an architecture called a **Special Transformer Layer**, but it will not be covered here, and a reference video link is provided instead.

