# CartoonGAN 

**Paper:** [*CartoonGAN: Generative Adversarial Networks for Photo Cartoonization*](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)
**Authors:** Yang Chen, Yu-Kun Lai, Yong-Jin Liu, et al.

---

## What this paper does 

This paper introduces **CartoonGAN**, a neural network method that turns real photos into cartoon-style images. The model learns from *unpaired* sets of photos and cartoon images (you do not need matching photo-cartoon pairs). The main goals are to keep the original content while producing clear cartoon edges and smooth, flat shading.

---

## Main ideas 

* Use a single **Generator** (photo → cartoon) and a **Discriminator** to tell real cartoons from generated ones.
* Design losses that suit cartoons: an **edge-promoting adversarial loss** and a **semantic content loss** based on VGG features with **L1 sparse regularization**.
* Add a simple **initialization (pre-train)** step that helps training converge faster and avoid bad local minima.

---

## Math 

The model solves a min-max problem between the generator (G) and discriminator (D):

```math (function)
(G^*, D^*) = \arg\min_G \max_D L(G, D)
```

The total loss is a sum of adversarial loss and weighted content loss:

```math (function)
L(G, D) = L_{adv}(G, D) + \omega \; L_{con}(G, D)
```

where (\omega) balances style vs. content (the paper uses (\omega = 10)).

The **edge-promoting adversarial loss** is:

```math (function)
L_{adv}(G, D) = \mathbb{E}_{c\sim S_c}[\log D(c)] + \mathbb{E}_{e\sim S_e}[\log(1 - D(e))] + \mathbb{E}_{p\sim S_p}[\log(1 - D(G(p)))]
```

* (c) = real cartoon image.
* (e) = edge-smoothed cartoon image (they create it by detecting edges, dilating, and then smoothing).
* (p) = photo image.

The **content loss** compares VGG feature maps between the generated image and the input photo, using L1 (sparse) norm:

```math (function)
L_{con}(G) = \mathbb{E}_{p\sim S_p} \big[ \| VGG_l(G(p)) - VGG_l(p) \|_1 \big]
```

They use the `conv4_4` layer of a pretrained VGG network for these feature maps.

---

## Why edge-smoothed cartoons

Cartoon images have strong, thin edges but edges occupy a small part of the image. A standard discriminator can be fooled by correct shading but missing edges. To fix this, the authors create a set of *edge-smoothed* cartoons (remove edges with Canny → dilate → Gaussian smooth) and make the discriminator also reject these. This forces the generator to produce images with clear edges.

---

## Network architecture 

* **Generator (G)**: conv layer → 2 down-conv blocks → 8 residual blocks → 2 up-conv blocks → final 7×7 conv.

  * Up-conv uses fractionally strided conv (stride 1/2) to upsample.
* **Discriminator (D)**: patch-level, shallow network (works on local patches because cartoon style is local). Uses LeakyReLU and batch normalization.

---

## Training and data

* Training images are resized/cropped to 256×256.
* Example datasets in the paper: ~6k photos from Flickr and several thousand cartoon frames from specific films/artists (Makoto Shinkai, Miyazaki, Paprika, Mamoru Hosoda).
* Implementation: Torch (Lua). Experiments run on an NVIDIA Titan Xp GPU.
* Pre-train the generator with only `L_con` for several epochs (initialization phase), then train adversarially.

---

## Experiments & findings 

* CartoonGAN produces clearer edges and smoother shading than NST and CycleGAN on cartoon styles.
* It learns artist-specific styles when trained on frames from a single director/film.
* Faster training than CycleGAN because CartoonGAN trains only one direction (photo → cartoon) and uses content loss rather than cycle consistency.
* Ablation studies show each part (initialization, L1 content loss, and edge-promoting adversarial loss) matters.

---

## Pros and cons 

**Pros:**

* Good cartoon-style outputs with clear edges.
* Requires unpaired data (easier to collect).
* More efficient than bidirectional cycle methods for this task.

**Cons / limits:**

* Not designed to map cartoons back to photos.
* May struggle with faces; authors propose future work to use local facial features.
* Original code used Torch/Lua (less common now); others may reimplement in PyTorch/TensorFlow.

---

## Future work 

* Improve portrait/face cartoonization using local facial features.
* Apply similar loss ideas to other image synthesis tasks.
* Extend the method to video by adding sequential constraints.

---

