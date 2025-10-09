# FlowNet: Learning Optical Flow with Convolutional Networks 

## 1. Introduction

-   **Optical flow** means estimating how each pixel in an image moves between two consecutive frames of a video.
-   Before this paper, most optical flow methods were **variational methods** (optimization-based), which were slow and required hand-crafted matching or interpolation techniques.
-   Deep learning (CNNs) had shown great success in classification, detection, segmentation but not yet in optical flow.
-   The authors asked: *Can we train a CNN end-to-end to directly predict dense optical flow from two images?*
-   Their answer: **Yes.** They built two CNN architectures (FlowNetS and FlowNetC) and trained them on a large synthetic dataset they created called Flying Chairs

------------------------------------------------------------------------

## 2. Related Work

-   **Classical approaches**: Variational methods (Horn & Schunck 1981) minimized data + smoothness terms. Improvements: large displacement handling, sparse-to-dense methods like DeepMatching, DeepFlow, EpicFlow.
-   **Machine learning approaches**: Used statistical models, classifiers, random forests to predict flow patches. These methods were limited and not competitive on large, realistic datasets.
-   **Earlier neural nets**: Unsupervised architectures (RBMs, autoencoders) could model motion in synthetic settings but failed on real videos.
-   This paper is **the first successful CNN approach** for supervised optical flow learning.

------------------------------------------------------------------------

## 3. Architectures

Two CNNs:

### 3.1 FlowNetSimple (FlowNetS)

-   Input: stack of two images `[H x W x 6]` (since each image is RGB).
-   Processed by 9 convolutional layers with stride=2 and ReLU activations.
-   Progressive downsampling → feature maps with semantic information but low resolution.
-   Then **upconvolutional layers** (unpooling + convolution) refine flow back to higher resolution.
-   Final output: dense optical flow map (x, y displacement for each pixel).

### 3.2 FlowNetCorr (FlowNetC)

-   Two parallel convolutional streams: one per input image.
-   Features combined with a **correlation layer**: compares feature
    patches across the two streams using dot products (like a
    sliding-window matching).
-   Helps with correspondence between images.
-   Weakness: correlation search limited to a fixed max displacement (20
    pixels), making large motion harder.

### 3.3 Refinement

-   Outputs are coarse (downsampled). They try two upsampling
    strategies:
    1.  **Bilinear upsampling** (fast but blurry).
    2.  **Variational refinement (+v)**: a traditional optical flow
        method applied to CNN output for smoother, subpixel-accurate
        flows.

------------------------------------------------------------------------

## 4. Training Data

### 4.1 Existing datasets

-   Middlebury: 8 pairs, small motions.
-   KITTI: 194 pairs, large displacements, sparse ground truth.
-   Sintel: 1,041 pairs, synthetic, dense ground truth, has *Clean* (no effects) and *Final* (motion blur, fog, etc.).
-   Problem: Too small to train CNNs.

### 4.2 Flying Chairs (new dataset)

-   Authors built a synthetic dataset:
    -   Backgrounds from Flickr images.
    -   3D chair models placed randomly, undergoing affine  transformations (translation, rotation, scaling).
    -   Generated **22,872 image pairs** with dense ground truth optical flow + occlusion masks.
-   Surprising result: Networks trained only on Flying Chairs generalize well to Sintel and KITTI.

### 4.3 Data Augmentation

-   Essential to avoid overfitting.
-   Applied online during training:
    -   Geometric: translation, rotation, scaling, flipping.
    -   Photometric: brightness, contrast, color changes, gamma, Gaussian noise.
-   Always applied consistently to both images in a pair, with some small relative offsets to simulate flow.

------------------------------------------------------------------------

## 5. Training Details

-   Architecture: 9 conv layers, kernel sizes vary (first layers use 7x7, 5x5, then smaller).
-   No fully connected layers.
-   Activation: ReLU.
-   Optimization: Adam optimizer.
-   Learning rate schedule:
    -   Start at 1e-4, drop by factor 2 every 100k iterations.
    -   FlowNetC: suffers exploding gradients if start LR too high →
        used LR warm-up (start at 1e-6).
-   Batch size: 8.
-   Loss function: **Endpoint Error (EPE)** (L2 distance between predicted and ground truth flow).

------------------------------------------------------------------------

## 6. Results

### 6.1 Evaluation metrics

-   EPE (pixels) on datasets: Sintel, KITTI, Middlebury, Flying Chairs.

### 6.2 Key findings

-   **Flying Chairs alone** is enough to learn good optical flow.
-   **Fine-tuning on Sintel** improves performance on Sintel and KITTI.
-   FlowNetS generalizes better to Sintel Final; FlowNetC better on Sintel Clean and Flying Chairs.
-   FlowNetC struggles with **large displacements** due to limited correlation search range.
-   Variational refinement (+v) improves smoothness, sometimes hurts on Flying Chairs (because CNN already produces good output).

### 6.3 Comparison with other methods

-   FlowNets are **faster**: \~10 frames per second on GPU (real-time).
-   Accuracy: Competitive with state-of-the-art like EpicFlow, DeepFlow.
-   Qualitatively: Networks preserve fine details (edges, small structures) better than EpicFlow, though global smoothness is worse.

------------------------------------------------------------------------

## 7. Analysis

-   Training only on Sintel: works, but Flying Chairs + fine-tuning is better.
-   Data augmentation is critical even with large datasets.
-   FlowNetS vs FlowNetC:
    -   S = better generalization, large motions.
    -   C = better detail on clean data, but overfits more.
-   Main limitation: correlation layer restricts maximum displacement.

------------------------------------------------------------------------

## 8. Conclusion

-   CNNs can directly learn dense optical flow prediction from two raw images.
-   Synthetic data (Flying Chairs) is surprisingly effective for training.
-   FlowNetS and FlowNetC achieve real-time performance with competitive accuracy against slow, hand-crafted classical methods.
-   Future: with better and more realistic datasets, deep learning methods for optical flow will surpass traditional ones further.

------------------------------------------------------------------------

## 9. Equations

-   Loss function:

    ```math
    EPE = \sqrt{(u - u^*)^2 + (v - v^*)^2}
    ```

    where ( (u, v) ) is predicted flow and ( (u\^*, v\^*) ) is ground
    truth.

-   Correlation layer:

    ```math
    C(x_1, x_2) = \sum_{o \in \mathcal{R}} f_1(x_1 + o) \cdot f_2(x_2 + o)
    ```

    where ( f_1, f_2 ) are feature maps, ( o ) runs over a patch.

------------------------------------------------------------------------

## 10. References

-   Dosovitskiy, A., Fischer, P., Ilg, E., et al. "FlowNet: Learning
    Optical Flow with Convolutional Networks." arXiv:1504.06852 (2015).
