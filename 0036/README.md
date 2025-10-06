# Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (AdaIN)

Paper: Xun Huang & Serge Belongie, 2017 .. [Paper](https://arxiv.org/abs/1703.06868)

## What is this paper about?

This paper introduces AdaIN (Adaptive Instance Normalization), a new
layer that makes style transfer fast, flexible, and real-time.

Before AdaIN: Gatys et al. (2015) introduced neural style transfer, but it was very slow (iterative optimization). 
Faster feed forward methods existed, but they were limited to a few fixed styles only

With AdaIN:We can apply any style image to any content image in real-time. It works with one single feed-forward network, not multiple networks for each style.

## Key Idea

Style = feature statistics (mean + variance) Content = spatial structure
of features

AdaIN works by replacing the mean and variance of the content features with those of the style features. This instantly makes the content image take on the style.

Equation:

``` math
AdaIN(x, y) = σ(y) * ( (x - μ(x)) / σ(x) ) + μ(y)
```

Where: x = content features,  y = style features , μ = mean (perchannel) ,  σ = standard deviation (per channel)

Steps: 1. Normalize the content features. 2. Rescale + shift them using the style's statistics.

## Architecture

The model is very simple:

1.  Encoder (VGG-19)
    -   Extracts features from content and style images.
2.  AdaIN Layer
    -   Aligns the statistics (mean & variance) of content features with
        style features.
3.  Decoder
    -   Reconstructs the stylized image from the modified features.

Diagram: Content Image → Encoder ┐ │ → AdaIN → Decoder → Stylized Image
Style Image → Encoder ┘

## Training

-   Content dataset: MS-COCO
-   Style dataset: WikiArt paintings
-   Loss function:

``` math
L = L_c + λ * L_s
```

Where: L_c: Content loss (difference between AdaIN output features and
decoder output features). L_s: Style loss (difference in mean & variance between style and output image)

## Results

-   Speed: Nearly real-time (56 FPS at 256px).
-   Flexibility: Works with any style image, not just pre-trained ones.
-   Quality: Comparable to optimization-based methods, but \~1000x faster.

## Extra Controls

AdaIN allows: - Content vs Style trade-off: adjust balance during
inference. - Style interpolation: mix multiple styles (for example, 50%
Van Gogh + 50% Monet). - Color control: preserve content colors if
desired. - Spatial control: apply different styles to different parts of the image.

