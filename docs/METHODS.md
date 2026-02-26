# EchoQual: Detailed Methodology

## Problem Statement

Echocardiogram image quality assessment currently requires expert sonographers to manually
grade acquisition quality — a process that is expensive, subjective (inter-rater κ ≈ 0.6–0.7),
and not scalable. We propose deriving quality scores entirely from the internal representations
of self-supervised foundation models, requiring **zero expert quality labels**.

## Three Proxy Quality Signals

### Signal 1: View Classification Confidence

**Intuition:** A model trained to recognize standard echocardiographic views will exhibit
lower confidence on poorly acquired images where expected anatomical structures are missing,
occluded, or distorted.

**Implementation:** We use the view classification head (trained self-supervisedly via
clustering in the contrastive embedding space) and compute:

- **Entropy method:** `quality = 1 - H(softmax(logits)) / log(K)` where K is number of views
- **Energy method:** `quality = T * logsumexp(logits / T)` (higher energy = more in-distribution)

**Prior art limitation:** Zhang et al. (2018) found only modest correlation between raw softmax
confidence and quality. We improve upon this by using entropy over the full softmax distribution
and combining with other signals.

### Signal 2: Embedding Space Density

**Intuition:** In a well-trained contrastive embedding space, high-quality images form tight
clusters. Degraded, off-axis, or poorly acquired images fall in low-density regions as outliers.

**Implementation options:**
- **k-NN distance:** Average cosine distance to k=10 nearest neighbors in the training set
- **Mahalanobis distance:** Distance to view-specific centroid using the full covariance structure
- **GMM likelihood:** Log-likelihood under a Gaussian Mixture Model fit to training embeddings

**Key insight:** The reference distribution is computed from the training set embeddings without
any filtering or quality labels. The assumption is that the *majority* of clinical echocardiograms
are of acceptable quality, so the density center represents "typical good quality."

### Signal 3: Vision-Language Alignment

**Intuition:** If a vision-language model is trained to align video embeddings with clinical
text descriptions, then a well-acquired A4C view should have high cosine similarity to the
text "Standard apical four-chamber view with clear visualization of all four chambers."
A poorly acquired image will diverge from these canonical descriptions.

**Implementation:**
1. Define canonical text descriptions for each standard view (5 good + 5 poor per view)
2. Encode texts using a sentence transformer (all-MiniLM-L6-v2)
3. Project video and text embeddings to a shared space
4. Compute differential score: `sim(video, good_texts) - sim(video, poor_texts)`

This is the most novel component — no prior work has used VL alignment as a quality proxy.

## Score Fusion

The three signals are fused via:
- **Min-max normalization** of each signal to [0, 1]
- **Weighted sum:** `Q = w₁·S_view + w₂·S_density + w₃·S_vl` with default weights (0.3, 0.4, 0.3)
- **Alternative: Borda rank aggregation** for robustness to outliers

## Evaluation Strategy

Since we have no expert quality labels, we use **proxy ground truth**:

1. **EF prediction confidence:** Videos where any EF prediction model makes large errors
   are likely of poor quality (hard to segment → hard to see structures)
2. **Synthetic perturbation test:** Add increasing Gaussian noise to clean videos and verify
   that quality scores decrease monotonically (sanity check)
3. **Cross-validation with available datasets:** When expert labels exist (e.g., CAMUS quality
   grades), compute Spearman rank correlation

## Self-Supervised Pretraining

The video encoder is pretrained using two complementary objectives:

1. **SimCLR contrastive learning:** Two augmented views of the same video are pulled together
   in embedding space while different videos are pushed apart (NT-Xent loss)
2. **Frame reordering pretext:** Video frames are randomly shuffled; the model must predict
   the correct temporal order. This encourages learning of cardiac cycle dynamics.

**Echo-specific augmentations:**
- Random crop (0.75–1.0x)
- Horizontal flip
- Brightness/contrast jitter (simulating gain differences)
- Gaussian noise (simulating speckle variation)
- Temporal frame dropping/reversal
