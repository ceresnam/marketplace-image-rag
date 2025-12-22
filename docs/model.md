# DINOv2

Using **DINOv2** for **both** Search and Classification allows you to build a **Unified Pipeline** where you run the heavy AI model only once, and get both features for "free."

Here is why this strategy beats having a separate EfficientNet for classification.

### 1. The "Unified Pipeline" Strategy

Instead of maintaining two different models (EfficientNet for Classifying + DINOv2 for Search), you use DINOv2 as a universal **Feature Extractor**.

- **Step 1 (Heavy):** Feed image to DINOv2 Get 768-dimensional Vector.
- **Step 2 (Light):**
- **Task A (Search):** Send vector to Qdrant/Milvus to find similar vectors.
- **Task B (Classify):** Multiply vector by a small matrix (Linear Layer) to get the Category.

**The Speed Gain:**

- **Old Way:** Run EfficientNet (20ms) + Run DINOv2 (25ms) = **45ms latency**.
- **Unified Way:** Run DINOv2 (25ms) + Linear Classifier (0.01ms) = **25ms latency**.

### 2. "Linear Probing"

Because DINOv2 understands objects so well, you **do not need to fine-tune the backbone**. You only need to train the final "decision" layer. This is called **Linear Probing**.

- **Training Time:** Minutes (not hours).
- **Compute Cost:** You can do this on a CPU.
- **Stability:** You cannot "break" the model because the backbone is frozen.

### The Only Downside (Latency)

**DINOv2 (ViT-Small)** is slightly slower than **EfficientNet-B0** on pure CPU.

- **EfficientNet-B0:** ~15-20ms
- **DINOv2-Small:** ~25-35ms

**Verdict:** The 10ms difference is negligible for a user upload, and the benefit of maintaining **one single model** instead of two is massive for your engineering sanity.

### Recommendation

**Switch to DINOv2 for EVERYTHING.**

1. Run the offline job to generate embeddings for all 150k products.
2. Train a `LogisticRegression` on those embeddings to predict the 150 categories.
3. Deploy the Unified Pipeline.

# EfficientNetV2

### Why EfficientNetV2-S

- **Modern architecture** - Faster training than original EfficientNet (up to 5-11x speedup)
- **Excellent accuracy** (~84.9% Top-1 accuracy on ImageNet)
- **Efficient training** - Uses Fused-MBConv blocks for better speed/accuracy trade-off
- **Better parameter efficiency** (~24M parameters)
- **Optimized for transfer learning** - Progressive learning during training
- **PyTorch ecosystem** provides excellent pre-trained models via `torchvision` and `timm` libraries

### Why Better Than AlexNet, ResNet, and Original EfficientNet?

| Model                | Year | Parameters | Accuracy | Training Speed | Why Inferior                                         |
| -------------------- | ---- | ---------- | -------- | -------------- | ---------------------------------------------------- |
| **AlexNet**          | 2012 | ~61M       | ~56.5%   | Fast           | ❌ Outdated architecture, poor accuracy, inefficient |
| **ResNet-50**        | 2015 | ~25M       | ~76.1%   | Medium         | ⚠️ Good but slower training, lower accuracy          |
| **ResNet-101**       | 2015 | ~44M       | ~77.4%   | Slow           | ⚠️ Heavyweight, diminishing returns                  |
| **EfficientNet-B0**  | 2019 | ~5.3M      | ~77.1%   | Medium         | ⚠️ Good accuracy but slower training                 |
| **EfficientNet-B4**  | 2019 | ~19M       | ~82.9%   | Slow           | ⚠️ High accuracy but very slow training              |
| **EfficientNetV2-S** | 2021 | ~24M       | ~84.9%   | Fast           | ✅ Best accuracy-to-efficiency ratio                 |

**Bottom Line:** For 150 marketplace categories, you need a model that captures fine-grained details (e.g., distinguishing "Leather Jacket" from "Faux Leather Jacket"). AlexNet is too primitive, ResNet is good but outdated, original EfficientNet is accurate but slow to train, and EfficientNetV2-S is the modern standard with the best speed-accuracy trade-off.
