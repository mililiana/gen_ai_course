# Generative AI Course 2026

A collection of lab assignments exploring different areas of generative AI — from image classification and anomaly detection to LoRA fine-tuning for images, text, and music generation.

**Author:** Liliana Mirchuk

## Labs Overview

### Lab 1 — Artifact Detection in AI-Generated Face Images

Binary classification of AI-generated face images to detect visual artifacts (distorted hands, extra fingers, text overlays, misaligned eyes, etc.). The dataset contains 2,000 images with a 10/90 class imbalance (artifact vs. clean).

**Approach:** Two pretrained models — EfficientNet-B0 (CNN, captures local texture patterns) and ViT-B/16 (Vision Transformer, captures global semantic artifacts) — fine-tuned and combined into a weighted ensemble. Class imbalance was addressed with WeightedRandomSampler, weighted CrossEntropyLoss, and threshold optimization.

**Key result:** ViT-B/16 achieved the best test macro F1 of 0.8302 with ROC-AUC 0.8722.

**Stack:** PyTorch, timm, albumentations, scikit-learn

`lab_1/`


### Lab 2 — Autoencoder-Based Anomaly Detection

An unsupervised approach to the same artifact detection problem using a convolutional autoencoder. The model is trained only on clean images and learns to reconstruct them; artifact images produce higher reconstruction errors, enabling anomaly detection without labeled artifact examples.

**Approach:** A symmetric convolutional autoencoder with a configurable bottleneck (tested sizes: 32, 64, 128, 256, 512). Extended with perceptual loss (VGG feature matching) and an ensemble combining autoencoder anomaly scores with Lab 1's supervised classifiers.

**Stack:** PyTorch, torchvision (VGG for perceptual loss), scikit-learn, matplotlib

`lab_2/`


### Lab 3 — LoRA Fine-Tuning for Image Generation (Stable Diffusion)

Training personal LoRA adapters for Stable Diffusion to generate images of a specific person/concept. Two architectures were explored: SDXL (1024×1024) and SD 1.5 (512×512), both trained using the Kohya-ss training framework on Kaggle T4 GPUs.

**Dataset preparation:** Raw photos (including HEIC format) were converted to center-cropped JPEGs at target resolution, then auto-captioned using BLIP-2 for text–image pairs.

**Stack:** Kohya-ss (sd-scripts), Stable Diffusion XL / 1.5, BLIP-2, Pillow, pillow-heif

`lab3/`


### Lab 4 — Tag Generation for Ukrainian News Articles

Predicting semantic tags for Ukrainian news articles using three progressively more capable approaches, evaluated on the [FIdo-AI/ua-news](https://huggingface.co/datasets/FIdo-AI/ua-news) dataset (~150K articles).

**Three approaches compared:**

1. **Classical baseline** — TF-IDF keyword extraction (extractive only, limited to words appearing in the text)
2. **Zero-shot LLM** — Qwen 2.5 3B via Unsloth (4-bit quantization), prompted in Ukrainian to generate abstractive tags without fine-tuning
3. **Fine-tuned LLM** — Same Qwen 2.5 3B model with LoRA adapters trained on the dataset to match the style and vocabulary of human-annotated tags

**Metrics:** Tag-level F1 and chrF++ (character n-gram overlap, suited for morphologically rich Ukrainian)

**Stack:** Unsloth, Transformers, PEFT, TRL, scikit-learn, sacrebleu

`lab4/`



### Lab 5 — Music Generation with LoRA Fine-Tuning (ACE-Step 1.5)

Fine-tuning ACE-Step 1.5 (a hybrid Language Model + Diffusion Transformer for music generation) with LoRA to generate Ukrainian folk music. The full pipeline covers dataset preparation, audio/lyrics/metadata annotation, two-pass tensor preprocessing, LoRA training of the DiT decoder, and generation.

**Dataset:** 10–20 Ukrainian folk songs with audio files, structured lyrics (with section tags), and JSON metadata (caption, BPM, key, time signature, language).

**Environment:** vast.ai with RTX 3090 (24 GB VRAM), using ACE-Step's built-in Gradio UI for preprocessing, training, and generation.

**Stack:** ACE-Step 1.5, LoRA, Gradio, vast.ai

`lab5/`


### Audio Project — Lyric-to-Melody Generator

A final course project building an end-to-end pipeline that takes raw lyrics as input and generates a complete song where the melody, tempo, genre, and instrumentation automatically match the emotional tone of the text.

**4-stage pipeline:**

1. **Emotion detection** — DistilRoBERTa fine-tuned on Ekman's 6 emotions + neutral (j-hartmann/emotion-english-distilroberta-base)
2. **Emotion-to-music mapping** — Rule-based translation of detected emotions into musical parameters (genre, tempo, instruments, vocal style)
3. **Music generation** — ACE-Step v1.5 with structured caption + lyrics input
4. **Interface** — Gradio UI with emotion visualization and audio playback

**Stack:** ACE-Step 1.5, DistilRoBERTa, Gradio

Individual labs may require GPU access (NVIDIA with CUDA or Apple MPS). Labs 3 and 5 were run on Kaggle T4 and vast.ai RTX 3090 respectively.
