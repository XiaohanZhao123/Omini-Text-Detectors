# Omini-Text

This project is part of the Omini-Detect project. It focuses on the text modality for AI-generated content detection.

## Roadmap

Dataset Creation
- [x] Implement some text detectors for benchmarking the difficulty of the dataset ✅ (4 methods implemented)
- [ ] Collect a small subset data for initial experiments
- [ ] Analyse the metrics for insights on attack methods, difficulty levels, modality, etc.

---
Detector Development

## Current Status: 4 Baseline Detectors Implemented

### Zero-Shot Detectors (Probability-based)
- [x] **Fast-DetectGPT** [[paper](https://arxiv.org/abs/2310.10830)] [[code](https://github.com/baoguangsheng/fast-detect-gpt)] - ICLR 2024
  - Conditional probability curvature detection
  - 340x speedup over original DetectGPT
  - Requires GPU, works with open-source models

- [x] **Glimpse** [[paper](https://arxiv.org/abs/2402.14809)] [[code](https://github.com/baoguangsheng/glimpse)] - ICLR 2025
  - Probability distribution estimation for proprietary models
  - Works with GPT-4, Claude, Gemini via API
  - Runs on CPU

### Supervised Detectors (Learning-based)
- [x] **e5-small LoRA** [[model](https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector)] - Microsoft Hackathon 2024
  - Fine-tuned e5-small with LoRA adaptation
  - **93.9% accuracy** on RAID benchmark (top performer)
  - **85.7% accuracy** with adversarial attacks
  - Robust across multiple LLMs (GPT-4: 99.3%, MPT: 94.0%, Mistral: 88.8%)

- [x] **Desklib AI Detector** [[model](https://huggingface.co/desklib/ai-text-detector-v1.01)]
  - Custom transformer with mean pooling
  - Simple inference interface
  - Threshold-tunable classification

## Next Steps

### Unified Interface Development 🎯
**Goal**: Provide standardized API across all 4 detection methods

**Features to implement**:
- [ ] Unified prediction interface with consistent input/output format
- [ ] Easy model switching (detector selection via parameter)
- [ ] Batch processing support for efficient inference
- [ ] Performance benchmarking framework for fair comparison
- [ ] Streamlined integration with simple import patterns
- [ ] Configuration management for model-specific parameters
- [ ] Documentation and usage examples

**Benefits**:
- Simplified detector comparison and evaluation
- Easy integration into downstream applications
- Consistent error handling and logging
- Reproducible benchmarking results

---

### Dataset Creation (Continued)
- [ ] Collect diverse subset data for comprehensive evaluation 