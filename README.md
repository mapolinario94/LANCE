# LANCE: Low-Rank Activation Compression for Efficient On-Device Continual Learning

This repository will host the official implementation of **LANCE**, introduced in our paper:  
**“LANCE: Low-Rank Activation Compression for Efficient On-Device Continual Learning”**

> 🚧 Code release in progress — we are currently cleaning and packaging experiments for public release.  
> Please ⭐ star or **Watch** the repository to stay updated.

---

## About LANCE
On-device learning requires efficient fine-tuning and continual adaptation under tight memory budgets.  
**LANCE** addresses this challenge by:
- Performing a **one-shot higher-order SVD** to obtain reusable low-rank activation subspaces.
- Eliminating costly per-iteration decompositions used in prior work.
- Enabling **continual learning** by allocating tasks to orthogonal subspaces, without large task-specific memory.

**Key results:**
- Up to **250× reduction** in activation storage with accuracy comparable to full backpropagation.  
- Strong performance on **continual learning benchmarks** (Split CIFAR-100, Split MiniImageNet, 5-Datasets).  
- Scalability demonstrated on datasets like CIFAR-10/100, Oxford-IIIT Pets, Flowers102, and CUB-200.

---

## Planned Release
- ✅ PyTorch implementation of LANCE
- ✅ Training & evaluation scripts for fine-tuning and continual learning
- ✅ Experiment configs and logging utilities

---

## Citation
If you find this work useful, please cite:
```bibtex
@article{Apolinario2025LANCE,
  title   = {LANCE: Low-Rank Activation Compression for Efficient On-Device Continual Learning},
  author  = {Apolinario, Marco and Roy, Kaushik},
  journal = {arXiv preprint arXiv:YYYY.NNNNN},
  year    = {2025}
}
```
