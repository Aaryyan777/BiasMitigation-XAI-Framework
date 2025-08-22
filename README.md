# XAI Framework for Bias Detection and Mitigation in Healthcare AI

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" alt="PyTorch Version"/>
  <img src="https://img.shields.io/badge/status-complete-brightgreen.svg" alt="Project Status"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
</p>

---

## 1. Introduction: The Problem of Trust in Medical AI

Artificial Intelligence holds immense promise for revolutionizing healthcare diagnostics. However, a critical barrier to its adoption is the "black box" problem and the risk of hidden biases. An AI model might achieve high accuracy in a lab setting, but what if it does so for the wrong reasons? What if it learns to associate diagnoses with demographic data like ethnicity or gender instead of relying purely on medical evidence? Such a model would be untrustworthy and potentially dangerous.

This project tackles this challenge head-on. It presents a complete, end-to-end framework to **detect, analyze, and actively mitigate** demographic bias in a sophisticated multimodal diagnostic AI. We don't just build a model; we build a system of accountability around it.

Our approach uses **Explainable AI (XAI)**, specifically the SHAP library, as a diagnostic tool for the model itself. By understanding *why* our model makes a decision, we can identify if it's "cheating" by using biased shortcuts. We then demonstrate how to train a new, fairer model and use the same XAI tools to **prove** that the new model is more robust and trustworthy.
