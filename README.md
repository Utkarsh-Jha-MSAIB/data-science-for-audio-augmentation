# Data Science for Audio Augmentation

**What if visual patterns could be converted into structured, controllable music?**

This project explores a **data science approach to audio generation**, where images (such as skylines) are transformed into **interpretable musical signals** instead of relying purely on learned models.

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz0.png" width="1050"/>
</p>

---

## Key Insight

Music generation does not have to depend entirely on historical datasets.

By treating audio as a **structured, controllable system**, we can generate **novel, pattern-driven music** from entirely non-musical sources such as images, curves, and environmental patterns.

---

## What Makes This Different?

- **Not end-to-end black-box learning** → fully interpretable pipeline  
- **Explicit feature control** → pitch, loudness, timbre engineered independently  
- **Regression-driven loudness modeling** → pitch-aware dynamic behavior  
- **Multi-mode synthesis** → formula-based, ML-based, and audio-guided outputs  
- **Non-musical inputs** → enables truly novel sound generation  

👉 The focus is on **understanding and controlling sound**, not just generating it.

---

## Motivation

Modern music AI systems (Transformers, diffusion models, etc.) are powerful, but:

- rely heavily on existing datasets  
- often reproduce learned styles  
- raise **ethical concerns** around data usage and originality  

This project takes a different direction:

👉 Generate **fresh, original audio** using **structured transformations** rather than imitation.

We start by decomposing music into core components:

- **Pitch** → melody  
- **Loudness** → dynamics  
- **Timbre** → harmonics + noise  

Then rebuild audio by controlling these components directly.

---

## Core Idea

**Visual Pattern → Time-Series → Engineered Controls → Synthesized Audio**

---

## Pipeline Overview

### 1. Image → Curve Extraction
- Extract continuous structure from images (e.g., skyline edges)
- Convert pixel space into normalized signal

### 2. Curve → Time Mapping
- X-axis → time  
- Y-axis → displacement → musical driver  

### 3. Feature Engineering
- Pitch derived from curve displacement  
- Loudness derived from height / energy  
- smoothing, normalization, and scaling  

### 4. Pitch Structuring
- Remove interpolation artifacts  
- Enforce discrete note transitions  
- Convert continuous signals → musical note blocks  

### 5. Loudness Modeling (Regression)

Instead of directly using loudness, we model it as a function of pitch:

- Convert pitch → feature space (Hz or MIDI)  
- Split into bands (low / mid / high)  
- Fit simple linear regression per band  
- Predict loudness from generated pitch  

👉 This ensures **musically consistent dynamics**, rather than purely visual mapping.

---

### 6. Synthesis Modes

Three parallel generation strategies:

- **Formula-based** → deterministic, structured, highly interpretable  
- **ML-based** → learned timbre, smoother and more natural  
- **Audio-based** → preserves donor audio texture while adapting structure  

---

## Song Breakdown & Harmonics-Noise Deep Dive

### 1. Pitch, Loudness & Amplitude

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz1.png" width="1050"/>
</p>

- **Pitch** → melody derived from visual structure  
- **Loudness** → dynamics shaped by regression + height mapping  
- **Amplitude (multi-mode)** → synthesis energy across generation modes  

👉 **Why plot amplitude when loudness exists?**

- Loudness = perceptual / modeled signal  
- Amplitude = actual waveform energy used in synthesis  

This distinction helps compare how different generation modes behave even under similar loudness patterns.

---

### 2. Harmonics Breakdown

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz2.png" width="1050"/>
</p>

- Tracks dominant harmonic components over time  
- Shows how timbre evolves independently of pitch  

👉 **Key role of harmonics:**  
They define tone quality — brightness, richness, and instrument identity — making them critical for generating expressive and distinguishable sounds.

---

## Outputs

- Generated audio across 3 synthesis modes  
- Frame-level pitch & loudness control signals  
- Harmonic and amplitude visualizations  
- Fully reconstructable audio representations  

---

## Future Directions

- **Iterative refinement loop**  
  Use perceptual feedback to continuously improve generated outputs  

- **Diverse pattern mapping**  
  Extend beyond skylines to cyclic, geometric, and abstract structures  

- **Multi-instrument pipeline**  
  Generate richer compositions by combining multiple instrument layers  

---

## Summary

This project reframes music generation as a **data science problem**:

- features are interpretable  
- transformations are controllable  
- outputs are analyzable  

Rather than imitating existing music, it enables **creation of entirely new musical forms** driven by patterns in the world around us.
