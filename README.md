# Data Science for Audio Augmentation

**Ever wondered how the New York skyline could sound? 🏙️**

What if patterns we see around us could be transformed into music?

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz0.png" width="1050"/>
</p>

## Contents

- [Motivation](#motivation)
- [Why Data Science (Not Just Deep Learning)?](#why-data-science-not-just-deep-learning)
- [Pipeline Overview](#pipeline-overview)
- [Song Breakdown & Harmonics-Noise Deep Dive](#song-breakdown--harmonics-noise-deep-dive)
- [Future Directions](#future-directions)

---

## Motivation

Music generation today is largely driven by models trained on existing data.  
While powerful, this raises two key challenges:

- **Limited novelty** → outputs often resemble training data  
- **Ethical concerns** → reliance on copyrighted datasets and stylistic imitation  

This project explores an alternative direction:

👉 generating **fresh, pattern-driven music** from non-musical sources like images.

To enable this, we treat music as a structured system:

- Pitch → melody  
- Loudness → dynamics  
- Timbre → harmonics + noise  

By controlling these explicitly, we move from imitation → **constructive generation**.

---

## Why Data Science, Not Just Deep Learning?

Instead of relying purely on learned mappings, this approach:

- decomposes audio into interpretable features  
- applies deterministic + statistical transformations  
- blends learned and engineered signals  

👉 The goal is not just to generate audio, but to **understand and control it**.

---

## Core Idea

**Visual pattern → time-series → engineered controls → synthesized audio**

---

## Pipeline Overview

### Step 1: Image → Curve Extraction
- Extract skyline / pattern as a continuous curve

### Step 2: Curve → Time Mapping
- X-axis → time  
- Y-axis → displacement signal  

### Step 3: Feature Engineering
- Pitch from curve displacement  
- Loudness from height profile  
- smoothing + normalization  

### Step 4: Pitch Structuring
- remove interpolation artifacts  
- enforce note-like transitions  
- convert continuous → discrete musical structure  

### Step 5: Loudness Modeling (Regression)
Instead of using loudness directly, we **learn its relationship with pitch**:

- Convert pitch → feature space (Hz / MIDI)
- Split into bands (low / mid / high)
- Fit simple linear regression per band
- Predict loudness from generated pitch

👉 This ensures loudness behaves **musically consistent with pitch**, not just visually driven.

### Step 6: Synthesis Modes
- **Formula-based** → rule-driven, structured but less expressive  
- **ML-based** → learned timbre, smoother and more natural  
- **Audio-based** → preserves donor audio texture while adapting structure  

---

## Song Breakdown & Harmonics-Noise Deep Dive

### 1. Pitch, Loudness & Amplitude

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz1.png" width="1050"/>
</p>

- **Pitch** → shows melody derived from visual structure  
- **Loudness** → shows expressive dynamics after regression + shaping  
- **Amplitude (multi-mode)** → shows energy used in synthesis across modes  

👉 Why amplitude if loudness exists?  
- Loudness = perceptual / modeled control  
- Amplitude = actual synthesis energy  

This helps compare how different generation modes behave even under similar loudness patterns.

---

### 2. Harmonics Breakdown

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz2.png" width="1050"/>
</p>

- Tracks contribution of dominant harmonics over time  
- Shows how timbre evolves even when pitch remains stable  

👉 Key insight:  
Harmonics define **tone quality** — brightness, richness, and instrument identity — making them critical for generating distinguishable and expressive sounds.

---

## Future Directions

- **Continuous Improvement**  
  Iterative refinement based on perceptual feedback enables progressive enhancement of generated audio.

- **Diverse Pattern Mapping**  
  Extending beyond skylines to cyclic, geometric, or abstract patterns could unlock rhythm and multi-dimensional musical structures.

- **Multi-Instrument Generation**  
  Expanding the pipeline to handle multiple instruments simultaneously can lead to richer compositions and more complex sound design.

---

## Summary

This work demonstrates that music generation can be approached as a **structured data science problem**, where:

- features are interpretable  
- transformations are controllable  
- outputs are analyzable  

Rather than mimicking existing music, the system enables **creation of entirely new musical forms driven by patterns in the world around us**.
