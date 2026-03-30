# Data Science for Audio Augmentation

**Can data science help us understand what the New York skyline might sound like? 🏙️**

What if patterns we see around us could be transformed into structured music?

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz0.png" width="1050"/>
</p>

## Contents

- [Motivation](#motivation)
- [Why Data Science Beyond Deep Learning?](#why-data-science-beyond-deep-learning?)
- [Pipeline Overview](#pipeline-overview)
- [Song Breakdown & Harmonics-Noise Deep Dive](#song-breakdown--harmonics-noise-deep-dive)
- [Future Directions](#future-directions)

---

## Motivation

AI music generation today is largely driven by models trained on existing data. 

While powerful, this raises two key challenges:

- **Limited novelty** → outputs often resemble training data  
- **Ethical concerns** → reliance on copyrighted datasets and stylistic imitation  

This project explores an alternative direction:

**Data Science** has true potential to create sounds never heard before. To do that, we must first understand how music itself can be represented: 
- Music is fundamentally a time-series signal
- It can be decomposed into interpretable components:
  - **Pitch** (melody)
  - **Loudness** (dynamics)
  - **Timbre** (harmonics + noise) .........and more

By isolating and controlling these components, we can unlock new possibilities for creative audio generation.

**But what if we go one step further?**

Instead of learning only from existing music, we **reverse-engineer** patterns from the world around us: 
- Skylines
- Ocean waves
- Natural curves
- Abstract visual patterns

…and convert them into musical structures.

---

## Why Data Science Beyond Deep Learning?

Modern AI (Transformers, RNNs, diffusion models) has made: 

- Text → Music
- Style transfer
- Long-range coherence possible

However:
- These systems are still heavily dependent on historical data 
- True novelty is limited by training distributions 
- Music remains a subjective domain (both strength & weakness)

<br/>

💡 **This is where Data Science comes in** 

Instead of treating music as a black box, we: 

- Decompose signals into interpretable features  
- Engineer transformations explicitly  
- Control each component independently  

This is analogous to real-world DS systems:

| Domain | Components |
|--------|--------| 
| Marketing | price, promotion, trade |
| Finance | risk, return, volatility | 
| **Music (our goal)** | **pitch, loudness, timbre** |

👉 We model the system, not just the output 

<br/>


**Core Idea**

**Convert visual structure → time-series → controllable audio features → synthesized sound**

---

## Pipeline Overview

**Step 1: Image → Curve Extraction**
- Extract skyline / pattern as a continuous curve

**Step 2: Curve → Time Mapping**
- X-axis → time  
- Y-axis → displacement signal  

**Step 3: Feature Engineering**
- Pitch from curve displacement  
- Loudness from height profile  
- Smoothing + normalization    

**Step 4: Pitch Structuring**
- Remove interpolation artifacts  
- Enforce note-like transitions  
- Convert continuous → discrete musical structure  

**Step 5: Loudness Modeling (Regression)**
Instead of using loudness directly, we **learn its relationship with pitch**:

- Convert pitch → feature space (Hz / MIDI)
- Split into bands (low / mid / high)
- Fit regression per band
- Predict loudness from generated pitch

**Step 6: Synthesis Modes**
- **Formula-based** → rule-driven, structured but less expressive  
- **ML-based** → learned timbre, smoother and more natural (model checkpoints from DL pipeline) 
- **Audio-based** → preserves donor audio texture while adapting structure  

---

## Song Breakdown & Harmonics-Noise Deep Dive

**1. Pitch, Loudness & Amplitude**

- **Pitch** → shows melody derived from visual structure  
- **Loudness** → shows expressive dynamics after regression + shaping  
- **Amplitude (multi-mode)** → shows energy used in synthesis across modes 

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz1.png" width="1050"/>
</p> 

👉 Why amplitude if loudness exists?  
- Loudness = perceptual / modeled control  
- Amplitude = actual synthesis energy  

This helps compare how different generation modes behave even under similar loudness conditions.

---

**2. Harmonics Breakdown**

- Tracks contribution of dominant harmonics over time  
- Shows how timbre evolves even when pitch remains stable 

<p align="center">
  <img src="https://github.com/Utkarsh-Jha-MSAIB/data-science-for-audio-augmentation/blob/main/output_sample/viz2.png" width="1050"/>
</p>
 

👉 Key insight:  
Harmonics define **tone quality**: brightness, richness, and instrument identity, making them critical for generating distinct and expressive sounds.

---

## Future Directions

- **Scope for Continuous Improvement**
  Music is inherently subjective, and this pipeline leverages that flexibility. By learning patterns from high-quality audio and adapting them to pattern-driven generation, the system enables iterative refinement. This allows for multiple cycles of listening, tuning, and enhancement, ensuring there is always room for improvement.

- **Ability to Utilize Diverse Patterns**
  While skyline structures provide a strong starting point for melodic instruments such as guitar and piano, other patterns can unlock new possibilities. For example, spherical or cyclic patterns could be mapped to rhythmic structures like drums. This opens up exciting Edge AI use cases where composers and users can discover new tunes simply by interpreting patterns they observe in the world around them.

- **Multi-Instrumentation Pipeline**
  By isolating and visualizing key components such as pitch, loudness, and timbre, the framework can be extended to support multiple instruments simultaneously. This enables the generation of richer, multi-instrument compositions, significantly expanding the scope for innovation and creation of fresh, original music.

- **Modeling Missing Musical Components**
  While the current pipeline focuses on pitch, loudness, and timbre, future work can incorporate additional musical components such as rhythm, articulation, and higher-level structure. Explicitly modeling these elements will enable the generation of more complete and coherent songs while preserving the interpretability and control of the system.
  

Overall, this work reinforces the idea that expressive music generation benefits from a combination of structured data science principles, modular design, and perceptual evaluation. These foundations position the system as a flexible and scalable framework for future research in novel, high-quality audio generation.
