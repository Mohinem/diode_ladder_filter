Diode Ladder Low-Pass Filter (Minimal Virtual-Analog DSP Demo)

Overview
This project implements a minimal nonlinear ladder-style low-pass filter inspired by classic analog designs. The goal is not to recreate a full synthesizer or plugin, but to demonstrate correct digital signal-processing reasoning around nonlinear feedback filters in a compact, reproducible form.

The implementation focuses on clarity, stability, and measurable behavior rather than UI or musical polish.

What is implemented
• Four cascaded one-pole low-pass stages
• Resonance feedback from the last stage to the input
• Smooth nonlinear saturation using tanh as a diode-like proxy
• Parameter smoothing to avoid zipper noise
• Offline audio rendering and objective analysis plots

This is intentionally not a circuit-accurate zero-delay feedback (ZDF) ladder. Instead, it is a minimal, stable model suitable for rapid experimentation and evaluation.

Why this project
Nonlinear ladder filters are a core building block of subtractive synthesis and virtual-analog modeling. Implementing even a simplified version requires understanding:

• Discrete-time integrators
• Nonlinear saturation in feedback loops
• Stability constraints at high resonance
• Practical evaluation using impulse and noise excitation

This project demonstrates those concepts without unnecessary engineering overhead.

Generated outputs
Running the script produces:

Audio files
• output_impulse.wav — impulse-train response (audible and player-safe)
• output_noise.wav — filtered white noise
• output_saw_low.wav — saw wave at low cutoff (acid-style behavior)
• output_saw_high_res.wav — high-resonance response near instability

Plots (saved under plots/)
• impulse_response.png — time-domain response
• frequency_response.png — estimated magnitude response using noise excitation

All audio and plots are fully regenerable and are therefore excluded from version control.

Signal flow summary
Input signal → resonance feedback subtraction → nonlinear saturation →
four cascaded one-pole low-pass stages → output

The one-pole stages use an integrator form:

y[n] = y[n−1] + g · (x[n] − y[n−1])

with g derived from the cutoff frequency using an exponential mapping.

Evaluation approach
Because the filter is nonlinear, its response depends on signal level. Frequency response is therefore estimated using low-level white-noise excitation and spectral magnitude ratios rather than assuming linear time-invariant behavior.

Impulse-train excitation is used instead of a single-sample impulse to ensure audibility and robust playback across audio players.

Limitations
• No zero-delay feedback or implicit solving
• No oversampling (aliasing may occur at high cutoff or drive)
• Simplified tanh nonlinearity instead of a circuit-derived diode model
• Offline processing only (not real-time or plugin-ready)

These limitations are intentional to keep the project focused and achievable in a single day.

Future extensions
• Zero-delay feedback ladder using Newton iteration
• Oversampling and proper antialiasing
• Circuit-accurate diode pair modeling
• Parameter automation examples (e.g., cutoff sweeps)
• VST/AU implementation once DSP behavior is validated

How to run
Install dependencies from requirements.txt and execute ladder_filter.py.
All outputs will be generated automatically.

Research intent
This project is designed as a minimal, reproducible demonstration of nonlinear audio DSP concepts relevant to virtual-analog modeling, suitable as a starting point for more advanced ladder-filter research or plugin development.