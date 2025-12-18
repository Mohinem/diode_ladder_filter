"""
Minimal nonlinear ladder low-pass filter demo:
- 4 cascaded one-pole filters
- resonance feedback
- tanh nonlinearity as diode-like saturation proxy
- generates audio outputs + plots

Run:
  python ladder_filter.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import get_window


# -----------------------------
# Utility: signals
# -----------------------------
def make_impulse(n: int, amp: float = 1.0) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    if n > 0:
        x[0] = amp
    return x


def make_impulse_train(n: int, fs: int, amp: float = 0.9, period_s: float = 0.25, pre_silence_s: float = 0.05) -> np.ndarray:
    """
    Creates an impulse train with a short pre-roll silence so players don't skip sample-0 transients.
    """
    x = np.zeros(n, dtype=np.float64)
    pre = int(pre_silence_s * fs)
    step = max(1, int(period_s * fs))

    start = min(pre, n - 1) if n > 0 else 0
    x[start::step] = amp
    return x


def make_white_noise(n: int, amp: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(n).astype(np.float64)


def make_saw(n: int, fs: int, freq: float = 110.0, amp: float = 0.3) -> np.ndarray:
    # Simple band-unlimited saw for demo (fine for 1-day MVP; aliasing may occur)
    t = np.arange(n, dtype=np.float64) / fs
    phase = (freq * t) % 1.0
    saw = 2.0 * phase - 1.0
    return amp * saw


def dc_block(x: np.ndarray, r: float = 0.995) -> np.ndarray:
    # Simple DC blocker: y[n] = x[n] - x[n-1] + r*y[n-1]
    y = np.zeros_like(x)
    x1 = 0.0
    y1 = 0.0
    for i in range(len(x)):
        y0 = x[i] - x1 + r * y1
        y[i] = y0
        x1 = x[i]
        y1 = y0
    return y


# -----------------------------
# Ladder Filter
# -----------------------------
@dataclass
class LadderParams:
    fs: int = 48000
    cutoff_hz: float = 800.0     # 20..(fs/2)
    resonance: float = 0.2       # 0..~1 (we'll clamp)
    drive: float = 1.0           # input drive into nonlinearity
    nonlinear_amount: float = 1.0  # scales tanh input
    cutoff_smooth_ms: float = 10.0
    res_smooth_ms: float = 10.0


class OnePole:
    """
    One-pole lowpass in integrator form:
      y[n] = y[n-1] + g * (x[n] - y[n-1])
    where g in (0,1).
    """
    def __init__(self):
        self.y = 0.0

    def reset(self):
        self.y = 0.0

    def process(self, x: float, g: float) -> float:
        self.y = self.y + g * (x - self.y)
        return self.y


class SmoothParam:
    """Exponential smoothing for parameters to avoid zipper noise."""
    def __init__(self, fs: int, smooth_ms: float, initial: float):
        self.fs = fs
        self.set_time_ms(smooth_ms)
        self.value = float(initial)

    def set_time_ms(self, smooth_ms: float):
        smooth_s = max(1e-6, smooth_ms / 1000.0)
        # alpha close to 0 -> heavy smoothing; close to 1 -> fast
        self.alpha = 1.0 - np.exp(-1.0 / (self.fs * smooth_s))

    def update(self, target: float) -> float:
        self.value = self.value + self.alpha * (float(target) - self.value)
        return self.value


class NonlinearLadderLPF:
    """
    Minimal 4-stage ladder-ish filter with:
      x_in = tanh(drive*(x - k*y4))
      stage i: yi = onepole( tanh(nl * something), g )
    This is *not* a circuit-accurate ZDF ladder, but it captures:
      - resonance feedback
      - nonlinear saturation
      - 4-pole lowpass character
    """
    def __init__(self, params: LadderParams):
        self.p = params
        self.stages = [OnePole(), OnePole(), OnePole(), OnePole()]

        self.cut_smoother = SmoothParam(params.fs, params.cutoff_smooth_ms, params.cutoff_hz)
        self.res_smoother = SmoothParam(params.fs, params.res_smooth_ms, params.resonance)

    def reset(self):
        for s in self.stages:
            s.reset()
        # Keep smoothers (they'll update anyway), but resetting stages is enough for this demo.

    @staticmethod
    def _tanh_sat(x: float) -> float:
        # tanh saturation; stable and smooth
        return float(np.tanh(x))

    def _g_from_cutoff(self, cutoff_hz: float) -> float:
        """
        Convert cutoff to one-pole coefficient g.
        A simple mapping:
          g = 1 - exp(-2*pi*fc/fs)
        Stable for fc>=0.
        """
        fc = np.clip(cutoff_hz, 5.0, 0.49 * self.p.fs)
        g = 1.0 - np.exp(-2.0 * np.pi * fc / self.p.fs)
        return float(np.clip(g, 0.0, 1.0))

    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=np.float64)

        for n in range(len(x)):
            cutoff = self.cut_smoother.update(self.p.cutoff_hz)
            res = self.res_smoother.update(self.p.resonance)

            # Clamp resonance to reduce blowups in this simplified model.
            res = float(np.clip(res, 0.0, 0.98))

            g = self._g_from_cutoff(cutoff)

            # feedback from last stage
            y4 = self.stages[3].y
            xin = x[n] - res * y4

            # input drive + nonlinearity
            xin = self._tanh_sat(self.p.nonlinear_amount * self.p.drive * xin)

            # cascade: add mild nonlinearity per stage too
            s1 = self.stages[0].process(self._tanh_sat(self.p.nonlinear_amount * xin), g)
            s2 = self.stages[1].process(self._tanh_sat(self.p.nonlinear_amount * s1), g)
            s3 = self.stages[2].process(self._tanh_sat(self.p.nonlinear_amount * s2), g)
            s4 = self.stages[3].process(self._tanh_sat(self.p.nonlinear_amount * s3), g)

            y[n] = s4

        return y


# -----------------------------
# Analysis: frequency response
# -----------------------------
def estimate_freq_response(
    filt: NonlinearLadderLPF,
    fs: int,
    cutoff_hz: float,
    resonance: float,
    n_fft: int = 16384,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate magnitude response using a low-level method:
    excite with white noise, compute output/input spectra ratio.

    Because the filter is nonlinear, "frequency response" depends on level.
    We keep level small and use drive to manage saturation.
    """
    rng = np.random.default_rng(seed)
    n = n_fft
    x = 0.08 * rng.standard_normal(n).astype(np.float64)

    # configure params
    filt.reset()
    filt.p.cutoff_hz = cutoff_hz
    filt.p.resonance = resonance

    # warm up with a short block to settle
    warm = 2048
    _ = filt.process_block(0.08 * rng.standard_normal(warm).astype(np.float64))

    y = filt.process_block(x)

    # windowing for FFT stability
    w = get_window("hann", n, fftbins=True).astype(np.float64)
    X = np.fft.rfft(x * w)
    Y = np.fft.rfft(y * w)

    eps = 1e-12
    H = (np.abs(Y) + eps) / (np.abs(X) + eps)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag_db = 20.0 * np.log10(H)
    return freqs, mag_db


# -----------------------------
# Main: generate outputs
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_wav(path: str, x: np.ndarray, fs: int) -> None:
    # Avoid clipping surprises; keep small headroom
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.999:
        x = 0.95 * x / peak
    sf.write(path, x.astype(np.float32), fs)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(root, "plots")
    ensure_dir(plots_dir)

    fs = 48000
    dur_s = 2.5
    n = int(fs * dur_s)

    # Signals
    # IMPORTANT: Use an impulse train + preroll so you can actually hear it and players don't skip the transient.
    x_imp = make_impulse_train(n, fs, amp=0.9, period_s=0.25, pre_silence_s=0.05)
    x_noise = make_white_noise(n, amp=0.18, seed=1)
    x_saw = make_saw(n, fs, freq=110.0, amp=0.25)

    # NEW: write input signals for A/B comparison
    write_wav(os.path.join(root, "input_impulse.wav"), x_imp, fs)
    write_wav(os.path.join(root, "input_noise.wav"), x_noise, fs)
    write_wav(os.path.join(root, "input_saw.wav"), x_saw, fs)

    # Filter instance
    p = LadderParams(
        fs=fs,
        cutoff_hz=800.0,
        resonance=0.2,
        drive=1.2,
        nonlinear_amount=1.2,
        cutoff_smooth_ms=10.0,
        res_smooth_ms=10.0,
    )
    filt = NonlinearLadderLPF(p)

    # --- Render 1: impulse response at moderate resonance
    filt.reset()
    filt.p.cutoff_hz = 800.0
    filt.p.resonance = 0.35

    # Warm-up to settle internal states/smoothers so the first click isn't “weird”
    _ = filt.process_block(np.zeros(2048, dtype=np.float64))

    y_imp = filt.process_block(x_imp)

    # DO NOT DC-block the impulse train: it can reduce the perceived transient.
    # y_imp = dc_block(y_imp)

    write_wav(os.path.join(root, "output_impulse.wav"), y_imp, fs)
    print("impulse peak:", float(np.max(np.abs(y_imp))))

    # --- Render 2: noise
    filt.reset()
    filt.p.cutoff_hz = 1200.0
    filt.p.resonance = 0.15
    y_noise = filt.process_block(x_noise)
    y_noise = dc_block(y_noise)
    write_wav(os.path.join(root, "output_noise.wav"), y_noise, fs)

    # --- Render 3: saw at low cutoff (acid-ish)
    filt.reset()
    filt.p.cutoff_hz = 300.0
    filt.p.resonance = 0.35
    y_saw_low = filt.process_block(x_saw)
    y_saw_low = dc_block(y_saw_low)
    write_wav(os.path.join(root, "output_saw_low.wav"), y_saw_low, fs)

    # --- Render 4: saw at higher resonance (edge-of-oscillation vibe)
    filt.reset()
    filt.p.cutoff_hz = 700.0
    filt.p.resonance = 0.85
    # slightly lower drive to keep it controlled
    filt.p.drive = 0.9
    y_saw_hi_res = filt.process_block(x_saw)
    y_saw_hi_res = dc_block(y_saw_hi_res)
    write_wav(os.path.join(root, "output_saw_high_res.wav"), y_saw_hi_res, fs)

    # -----------------------------
    # Plots: impulse response
    # -----------------------------
    t = np.arange(n) / fs
    show_n = min(n, int(0.80 * fs))  # show 0.8s so you see multiple impulses

    plt.figure()
    plt.plot(t[:show_n] * 1000.0, y_imp[:show_n])
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("Impulse train response (cutoff=800 Hz, resonance=0.35)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "impulse_response.png"), dpi=160)
    plt.close()

    # -----------------------------
    # Plots: estimated frequency response
    # -----------------------------
    resp_specs = [
        ("low cutoff / low res", 300.0, 0.1),
        ("mid cutoff / mid res", 800.0, 0.35),
        ("high cutoff / high res", 2000.0, 0.85),
    ]

    plt.figure()
    for label, fc, r in resp_specs:
        # restore drive (important for comparability)
        filt.p.drive = 1.0
        freqs, mag_db = estimate_freq_response(filt, fs, fc, r, n_fft=16384, seed=10)
        # show up to 12 kHz for readability
        max_f = 12000.0
        idx = freqs <= max_f
        plt.plot(freqs[idx], mag_db[idx], label=f"{label} (fc={fc:.0f}Hz, r={r:.2f})")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Estimated magnitude response (noise excitation; nonlinear)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "frequency_response.png"), dpi=160)
    plt.close()

    print("Done.")
    print("Audio written (inputs):")
    print(" - input_impulse.wav")
    print(" - input_noise.wav")
    print(" - input_saw.wav")
    print("Audio written (outputs):")
    print(" - output_impulse.wav (impulse train response)")
    print(" - output_noise.wav")
    print(" - output_saw_low.wav")
    print(" - output_saw_high_res.wav")
    print("Plots written to plots/:")
    print(" - impulse_response.png")
    print(" - frequency_response.png")


if __name__ == "__main__":
    main()
