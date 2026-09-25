"""
Whistle detector: listens to the mic and shouts a warning when someone whistles.

Dependencies:
    pip install numpy sounddevice

Run:
    python no_whistle.py          # calibrate 3 s, then listen
    python no_whistle.py --test   # synthetic self-check, no mic needed
"""
import queue
import sys
import time
import wave

import numpy as np

# ---- Settings -------------------------------------------------------------
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048            # ~46 ms per block
WHISTLE_MIN_HZ = 1000
WHISTLE_MAX_HZ = 4000
PEAK_HALF_WIDTH_BINS = 3     # narrow window around the peak
PEAK_RATIO_THRESHOLD = 0.5   # share of total energy inside that window
MIN_RMS = 0.01               # ignore near-silence (raise if background is loud)
CONSECUTIVE_FRAMES = 4       # 4 * 46 ms ~ 185 ms of sustained whistle
COOLDOWN_SEC = 4.0

SOUND_FILE = "audio_2026-09-26_00-54-02.wav"
VOLUME = 1.0                 # playback gain, 1.0 = original
# ---------------------------------------------------------------------------

_WINDOW = np.hanning(BLOCK_SIZE)
_FREQS = np.fft.rfftfreq(BLOCK_SIZE, 1 / SAMPLE_RATE)


def detect_whistle(block):
    """Return (is_whistle, peak_hz, peak_ratio) for one mono float block."""
    rms = float(np.sqrt(np.mean(block ** 2)))
    spec = np.abs(np.fft.rfft(block * _WINDOW)) ** 2
    spec[0] = 0  # drop DC
    peak = int(np.argmax(spec))
    peak_hz = float(_FREQS[peak])
    lo, hi = max(peak - PEAK_HALF_WIDTH_BINS, 0), peak + PEAK_HALF_WIDTH_BINS + 1
    ratio = float(spec[lo:hi].sum() / (spec.sum() + 1e-12))
    ok = (rms >= MIN_RMS and WHISTLE_MIN_HZ <= peak_hz <= WHISTLE_MAX_HZ
          and ratio >= PEAK_RATIO_THRESHOLD)
    return ok, peak_hz, ratio


def speak_warning():
    """Play the pre-recorded WAV file (blocking)."""
    import sounddevice as sd
    with wave.open(SOUND_FILE) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=f"int{8 * w.getsampwidth()}")
        data = data.reshape(-1, w.getnchannels()) / float(2 ** (8 * w.getsampwidth() - 1))
        rate = w.getframerate()
    sd.play(data * VOLUME, rate, device=sd.default.device[1])
    sd.wait()


def calibrate(q, seconds=3):
    print(f"Calibrating: stay quiet for {seconds}s...")
    end, levels = time.time() + seconds, []
    while time.time() < end:
        try:
            levels.append(float(np.sqrt(np.mean(q.get(timeout=1) ** 2))))
        except queue.Empty:
            pass
    avg = np.mean(levels) if levels else 0.0
    print(f"Background RMS: {avg:.5f} (MIN_RMS is {MIN_RMS}; set it ~3x above background)")


def main():
    import sounddevice as sd
    q = queue.Queue()
    stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1,
                            callback=lambda indata, *_: q.put(indata[:, 0].copy()))
    streak, last_trigger = 0, 0.0
    with stream:
        calibrate(q)
        print("Listening... Ctrl+C to stop.")
        while True:
            ok, hz, ratio = detect_whistle(q.get())
            streak = streak + 1 if ok else 0
            trigger = streak >= CONSECUTIVE_FRAMES and time.time() - last_trigger > COOLDOWN_SEC
            print(f"{time.strftime('%H:%M:%S')} peak={hz:7.1f} Hz ratio={ratio:.2f} "
                  f"whistle={ok} streak={streak} trigger={trigger}")
            if trigger:
                speak_warning()
                last_trigger, streak = time.time(), 0
                q.queue.clear()  # drop audio captured while we were talking


def selftest():
    t = np.arange(BLOCK_SIZE) / SAMPLE_RATE
    whistle = 0.3 * np.sin(2 * np.pi * 2500 * t)
    noise = np.random.default_rng(0).normal(0, 0.1, BLOCK_SIZE)
    low_tone = 0.3 * np.sin(2 * np.pi * 300 * t)
    assert detect_whistle(whistle)[0], "2.5 kHz tone should be a whistle"
    assert not detect_whistle(noise)[0], "white noise should not trigger"
    assert not detect_whistle(low_tone)[0], "300 Hz tone is out of band"
    print("selftest ok")


if __name__ == "__main__":
    if "--test" in sys.argv:
        selftest()
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nBye.")
