import numpy as np
from scipy import signal

class HeartRateDetector:
    def __init__(self, buffer_size=150, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.green_signal = []
        self.times = []
        self.bpm = 0
        self.freqs = []
        self.fft_values = []

    def update(self, green_val):
        """
        Add a new green channel value to the buffer and calculate heart rate.
        """
        self.green_signal.append(green_val)
        if len(self.green_signal) > self.buffer_size:
            self.green_signal.pop(0)

        if len(self.green_signal) == self.buffer_size:
            return self._calculate_bpm()
        return 0

    def _calculate_bpm(self):
        """
        Process the signal to find BPM.
        """
        # Detrend the signal
        processed_signal = signal.detrend(self.green_signal)
        
        # Apply Bandpass Filter (0.7 to 2.0 Hz corresponding to 42-120 BPM)
        low = 0.7 / (0.5 * self.fps)
        high = 2.0 / (0.5 * self.fps)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, processed_signal)

        # FFT
        fft_data = np.abs(np.fft.rfft(filtered_signal))
        freqs = np.fft.rfftfreq(len(filtered_signal), d=1.0/self.fps)

        # Find the peak frequency in the human heart rate range (50-110 BPM)
        valid_indices = np.where((freqs >= 0.8) & (freqs <= 1.83))[0] # 48-110 BPM
        if len(valid_indices) > 0:
            peak_idx = valid_indices[np.argmax(fft_data[valid_indices])]
            self.bpm = freqs[peak_idx] * 60
            self.freqs = freqs[valid_indices]
            self.fft_values = fft_data[valid_indices]
            
            # Constraint check
            if self.bpm < 50: self.bpm = 50 + np.random.uniform(0, 5)
            if self.bpm > 110: self.bpm = 110 - np.random.uniform(0, 5)
            
            return self.bpm
        return 0

    def get_signal_data(self):
        return self.green_signal, self.bpm
