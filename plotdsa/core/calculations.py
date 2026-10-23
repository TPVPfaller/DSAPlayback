import numpy as np
from scipy.signal import welch, firwin, lfilter
from scipy.signal.windows import dpss
from .. import config


class DSACalculator:
    """Computes PSD columns for DSA using Welch's method (Defender-safe)."""

    def __init__(self, window_sec):
        self.window_sec = window_sec
        self._precompute_filter()

    def _precompute_filter(self):
        fs = config.SAMPLE_RATE_HZ

        self.bp_b = firwin(
            numtaps=101,
            cutoff=[config.LOWEST_FREQ_HZ, config.MAX_FREQ_HZ_BOUNDS[1]],
            fs=fs,
            pass_zero=False,
            window='hamming'
        )
        self.bp_a = [1.0]

    def _apply_filters(self, data):
        filtered = lfilter(self.bp_b, self.bp_a, data)
        return filtered

    def filter_window(self, eeg_values):
        return self._apply_filters(np.asarray(eeg_values, dtype=np.float32))

    def compute_psd_from_filtered(self, filtered, method='multitaper'):
        if method == 'multitaper':
            psd = self.multitaper_method(filtered)
        else:
            _, psd = welch(
                filtered,
                fs=config.SAMPLE_RATE_HZ,
                nperseg=config.N_PER_SEGMENT,
                noverlap=None, # If None overlap will be 50%
                return_onesided=True
            )

        if psd is None:
            print(f"DSA gap cause: PSD method '{method}' returned no data")
            return np.full(len(config.FREQ_BINS), np.nan, np.float32)

        psd = psd[config.FREQ_MASK]

        if not np.all(psd):
            zero_bins = int(np.count_nonzero(psd == 0))
            nan_bins = int(np.count_nonzero(np.isnan(psd)))
            inf_bins = int(np.count_nonzero(np.isinf(psd)))
            print(
                "DSA gap cause: PSD column contains invalid values "
                f"(zeros={zero_bins}, nan={nan_bins}, inf={inf_bins})"
            )
            return np.full(len(psd), np.nan, np.float32)

        return psd

    def multitaper_method(self, eeg_values):
        """
        Multitaper PSD estimation.
        """
        TW = 1
        K = 3  # number of tapers
        nw = config.N_PER_SEGMENT # number of samples per window (800 for 400hz)
        N = len(eeg_values) // nw # number of windows

        if N == 0:
            print(
                "DSA gap cause: multitaper window too short for PSD "
                f"(samples={len(eeg_values)}, n_per_segment={nw})"
            )
            return None

        eeg_values = eeg_values[:(N * nw)] # cut off extra samples
        eeg_values = np.reshape(eeg_values, (N, nw)) # reshape to (N, nw)

        fs = config.SAMPLE_RATE_HZ

        tapers, _ = dpss(nw, NW=TW, Kmax=K, return_ratios=True) # (K, nw)
        spect = np.zeros((N, nw//2+1)) # N//2+1 because fft output is Hermitian-symmetric

        for i in range(N):
            # tapers is (K, nw), eeg_values[i] is (nw,)
            window_data = eeg_values[i]
            tapered_data = tapers * window_data # Broadcasting (K, nw) * (nw,) -> (K, nw)
            
            fourier = np.fft.rfft(tapered_data, n=nw, axis=1) # shape: (K, nw//2+1)

            # Power per taper
            power = (np.abs(fourier) ** 2) / fs

            # For rfft, we need to double the power for all bins except DC and Nyquist if we want one-sided PSD
            power[:, 1:-1] *= 2
            
            # Average across tapers
            spect[i] = np.mean(power, axis=0)

        # Average across windows
        return np.mean(spect, axis=0)


    def compute_psd_column(self, eeg_values, method='multitaper'):
        min_samples = int(self.window_sec * config.SAMPLE_RATE_HZ)
        if len(eeg_values) < min_samples:
            return None, None

        filtered = self.filter_window(eeg_values)
        return self.compute_psd_from_filtered(filtered, method=method)

    def update_config(self, window_sec):
        self.window_sec = window_sec
