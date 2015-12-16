import numpy as np
import librosa
import scipy
import scipy.signal
from math import sqrt
from math import log
import scipy.signal.spectral


def compute_variance(signal, mean):
    N = len(signal)
    return 1.0 / (N-1) * np.sum(signal-mean)


def normalize_signal(signal, mean, variance):
    for index in xrange(len(signal)):
        signal[index] = (signal[index] - mean)/variance
    return signal


def pre_emphasis(signal, alpha=0.9):
    new_sig = signal[1:]-alpha*signal[:-1]
    return np.append(signal[0], new_sig)


def window_signal(signal, window_len=25):
    w = np.hamming(window_len)
    return np.convolve(w/w.sum(), signal, mode='same')


def zero_crossing_rate(signal):
    N = len(signal)
    zcr = 0
    for i in xrange(1, N):
        zcr += np.abs(np.sign(signal[i]) - np.sign(signal[i-1]))
    zcr /= (2.0 * N)
    return zcr


def temporal_centroid(signal):
    # returns centoid in ticks
    tc = np.sum(map(lambda (x, y): x*(y*y), enumerate(signal)))/     \
        np.sum(map(lambda (x, y): (y*y), enumerate(signal)))
    return tc


def root_mean_square(signal):
    return np.sqrt(short_time_energy(signal))


def short_time_energy(signal):
    N = len(signal)
    s = 0.0
    for i in xrange(N):
        s += abs(signal[i]**2)
    return s/N


def autocorellation(signal, sample_rate):
    #autocorellation_value = max(librosa.autocorrelate(signal))
    signal_length = len(signal)
    power_spectrum = np.abs(scipy.fft(signal, n=2 * signal_length + 1 ))**2
    auto_corr = max(np.real(scipy.ifft(power_spectrum))[:])
    return auto_corr, auto_corr/2


def energy_entropy(signal, num_of_frames = 22050 / 50):
    subframes = np.array_split(signal, num_of_frames)
    frame_energy = short_time_energy(signal)
    e = map(lambda (x, y): short_time_energy(y)/frame_energy,
                          enumerate(subframes))
    entropy = 0
    for frame_index in xrange(num_of_frames):
        if e[frame_index] != 0:
            entropy -= e[frame_index] * log(e[frame_index],2)

    return entropy


def spectral_spread(centroid, signal = None, fft_spectrum = None):
    if fft_spectrum is None and signal is not None:
        fft_spectrum = scipy.real(scipy.fft(signal))
    sum_abs_squared = sum(map(lambda (y,x): abs(x)**2, enumerate(fft_spectrum)))
    ss = sqrt((centroid ** 2 * sum_abs_squared) / sum_abs_squared)
    return ss


def spectral_flux(signal = None, fft_spectrum  = None):
    if fft_spectrum is None and signal is not None:
        fft_spectrum = scipy.real(scipy.fft(signal))
    flux_values = []
    flux = 0
    for element in fft_spectrum:
        flux += abs(element)

    flux_values.append(flux)

    for element in xrange(1, len(fft_spectrum)):
        prev = fft_spectrum[element-1]
        current = fft_spectrum[element]
        flux = abs(abs(current) - abs(prev))
        flux_values.append(flux)
    return np.array(flux_values)


def spectral_entropy(signal = None, fft_signal = None, num_of_frames =22050 / 50):
    if fft_signal is not None and signal is not None:
        fft_signal = scipy.real(scipy.fft(signal))
    subframes = np.array_split(fft_signal, num_of_frames)
    frame_energy = short_time_energy(fft_signal)
    e = map(lambda (x, y): short_time_energy(y)/frame_energy,
            enumerate(subframes))
    entropy = 0
    for frame_index in xrange(num_of_frames):
        if e[frame_index] != 0:
            entropy -= e[frame_index] * log(e[frame_index],2)

    return entropy


def compute_mfcc(signal, sample_rate, number_expected=13, num_of_triangular_features=23):
    return librosa.feature.mfcc(signal,
                                sample_rate,
                                n_mfcc=num_of_triangular_features)[:number_expected]


def spectral_roloff(signal, sample_rate):
    return librosa.feature.spectral_rolloff(signal, sample_rate,)


def spectral_centroid(signal, sample_rate):
    return librosa.feature.spectral_centroid(signal,sample_rate)


class Sampler(object):
    def __init__(self, source, duration = None, sample_rate = None, offset = 0.0):
        if isinstance(source, basestring):
            self.signal, self.sample_rate = librosa.load(source, duration=duration, offset=offset)
            if duration is None:
                self.duration = librosa.get_duration(self.signal, sr=self.sample_rate)
            else:
                self.duration = duration
        else:
            self.signal = source
            self.sample_rate = sample_rate
            self.duration = librosa.get_duration(self.signal, sr=sample_rate)
        self.pre_process()

    def split(self, part_len):
        parts_count = self.duration // part_len
        return np.array_split(self.signal_emphased, parts_count)

    def pre_process(self):
        #plt.plot(self.signal,'r',)
        #plt.show()

        self.normalized_signal = librosa.util.normalize(self.signal)
        self.signal_emphased = pre_emphasis(self.normalized_signal)

    def compute_features(self):
        self.signal_hammed = window_signal(self.signal_emphased)
        self.fft = scipy.real(scipy.fft(self.signal_hammed))
        self.spectral_flux = spectral_flux(fft_spectrum=self.fft).mean()

        self.spectral_cenroid_mean = spectral_centroid\
            (self.signal_hammed, self.sample_rate).mean()

        self.spectral_spread = spectral_spread(centroid=self.spectral_cenroid_mean, fft_spectrum=self.fft)
        self.spectral_roloff_mean = spectral_roloff\
            (self.signal_hammed, self.sample_rate).mean()
        self.spectral_entropy = spectral_entropy(fft_signal=self.fft)

        self.energy_entropy = energy_entropy(self.signal_hammed)
        self.zero_crossing_rate = zero_crossing_rate(self.signal_hammed)
        self.temporal_centroid = temporal_centroid(self.signal_hammed)
        self.short_time_energy = short_time_energy(self.signal)
        self.root_mean_square = root_mean_square(self.signal)
        self.fundamental_period, self.autocorellation = autocorellation(self.signal_hammed, self.sample_rate)
        self.mfcc = compute_mfcc(self.signal_hammed, self.sample_rate)

    def extract_features(self):
        vector = list()
        vector += [self.zero_crossing_rate]

        vector += [self.temporal_centroid]
        vector += [self.energy_entropy]
        vector += [self.root_mean_square]
        vector += [self.fundamental_period]
        vector += [self.autocorellation]
        vector += [self.spectral_roloff_mean]
        vector += [self.spectral_spread]
        vector += [self.spectral_flux]
        vector += [self.spectral_entropy]
        vector += [self.spectral_cenroid_mean]
        mean = self.mfcc.mean(axis = 1)
        vector = np.append(np.array(vector), mean)
        return vector



'''
another type of multiprocessing

global song

def convert(path, duration=20.0, half_part_length = 0.1, offset = 0):
    whole_song = Sampler(path, duration=duration,offset = offset)
    global song
    song = whole_song
    parts = whole_song.split(half_part_length)
    part_arr = [np.append(parts[i-1], parts[i]) for i in xrange(1, len(parts))]
    #pl = Pool(4)
    samples = map(take_feature, part_arr)
    return np.array(samples)
# end of process_converting
'''

if __name__ == "__main__":
    pass
    #path = "/media/files/musicsamples/genres/pop/pop.00002.au"
    #t = time.time()
    #df = pandas.DataFrame(convert_with_processes(path))
    #print time.time() - t
    #df.to_csv("/media/files/file.csv")