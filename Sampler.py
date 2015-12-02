import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from math import sqrt
import aubio
import audiodev
import pymir.SpectralFlux as SpectralFlux
from math import log

def compute_variance(signal, mean):
    N = signal.shape[0]
    return 1.0 / (N-1) * sum(signal-mean)


def normalize_signal(signal, mean, variance):
    for index in xrange(signal.shape[0]):
        signal[index] = (signal[index] - mean)/variance
    return signal


def pre_emphasis(signal, alpha= 0.9):
    new_sig = signal[1:]-alpha*signal[:-1]
    return np.append(signal[0], new_sig)


def window_signal(signal, window_len=25):
    w = np.hamming(window_len)

    return np.convolve(w/w.sum(), signal, mode='same')


def compute_mfcc(signal, sample_rate, number_expected=13, num_of_triangular_features=23):
    return librosa.feature.mfcc(signal,
                                sample_rate,
                                n_mfcc=num_of_triangular_features)\
                                            [:number_expected]


def zero_crossing_rate(signal, frame_length=2048):
    return librosa.feature.zero_crossing_rate(signal, frame_length=frame_length).mean()


def temporal_cenroid(signal):
    # returns centoid in ticks
    tc = sum(map(lambda (x, y): x*(y*y), enumerate(signal)))/     \
        sum(map(lambda (x, y): (y*y), enumerate(signal)))
    return tc


def root_mean_square(signal):
    return sqrt(short_time_energy(signal))


def short_time_energy(signal):
    N = signal.shape[0]
    s = 0.0
    for i in xrange(N):
        s += signal[i]**2
    return s/N


def autocorellation(signal, sample_rate):
    autocorellation_value = max(librosa.autocorrelate(signal))
    return autocorellation_value, autocorellation_value/2


def energy_entropy(signal, num_of_frames = 22050/50):

    subframes = np.split(signal,num_of_frames)
    frame_energy = short_time_energy(signal)
    e = map(lambda (x,y): short_time_energy(y)/frame_energy,
                          enumerate(subframes))
    entropy = 0
    for frame_index in xrange(num_of_frames):
        entropy -= e[frame_index] * log(e[frame_index],2)

    return entropy


def spectal_roloff(signal, sample_rate):
    return librosa.feature.spectral_rolloff(signal, sample_rate)


def spectral_spread(spectral_centroid, signal = None, fft_spectrum = None):

    if fft_spectrum is None and signal is not None:
        fft_spectrum = scipy.real(scipy.fft(signal))
    sum_abs_squared = sum(map(lambda (y,x): abs(x)**2, enumerate(fft_spectrum)))
    ss = sqrt((spectral_centroid ** 2 * sum_abs_squared) / sum_abs_squared)
    return ss


def spectral_flux(signal = None, fft_spectrum  = None):
    if fft_spectrum is None and signal is not None:
        fft_spectrum = scipy.real(scipy.fft(signal))
    spectralFlux = []
    flux = 0
    for element in fft_spectrum:
        flux += abs(element)

    spectralFlux.append(flux)

    for element in xrange(1, fft_spectrum.shape[0]):
        prev = fft_spectrum[element-1]
        current = fft_spectrum[element]
        flux = abs(abs(current) - abs(prev))
        spectralFlux.append(flux)
    return np.array(spectralFlux)


def spectral_entropy(signal = None, fft_signal = None, num_of_frames =22050 / 50):
    if fft_signal is not None and signal is not None:
        fft_signal = scipy.real(scipy.fft(signal))
    subframes = np.split(fft_signal, num_of_frames)
    frame_energy = short_time_energy(fft_signal)
    e = map(lambda (x,y): short_time_energy(y)/frame_energy,
                          enumerate(subframes))
    entropy = 0
    for frame_index in xrange(num_of_frames):
        entropy -= e[frame_index] * log(e[frame_index],2)

    return entropy

def spectral_centroid(signal, sample_rate):
        return librosa.feature.spectral_centroid(signal,sample_rate)


class Sampler(object):
    def __init__(self, filepath, duration = None):
        self.mfcc = None
        self.signal_hammed = None
        self.normalized_signal = None
        self.signal_emphased = None
        self.signal, self.sample_rate = librosa.load(filepath, duration=duration)
        if not duration:
            self.duration = librosa.get_duration(self.signal, sr=self.sample_rate)
        else:
            self.duration = duration
        self.pre_process()
        self.compute_features()

        '''
        spectral_centroid = librosa.feature.\
            spectral_centroid(self.sample,sr = self.frequency)
        plt.plot(range(spectral_centroid.size),spectral_centroid.T)
        plt.show()

        spectral_roloff = librosa.feature.spectral_rolloff\
            (self.sample,sr=self.frequency)
        plt.plot(range(spectral_roloff.size),spectral_roloff.T)
        plt.show()

        #flux
        zero_crossings = librosa.zero_crossings(self.sample,)
        plt.plot(range(zero_crossings.size),zero_crossings.T)
        plt.show()
        #librosa.feature.mfcc()
        tempo, beats = librosa.beat.beat_track(self.sample, sr  = self.frequency)
        '''


    def pre_process(self):
        #plt.plot(self.signal,'r',)
        #plt.show()

        self.normalized_signal = librosa.util.normalize(self.signal)

        self.signal_emphased = pre_emphasis(self.normalized_signal)
        #plt.plot(self.signal_emphased,'r')
        #plt.show()

        #self.spectrum = scipy.fft(self.normalized_signal,)
        #plt.plot(self.spectrum,'r')
        #plt.show()
        #self.spectrum = scipy.fft(self.signal_emphased,)
        #plt.plot(self.spectrum,'y')
        #plt.show()

        self.signal_hammed = window_signal(self.signal_emphased)
        print self.signal_hammed.shape

        #plt.plot(self.signal_hammed)
        #plt.show()

        #self.signal_mean = self.signal.mean()
        #self.signal_variance = compute_variance(self.signal, self.signal_mean)
        #self.normalized_signal = normalize_signal(self.signal, self.signal_mean, self.signal_variance)

    def compute_features(self):
        self.fft = scipy.real(scipy.fft(self.signal_hammed))
        print ".",

        self.spectral_flux = spectral_flux(fft_spectrum=self.fft).mean()
        print ".",

        self.spectral_cenroid_mean = spectral_centroid\
            (self.signal_hammed, self.sample_rate).mean()
        print ".",

        self.spectral_spread = spectral_spread(spectral_centroid=self.spectral_cenroid_mean, fft_spectrum=self.fft)
        print ".",
        self.spectral_roloff_mean = spectal_roloff\
            (self.signal_hammed, self.sample_rate).mean()
        print ".",
        self.spectral_entropy = spectral_entropy(fft_signal=self.fft)
        print "spectral features calc!"

        self.energy_entropy = energy_entropy(self.signal_hammed, self.sample_rate)
        print ".",
        self.zero_crossing_rate = zero_crossing_rate(self.signal_hammed)
        print ".",
        self.temporal_centroid = temporal_cenroid(self.signal_hammed)
        print ".",
        self.root_mean_square = root_mean_square(self.signal)
        print ".",
        self.short_time_energy = short_time_energy(self.signal)
        print ".",
        self.fundamental_period, self.autocorellation = autocorellation(self.signal_hammed, self.sample_rate)
        print ".",
        self.mfcc = compute_mfcc(self.signal_hammed, self.sample_rate)
        print "signal features calc!"
        print self.extract_features()

    def extract_features(self):
        vector = [0] * 11
        vector[0] = self.zero_crossing_rate
        vector[1] = self.temporal_centroid
        vector[2] = self.energy_entropy
        vector[3] = self.root_mean_square
        vector[4] = self.fundamental_period
        vector[5] = self.autocorellation
        vector[6] = self.spectral_roloff_mean
        vector[7] = self.spectral_spread
        vector[8] = self.spectral_flux
        vector[9] = self.spectral_entropy
        vector[10] = self.spectral_cenroid_mean
        mean = self.mfcc.mean(axis = 1)
        vector = np.append(np.array(vector),mean)
        print "features extracted"
        return vector



if __name__ == "__main__":
    path = "/media/files/musicsamples/genres/pop/pop.00002.au"
    sampler = Sampler(path, duration=30)