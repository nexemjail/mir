import warnings
warnings.filterwarnings('ignore')

# numerical processing and scientific libraries
import numpy as np
import scipy

# signal processing
from scipy.io                     import wavfile
from scipy                        import stats, signal
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scikits.talkbox              import segment_axis
from scikits.talkbox.features     import mfcc

# general purpose
import collections

# plotting
import matplotlib.pyplot as plt
from numpy.lib                    import stride_tricks

from IPython.display              import HTML
from base64                       import b64encode

# Classification and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd\


ABBRIVATIONS = {}

# features
ABBRIVATIONS["zcr"] = "Zero Crossing Rate"
ABBRIVATIONS["rms"] = "Root Mean Square"
ABBRIVATIONS["sc"]  = "Spectral Centroid"
ABBRIVATIONS["sf"]  = "Spectral Flux"
ABBRIVATIONS["sr"]  = "Spectral Rolloff"

# aggregations
ABBRIVATIONS["var"] = "Variance"
ABBRIVATIONS["std"] = "Standard Deviation"
ABBRIVATIONS["mean"] = "Average"

PLOT_WIDTH  = 15
PLOT_HEIGHT = 3.5


def show_mono_waveform(samples):

	fig = plt.figure(num=None, figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=72, facecolor='w', edgecolor='k')

	channel_1 = fig.add_subplot(111)
	channel_1.set_ylabel('Channel 1')
	#channel_1.set_xlim(0,song_length) # todo
	channel_1.set_ylim(-32768,32768)

	channel_1.plot(samples)

	plt.show();
	plt.clf();

def show_stereo_waveform(samples):

	fig = plt.figure(num=None, figsize=(PLOT_WIDTH, 5), dpi=72, facecolor='w', edgecolor='k')

	channel_1 = fig.add_subplot(211)
	channel_1.set_ylabel('Channel 1')
	#channel_1.set_xlim(0,song_length) # todo
	channel_1.set_ylim(-32768,32768)
	channel_1.plot(samples[:,0])

	channel_2 = fig.add_subplot(212)
	channel_2.set_ylabel('Channel 2')
	channel_2.set_xlabel('Time (s)')
	channel_2.set_ylim(-32768,32768)
	#channel_2.set_xlim(0,song_length) # todo
	channel_2.plot(samples[:,1])

	plt.show();
	plt.clf();


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(samples, samplerate, binsize=2**10, plotpath=None, colormap="jet", ax=None, fig=None):

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    if ax is None:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(PLOT_WIDTH, 3.5))

    #ax.figure(figsize=(15, 7.5))
    cax = ax.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1], cax=ax)
    #ax.set_colorbar()

    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (hz)")
    ax.set_xlim([0, timebins-1])
    ax.set_ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    ax.set_xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    ax.set_yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    #plt.clf();
    b = ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate]
    return xlocs, b, timebins


def list_audio_samples(sound_files):

	src = ""

	for genre in sound_files.keys():

		src += "<b>" + genre + "</b><br><br>"
		src += "<object width='600' height='90'><param name='movie' value='http://freemusicarchive.org/swf/trackplayer.swf'/><param name='flashvars' value='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml'/><param name='allowscriptaccess' value='sameDomain'/><embed type='application/x-shockwave-flash' src='http://freemusicarchive.org/swf/trackplayer.swf' width='600' height='50' flashvars='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml' allowscriptaccess='sameDomain' /></object><br><br>".format(sound_files[genre]["online_id"])

	return HTML(src)

def play_sample(sound_files, genre):

	src = "<object width='600' height='90'><param name='movie' value='http://freemusicarchive.org/swf/trackplayer.swf'/><param name='flashvars' value='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml'/><param name='allowscriptaccess' value='sameDomain'/><embed type='application/x-shockwave-flash' src='http://freemusicarchive.org/swf/trackplayer.swf' width='600' height='50' flashvars='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml' allowscriptaccess='sameDomain' /></object><br><br>".format(sound_files[genre]["online_id"])

	return HTML(src)


def plot_compairison(data, feature, aggregators):

    width = 0.35

    features = {}

    for aggr_name in aggregators:

        features[aggr_name] = []

        for genre in data.keys():

            if aggr_name == "mean":
                features[aggr_name].append(np.mean(data[genre][feature]))
            elif aggr_name == "std":
                features[aggr_name].append(np.std(data[genre][feature]))
            elif aggr_name == "var":
                features[aggr_name].append(np.var(data[genre][feature]))
            elif aggr_name == "median":
                features[aggr_name].append(np.median(data[genre][feature]))
            elif aggr_name == "min":
                features[aggr_name].append(np.min(data[genre][feature]))
            elif aggr_name == "max":
                features[aggr_name].append(np.max(data[genre][feature]))

    fig, ax = plt.subplots()
    ind     = np.arange(len(features[aggregators[0]]))
    rects1  = ax.bar(ind, features[aggregators[0]], 0.7, color='b')
    ax.set_xticklabels( data.keys() )
    ax.set_xticks(ind+width)
    ax.set_ylabel(ABBRIVATIONS[aggregators[0]])
    ax.set_title("{0} Results".format(ABBRIVATIONS[feature]))


    if len(aggregators) == 2:

        ax2 = ax.twinx()
        ax2.set_ylabel(ABBRIVATIONS[aggregators[1]])
        rects2 = ax2.bar(ind+width, features[aggregators[1]], width, color='y')
        ax.legend( (rects1[0], rects2[0]), (ABBRIVATIONS[aggregators[0]], ABBRIVATIONS[aggregators[1]]) )

    plt.show()

def show_feature_superimposed(genre, feature_data, timestamps, squared_wf=False):

    # plot spectrogram
    a,b,c = plotstft(sound_files[genre]["wavedata"], sound_files[genre]["samplerate"]);

    fig = plt.figure(num=None, figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=72, facecolor='w', edgecolor='k');
    channel_1 = fig.add_subplot(111);
    channel_1.set_ylabel('Channel 1');
    channel_1.set_xlabel('time');

    # plot waveform
    scaled_wf_y = ((np.arange(0,sound_files[genre]["wavedata"].shape[0]).astype(np.float)) / sound_files[genre]["samplerate"]) * 1000.0

    if squared_wf:
        scaled_wf_x = (sound_files[genre]["wavedata"]**2 / np.max(sound_files[genre]["wavedata"]**2))
    else:
        scaled_wf_x = (sound_files[genre]["wavedata"] / np.max(sound_files[genre]["wavedata"]) / 2.0 ) + 0.5

    #scaled_wf_x = scaled_wf_x**2

    plt.plot(scaled_wf_y, scaled_wf_x, color='lightgrey');

    # plot feature-data
    scaled_fd_y = timestamps * 1000.0
    scaled_fd_x = (feature_data / np.max(feature_data))

    plt.plot(scaled_fd_y, scaled_fd_x, color='r');

    plt.show();
    plt.clf();

def nextpow2(num):
    n = 2
    i = 1
    while n < num:
        n *= 2
        i += 1
    return i

def periodogram(x,win,Fs=None,nfft=1024):

    if Fs == None:
        Fs = 2 * np.pi

    U  = np.dot(win.conj().transpose(), win) # compensates for the power of the window.
    Xx = fft((x * win),nfft) # verified
    P  = Xx*np.conjugate(Xx)/U

    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.

    # Generate the one-sided spectrum [Power] if so wanted
    if nfft % 2 != 0:
        select = np.arange((nfft+1)/2)  # ODD
        P_unscaled = P[select,:] # Take only [0,pi] or [0,pi)
        P[1:-1] = P[1:-1] * 2 # Only DC is a unique point and doesn't get doubled
    else:
        select = np.arange(nfft/2+1);    # EVEN
        P = P[select]         # Take only [0,pi] or [0,pi) # todo remove?
        P[1:-2] = P[1:-2] * 2

    P = P / (2* np.pi)

    return P

def map_labels_to_numbers(eval_data):

    for df_name in eval_data.keys():

        # create label mapping
        label_mapping = {}
        num_to_label  = []

        i = 0
        for l in set(eval_data[df_name]["labels"]):
            label_mapping[l] = i
            num_to_label.append(l)
            i += 1

        eval_data[df_name]["label_mapping"] = label_mapping
        eval_data[df_name]["num_to_label"] = num_to_label

        mapped_labels = []

        for i in range(eval_data[df_name]["labels"].shape[0]):
            #print label_mapping[ls[i]]
            mapped_labels.append(label_mapping[eval_data[df_name]["labels"][i]])

        #transformed_label_space.append(mapped_labels)

        eval_data[df_name]["num_labels"] = np.asarray(mapped_labels)

styles = "<style>div.cell{ width:900px; margin-left:0%; margin-right:auto;} </style>"
HTML(styles)