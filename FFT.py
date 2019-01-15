import matplotlib
matplotlib.use('AGG')
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.signal import argrelextrema
from scipy.signal import spectrogram
import numpy as np
import pyaudio
import wave
import cv2
from matplotlib import rc
# rc('font', family="AppleGothic")#Malgun Gothic
matplotlib.rcParams.update({'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix'})

def play(sound, sample_rate):
    wav.write("_sample_sound.wav", sample_rate, sound)
    f = wave.open(r"_sample_sound.wav","rb")
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    data = f.readframes(1024)
    while data:
        stream.write(data)
        data = f.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
        raise ValueError

    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
        raise ValueError

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        raise ValueError

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    #print(len(s))
    if window == 'flat':  #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[window_len//2:-(window_len//2)]


def process_wav(filename, comment, delta_t, inc_t=None, specific_lim = (20, 2000), fft_color = 'C2', scatter_color = 'red', max_color = 'r'):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        print("%d CHANNELS in %s"%(data.shape[1], comment))
        # channel = int(input("USE: (0, ..., %d): "%(data.shape[1]-1)))
        print("Using 0")
        channel = 0
        data = data[:,channel]
    # play(data, rate)

    # data = data[::10]
    # rate //= 10
    N = len(data)


    # #spectogram
    # specto_f, specto_t, Sxx = spectrogram(data, rate)
    # specto_fig = plt.figure('Spectogram %s'%comment, figsize=(5, 4), dpi=150)
    # ax_specto = specto_fig.add_subplot(111)
    # ax_specto.set_title("Spectrogram %s"%comment)
    # ax_specto.pcolormesh(specto_t, specto_f, Sxx)
    # ax_specto.set_ylabel("Frequency [Hz]")
    # ax_specto.set_xlabel("Time [s]")
    # ax_specto.set_ylim(0, 2500) #Min, Max frequency
    # specto_fig.tight_layout()
    # specto_fig.canvas.draw()
    # specto_fig.savefig('spectro(%s).png'%comment)
    # specto = cv2.imread('spectro(%s).png'%comment, cv2.IMREAD_COLOR)
    # cv2.imshow("spectrogram %s"%comment, specto)
    # plt.close(specto_fig)
    

    fs = rate
    dt = 1/fs

    # delta_t = 1# 한 구간의 길이 (초)
    # inc_t = delta_t/2 # 구간과 구간 사이의 거리 (초)
    if inc_t is None:
        inc_t = delta_t/2
    inc_N = int(rate*inc_t)
    delta_N = int(rate*delta_t)
    segment = np.arange(0, N//inc_N)*inc_N
    data_segment = [data[start_N:start_N+delta_N] for start_N in segment]
    del data_segment[-1]
    if not data_segment:
        import sys
        print("%s Too Short!"%comment)
        sys.exit(0)
    data_info = np.iinfo(type(data[0]))

    whole_df = fs/N
    whole_f = (np.arange(0, N)*whole_df)[0:int(N/2+1)]
    whole_fourier = np.abs(fft(data)*dt)[0:int(N/2+1)]
    smooth_whole_fourier = smooth(whole_fourier, window='hanning')

    local_maximum_points, = argrelextrema(smooth_whole_fourier, np.greater)
    local_maximum_points = local_maximum_points[whole_fourier[local_maximum_points] > 20]
    local_maximum_points = list(local_maximum_points)
    local_maximum_points.sort(key=lambda num: whole_fourier[num], reverse=True)


    global ax_whole
    ax_whole.set_title("FFT")
    ax_whole.plot(whole_f, whole_fourier, fft_color+"-")
    ax_whole.set_xlabel('frequency(Hz)')
    ax_whole.set_ylabel('amplitude')
    ax_whole.grid()
    ax_whole.legend(legend_setting)
    ax_whole.relim()
    ax_whole.set_xlim(whole_f[0], whole_f[-1]+100)
    whole_fig.tight_layout()
    whole_fig.canvas.draw()
    whole_fig.savefig("fourier.png")
    ax_whole.set_xlim(specific_lim[0], specific_lim[1])
    whole_fig.canvas.draw()
    whole_fig.savefig("fourier_specific.png")
    whole = cv2.imread('fourier.png', cv2.IMREAD_COLOR)
    whole_specific = cv2.imread("fourier_specific.png", cv2.IMREAD_COLOR)
    whole_specific = cv2.copyMakeBorder(whole_specific, 0, 100, 0, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(whole_specific, "%d PEAKS: " % len(local_maximum_points) +
            ", ".join(["%.2f" % whole_f[num] for num in local_maximum_points[:6]]),
            (20, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
    if len(local_maximum_points) > 6:
        cv2.putText(whole_specific, ", ".join(["%.2f" % whole_f[num] for num in local_maximum_points[6:13]]) + "...",
                    (20, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
    cv2.imwrite('fourier_specific.png', whole_specific)
    cv2.imshow("whole fourier", whole)
    cv2.imshow("specific fourier", whole_specific)

    # xf = fft(data) * dt
    fourier_segment = [np.abs(fft(x_t)*dt)[0:int(delta_N/2)+1] for x_t in data_segment]
    fig = plt.figure('Fourier Transform %s'%comment, figsize=(10, 4), dpi=150)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(224)
    ax2.set_xscale('log')
    ax_all_signal = fig.add_subplot(222)
    ax_specific_freq = fig.add_subplot(223)


    ax_all_signal.set_title("Fast Fourier Transform: %s"%comment)
    ax_all_signal.plot(np.arange(0, N)*dt, data, 'C1-')
    bar_left, = ax_all_signal.plot([0, 0], [data_info.min, data_info.max], 'r-')
    bar_right, = ax_all_signal.plot([0, 0], [data_info.min, data_info.max], 'r-')
    ax_all_signal.set_xlabel("time(s)")
    ax_all_signal.set_ylabel("signal")
    ax_all_signal.grid()
    ax_all_signal.set_ylim(data_info.min, data_info.max)
    t = np.arange(0, delta_N)*dt

    t_start = [start_N/rate for start_N in segment]
    # data = np.sin(2*np.pi*60*t)
    signal, = ax1.plot(t, data_segment[0], 'C1-')
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("signal")
    ax1.grid()
    # ax1.legend(["sound"])

    df = fs/delta_N
    f = (np.arange(0,delta_N)*df)[0:int(delta_N/2+1)]
    # xf = fft(data) * dt
    fourier_segment = [np.abs(fft(x_t)*dt)[0:int(delta_N/2)+1] for x_t in data_segment]
    flat_fourier_segment = [smooth(f_seg, window='hanning') for f_seg in fourier_segment]
    localmaxes = [argrelextrema(flat_f_seg, np.greater)[0] for flat_f_seg in flat_fourier_segment]
    localmaxes = [list(localmaxes[i][flat_fourier_segment[i][localmaxes[i]] > 20]) for i in range(len(localmaxes))]
    freq_time_x = []
    freq_time_y = []
    freq_time_maxes_x = []
    freq_time_maxes_y = []
    for i in range(len(localmaxes)):
        localmaxes[i].sort(key= lambda num: flat_fourier_segment[i][num], reverse=True)
        if len(localmaxes[i]) > 0:
            freq_time_x.extend([t_start[i] + inc_t / 2] * min(len(localmaxes[i]), 50))
            freq_time_y.extend(f[localmaxes[i][:50]])
            freq_time_maxes_x.append(t_start[i]+inc_t/2)
            freq_time_maxes_y.append(f[localmaxes[i][0]])

    global ax_result
    ax_result.set_yscale('log')
    ax_result.scatter(freq_time_x, freq_time_y, 1, scatter_color)
    if len(freq_time_maxes_x) > 0:
        ax_result.plot(freq_time_maxes_x, freq_time_maxes_y, max_color+'-')
    # ax_result.legend(["Max amplitudes", "Scatter top 50"])
    ax_result.set_ylabel('frequency(Hz)')
    ax_result.set_xlabel("time(s)")
    ax_result.grid()
    result_fig.tight_layout()
    result_fig.canvas.draw()
    result_fig.savefig("result.png")
    result = cv2.imread("result.png", cv2.IMREAD_COLOR)
    cv2.imshow("result", result)
    
    fourier, = ax2.plot(f, fourier_segment[0], 'C2-')
    specific_fourier, = ax_specific_freq.plot(f, fourier_segment[0], 'C2-')
    ax_specific_freq.set_xscale('log')
    ax2.set_xlabel('frequency(Hz)')
    ax2.set_ylabel('amplitude')
    ax2.grid()
    ax2.plot([0], [100], 'k,')

    ax_specific_freq.set_xlabel('frequency(Hz)')
    ax_specific_freq.set_ylabel('amplitude')
    ax_specific_freq.grid()
    ax_specific_freq.plot([0], [100], 'k,')
    ax_specific_freq.set_xlim(specific_lim[0], specific_lim[1]) # NOTE change left_low view

    ax1.set_ylim(data_info.min, data_info.max)
    ax2.set_autoscale_on(True)
    ax_specific_freq.set_autoscale_on(True)
    fig.tight_layout()
    def animate(i, playing=True):
        signal.set_xdata(t_start[i] + t)
        signal.set_ydata(data_segment[i])
        fourier.set_ydata(fourier_segment[i])
        specific_fourier.set_ydata(fourier_segment[i])
        ax1.set_xlim(t_start[i], t_start[i] + delta_t)
        bar_left.set_xdata([t_start[i]]*2)
        bar_right.set_xdata([t_start[i] + delta_t]*2)
        ax_specific_freq.relim()
        ax_specific_freq.autoscale_view(True, False, True)
        ax2.relim()
        ax2.autoscale_view(True, False, True)

        fig.canvas.draw()
        fig.savefig('graph(%s).png'%comment)
        graph = cv2.imread('graph(%s).png'%comment, cv2.IMREAD_COLOR)
        # graph = cv2.resize(graph, (1080, 720))
        graph = cv2.copyMakeBorder(graph, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(graph, "%d PEAKS: " % len(localmaxes[i]) +
                ", ".join(["%.2f" % f[num] for num in localmaxes[i][:9]]),
                (20, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
        if len(localmaxes[i]) > 8:
            cv2.putText(graph, ", ".join(["%.2f" % f[num] for num in localmaxes[i][9:16]]) + "...",
                    (20, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
        cv2.imshow('graph %s'%comment, graph)
        key = cv2.waitKey(1)
        if playing:
            play(data_segment[i], rate) # Keys
        return key
    return len(data_segment), animate, f, fourier_segment

whole_fig = plt.figure("Whole", figsize=(6,4),dpi=150)
ax_whole = whole_fig.add_subplot(111)
ax_whole.set_xscale('log')
ax_whole.set_autoscale_on(True)
result_fig = plt.figure('Results', figsize=(6, 4), dpi=150)
ax_result = result_fig.add_subplot(111)

# process_wav(filename, comment, delta_t: 한 간격의 길이, inc_t=None, specific_lim = (20, 2000), fft_color = 'C2', scatter_color = 'C4', max_color = 'r')

legend_setting = ["high", "low"]
filenames = ["sample piano.wav", "IMG_2252.wav"]

reverse = False
if reverse:
    legend_setting.reverse()
    filenames.reverse()

N1, anim1, f1, fs1 = process_wav(filenames[0], legend_setting[0], 3, 1, (0, 2000)) # What I want to see

second = False
if second:
    N2, anim2, f2, fs2 = process_wav(filenames[1], legend_setting[1], 1, fft_color = 'C3', scatter_color = 'blue', max_color = 'b')

cv2.destroyAllWindows() #NOTE destroying
plt.close(whole_fig)
plt.close(result_fig)
# plt.clf()
if second:
    fig_merge = plt.figure('Merged', figsize=(10, 4), dpi=150)
    ax_merge_specific = fig_merge.add_subplot(121)
    ax_merge_specific.set_xscale('log')
    ax_merge = fig_merge.add_subplot(122)
    ax_merge.set_xscale('log')

    ax_merge_specific.set_xlabel('frequency(Hz)')
    ax_merge_specific.set_ylabel('amplitude')
    ax_merge_specific.grid()

    ax_merge.set_xlabel('frequency(Hz)')
    ax_merge.set_ylabel('amplitude')
    ax_merge.grid()

    merge_fr_1, = ax_merge.plot(f1, fs1[0], 'C2-')
    merge_fr_s_1, = ax_merge_specific.plot(f1, fs1[0], 'C2-')
    merge_fr_2, = ax_merge.plot(f2, fs2[0], 'C3-')
    merge_fr_s_2, = ax_merge_specific.plot(f2, fs2[0], 'C3-')

    ax_merge_specific.plot([0], [100], 'k,')
    ax_merge.plot([0], [100], 'k,')

    ax_merge.legend(legend_setting)
    ax_merge_specific.legend(legend_setting)

    specific_lim = (0, 3000)
    ax_merge_specific.set_xlim(specific_lim[0], specific_lim[1])

    ax_merge.set_autoscale_on(True)
    ax_merge_specific.set_autoscale_on(True)

    fig_merge.tight_layout()

i = 0
j = 0
wait1 = 0
wait2 = 0
play_1 = True
play_2 = True

while True:
    key = anim1(i, playing=play_1)
    play_1 = False
    if second:
        key = anim2(j, playing=play_2)
        play_2 = False
        if not reverse:
            merge_fr_1.set_ydata(fs1[i])
            merge_fr_s_1.set_ydata(fs1[i])
            merge_fr_2.set_ydata(fs2[j])
            merge_fr_s_2.set_ydata(fs2[j])
            ax_merge.legend(legend_setting)
            ax_merge_specific.legend(legend_setting)
        else:
            merge_fr_1.set_ydata(fs2[j])
            merge_fr_s_1.set_ydata(fs2[j])
            merge_fr_2.set_ydata(fs1[i])
            merge_fr_s_2.set_ydata(fs1[i])
            ax_merge.legend([legend_setting[1], legend_setting[0]])
            ax_merge_specific.legend([legend_setting[1], legend_setting[0]])
        ax_merge_specific.relim()
        ax_merge_specific.autoscale_view(True, False, True)
        ax_merge.relim()
        ax_merge.autoscale_view(True, False, True)

        fig_merge.canvas.draw()
        fig_merge.savefig('graph_merge.png')
        graph_merge = cv2.imread('graph_merge.png', cv2.IMREAD_COLOR)
        # graph = cv2.resize(graph, (1080, 720))
        cv2.imshow('graph_merge', graph_merge)

    if key == -1:
        key = cv2.waitKey(min(wait1, wait2)) & 0xFF
    if key == ord('q') or key == ord('ㅂ') or key == 27:  # 27 ESC
        break
    elif key == ord('a'):
        i = i - 1
        if i < 0:
            i += N1
        play_1 = True
    elif key == ord('w'):
        i = (i + 10) % N1
        play_1 = True
    elif key == ord('s'):
        i = i - 10
        if i < 0:
            i += N1
        play_1 = True
    elif key == ord('d'):
        i = (i + 1) % N1
        play_1 = True
    elif key == ord('z'):
        wait1 = 10
    elif key == ord('x'):
        wait1 = 0
    elif key == ord('k'):
        j -= 1
        if j < 0:
            j += N2
        play_2 = True
    elif key == ord(';'):
        j = (j + 1) % N2
        play_2 = True
    elif key == ord('o'):
        j = (j + 10) % N2
        play_2 = True
    elif key == ord('l'):
        j -= 10
        if j < 0:
            j += N2
        play_2 = True
    elif key == ord('h'):
        i = (i + 1) % N1
        j = (j + 1) % N2
        play_1 = True
        play_2 = True
    elif key == ord('g'):
        i = i - 1
        if i < 0:
            i += N1
        play_1 = True
        j -= 1
        if j < 0:
            j += N2
        play_1 = True
        play_2 = True
    elif key == ord(' '):
        reverse = not reverse
    elif key == ord('.'):
        play_1 = True
        play_2 = True
    elif key == -1:
        if wait1:
            i = (i + 1)%N1
            play_1 = True
        if wait2:
            j = (j + 1)%N2
            play_2 = True
plt.close('all')
cv2.destroyAllWindows()