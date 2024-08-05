import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def read_voice(path):
    rate, data = wavfile.read(path)
    Amplitude = rfft(data)
    Frequency = rfftfreq(len(data), 1 / rate)
    return Amplitude, Frequency , data , rate

def write_voice(data, rate, path):
    wavfile.write(path, rate, data.astype(np.int16))

def change_voice_speed(data, rate, speed_factor):
    new_rate = int(rate * speed_factor)
    indices = np.round(np.arange(0, len(data), speed_factor))
    indices = indices[indices < len(data)].astype(int)
    new_data = data[indices]
    return new_data, new_rate

def low_pass_filter(Frequency, Amplitude, F, t):
    lower_bound = F - t / 2
    upper_bound = F + t / 2
    for i in range(len(Frequency)):
        if  Frequency[i] > upper_bound or Frequency[i]<lower_bound  or np.abs(Amplitude[i]) > 5*10**8 :
            Amplitude[i] = 0
    return Amplitude

def reverse_voice(data):
    new_data = data[::-1]
    return new_data

def mix_voices(Datas, Rates):
    Rate = Rates[0]
    min_length = min([len(data) for data in Datas])
    Data = np.zeros(min_length)
    for data in Datas:
        Data += data[:min_length]
    Data = Data / len(Datas) 
    return Data, Rate



path = '/Users/arasvalizadeh/Downloads/audio/potc.wav'
amplitude, frequency , data , rate = read_voice(path)
amplitudecopy , frequencycopy , datacopy , ratecopy = read_voice(path)


plt.figure()

time = []
for i in range(len(data)):
    time.append(i/rate)
plt.plot(time, data)
plt.title("waveform")
plt.xlabel("time")
plt.ylabel("Main_Data")
plt.show()

plt.plot(frequency, np.abs(amplitude))
plt.title("amplitude spectrum")
plt.xlabel("frequency")
plt.ylabel("Main amplitude")
plt.show()

f, t, Sxx = spectrogram(data, rate)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('frequency')
plt.xlabel('time')
plt.title('Main spectrogram')
plt.show()

good = low_pass_filter(frequency,amplitude,4000,8000)
filteredVoice = irfft(good)
write_voice(filteredVoice,rate,'/Users/arasvalizadeh/Downloads/audio/FinalVoices/cleanpotc.wav')
amplitude_filter, frequency_filter , data_filter , rate_filter = read_voice("/Users/arasvalizadeh/Downloads/audio/FinalVoices/cleanpotc.wav")

time = []
for i in range(len(data_filter)):
    time.append(i/rate_filter)

plt.plot(time, data_filter)
plt.title("waveform")
plt.xlabel("time")
plt.ylabel("filtered Voice")
plt.show()

plt.plot(frequency_filter , np.abs(amplitude_filter))
plt.title("amplitude spectrum")
plt.xlabel("frequency")
plt.ylabel("filtered amplitude")
plt.show()

f, t, Sxx = spectrogram(data_filter, rate_filter)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('frequency')
plt.xlabel('time')
plt.title('filtered spectrogram')
plt.show()


slow_speed_factor = 0.8
slowed_data, slowed_rate = change_voice_speed(filteredVoice, rate, slow_speed_factor)
fast_speed_factor = 1.5
fast_data, fast_rate = change_voice_speed(filteredVoice, rate, fast_speed_factor)


write_voice(slowed_data, slowed_rate, '/Users/arasvalizadeh/Downloads/audio/FinalVoices/slowpotc.wav')
slow_amplitude, slow_frequency , slow_data , slow_rate = read_voice('/Users/arasvalizadeh/Downloads/audio/FinalVoices/slowpotc.wav')


time = []
for i in range(len(slow_data)):
    time.append(i/slow_rate)

plt.plot(time, slow_data)
plt.title("waveform")
plt.xlabel("time")
plt.ylabel("slowed Voice")
plt.show()

plt.plot(slow_frequency , np.abs(slow_amplitude))
plt.title("amplitude spectrum")
plt.xlabel("frequency")
plt.ylabel("slowed amplitude")
plt.show()

f, t, Sxx = spectrogram(slow_data, slow_rate)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('frequency')
plt.xlabel('time')
plt.title('slowed spectrogram')
plt.show()



write_voice(fast_data, fast_rate, '/Users/arasvalizadeh/Downloads/audio/FinalVoices/fastpotc.wav')
fast_amplitude, fast_frequency , fast_data , fast_rate = read_voice('/Users/arasvalizadeh/Downloads/audio/FinalVoices/fastpotc.wav')

time = []
for i in range(len(fast_data)):
    time.append(i/fast_rate)

plt.plot(time, fast_data)
plt.title("waveform")
plt.xlabel("time")
plt.ylabel("fasted Voice")
plt.show()

plt.plot(fast_frequency , np.abs(fast_amplitude))
plt.title("amplitude spectrum")
plt.xlabel("frequency")
plt.ylabel("fasted amplitude")
plt.show()

f, t, Sxx = spectrogram(fast_data, fast_rate)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('frequency')
plt.xlabel('time')
plt.title('fasted spectrogram')
plt.show()



path2="/Users/arasvalizadeh/Downloads/audio/FinalVoices/cleanpotc.wav"
rateclean , dataclean = wavfile.read(path2)
reversed_data = reverse_voice(dataclean)
write_voice(reversed_data, rateclean, '/Users/arasvalizadeh/Downloads/audio/FinalVoices/reversepotc.wav')
reverse_amplitude, reverse_frequency , reverse_data , reverse_rate = read_voice('/Users/arasvalizadeh/Downloads/audio/FinalVoices/reversepotc.wav')



time = []
for i in range(len(reverse_data)):
    time.append(i/reverse_rate)

plt.plot(time, reverse_data)
plt.title("waveform")
plt.xlabel("time")
plt.ylabel("reversed Voice")
plt.show()

plt.plot(reverse_frequency , np.abs(reverse_amplitude))
plt.title("amplitude spectrum")
plt.xlabel("frequency")
plt.ylabel("reverse amplitude")
plt.show()

f, t, Sxx = spectrogram(reverse_data, reverse_rate)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('frequency')
plt.xlabel('time')
plt.title('reverse spectrogram')
plt.show()





rateslow , dataslow = wavfile.read("/Users/arasvalizadeh/Downloads/audio/FinalVoices/slowpotc.wav")
ratefast , datafast = wavfile.read("/Users/arasvalizadeh/Downloads/audio/FinalVoices/fastpotc.wav")
ratereverse , datareverse = wavfile.read("/Users/arasvalizadeh/Downloads/audio/FinalVoices/slowpotc.wav")

Datas = [dataclean,datafast,datareverse,dataslow]
Rates = [rateclean,ratefast,ratereverse,rateslow]
mixed_data, mixed_rate = mix_voices(Datas, Rates)
write_voice(mixed_data, mixed_rate, '/Users/arasvalizadeh/Downloads/audio/FinalVoices/mixpotc.wav')
mix_amplitude, mix_frequency , mix_data , mix_rate = read_voice('/Users/arasvalizadeh/Downloads/audio/FinalVoices/mixpotc.wav')

time = []
for i in range(len(mix_data)):
    time.append(i/mix_rate)

plt.plot(time, mix_data)
plt.title("waveform")
plt.xlabel("time")
plt.ylabel("mixed Voice")
plt.show()

plt.plot(mix_frequency , np.abs(mix_amplitude))
plt.title("amplitude spectrum")
plt.xlabel("frequency")
plt.ylabel("mixed amplitude")
plt.show()

f, t, Sxx = spectrogram(mix_data, mix_rate)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('frequency')
plt.xlabel('time')
plt.title('mixed spectrogram')
plt.show()



