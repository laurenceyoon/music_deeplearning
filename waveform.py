import matplotlib.pyplot as plt
import librosa.display
import numpy as np


train_audio_path = 'input/'
file_path = 'jazz.00007.wav'
y, sr = librosa.load(str(train_audio_path) + file_path, offset=15.0, duration=30.0)
# print(y, len(y))

fig = plt.figure(figsize = (14, 5))
librosa.display.waveplot(y, sr=sr)

S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024, window='hann')
# print(f"{S}, {S.shape}, {len(S[0])}, {S[0][0]}")  # 시간의 흐름에 따른  Frequency 영역별 Amplitude 를 반환.

D = np.abs(S) ** 2
# print(D)

# mel spectrogram
mel_basis = librosa.filters.mel(sr, 1024, n_mels=128)
mel_S = np.dot(mel_basis, D)
# print(mel_S.shape)

# log compression
log_mel_S = librosa.power_to_db(mel_S)
print(log_mel_S, log_mel_S.shape)

S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel power spectrogram')
plt.colorbar(format="%+02.0f dB")

plt.tight_layout()
