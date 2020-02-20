# music_deeplearning

Pytorch, Librosa based training project for music genre classification.

### Requirement
- Python 3.6 이상  
- Numpy
- Librosa
- PyTorch 1.0

### Dataset
[GTZAN subset](https://drive.google.com/file/d/1rHw-1NR_Taoz6kTfJ4MPR5YTxyoCed1W/view)

### Run
Before you run this code, you need to check you have all the directories propertly in `hyper_parameters.py` 

```
$ virtualenv -p python3 venv
$ source venv/bin/activate

(venv) $ pip install -r requirements.txt
(venv) $ mkdir input  # 학습용 음원 데이터를 넣을 위치 
(venv) $ cp gtzan/classical/classical.00000.wav input/  # 음원 데이터 이동 

(venv) $ python train_test.py
```

### Reference
2019년 파이콘 튜토리얼 [Music과 Deeplearning의 만남](https://github.com/Dohppak/Pycon_Tutorial_Music_DeepLearing)