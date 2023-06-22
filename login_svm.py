import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import subprocess
from string import Template
import joblib

data_dir = './login-captcha-example'
train_material_dir = './material'
model_file = 'svm_model.pkl'

def train_model():
  X = []
  y = []
  for file in os.listdir(train_material_dir):
      if file.endswith('.wav'):
          file_path = os.path.join(train_material_dir, file)
          label = file.split('.')[0]  # Extract the label from the file name
          y.extend(list(label))  # Extend the label list for each digit
          feature = extract_features(file_path)
          X.append(feature)

  # Training the SVM model
  svm = SVC(kernel='rbf')
  svm.fit(X, y)

  # Model evaluation
  y_pred = svm.predict(X)
  accuracy = accuracy_score(y, y_pred)
  print("Accuracy:", accuracy)
  dump_model(svm)

def dump_model(svm):
  joblib.dump(svm, model_file)

temp_dir = 'temp'
def split_by_silence(file_path):
  os.makedirs(temp_dir, exist_ok=True)
  sox_template = Template('sox "${src}" "${dst}" silence 1 0.02 1% 1 0.02 1% : newfile : restart')
  sox_cmd = sox_template.substitute(src=file_path, dst=f'temp/%n.wav')
  subprocess.run(sox_cmd, shell=True)

def extract_features(file_path):
  audio, sr = librosa.load(file_path)
  # By using MFCC to extract features, the item in the audio could reduce from 5000+ to 20*30 2D array
  mfcc = librosa.feature.mfcc(y=audio, sr=sr)
  # By np.mean, it could reduce from 20*30 2D array to 20 in 1D array, which could be used as the input of SVM
  mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean across MFCC coefficients
  return mfcc_mean

def predict(file_path):
  split_by_silence(file_path)
  loaded_model = joblib.load(model_file)
  result = ''
  # only 5 digit in the audio file, but SOX is split by silence into 6 file
  for i in range(1, 6):
    filename = f"{str(i).zfill(2)}.wav"
    chunk_file_path = os.path.join(temp_dir, filename)
    X = extract_features(chunk_file_path)
    result += loaded_model.predict([X])[0]

  return result

train_model()
for file in os.listdir(data_dir):
  result = predict(os.path.join(data_dir, file))
  print(result, file)