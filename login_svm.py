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
          X.extend(feature)

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
  mfcc = librosa.feature.mfcc(y=audio, sr=sr)
  mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean across MFCC coefficients
  X = np.array([mfcc_mean])  # Convert to a 2D array for prediction
  return X

def predict(file_path):
  split_by_silence(file_path)
  loaded_model = joblib.load(model_file)
  result = ''
  for i in range(1, 6):
    filename = f"{str(i).zfill(2)}.wav"
    chunk_file_path = os.path.join(temp_dir, filename)
    X = extract_features(chunk_file_path)
    result += loaded_model.predict(X)[0]

  return result

train_model()
for file in os.listdir(data_dir):
  result = predict(os.path.join(data_dir, file))
  print(result, file)