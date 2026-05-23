import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import cv2
import pywt
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from scipy.signal import spectrogram
from skimage.feature import hog

# =========================
# CONFIG
# =========================

app = Flask(__name__)

UPLOAD_FOLDER = "static/generated"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load("models/model.sav")
scaler = joblib.load("models/scaler.pkl")

fs = 5000
duration = 1
t = np.linspace(0, duration, fs)

# =========================
# FEATURE FUNCTIONS
# =========================

def calculate_thd(signal):
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    fundamental_idx = np.argmin(np.abs(freqs - 50))
    V1 = fft_vals[fundamental_idx]

    harmonic_power = np.sum(fft_vals[2*fundamental_idx:10*fundamental_idx]**2)
    thd = np.sqrt(harmonic_power) / (V1 + 1e-10)
    return thd


def compute_dvneg_dt(Va, Vb, Vc):
    a = np.exp(1j * 2*np.pi/3)
    Vneg = (Va + a**2 * Vb + a * Vc) / 3
    Vneg_mag = np.abs(Vneg)
    dvneg_dt = np.gradient(Vneg_mag)
    return dvneg_dt


# =========================
# SIGNAL GENERATION
# =========================

def build_feature_signal(dv_input, vthd_input, cthd_input):

    Va = np.sin(2*np.pi*50*t)
    Vb = np.sin(2*np.pi*50*t - 2*np.pi/3)
    Vc = np.sin(2*np.pi*50*t + 2*np.pi/3)

    if dv_input > 0.15:  # Islanding condition

        Va[int(0.5*fs):] *= 0.6
        Vb[int(0.5*fs):] *= 0.5
        Vc[int(0.5*fs):] *= 0.8

        Va[int(0.6*fs):] = np.sin(2*np.pi*52*t[int(0.6*fs):])

        Va += 0.2*np.sin(2*np.pi*150*t)
        Vb += 0.15*np.sin(2*np.pi*150*t)
        Vc += 0.1*np.sin(2*np.pi*150*t)

        Ia = Va + 0.1*np.random.randn(len(t))

    else:  # Non-Islanding

        Va += 0.05*np.sin(2*np.pi*150*t)
        Vb += 0.05*np.sin(2*np.pi*150*t)
        Vc += 0.05*np.sin(2*np.pi*150*t)

        Ia = Va + 0.02*np.random.randn(len(t))

    dvneg_dt = compute_dvneg_dt(Va, Vb, Vc)
    voltage_thd = calculate_thd(Va)
    current_thd = calculate_thd(Ia)

    feature_signal = dvneg_dt + voltage_thd + current_thd
    return feature_signal


# =========================
# IMAGE GENERATION
# =========================

def generate_images(signal):

    signal = (signal - np.mean(signal)) / (np.std(signal)+1e-10)

    # Spectrogram
    f, tt, Sxx = spectrogram(signal, fs)
    fig1, ax1 = plt.subplots(figsize=(3,3))
    ax1.pcolormesh(tt, f, 10*np.log10(Sxx+1e-10))
    ax1.axis('off')

    spec_filename = "generated/spectrogram.png"
    spec_path = os.path.join("static", spec_filename)
    fig1.savefig(spec_path, bbox_inches='tight', pad_inches=0, dpi=80)
    plt.close(fig1)

    # Scalogram
    scales = np.arange(1,64)
    coeffs, _ = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)
    fig2, ax2 = plt.subplots(figsize=(3,3))
    ax2.imshow(np.abs(coeffs), aspect='auto', cmap='jet')
    ax2.axis('off')

    scal_filename = "generated/scalogram.png"
    scal_path = os.path.join("static", scal_filename)
    fig2.savefig(scal_path, bbox_inches='tight', pad_inches=0, dpi=80)
    plt.close(fig2)

    return spec_filename, scal_filename


# =========================
# PREDICTION
# =========================

def predict_from_image(image_filename):

    full_path = os.path.join("static", image_filename)

    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))

    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')

    features = scaler.transform([features])

    probs = model.predict_proba(features)[0]

    class_index = np.argmax(probs)
    label_map = {
        0: "Islanding",
        1: "Non_Islanding"
    }
    numeric_label = model.classes_[class_index]
    label = label_map[numeric_label]
    confidence = round(float(np.max(probs)) * 100, 2)

    return label, confidence, probs


# =========================
# ROUTES
# =========================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    dvdt = float(request.form['dvdt'])
    vthd = float(request.form['vthd'])
    cthd = float(request.form['cthd'])

    feature_signal = build_feature_signal(dvdt, vthd, cthd)

    spec_img, scal_img = generate_images(feature_signal)

    # Spectrogram Prediction
    label_spec, conf_spec, prob_spec = predict_from_image(spec_img)

    # Scalogram Prediction
    label_scal, conf_scal, prob_scal = predict_from_image(scal_img)

    # Spectrogram description
    description_spec = (
        "🔴 Grid Disconnection Detected (Islanding Condition)"
        if label_spec == "Islanding"
        else "🟢 Grid Operating Normally (Non-Islanding)"
    )

    # Scalogram description
    description_scal = (
        "🔴 Grid Disconnection Detected (Islanding Condition)"
        if label_scal == "Islanding"
        else "🟢 Grid Operating Normally (Non-Islanding)"
    )

    return render_template("result.html",
                           waveform=feature_signal[:500].tolist(),
                           spectrogram=spec_img,
                           scalogram=scal_img,

                           prediction_spec=label_spec,
                           confidence_spec=conf_spec,
                           description_spec=description_spec,
                           prob_islanding_spec=round(prob_spec[0],3),
                           prob_non_islanding_spec=round(prob_spec[1],3),

                           prediction_scal=label_scal,
                           confidence_scal=conf_scal,
                           description_scal=description_scal,
                           prob_islanding_scal=round(prob_scal[0],3),
                           prob_non_islanding_scal=round(prob_scal[1],3)
                           )


if __name__ == "__main__":
    app.run()
