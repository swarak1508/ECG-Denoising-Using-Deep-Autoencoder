import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from tensorflow.keras import layers, Model


# ---------------------------------------------------------
# File path
# ---------------------------------------------------------

FILE_PATH = "scientisst_chest.edf"


# ---------------------------------------------------------
# Load and preprocess ECG data
# ---------------------------------------------------------

def load_and_preprocess(path):

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    ch_names = raw.ch_names
    data = raw.get_data()

    # Clean ECG from gel electrode
    s1_clean = data[ch_names.index("ecg:gel")]

    # Noisy ECG from dry electrode
    s2_noisy = data[ch_names.index("ecg:dry")]

    # Chest accelerometer
    acc_x = data[ch_names.index("acc_chest:x")]
    acc_y = data[ch_names.index("acc_chest:y")]
    acc_z = data[ch_names.index("acc_chest:z")]

    # Calculate accelerometer magnitude
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Band-pass filter: 0.5–40 Hz
    fs = int(raw.info["sfreq"])

    nyq = 0.5 * fs
    b, a = butter(3, [0.5 / nyq, 40 / nyq], btype="band")

    s1_clean = filtfilt(b, a, s1_clean)
    s2_noisy = filtfilt(b, a, s2_noisy)

    # Standardization
    scaler_input = StandardScaler()
    scaler_target = StandardScaler()

    # Input: noisy ECG + accelerometer magnitude
    X_raw = np.stack([s2_noisy, acc_mag], axis=1)

    X_scaled = scaler_input.fit_transform(X_raw)

    # Target: clean ECG
    y_scaled = scaler_target.fit_transform(
        s1_clean.reshape(-1, 1)
    )

    return X_scaled, y_scaled, fs


# ---------------------------------------------------------
# Create windows
# ---------------------------------------------------------

def create_windows(X, y, window_size=512):

    n_windows = len(X) // window_size

    X_win = X[:n_windows * window_size].reshape(
        -1, window_size, 2
    )

    y_win = y[:n_windows * window_size].reshape(
        -1, window_size, 1
    )

    return X_win, y_win


# ---------------------------------------------------------
# U-Net model
# ---------------------------------------------------------

def build_unet(window_size=512):

    inputs = layers.Input(shape=(window_size, 2))

    # Encoder
    c1 = layers.Conv1D(
        32, 3, activation="relu", padding="same"
    )(inputs)

    p1 = layers.MaxPooling1D(2)(c1)

    c2 = layers.Conv1D(
        64, 3, activation="relu", padding="same"
    )(p1)

    p2 = layers.MaxPooling1D(2)(c2)

    # Bottleneck
    b1 = layers.Conv1D(
        128, 3, activation="relu", padding="same"
    )(p2)

    # Decoder
    u1 = layers.UpSampling1D(2)(b1)

    m1 = layers.Concatenate()([u1, c2])

    c3 = layers.Conv1D(
        64, 3, activation="relu", padding="same"
    )(m1)

    u2 = layers.UpSampling1D(2)(c3)

    m2 = layers.Concatenate()([u2, c1])

    c4 = layers.Conv1D(
        32, 3, activation="relu", padding="same"
    )(m2)

    outputs = layers.Conv1D(
        1, 3, activation="linear", padding="same"
    )(c4)

    return Model(inputs, outputs)


# ---------------------------------------------------------
# Custom correlation loss
# ---------------------------------------------------------

def correlation_loss(y_true, y_pred):

    x = y_true - tf.reduce_mean(y_true)
    y = y_pred - tf.reduce_mean(y_pred)

    corr = tf.reduce_sum(x * y) / (
        tf.sqrt(tf.reduce_sum(x**2)) *
        tf.sqrt(tf.reduce_sum(y**2)) + 1e-8
    )

    return 1 - corr


def total_loss(y_true, y_pred):

    mse = tf.reduce_mean(
        tf.square(y_true - y_pred)
    )

    return mse + (0.5 * correlation_loss(y_true, y_pred))


# ---------------------------------------------------------
# SNR calculation
# ---------------------------------------------------------

def calculate_snr(signal, noise):

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    return 10 * np.log10(
        signal_power / noise_power
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    if not os.path.exists(FILE_PATH):
        print(
            f"Dataset not found: {FILE_PATH}"
        )
        print(
            "Place the EDF file in the same folder "
            "as main.py."
        )
        exit()

    print("Loading and preprocessing ECG data...")

    X_data, y_data, fs = load_and_preprocess(
        FILE_PATH
    )

    X_train, y_train = create_windows(
        X_data, y_data
    )

    print("Input shape:", X_train.shape)
    print("Target shape:", y_train.shape)
    print("Sampling frequency:", fs, "Hz")

    # Build model
    model = build_unet()

    model.compile(
        optimizer="adam",
        loss=total_loss
    )

    # Train
    print("\nStarting training...")

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Select a sample
    sample_idx = np.random.randint(
        0, len(X_train)
    )

    prediction = model.predict(
        X_train[sample_idx:sample_idx + 1],
        verbose=0
    )

    # Extract signals
    ground_truth = y_train[
        sample_idx, :, 0
    ]

    reconstructed = prediction[
        0, :, 0
    ]

    noisy_input = X_train[
        sample_idx, :, 0
    ]

    # Pearson correlation
    correlation, _ = pearsonr(
        ground_truth,
        reconstructed
    )

    overlap_percentage = correlation * 100

    # SNR
    snr_initial = calculate_snr(
        ground_truth,
        ground_truth - noisy_input
    )

    snr_final = calculate_snr(
        ground_truth,
        ground_truth - reconstructed
    )

    snr_improvement = (
        snr_final - snr_initial
    )

    print("\n------------------------------")
    print("ECG DENOISING RESULTS")
    print("------------------------------")
    print(
        f"Pearson Correlation: "
        f"{correlation:.4f}"
    )
    print(
        f"Morphological Overlap: "
        f"{overlap_percentage:.2f}%"
    )
    print(
        f"Initial SNR: "
        f"{snr_initial:.2f} dB"
    )
    print(
        f"Denoised SNR: "
        f"{snr_final:.2f} dB"
    )
    print(
        f"SNR Improvement: "
        f"{snr_improvement:.2f} dB"
    )

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(
        noisy_input,
        label="Noisy ECG (Dry Electrode)"
    )
    plt.legend()
    plt.title("Input ECG")

    plt.subplot(3, 1, 2)
    plt.plot(
        X_train[sample_idx, :, 1],
        label="Accelerometer Magnitude"
    )
    plt.legend()
    plt.title("Motion Reference")

    plt.subplot(3, 1, 3)
    plt.plot(
        ground_truth,
        label="Ground Truth (Gel Electrode)"
    )

    plt.plot(
        reconstructed,
        label="Reconstructed ECG"
    )

    plt.legend()
    plt.title("ECG Denoising Result")

    plt.tight_layout()
    plt.show()
