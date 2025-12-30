# Spectral Contrast Field Scanner
# Field-based, scan-oriented multi-dimensional sound visualiser
# Core metaphor: a persistent energy field + a scanning plane
# macOS / Apple Silicon compatible
import os
import streamlit as st

PORT = int(os.environ.get("PORT", 8501))

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
import csv

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Spectral Contrast Field Scanner", layout="wide")

st.title("Spectral Contrast Field Scanner")
st.caption("Field-based representation · scanning as perspective · MDR")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Field & Scan Controls")

    slice_count = st.number_input(
        "Number of time slices",
        min_value=20,
        max_value=200,
        value=51,
        step=1
    )

    scan_index = st.slider(
        "Scan position (time slice)",
        min_value=1,
        max_value=slice_count,
        value=1,
        step=1
    )

    st.subheader("Field colour")
    field_cmap = st.selectbox(
        "Field colormap",
        ["magma", "inferno", "plasma", "viridis"]
    )

    st.subheader("Silence handling")
    silence_db_thresh = st.slider("Silence threshold (dB)", -80.0, -20.0, -40.0, 1.0)

# -----------------------------
# File upload
# -----------------------------
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

# -----------------------------
# Analysis
# -----------------------------
if audio_file is not None:
    with st.spinner("Constructing field..."):
        # Load audio
        y, sr = librosa.load(audio_file, sr=None, mono=True)

        # Short-time Fourier transform magnitude
        S = np.abs(librosa.stft(y))

        # -----------------------------
        # Core dimensions
        # -----------------------------

        # Spectral contrast (identity / structure)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-9)

        # RMS energy (presence)
        rms = librosa.feature.rms(S=S)[0]
        rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

        # Spectral flatness (texture)
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flatness = (flatness - flatness.min()) / (flatness.max() - flatness.min() + 1e-9)

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        centroid = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-9)

        # Novelty / onset strength (change)
        novelty = librosa.onset.onset_strength(S=S, sr=sr)
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-9)

        # -----------------------------
        # Time quantisation
        # -----------------------------
        contrast_blocks = np.array_split(contrast, slice_count, axis=1)
        rms_blocks = np.array_split(rms, slice_count)
        flat_blocks = np.array_split(flatness, slice_count)
        cent_blocks = np.array_split(centroid, slice_count)
        nov_blocks = np.array_split(novelty, slice_count)

        rms_vals = np.array([b.mean() for b in rms_blocks])
        flat_vals = np.array([b.mean() for b in flat_blocks])
        cent_vals = np.array([b.mean() for b in cent_blocks])
        nov_vals = np.array([b.max() for b in nov_blocks])

        # -----------------------------
        # Persistent field construction
        # -----------------------------
        field_slices = []
        for i, block in enumerate(contrast_blocks):
            base = block.mean(axis=1)
            # texture subtly modulates the field
            base = base * (0.7 + 0.3 * flat_vals[i])
            field_slices.append(base)

        field = np.stack(field_slices, axis=1)

        # Energy and brightness modulate the field without creating it
        modulation = (0.5 + rms_vals) * (0.7 + 0.3 * cent_vals)
        field = field * modulation

        # Silence softens but does not erase the field
        silence_mask = rms_vals < 10 ** (silence_db_thresh / 20)
        field[:, silence_mask] *= 0.6

        # Final normalisation
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)

    # -----------------------------
    # Visual output
    # -----------------------------
    col1, col2 = st.columns([3, 1])

    # Frontal field view
    with col1:
        fig, ax = plt.subplots(figsize=(14, 4))
        img = ax.imshow(field, aspect='auto', origin='lower', cmap=field_cmap)
        ax.set_title("Frontal view · full field")
        ax.set_xlabel("Time (slices)")
        ax.set_ylabel("Spectral structure")
        ax.axvline(scan_index - 1, color='white', linewidth=1.5, alpha=0.9)
        fig.colorbar(img, ax=ax, fraction=0.025)
        st.pyplot(fig)

    # Scan slice (side/profile view)
    with col2:
        slice_img = field[:, scan_index - 1].reshape(-1, 1)
        fig2, ax2 = plt.subplots(figsize=(3, 4))
        ax2.imshow(slice_img, aspect='auto', origin='lower', cmap=field_cmap)
        ax2.set_title(f"Scan slice {scan_index}")
        ax2.set_xticks([])
        ax2.set_yticks([])
        st.pyplot(fig2)

    # -----------------------------
    # Export
    # -----------------------------
    st.subheader("Export")

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow([
        "slice",
        "rms_energy",
        "spectral_centroid",
        "spectral_flatness",
        "novelty"
    ])

    for i in range(slice_count):
        writer.writerow([
            i + 1,
            float(rms_vals[i]),
            float(cent_vals[i]),
            float(flat_vals[i]),
            float(nov_vals[i])
        ])

    st.download_button(
        "Download slice data (CSV)",
        data=csv_buffer.getvalue(),
        file_name="field_scan_data.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a WAV file to construct the field.")
