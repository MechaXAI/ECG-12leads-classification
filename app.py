import streamlit as st
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tempfile import NamedTemporaryFile

def calculate_heart_rate(signal, sampling_rate):
    """
    Detect R-peaks & compute average heart rate (bpm).
    """
    # nk.ecg_peaks returns (signals, info)
    signals, info = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
    r_peaks = info["ECG_R_Peaks"]
    # RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / sampling_rate
    heart_rate = 60 / rr_intervals.mean() if rr_intervals.size else np.nan
    return heart_rate, r_peaks

def graficos(uploaded_hea, uploaded_mat):
    """
    1) Parse sampling_rate from .hea  
    2) Load signal from .mat  
    3) Plot & annotate  
    """
    if not (uploaded_hea and uploaded_mat):
        return

    try:
        # — Save uploads to temp files —
        with NamedTemporaryFile(delete=False, suffix=".hea") as t1:
            t1.write(uploaded_hea.read())
            hea_path = t1.name
        with NamedTemporaryFile(delete=False, suffix=".mat") as t2:
            t2.write(uploaded_mat.read())
            mat_path = t2.name

        # — Read sampling rate from header —
        with open(hea_path, "r") as f:
            first_line = f.readline().strip().split()
        # WFDB header: <recname> <n_signals> <fs> ...
        sampling_rate = float(first_line[2])

        # — Load the .mat signal array —
        mat = loadmat(mat_path)
        st.write("**Variables en .mat:**", list(mat.keys()))
        signal = mat.get("val", None)
        if signal is None:
            raise ValueError("No 'val' variable found in .mat")
        # If 2D, pick first channel
        if signal.ndim > 1:
            signal = signal[0]
        signal = signal.flatten()

        # — Time axis in seconds for first N samples —
        N = min(len(signal), 2000)
        times = np.arange(N) / sampling_rate

        # — Plot raw ECG —
        st.subheader("ECG bruto (primeras {} muestras)".format(N))
        plt.figure(figsize=(8, 3))
        plt.plot(times, signal[:N])
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        st.pyplot(plt)

        # — Compute HR & R-peaks —
        hr, r_peaks = calculate_heart_rate(signal, sampling_rate)

        st.markdown(f"### Frecuencia cardíaca promedio: **{hr:.1f} lpm**")
        if hr < 60 or hr > 100:
            st.warning(f"⚠️ Fuera de rango normal (60–100 bpm): {hr:.1f} lpm")
        else:
            st.success("✅ Dentro de rango normal (60–100 bpm).")

        # — Overlay R-peaks —
        st.subheader("R-peaks detectados")
        plt.figure(figsize=(8, 3))
        plt.plot(times, signal[:N], label="ECG")
        # Only show peaks within the first N samples
        rp = r_peaks[r_peaks < N]
        plt.scatter(rp / sampling_rate, signal[rp], color="r", label="R-peaks")
        plt.xlabel("Tiempo (s)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error al procesar ECG: {e}")
        st.stop()


# ——— Main App ———

def main():
    st.set_page_config(
        page_title="Analizador ECG",
        page_icon="❤️",
        layout="wide",
        menu_items={
            "About": "Visualizador de ECG con detección de picos R y cálculo de frecuencia cardíaca.",
        },
    )

    st.title("📈 ECG Dashboard")
    st.markdown(
        """
        **Carga tu señal de ECG** en formato WFDB (`.hea`) + MATLAB (`.mat`).  
        Esta herramienta detecta picos R, calcula la frecuencia cardíaca y alerta
        si está fuera del rango normal (60–100 bpm).
        """
    )

    with st.sidebar:
        st.header("Sube tus archivos")
        hea = st.file_uploader("Archivo header (.hea)", type="hea")
        mat = st.file_uploader("Datos (.mat)", type="mat")

    graficos(hea, mat)


if __name__ == "__main__":
    main()
