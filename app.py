import streamlit as st
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tempfile import NamedTemporaryFile
import pickle
from tensorflow.keras.models import load_model
import wfdb
from tempfile import TemporaryDirectory
import os

st.set_page_config(
        page_title="Analizador ECG",
        page_icon="❤️",
        layout="wide",
        menu_items={
            "About": "Visualizador de ECG con detección de picos R y cálculo de frecuencia cardíaca.",
        },
    )
def load_ecg_model(model_upl, map_upl):
    # 1) Write the uploaded model .h5 to disk
    with NamedTemporaryFile(delete=False, suffix=".h5") as tmp_mod:
        tmp_mod.write(model_upl.read())
        model_path = tmp_mod.name

    # 2) Write the uploaded label map .pkl to disk
    with NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_map:
        tmp_map.write(map_upl.read())
        map_path = tmp_map.name

    # 3) Load from those temporary files
    model = load_model(model_path)
    with open(map_path, "rb") as f:
        label_to_index = pickle.load(f)

    index_to_label = {v: k for k, v in label_to_index.items()}
    return model, index_to_label

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
    
    uploaded_hea.seek(0)
    uploaded_mat.seek(0)

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

def main():
    
    st.title("📈 ECG Dashboard")
    st.markdown(
        """
        **Carga tu señal de ECG** en formato WFDB (`.hea`) + MATLAB (`.mat`).  
        Esta herramienta detecta picos R, calcula la frecuencia cardíaca y alerta
        si está fuera del rango normal (60–100 bpm).
        """
    )

    with st.sidebar:
        st.header("1) Modelo y mapeo de etiquetas")
        model_upl    = st.file_uploader("Modelo Keras (.h5)", type=["h5"])
        map_upl         = st.file_uploader("Label map (.pkl)", type=["pkl"])
        st.header("2) Señal WFD")
        hea_upl = st.file_uploader("Archivo header (.hea)", type="hea")
        mat_upl = st.file_uploader("Datos (.mat)", type="mat")
        st.caption("Debe subir ambos .hea + .dat del mismo record")

    if hea_upl and mat_upl:
        hea_upl.seek(0)
        mat_upl.seek(0)
        graficos(hea_upl, mat_upl)
    else:
        st.info("Sube primero tu header (.hea) y datos (.mat) para ver la señal.")

    if model_upl and map_upl and hea_upl and mat_upl:
        try:
            
            with NamedTemporaryFile(delete=False, suffix=".h5") as t:
                t.write(model_upl.read())
                model_path = t.name
            with NamedTemporaryFile(delete=False, suffix=".pkl") as t2:
                t2.write(map_upl.read())
                map_path = t2.name
            model = load_model(model_path)
            with open(map_path, "rb") as f:
                label_to_index = pickle.load(f)
            index_to_label = {v:k for k,v in label_to_index.items()}

            # 2) Save WFDB files into a temp folder with the same prefix
            with TemporaryDirectory() as tmpdir:
                prefix = os.path.join(tmpdir, "ecg_record")
                # write .hea and .dat
                open(prefix + ".hea", "wb").write(hea_upl.read())
                open(prefix + ".mat", "wb").write(mat_upl.read())

                # 3) Read the record
                record = wfdb.rdrecord(prefix)
                ecg = record.p_signal[:, 0]  # lead 0

            # 4) Plot the raw ECG
            st.subheader("📈 Señal ECG (lead 0)")
            fig, ax = plt.subplots(figsize=(8,2))
            times = np.arange(len(ecg)) / record.fs
            ax.plot(times, ecg)
            ax.set(xlabel="Tiempo (s)", ylabel="Amplitud (mV)")
            st.pyplot(fig)

            # 5) Prepare sample and predict
            sample = ecg[np.newaxis, ..., np.newaxis].astype(np.float32)
            pred_probs = model.predict(sample)[0]
            pred_idx   = np.argmax(pred_probs)
            pred_label = index_to_label[pred_idx]
            pred_conf  = pred_probs[pred_idx]

            # 6) Show result
            st.subheader("📊 Resultado de clasificación")
            st.markdown(f"**Clase:** {pred_label}  \n**Confianza:** {pred_conf:.2%}")

        except Exception as e:
            st.error(f"❌ Ocurrió un error: {e}")

    else:
        if not (model_upl and map_upl):
            st.info("Sube tu modelo (.h5) y tu mapeo de etiquetas (.pkl) para clasificar.")


if __name__ == "__main__":
    main()
