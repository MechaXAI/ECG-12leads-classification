import streamlit as st
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.io import loadmat
import wfdb
from tempfile import NamedTemporaryFile

# Función para calcular la frecuencia cardíaca
def calculate_heart_rate(signal):
    # Detectar los picos R usando nk.ecg_peaks
    r_peaks = nk.ecg_peaks(signal)

    # Extraer los índices de los picos R
    r_peak_indices = r_peaks[1]['ECG_R_Peaks']

    # Calcular los intervalos RR (tiempo entre los picos R)
    rr_intervals = np.diff(r_peak_indices) / 1000  # Convertir de ms a segundos

    # Calcular la frecuencia cardíaca (lpm)
    heart_rate = 60 / rr_intervals.mean()  # Promedio de la frecuencia
    return heart_rate, r_peaks

def graficos(uploaded_hea_file,uploaded_mat_file):
    # Si los archivos .hea y .mat son cargados
    if uploaded_hea_file is not None and uploaded_mat_file is not None:
        try:
            # Guardar temporalmente el archivo .hea
            with NamedTemporaryFile(delete=False, suffix='.hea') as temp_hea_file:
                temp_hea_file.write(uploaded_hea_file.read())
                temp_hea_file.close()  # Cerrar el archivo para que pueda ser leído

            # Guardar temporalmente el archivo .mat
            with NamedTemporaryFile(delete=False, suffix='.mat') as temp_mat_file:
                temp_mat_file.write(uploaded_mat_file.read())
                temp_mat_file.close()  # Cerrar el archivo para que pueda ser leído

            # Leer el archivo .mat usando scipy.io.loadmat
            mat_data = loadmat(temp_mat_file.name)
        
            # Inspeccionar las variables en el archivo .mat
            st.write("### Variables en el archivo .mat:")
            st.write(mat_data.keys())  # Mostrar las claves del archivo .mat
            
            # Acceder a la variable 'val' que contiene los datos
            signal_array = mat_data['val']  # Cambiar 'val' si es necesario
            
            # Si hay múltiples señales, seleccionamos la primera
            if isinstance(signal_array, np.ndarray):
                signal_array = signal_array[0]  # Tomar solo la primera señal si hay más de una

            # Mostrar la señal de ECG
            st.write("### Señal de ECG")
            st.line_chart(signal_array[:1000])  # Mostrar las primeras 1000 muestras

            # Calcular la frecuencia cardíaca
            heart_rate, r_peaks = calculate_heart_rate(signal_array)
        
            # Mostrar la frecuencia cardíaca
            st.write(f"### Frecuencia Cardíaca Promedio: {heart_rate:.2f} lpm")

            # Alerta de frecuencia cardíaca fuera del rango
            if heart_rate < 60 or heart_rate > 100:
                st.warning(f"**ALERTA**: La frecuencia cardíaca está fuera del rango normal (60-100 lpm): {heart_rate:.2f} lpm")
            else:
                st.success(f"La frecuencia cardíaca está dentro del rango normal: {heart_rate:.2f} lpm")

            # Visualización de los picos R
            st.write("### Visualización de los Picos R")
            plt.figure(figsize=(10, 6))
            plt.plot(signal_array[:1000], label="Señal ECG")  # Ajusta el número de muestras
            plt.scatter(r_peaks[1]['ECG_R_Peaks'], signal_array[r_peaks[1]['ECG_R_Peaks']], color='r', label='Picos R')
            plt.title("Señal ECG con Picos R")
            plt.xlabel("Tiempo (segundos)")
            plt.ylabel("Amplitud")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error al leer el archivo .mat o .hea de ECG: {e}")
            st.stop()

def main():

    st.set_page_config(page_title="Miraflores Respira", page_icon="🌬️", layout="wide", menu_items={
        'About': "MirafloresRespira – 📊 Monitoreo de Calidad del Aire en Miraflores, Lima, Perú | Dashboard en \
        Tiempo Real 🌍. Consulta en tiempo real los niveles de contaminación en el aire de Miraflores: \
        PM2.5, PM10, CO, NO2, O3, y otros indicadores clave para la salud. Conoce cómo la calidad del \
        aire impacta tu bienestar y toma decisiones informadas para protegerte a ti y a tu familia. \
        🔍 Actualizaciones continuas y alertas para que estés siempre al tanto. Ideal para residentes, visitantes, y profesionales en salud ambiental"
    })

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://ibb.co/d70C1dn);
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Miraflores Respira 🍃")

    with st.sidebar:
        st.markdown(
            """
                <style>
                .rounded-image {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 150px;
                    height: 150px;
                    border-radius: 50%; 
                    object-fit: cover;
                }
                </style>
                """,
            unsafe_allow_html=True
        )
        st.markdown(
            f'<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTIwzSGjLmVJXhHO2r8MmF4mzvpcIyWoyUyg&s" class="rounded-image">',
            unsafe_allow_html=True
        )
        st.title("Análisis de Electrocardiograma (ECG)")
        st.write("""
            Esta aplicación permite visualizar señales de ECG, detectar los picos R, 
            calcular la frecuencia cardíaca y mostrar alertas si la frecuencia cardíaca 
            está fuera del rango normal (60-100 lpm).
        """)

        # Subir archivo de ECG
        uploaded_hea_file = st.file_uploader("Cargar archivo .hea", type=["hea"])
        uploaded_mat_file = st.file_uploader("Cargar archivo .mat", type=["mat"])


    graficos(uploaded_hea_file, uploaded_mat_file)


if __name__ == "__main__":

    main()