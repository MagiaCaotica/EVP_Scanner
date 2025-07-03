# SPIRICOM EVP Capture System

Un sistema avanzado para la captura y análisis de Electronic Voice Phenomena (EVP), implementando técnicas inspiradas en el protocolo SPIRICOM Mark IV con mejoras modernas de procesamiento de señal.

## 🔍 Descripción

Este proyecto es una interfaz completa para la captura de EVPs que combina:
- Generación de tonos SPIRICOM específicos (13 frecuencias armónicamente relacionadas)
- Procesamiento de audio con emulación de características analógicas (hiss de cinta, saturación, wow/flutter)
- Visualización en tiempo real con detección de anomalías espectrales
- Exportación de resultados en múltiples formatos (WAV, PNG, MP4)

El sistema está diseñado para maximizar la posibilidad de capturar fenómenos de voz electrónica mientras proporciona herramientas avanzadas para su análisis.

## ✨ Características Principales

### 🎛️ Procesamiento de Audio Avanzado
- **Tonos SPIRICOM**: 13 frecuencias específicas (131Hz a 871Hz) generadas con modulación
- **Emulación de cinta analógica**:
  - Hiss de cinta sutil
  - Saturación por soft-clipping
  - Efectos wow/flutter para emular variaciones de velocidad
- **Filtrado inteligente**:
  - Filtro de muesca a 60Hz (elimina interferencia eléctrica)
  - Filtro paso banda (200Hz-4kHz) optimizado para voz
  - Reducción de ruido espectral adaptable

### 📊 Visualización en Tiempo Real
- Espectrograma con:
  - Detección automática de anomalías (resaltadas en colores)
  - Escala de frecuencia personalizada
  - Marcadores de tiempo
- Indicadores de:
  - Frecuencia predominante
  - Estado de grabación
  - Nivel de señal

### 💾 Exportación de Resultados
- Formatos soportados:
  - Audio WAV (procesado)
  - Imagen PNG del espectrograma
  - Video MP4 (espectrograma + audio)
- Metadatos automáticos con timestamp
- Organización en directorio `evp_sessions`

## 🛠️ Configuración Técnica

### 📋 Requisitos
- Python 3.8+
- Bibliotecas principales:
  - `numpy`, `pyaudio`, `librosa`, `pygame`, `moviepy`, `scipy`
- Recomendado: Entorno virtual (venv o conda)

### ⚙️ Parámetros Ajustables
```python
# Configuración de audio
SAMPLE_RATE = 44100       # Frecuencia de muestreo (Hz)
CHUNK = 2048              # Tamaño del buffer
MAX_DURATION = 300        # Duración máxima por sesión (segundos)

# Tonos SPIRICOM
TONE_FREQUENCIES = [131, 192, 241, 296, 364, 422, 483, 534, 587, 643, 704, 767, 871]

# Emulación analógica
TAPE_HISS_AMPLITUDE = 0.003
SATURATION_THRESHOLD = 0.6
WOW_FLUTTER_RATE = 0.5    # Hz
WOW_FLUTTER_DEPTH = 0.002 # 0.2%

# Detección de anomalías
ANOMALY_THRESHOLD_MULTIPLIER = 3.0
ANOMALY_COLOR_HIGH = (255, 0, 0)    # Rojo
ANOMALY_COLOR_MEDIUM = (255, 165, 0) # Naranja
ANOMALY_COLOR_LOW = (255, 255, 0)   # Amarillo
