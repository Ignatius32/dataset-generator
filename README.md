# 🤖 Pipeline Avanzado de Base de Datos Vectorial y Generación de Datasets DPO

Un pipeline de grado profesional para crear bases de datos vectoriales a partir de documentos y generar datasets de alta calidad para el entrenamiento de optimización de preferencias directas (DPO) de agentes de IA.

## 🌟 Características Principales

- 📄 **Soporte multi-formato**: PDFs, TXT, MD con segmentación inteligente
- 🌍 **Multilingüe**: Optimizado para español con respaldo en inglés
- 🔍 **Búsqueda vectorial**: Búsqueda rápida por similitud usando embeddings FAISS
- 🎯 **Generación DPO avanzada**: Pipeline de IA multi-etapa para calidad premium de datasets
- 🧠 **Generación inteligente de consultas**: 5 modos especializados (analítico, factual, conceptual, procedimental, comparativo)
- ⚖️ **Puntuación impulsada por IA**: Evaluación de calidad multi-dimensional para consultas y respuestas
- 🎚️ **Balanceado de dificultad**: Distribución automática a través de niveles de complejidad cognitiva
- 🔄 **Validación multi-etapa**: Validación cruzada y puntuación de calidad en cada paso
- 🏆 **Salida de grado profesional**: Datasets listos para producción para entrenar agentes de IA
- 🔗 **Diseño modular**: Fácil de extender y personalizar

## 🎭 Modos de Generación de Consultas

El generador avanzado crea consultas diversas usando prompts de IA especializados:

- **Analítico**: Análisis profundo, pensamiento crítico, relaciones causa-efecto
- **Factual**: Datos precisos, definiciones, información verificable
- **Conceptual**: Principios fundamentales, teorías, conceptos abstractos
- **Procedimental**: Procesos paso a paso, metodologías, aplicaciones prácticas
- **Comparativo**: Contrastes, similitudes, evaluaciones relativas

## 🎚️ Aseguramiento de Calidad

- **Puntuación multi-dimensional**: Evaluación de relevancia, dificultad y claridad
- **Filtrado de calidad**: Solo las mejores consultas y respuestas pasan al dataset final
- **Balanceado de dificultad**: Distribución automática entre niveles fácil/medio/difícil
- **Validación cruzada**: Verificación múltiple de coherencia y calidad

## 🚀 Instalación Rápida

### Prerrequisitos

- Python 3.8+
- Tesseract OCR (para procesamiento de PDFs)
- Clave API de GradienteSur

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd agentic-gob
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Instalar Tesseract OCR

**Windows:**
1. Descargar desde [GitHub de Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Instalar en `C:\Program Files\Tesseract-OCR\`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### 4. Configurar variables de entorno

Crear un archivo `.env` en el directorio raíz:

```env
GS_API_KEY=tu_clave_api_aqui
API_URL=https://api.gradientesur.com/functions/v1/embeddings
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
CHAT_API_URL=https://api.gradientesur.com/functions/v1/chat/completions
CHAT_MODEL=Qwen/Qwen3-1.7B
```

## 📖 Guía de Uso

### Pipeline Completo (Recomendado)

Ejecuta todo el pipeline desde documentos hasta dataset DPO:

```bash
python robust_pipeline.py --input_dir data/documents --num_samples 500
```

**Opciones del pipeline:**
- `--input_dir`: Directorio con documentos a procesar
- `--output_dir`: Directorio base de salida (por defecto: `data`)
- `--num_samples`: Número de muestras DPO a generar
- `--chunk_size`: Tamaño de segmentos de texto
- `--skip_vectorstore`: Omitir creación de base vectorial (usar existente)
- `--max_retries`: Reintentos máximos de API
- `--batch_size`: Tamaño de lote para API de embeddings

### Uso por Módulos

#### 1. Crear Base de Datos Vectorial

```bash
python src/create_vectorstore.py --input_dir data/documents --output_dir data/vectorstore
```

#### 2. Generar Dataset DPO

```python
from src.vector_store import VectorStore
from src.advanced_dataset_generator import AdvancedDatasetGenerator

# Cargar base vectorial
vector_store = VectorStore()
vector_store.load("data/vectorstore")

# Generar dataset
generator = AdvancedDatasetGenerator(vector_store)
dataset = generator.generate_dataset(num_samples=500)
```

## 📁 Estructura del Proyecto

```
agentic-gob/
├── data/                          # Datos del proyecto
│   ├── documents/                 # Documentos de entrada
│   ├── vectorstore/              # Base de datos vectorial
│   └── dpo_dataset.json          # Dataset DPO generado
├── src/                          # Código fuente
│   ├── __init__.py
│   ├── config.py                 # Configuración central
│   ├── document_processor.py     # Procesamiento de documentos
│   ├── advanced_pdf_processor.py # Procesador PDF avanzado
│   ├── embeddings_api.py        # Cliente API de embeddings
│   ├── vector_store.py          # Gestión de base vectorial
│   ├── advanced_chat_api.py     # Cliente API de chat
│   ├── advanced_dataset_generator.py # Generador DPO avanzado
│   └── create_vectorstore.py    # Script de creación vectorial
├── robust_pipeline.py            # Pipeline principal robusto
├── analyze_pdf.py               # Herramienta de análisis PDF
├── requirements.txt             # Dependencias Python
├── .env                        # Variables de entorno
├── .gitignore                  # Archivos ignorados por Git
└── README.md                   # Esta documentación
```

## ⚙️ Configuración Avanzada

### Parámetros del Procesador de Documentos

```python
# En src/config.py
CHUNK_SIZE = 1500           # Tamaño de segmento (caracteres)
CHUNK_OVERLAP = 300         # Solapamiento entre segmentos
MAX_TOKENS_PER_CHUNK = 1024 # Límite de tokens por segmento
MIN_CHUNK_SIZE = 100        # Tamaño mínimo de segmento
```

### Parámetros de Generación de Dataset

```python
QUERIES_PER_CHUNK = 5              # Consultas iniciales por segmento
CANDIDATE_QUERIES_PER_CHUNK = 3    # Mejores consultas seleccionadas
MAX_RETRIEVED_DOCS = 7             # Documentos para contexto de respuesta
```

### Configuración de API

```python
MAX_RETRIES = 5                # Reintentos máximos
BASE_RETRY_DELAY = 2.0        # Delay base entre reintentos
REQUEST_TIMEOUT = 60          # Timeout de peticiones
```

## 🔧 Herramientas Incluidas

### Análisis de PDF

Analiza la estructura y contenido de archivos PDF:

```bash
python analyze_pdf.py
```

### Procesamiento Robusto de PDFs

El sistema incluye un procesador PDF avanzado que maneja:
- ✅ PDFs rotados (90°, 180°, 270°)
- ✅ PDFs basados en imágenes (OCR automático)
- ✅ PDFs con texto extraíble
- ✅ Múltiples estrategias de extracción
- ✅ Limpieza inteligente de texto

## 📊 Formato del Dataset DPO

El dataset generado sigue el formato estándar DPO:

```json
{
  "dataset_info": {
    "generation_timestamp": "2025-06-01T14:12:19.885494",
    "generator_version": "AdvancedDatasetGenerator_v1.0",
    "total_samples": 500,
    "quality_settings": {
      "min_query_score": 0.7,
      "min_response_score": 0.75,
      "min_answer_quality_score": 0.2
    }
  },
  "samples": [
    {
      "query": "¿Cuáles son los principales beneficios...",
      "chosen": "Los principales beneficios incluyen...",
      "rejected": "Algunos beneficios son...",
      "context": "Texto del documento relevante...",
      "metadata": {
        "source": "documento.pdf",
        "difficulty": "medium",
        "query_mode": "analytical",
        "scores": {...}
      }
    }
  ]
}
```

## 🔍 Solución de Problemas

### Error: "No se puede encontrar Tesseract"

```bash
# Verificar instalación
tesseract --version

# Windows: Agregar al PATH
# C:\Program Files\Tesseract-OCR\

# O especificar ruta en analyze_pdf.py
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Error: "API Key no válida"

1. Verificar que el archivo `.env` existe
2. Confirmar que `GS_API_KEY` está configurada correctamente
3. Verificar que la clave no tiene espacios extra

### PDFs no se procesan correctamente

1. Usar `python analyze_pdf.py` para diagnosticar
2. Verificar que Tesseract está instalado para OCR
3. El sistema probará múltiples métodos automáticamente

### Memoria insuficiente

1. Reducir `batch_size` en el pipeline
2. Reducir `CHUNK_SIZE` en configuración
3. Procesar menos documentos a la vez

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit de cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para detalles.

## 🆘 Soporte

Para problemas y preguntas:

1. Revisar esta documentación
2. Buscar en issues existentes
3. Crear nuevo issue con:
   - Descripción del problema
   - Pasos para reproducir
   - Logs de error
   - Configuración del sistema

## 🔄 Actualizaciones

### v1.0.0 (Actual)
- ✅ Pipeline completo documentos → vectorstore → DPO
- ✅ Procesamiento avanzado de PDFs con rotación
- ✅ Generación inteligente de consultas (5 modos)
- ✅ Puntuación de calidad multi-dimensional
- ✅ Soporte completo para español
- ✅ Manejo robusto de errores
- ✅ Documentación completa

---

**¿Listo para crear datasets de IA de clase mundial? 🚀**

Ejecuta `python robust_pipeline.py` y observa cómo tus documentos se transforman en un dataset DPO de alta calidad para entrenar agentes de IA inteligentes.
