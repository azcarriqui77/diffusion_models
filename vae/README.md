# Fichero `blocks.py`

## Clase `DownBlock`

Este código define una clase llamada `DownBlock`, que implementa un bloque de convolución y atención utilizado comúnmente en arquitecturas de modelos generativos o de segmentación, como redes neuronales de tipo **U-Net**. Aquí está el desglose de lo que hace cada parte:

---

### **Descripción general del bloque**
- Este bloque combina tres elementos clave:
  1. **Bloques ResNet**: Realizan convoluciones profundas con embeddings de tiempo opcionales.
  2. **Bloques de Atención**: Aplican mecanismos de atención (atención normal y cruzada, si se especifica) para capturar relaciones espaciales o condicionales en los datos.
  3. **Downsampling**: Reduce la resolución espacial de los mapas de características para permitir una mayor abstracción.

El objetivo del bloque es procesar y condensar características mientras preserva información clave a través de convoluciones residuales, normalizaciones y atención.

---

### **Detalles clave del código**

#### **Constructor (`__init__`)**
El constructor inicializa todos los componentes necesarios para las operaciones:

1. **ResNet Blocks:**
   - `resnet_conv_first` y `resnet_conv_second`: Convoluciones residuales para procesar las características de entrada y salida. 
   - Opcionalmente utiliza un embedding de tiempo (`t_emb_dim`) para modelar tareas temporales, como en redes de difusión.

2. **Attention Blocks:**
   - Condicionalmente agrega capas de atención:
     - `attentions` (atención normal) para capturar relaciones internas en el mapa de características.
     - `cross_attentions` (atención cruzada) para fusionar información del contexto adicional, como un embedding de texto o una condición.
   - Normaliza las características antes de aplicar atención mediante `GroupNorm`.

3. **Residual Connections:**
   - `residual_input_conv` aplica una convolución de ajuste para añadir las conexiones residuales de entrada directamente a la salida de cada capa.

4. **Downsampling:**
   - `down_sample_conv`: Realiza una reducción de resolución espacial mediante una convolución con stride=2 si `down_sample` es `True`, o actúa como identidad si no.

---

#### **Método `forward`**
Procesa los datos de entrada en pasos:

1. **Iteración por capas (`self.num_layers`):**
   Cada capa realiza:
   - **ResNet Block**:
     1. Aplicar `resnet_conv_first`.
     2. Sumar el embedding de tiempo (`t_emb`) si está habilitado.
     3. Aplicar `resnet_conv_second`.
     4. Añadir conexión residual (`residual_input_conv`).
   - **Attention Block** (si `attn` es `True`):
     - Normaliza el mapa de características y lo reestructura para el mecanismo de atención.
     - Aplica atención multi-cabeza (`MultiheadAttention`) y reestructura los resultados al formato original.
     - Suma el resultado de la atención al mapa de características.
   - **Cross Attention Block** (si `cross_attn` es `True`):
     - Similar al bloque de atención, pero en este caso, se utiliza un contexto externo (`context`), como embeddings de texto.

2. **Downsampling:**
   - Reduce la resolución espacial del mapa de características usando `down_sample_conv`.

---

### **Parámetros importantes**

- **`in_channels` y `out_channels`**: Dimensiones de entrada y salida del bloque.
- **`t_emb_dim`**: Dimensión del embedding de tiempo para tareas condicionales dependientes del tiempo.
- **`num_heads`**: Número de cabezas en la atención multi-cabeza.
- **`cross_attn` y `context_dim`**: Determinan si se incluye un mecanismo de atención cruzada y la dimensión del contexto adicional.
- **`norm_channels`**: Número de grupos usados en `GroupNorm`, una técnica de normalización.

---

### **Resumen del flujo de datos**

1. **Entrada**: Tensor `x` de forma `(batch_size, in_channels, height, width)`, opcionalmente con `t_emb` (embedding de tiempo) y `context` (para atención cruzada).
2. **Procesamiento**:
   - Las capas procesan las características usando convoluciones residuales.
   - Aplica atención normal y/o cruzada, según los parámetros.
3. **Salida**: Tensor procesado de menor resolución si `down_sample` es `True`.

---
