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

## Clase `MidBlock`

Esta clase define un **bloque intermedio (MidBlock)**, que se utiliza dentro de arquitecturas como las VAE para procesar características a través de bloques residuales y mecanismos de atención. A continuación, explicaré paso a paso cómo está diseñado y qué hace cada parte del código:

---

### **Estructura general**

El bloque intermedio (`MidBlock`) tiene la siguiente secuencia de operaciones:
1. **Bloque ResNet con embedding temporal (`t_emb`)**.
2. **Mecanismo de atención (opcionalmente atención cruzada)**.
3. **Otro bloque ResNet con embedding temporal**.

Estas operaciones se repiten varias veces dependiendo de `num_layers`.

---

### **Parámetros del constructor**

```python
def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
```

1. **`in_channels`**: Número de canales del tensor de entrada.
2. **`out_channels`**: Número de canales del tensor de salida.
3. **`t_emb_dim`**: Dimensión del vector de embedding temporal. Si se usa, introduce información temporal en el modelo.
4. **`num_heads`**: Número de cabezas en las capas de atención (multi-head attention).
5. **`num_layers`**: Número de veces que se repiten los bloques ResNet y atención.
6. **`norm_channels`**: Número de grupos en la normalización por grupos (`GroupNorm`).
7. **`cross_attn`**: Booleano que indica si se usará **atención cruzada** (útil para incluir información contextual adicional, como texto en modelos de texto a imagen).
8. **`context_dim`**: Dimensión del vector de contexto para la atención cruzada. Requerido si `cross_attn=True`.

---

### **Componentes principales**

#### 1. **Bloques ResNet**
- Los bloques ResNet consisten en:
  - Normalización (`GroupNorm`).
  - Activación no lineal (`SiLU`).
  - Convoluciones 2D.
- Hay dos bloques ResNet en cada iteración:
  - **`resnet_conv_first`**: Primera convolución del bloque ResNet.
  - **`resnet_conv_second`**: Segunda convolución del bloque ResNet.
  - **`residual_input_conv`**: Convolución para procesar la conexión residual.

#### 2. **Embedding temporal (`t_emb`)**
- Si se proporciona `t_emb_dim`, el modelo incluye un término adicional que incorpora un vector de embedding temporal a través de una capa lineal y `SiLU`.

#### 3. **Mecanismos de atención**
- Hay dos tipos de atención:
  - **Atención estándar**: Un mecanismo `MultiheadAttention` procesa las características espaciales del tensor.
  - **Atención cruzada (opcional)**: Procesa un vector de contexto (por ejemplo, texto en aplicaciones multimodales).
  - **Normalización previa a la atención**: Antes de aplicar atención, se normalizan las características usando `GroupNorm`.

#### 4. **Cross-attention**
- Si `cross_attn=True`, se incluyen módulos adicionales para procesar el contexto:
  - **`context_proj`**: Proyecta el vector de contexto a la dimensión de las características espaciales.
  - **`cross_attentions`**: Aplica atención cruzada entre las características y el contexto.

---

### **Método `forward`**

El método `forward` define cómo se aplica el bloque intermedio al tensor de entrada.

```python
def forward(self, x, t_emb=None, context=None):
```

1. **Primer bloque ResNet:**
   - Procesa el tensor de entrada con el primer bloque ResNet.
   - Si se usa `t_emb`, se suma al tensor procesado.

2. **Iteración sobre los bloques de atención y ResNet:**
   - Para cada capa (\(i\)) dentro de las \(num\_layers\):
     - **Atención estándar**:
       - Se aplica normalización (`attention_norms`) y atención (`attentions`) al tensor.
       - La salida se suma al tensor original (\(residual\)).
     - **Atención cruzada (opcional)**:
       - Si `cross_attn=True`, se proyecta el vector de contexto (`context_proj`) y se aplica atención cruzada (`cross_attentions`).
       - La salida también se suma al tensor.
     - **Segundo bloque ResNet**:
       - Procesa las características con un bloque ResNet adicional.
       - Se suma al tensor procesado.

3. **Salida:**
   - El tensor procesado después de pasar por todas las capas se devuelve como salida.

---

### **Flujo simplificado del bloque**

Para cada iteración en las capas:
1. **Entrada inicial**:
   - Procesa el tensor \([B, C, H, W]\) con el primer bloque ResNet.
2. **Atención**:
   - Reorganiza \([B, C, H, W]\) a una secuencia para aplicar atención estándar.
   - Si hay contexto, incluye atención cruzada.
3. **Segundo ResNet**:
   - Ajusta el tensor con otro bloque ResNet y lo combina con conexiones residuales.
4. Repite el proceso para el número de capas definido (\(num\_layers\)).

---

### **Visualización resumida**

```
Entrada -> ResNet Block (1)
       -> Atención estándar (y opcionalmente cruzada)
       -> ResNet Block (2)
       -> Repetir por num_layers
       -> Salida
```

---


# Fichero `VAE.py`

## Clase `VAE`

El código define una clase `VAE` (Variational Autoencoder) como una implementación avanzada de un modelo generativo que utiliza convoluciones, bloques de atención y arquitecturas basadas en U-Net. Este modelo está diseñado para codificar imágenes en un espacio latente y reconstruirlas, con capacidad para aprender distribuciones complejas de datos. Vamos a desglosarlo:

---

### **Descripción general**
Un VAE tiene tres componentes principales:
1. **Codificador (`encode`)**: Comprime la imagen de entrada en un espacio latente (`z`).
2. **Espacio latente**: Genera una representación probabilística (`mean` y `logvar`) para aprender distribuciones.
3. **Decodificador (`decode`)**: Reconstruye una imagen desde el espacio latente.

En este caso, el VAE es más avanzado porque:
- Usa **DownBlocks** (bloques residuales con atención) en el codificador.
- Usa **UpBlocks** en el decodificador, que son equivalentes pero operan de manera inversa.
- Soporta configuraciones flexibles gracias a los parámetros en `model_config`.

---

### **Componentes clave**

#### **`__init__`**
El constructor inicializa todos los bloques necesarios para el codificador, el espacio latente y el decodificador.

1. **Parámetros del modelo**:
   - `im_channels`: Número de canales en la entrada (por ejemplo, 3 para imágenes RGB).
   - `model_config`: Diccionario con configuraciones como:
     - `down_channels`: Lista con los canales usados en cada nivel del codificador.
     - `mid_channels`: Canales en los bloques intermedios.
     - `z_channels`: Dimensión del espacio latente.
     - `down_sample`: Indicadores para aplicar downsampling en cada nivel del codificador.
     - `attn_down`: Flags para habilitar o deshabilitar la atención en los bloques.
     - `norm_channels`: Número de grupos para normalización (`GroupNorm`).

2. **Codificador**:
   - **Entrada**: Una capa convolucional inicial (`encoder_conv_in`).
   - **DownBlocks**: Una lista de bloques residuales con downsampling opcional.
   - **MidBlocks**: Procesamiento adicional en el nivel intermedio del codificador.
   - **Salida**: Normalización y convolución para proyectar al espacio latente (`mean` y `logvar`).

3. **Espacio latente**:
   - La salida del codificador se divide en `mean` y `logvar`.
   - La muestra `z` se genera con ruido gaussiano para garantizar continuidad en el espacio latente.

4. **Decodificador**:
   - **Entrada**: Una capa convolucional inicial (`decoder_conv_in`).
   - **MidBlocks**: Bloques intermedios que procesan el espacio latente.
   - **UpBlocks**: Bloques residuales con upsampling para reconstruir la imagen.
   - **Salida**: Normalización y convolución para proyectar al espacio de salida.

---

### **Métodos principales**

#### **`encode`**
Convierte la imagen de entrada en una representación latente probabilística (`mean` y `logvar`):
1. Aplica convoluciones iniciales (`encoder_conv_in`).
2. Pasa por una secuencia de `DownBlocks` para reducir la resolución espacial.
3. Procesa los datos comprimidos a través de `MidBlocks` para aprender representaciones abstractas.
4. Calcula `mean` y `logvar` con una convolución final y los divide en dos partes.
5. Genera la muestra `z` usando reparametrización:
   \[
   z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
   \]

#### **`decode`**
Reconstruye una imagen a partir de un vector latente `z`:
1. Pasa `z` por una convolución inicial (`decoder_conv_in`).
2. Procesa los datos a través de `MidBlocks` y luego `UpBlocks`.
3. Reconstruye la imagen con una convolución final (`decoder_conv_out`).

#### **`forward`**
Integra el flujo completo del modelo:
1. Llama a `encode` para obtener `z`.
2. Llama a `decode` para reconstruir la imagen.
3. Devuelve la reconstrucción y la salida del codificador.

---

### **Puntos importantes**

1. **Atención condicional**:
   - Usa atención en los bloques para capturar relaciones no locales entre píxeles. Esto mejora la capacidad del modelo para trabajar con dependencias espaciales complejas.
   - La atención es configurable y puede activarse/desactivarse según las necesidades del modelo.

2. **Espacio latente probabilístico**:
   - `mean` y `logvar` definen una distribución gaussiana en el espacio latente.
   - El muestreo probabilístico garantiza que las reconstrucciones sean continuas y permitan interpolaciones suaves.

3. **Convoluciones residuales**:
   - Usa conexiones residuales para facilitar el entrenamiento y permitir que los gradientes fluyan más fácilmente.

4. **Flexibilidad**:
   - El modelo puede ajustarse para diferentes resoluciones, dimensiones del espacio latente y configuraciones de atención.

---
