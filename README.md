# VoicePanel-Assistant-AI

## Installation
**IMPORTANTE:** Este proyecto requiere **Python 3.12**
Revisar con python --version

### Entorno Virtual

<details>
<summary><strong>Linux (Fedora / Ubuntu / Arch / etc.)</strong></summary>

### 1. Crear el entorno virtual

```bash
python3.12 -m venv .venv
```

### 2. Activarlo

```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```
</details>

---

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

### 1. Crear el entorno virtual

```powershell
python -m venv .venv
```

### 2. Activarlo

```powershell
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego intenta activarlo otra vez.

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

</details>

---

## Executar Proyecto

VoicePanel-Assistant-AI se compone de 10 módulos principales. Que van desde el tratamiento de los audios que se usaran para el entrenamiento del modelo, la creación de dicho modelo y finalmente la implementación de este en un Raspberry Pi 4.

Por lo tanto es importante ejecutar los 10 módulos en el orden correcto.

### Módulos

**Nota.** Recuerde ejecutar el proyecto desde la raiz de la siguiente forma:
```bash
python src/nombre_de_modulo.py
```
#### Creación de Modelo
1. explore_dataset.py
2. feature_extraction.py
3. augment_dataset.py
4. prepare_splits.py
5. train_cnn.py
6. evaluate_cnn.py
7. train_lstm.py

#### Modulos de Raspberry
8. gpio_controller.py
9. **realtime_pipeline.py**
10. benchmark.py

