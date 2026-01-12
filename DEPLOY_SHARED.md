# Guía de Despliegue en Hosting Compartido (Shared Hosting / cPanel)

> [!WARNING]
> **Streamlit no es una a web tradicional (como WordPress o Django/WSGI).**  
> Funciona como un servidor propio (Websocket). La mayoría de hostings compartidos están diseñados para scripts PHP o WSGI (Python) efímeros.  
> **Para que funcione, necesitas acceso SSH y poder ejecutar procesos en background.**

---

## 1. Requisitos Previos
1. **Acceso SSH** a tu hosting.
2. **Python 3.9+** instalado en el servidor (muchos cPanel ya traen "Setup Python App").
3. **Un puerto libre** (ej. 8501, 9000...) o permiso para mapear puertos.

---

## 2. Preparar el Entorno (vía SSH)

Conéctate a tu terminal y ve a la carpeta donde subirás los archivos (ej. `public_html/dashboard` o una carpeta privada).

```bash
# 1. Crear entorno virtual
python3 -m venv .venv

# 2. Activarlo
source .venv/bin/activate

# 3. Instalar librerías
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configuración Específica para Streamlit

Streamlit intenta abrir un navegador, desactívalo. Crea un archivo `.streamlit/config.toml` o usa flags.

Crea la carpeta de configuración:
```bash
mkdir -p .streamlit
nano .streamlit/config.toml
```

Pega esto dentro:
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
# Puerto: en hosting compartido NO SIEMPRE puedes usar el 80/443 directo.
# Si tu hosting te asigna un puerto específico, ponlo aquí.
# Si no, prueba uno alto como 8501.
port = 8501
address = "0.0.0.0"

[browser]
gatherUsageStats = false
```

---

## 4. Ejecutar en Segundo Plano (Background)

En SSH, si cierras la terminal, se apaga Streamlit. Usa `nohup` para mantenerlo vivo.

### Opción A: Script de arranque (`run.sh`)

Crea un archivo `run.sh`:
```bash
#!/bin/bash
source .venv/bin/activate
nohup streamlit run app/app.py > streamlit.log 2>&1 &
echo "Streamlit iniciado. Revisa streamlit.log"
```

Dale permisos y ejecuta:
```bash
chmod +x run.sh
./run.sh
```

### Opción B: Usar `screen` (si está disponible)
```bash
screen -S dashboard
source .venv/bin/activate
streamlit run app/app.py
# Presiona Ctrl+A, luego D para salir sin cerrar.
```

---

## 5. Acceder a la App

Si tu servidor tiene IP `123.45.67.89` y usaste el puerto `8501`:
Entra a: `http://123.45.67.89:8501`

### Problema común: El Firewall del Hosting
Muchos hostings bloquean el puerto 8501.  
**Solución:** Pide a soporte que "abran el puerto 8501 para conexiones entrantes TCP" o intenta usar puertos como 8080.

---

## 6. Método Alternativo: .htaccess (Reverse Proxy / Passenger)

**Solo si tu hosting permite `mod_proxy` (raro en planes baratos).**  
Puedes redirigir tu dominio `midominio.com/dashboard` al puerto interno de localhost.

En tu `.htaccess`:
```apache
RewriteEngine On
RewriteRule ^$ http://127.0.0.1:8501/ [P,L]
RewriteRule ^(.*) http://127.0.0.1:8501/$1 [P,L]
```

*Nota: Esto suele fallar con los WebSockets de Streamlit. Si ves "Please wait...", el WebSocket está bloqueado.*

---

## Resumen
La forma más robusta en compartido es:
1. Subir archivos.
2. Instalar deps en venv.
3. Ejecutar con `nohup` en un puerto abierto.
4. Entrar vía `IP:PUERTO`.
