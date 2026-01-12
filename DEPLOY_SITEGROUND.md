# Guía de Despliegue en Siteground (Plan GoGeek)

Esta guía detalla cómo desplegar tu aplicación **Streamlit** en Siteground utilizando las características avanzadas del plan **GoGeek** (Acceso SSH y Git).

> [!WARNING]
> **Aviso Importante**: Siteground es un hosting compartido, optimizado para PHP (WordPress). **Streamlit** es un servidor de aplicaciones que requiere un proceso de larga duración y soporte de WebSockets.
> 
> Aunque el plan GoGeek permite acceso SSH, es posible que encuentres limitaciones:
> 1. **Puertos Cerrados**: No puedes abrir el puerto 8501 al público. Usaremos un "Truco" (Proxy Reverso) para verlo.
> 2. **Procesos Background**: Siteground monitoriza los procesos. Si tu app consume mucha RAM, podrían matarla.
> 3. **WebSockets**: Si la configuración de Apache de Siteground no permite `WS Tunnel`, la app cargará pero se quedará en "Please wait..." o "Connecting...".

---

## Paso 1: Subir los Archivos

Puedes usar **FTP** (FileZilla) o el **Gestor de Archivos** de Site Tools, pero recomendamos **Git** si ya lo tienes configurado.

1. Entra a **Site Tools** > **Site** > **File Manager**.
2. Navega a `public_html`.
3. Crea una carpeta para tu app, por ejemplo `dashboard`.
4. Sube todos los archivos de tu proyecto (`app/`, `requirements.txt`, `packages.txt` si existe) dentro de `public_html/dashboard`.

Ruta final típica: `/home/customer/www/tudominio.com/public_html/dashboard`

---

## Paso 2: Preparar el Entorno Python (vía SSH)

El plan GoGeek incluye acceso SSH.

1. **Activar SSH**: Ve a **Site Tools** > **Devs** > **SSH Keys Manager**. Crea una llave o usa una existente.
2. **Conectarse**: Usa tu terminal (o PuTTY):
   ```bash
   ssh tu_usuario@tudominio.com -p 18765
   ```
   *(El puerto SSH de Siteground suele ser 18765)*.

3. **Navegar a la carpeta**:
   ```bash
   cd www/tudominio.com/public_html/dashboard
   ```

4. **Crear Entorno Virtual**:
   Siteground tiene Python instalado. Verifica la versión con `python3 --version`.
   ```bash
   # Crear entorno virtual llamado 'venv'
   python3 -m venv venv
   
   # Activarlo
   source venv/bin/activate
   
   # Actualizar pip
   pip install --upgrade pip
   ```

5. **Instalar Dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
   *Nota: Si alguna librería falla al compilar (como numpy o pandas antiguos), intenta instalar versiones pre-compiladas o más recientes.*

---

## Paso 3: Configurar Streamlit (Headless)

Crea la configuración para que Streamlit sepa que corre en un servidor.

1. Crea la carpeta y el archivo:
   ```bash
   mkdir -p .streamlit
   nano .streamlit/config.toml
   ```

2. Pega el siguiente contenido (Ctrl+V):
   ```toml
   [server]
   headless = true
   enableCORS = false
   enableXsrfProtection = false
   address = "127.0.0.1"
   port = 8501
   
   [browser]
   gatherUsageStats = false
   ```
   *Presta atención a `address = "127.0.0.1"`. Esto hace que solo escuche internamente, lo cual es más seguro ya que usaremos un proxy.*

3. Guarda y sal (Ctrl+O, Enter, Ctrl+X).

---

## Paso 4: Configurar el Acceso Web (.htaccess)

Como no podemos entrar a `tudominio.com:8501`, configuraremos el servidor web (Apache) para que redirija `tudominio.com/dashboard` internamente a tu app Streamlit.

1. En la carpeta `dashboard`, crea o edita el archivo `.htaccess`:
   ```bash
   nano .htaccess
   ```

2. Agrega estas reglas. **IMPORTANTE**: Esto requiere que Siteground tenga activos `mod_proxy` y `mod_proxy_http`. Generalmente en GoGeek están disponibles, pero `mod_proxy_wstunnel` (para websockets) a veces no.

   ```apache
   DirectoryIndex disabled
   RewriteEngine On
   
   # Reglas para Redirigir el tráfico al puerto 8501 interno
   
   # Soporte para WebSockets (Vital para Streamlit)
   RewriteCond %{HTTP:Upgrade} =websocket [NC]
   RewriteRule /(.*)           ws://127.0.0.1:8501/$1 [P,L]
   RewriteCond %{HTTP:Upgrade} !=websocket [NC]
   RewriteRule /(.*)           http://127.0.0.1:8501/$1 [P,L]
   
   # Proxy inverso general
   ProxyPass / http://127.0.0.1:8501/
   ProxyPassReverse / http://127.0.0.1:8501/
   ```

   *Nota: Si `.htaccess` te da "Internal Server Error", es probable que Siteground no permita estas directivas proxy en tu plan. En ese caso, contacta a soporte.*

---

## Paso 5: Ejecutar la Aplicación

Ahora lanzamos la aplicación en segundo plano.

1. **Prueba rápida** (Mantenla abierta un momento):
   ```bash
   ./venv/bin/streamlit run app/app.py
   ```
   Visita tu web (`tudominio.com/dashboard`). Si carga, ¡éxito!

2. **Ejecutar persistente (nohup)**:
   Si cierras la terminal, la app muere. Usa `nohup` para dejarla viva.

   Crea un script `start.sh`:
   ```bash
   nano start.sh
   ```
   Contenido:
   ```bash
   #!/bin/bash
   cd /home/customer/www/tudominio.com/public_html/dashboard
   source venv/bin/activate
   nohup streamlit run app/app.py > streamlit.log 2>&1 &
   echo "Streamlit iniciado en background."
   ```
   
   Dale permisos y correlo:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

---

## Paso 6: Verificación y Mantenimiento

- **Ver logs**: `tail -f streamlit.log` para ver errores en vivo.
- **Detener la app**: 
  1. Busca el proceso: `ps aux | grep streamlit`
  2. Mata el ID: `kill -9 [PID]`
- **Si la web dice "Please Wait" infinitamente**: 
  Significa que los **WebSockets** están bloqueados por el proxy de Siteground. 
  *Solución*: Intenta configurar `server.enableWebsocketCompression = false` en `config.toml`, pero generalmente requiere soporte del hosting. Si no funciona, considera desplegar en **Streamlit Cloud** (gratis/fácil) o un VPS.
