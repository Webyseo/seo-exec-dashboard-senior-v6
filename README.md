# SEO Executive Dashboard (Senior)

Dashboard ejecutivo SEO que interpreta (no solo muestra) datos:
- SoV / Market Share SEO (pastel)
- Tráfico estimado por CTR según posición (actual vs potencial Top 3)
- KEI / Easy Wins
- Striking Distance (pos 4–10)
- Threats (pos 1–3 con competidor 4–6)
- Content Gap (competidor Top10 y tú >20)
- Heatmap por clúster vs dominios
- Secciones (carpeta raíz de URL)
- Cannibalización (si existe)

## Arranque (Docker)
```bash
docker compose up --build
```
Abrir: http://localhost:8501

## Arranque (sin Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

### 🌍 Despliegue en Servidor Compartido (cPanel)
Si usas hosting compartido (GoDaddy, Namecheap, Banahosting...), [lee la guía detallada aquí](DEPLOY_SHARED.md).


## Datos
Sube 1 o varios CSV con la MISMA estructura (como el ejemplo).
Cada import se guarda y puedes navegar el histórico.

## Variables de entorno
- PRIMARY_DOMAIN: dominio principal por defecto (ej. radiofonics.com)
- NON_RANK_VALUE: valor numérico para "No está entre las primeras 20" (default 21)


## Fixes v2
- Visibilidad/CTR soportan valores tipo `75%`.
- Vista de Data Hygiene sin columnas duplicadas.



## IA (opcional): interpretaciones y briefs
- El dashboard funciona **sin IA**.
- Si quieres usar IA, pega tu API key en la pestaña **🤖 IA (interpretación)** o define una variable de entorno en el servidor:

```bash
export OPENAI_API_KEY="TU_API_KEY"
```

Notas de seguridad:
- En **producción**, no expongas la API key en frontend. Este dashboard la usa en **server-side** (Streamlit).
- Para equipos: usa una key de **Project** con permisos y límites (rate limits) adecuados.

## Estructura de datos
- Puedes subir múltiples CSV con la misma estructura.
- Los CSV quedan guardados en `data/csv/` y el índice en SQLite `data/db/dashboard.sqlite3`.

