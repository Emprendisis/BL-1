# Black-Litterman Portfolio Lab

Aplicación didáctica en Python y Streamlit para construir y comparar:

- Portafolio de mercado.
- Portafolio Markowitz de máximo Sharpe.
- Portafolio Black-Litterman de máximo Sharpe.

## Características

- Fuente de precios:
  - Yahoo Finance.
  - Archivo Excel.
- Horizonte histórico:
  - 1, 3 o 5 años.
- Periodicidad:
  - diaria, semanal o mensual.
- Entre 4 y 15 activos en modo Yahoo Finance.
- Prior de mercado por capitalización.
- Coeficiente de aversión al riesgo calculado desde un benchmark y editable.
- Tau configurable.
- Views absolutas y relativas.
- Confianza por view entre 1% y 99%.
- Matriz Omega construida con una aproximación del método de Idzorek.
- Restricción long-only.
- Peso máximo configurable por activo.
- Comparación Mercado vs. Markowitz vs. Black-Litterman.
- Frontera eficiente aproximada.
- Matrices de correlación y covarianza.
- Contribución al riesgo.
- Descarga de resultados en Excel y CSV.

## Estructura del Excel de precios

La primera columna debe contener fechas. Las demás columnas deben contener precios históricos de los activos.

Ejemplo:

| Fecha | AAPL | MSFT | NVDA | AMZN |
|---|---:|---:|---:|---:|
| 2025-01-02 | 243.85 | 418.58 | 138.31 | 220.22 |
| 2025-01-03 | 243.36 | 423.35 | 144.47 | 224.19 |

## Lógica Black-Litterman

El modelo parte de los retornos implícitos de equilibrio:

`Pi = delta * Sigma * w_mkt + rf`

donde:

- `delta` = aversión al riesgo;
- `Sigma` = matriz de covarianza;
- `w_mkt` = pesos de mercado;
- `rf` = tasa libre de riesgo.

Las views se representan mediante:

- `P` = matriz de selección de activos;
- `Q` = retornos o diferenciales esperados;
- `Omega` = incertidumbre de las views.

### Confianza e Idzorek

Para cada view se asigna una confianza entre 1% y 99%.

La aplicación transforma esa confianza mediante:

`alpha = (1 - confianza) / confianza`

y construye la incertidumbre aproximada:

`Omega_i = tau * alpha * (P_i * Sigma * P_i')`

Por tanto:

- mayor confianza -> menor Omega;
- menor Omega -> mayor influencia de la view;
- menor confianza -> mayor Omega;
- mayor Omega -> menor influencia de la view.

## Optimización

Markowitz y Black-Litterman se optimizan mediante SLSQP buscando máximo Sharpe bajo:

- suma de pesos = 100%;
- pesos >= 0%;
- peso máximo por activo configurable.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir `app.py`, `requirements.txt` y `README.md`.
3. Conectar el repositorio con Streamlit Community Cloud.
4. Seleccionar `app.py` como archivo principal.
5. Desplegar.

## Nota didáctica

El modelo está diseñado para enseñanza y demostración. No constituye una recomendación de inversión ni sustituye procesos profesionales de validación de datos, estimación, riesgo, compliance o suitability.
