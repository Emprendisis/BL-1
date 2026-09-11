import io
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Black-Litterman Portfolio Lab",
    layout="wide",
)

st.title("Black-Litterman Portfolio Lab")
st.caption(
    "Modelo didáctico para combinar equilibrio de mercado, views del inversionista "
    "y optimización media-varianza."
)

TRADING_DAYS = {
    "Diaria": 252,
    "Semanal": 52,
    "Mensual": 12,
}

INTERVALS = {
    "Diaria": "1d",
    "Semanal": "1wk",
    "Mensual": "1mo",
}

HORIZONS = {
    "1 año": 1,
    "3 años": 3,
    "5 años": 5,
}

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]


# =========================================================
# FUNCIONES DE DATOS
# =========================================================

def normalize_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if price_col not in raw.columns:
            return pd.DataFrame()
        name = tickers[0] if len(tickers) == 1 else "Precio"
        prices = raw[[price_col]].rename(columns={price_col: name})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna(how="any")
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices


@st.cache_data(show_spinner=False, ttl=3600)
def download_prices(
    tickers: tuple[str, ...],
    start: dt.date,
    end: dt.date,
    interval: str,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end + dt.timedelta(days=1),
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    return normalize_prices(raw, list(tickers))


@st.cache_data(show_spinner=False, ttl=3600)
def download_market_caps(tickers: tuple[str, ...]) -> dict:
    caps = {}
    for ticker in tickers:
        cap = np.nan
        try:
            t = yf.Ticker(ticker)
            try:
                cap = t.fast_info.get("market_cap", np.nan)
            except Exception:
                cap = np.nan

            if pd.isna(cap):
                try:
                    cap = t.info.get("marketCap", np.nan)
                except Exception:
                    cap = np.nan
        except Exception:
            cap = np.nan

        caps[ticker] = cap

    return caps


@st.cache_data(show_spinner=False, ttl=3600)
def download_benchmark_returns(
    ticker: str,
    start: dt.date,
    end: dt.date,
    interval: str,
) -> pd.Series:
    prices = download_prices((ticker,), start, end, interval)
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.iloc[:, 0].pct_change().dropna()


def read_excel_prices(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)

    if df.empty or df.shape[1] < 3:
        raise ValueError(
            "El Excel debe contener una columna de fecha y al menos dos activos."
        )

    first_col = df.columns[0]
    dates = pd.to_datetime(df[first_col], errors="coerce")

    if dates.isna().all():
        raise ValueError("La primera columna debe contener fechas válidas.")

    data = df.iloc[:, 1:].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.index = dates
    data = data.dropna(axis=1, how="all").ffill().dropna(how="any")

    if data.shape[1] < 2:
        raise ValueError("Se requieren al menos dos activos con datos numéricos.")

    if len(data) < 12:
        raise ValueError("Se requieren al menos 12 observaciones históricas.")

    return data


# =========================================================
# FUNCIONES FINANCIERAS
# =========================================================

def annualized_stats(prices: pd.DataFrame, annualization: int):
    returns = prices.pct_change().dropna()
    mu = returns.mean() * annualization
    cov = returns.cov() * annualization
    return returns, mu, cov


def market_weights_from_caps(caps: pd.Series) -> pd.Series:
    caps = pd.to_numeric(caps, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if caps.isna().any() or (caps <= 0).any():
        raise ValueError(
            "Todas las capitalizaciones deben ser numéricas y mayores que cero."
        )

    return caps / caps.sum()


def infer_risk_aversion(
    benchmark_returns: pd.Series,
    annualization: int,
    risk_free_rate: float,
    fallback: float = 2.5,
) -> float:
    if benchmark_returns is None or len(benchmark_returns) < 3:
        return fallback

    mean_ann = benchmark_returns.mean() * annualization
    var_ann = benchmark_returns.var() * annualization

    if var_ann <= 0:
        return fallback

    delta = (mean_ann - risk_free_rate) / var_ann

    if not np.isfinite(delta) or delta <= 0:
        return fallback

    return float(delta)


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
    risk_free_rate: float,
) -> pd.Series:
    w = market_weights.loc[cov.index].values
    pi_excess = risk_aversion * cov.values @ w
    return pd.Series(
        pi_excess + risk_free_rate,
        index=cov.index,
        name="Retorno equilibrio",
    )


def build_views(
    assets: list[str],
    view_rows: list[dict],
):
    if not view_rows:
        return (
            np.empty((0, len(assets))),
            np.array([]),
            np.array([]),
            [],
        )

    P = []
    Q = []
    confidences = []
    labels = []

    asset_pos = {a: i for i, a in enumerate(assets)}

    for row in view_rows:
        p = np.zeros(len(assets))

        if row["Tipo"] == "Absoluta":
            a = row["Activo A"]
            p[asset_pos[a]] = 1.0
            q = row["Retorno / diferencia (%)"] / 100.0
            label = f"{a} = {row['Retorno / diferencia (%)']:.2f}%"
        else:
            a = row["Activo A"]
            b = row["Activo B"]
            if a == b:
                raise ValueError(
                    "En una view relativa, Activo A y Activo B deben ser distintos."
                )
            p[asset_pos[a]] = 1.0
            p[asset_pos[b]] = -1.0
            q = row["Retorno / diferencia (%)"] / 100.0
            label = (
                f"{a} supera a {b} en "
                f"{row['Retorno / diferencia (%)']:.2f}%"
            )

        c = np.clip(row["Confianza (%)"] / 100.0, 0.01, 0.99)

        P.append(p)
        Q.append(q)
        confidences.append(c)
        labels.append(label)

    return (
        np.array(P, dtype=float),
        np.array(Q, dtype=float),
        np.array(confidences, dtype=float),
        labels,
    )


def idzorek_omega(
    cov: pd.DataFrame,
    P: np.ndarray,
    confidences: np.ndarray,
    tau: float,
) -> np.ndarray:
    if len(P) == 0:
        return np.empty((0, 0))

    sigma = cov.values
    diag = []

    for i, p in enumerate(P):
        c = float(np.clip(confidences[i], 0.01, 0.99))
        alpha = (1.0 - c) / c
        base_uncertainty = float(p @ sigma @ p.T)
        omega_i = tau * alpha * base_uncertainty
        diag.append(max(omega_i, 1e-12))

    return np.diag(diag)


def black_litterman_posterior(
    cov: pd.DataFrame,
    pi: pd.Series,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float,
):
    sigma = cov.values
    pi_vec = pi.loc[cov.index].values

    if len(Q) == 0:
        return pi.copy(), cov.copy()

    tau_sigma = tau * sigma
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega)

    middle = np.linalg.pinv(
        inv_tau_sigma + P.T @ inv_omega @ P
    )

    posterior_mu = middle @ (
        inv_tau_sigma @ pi_vec + P.T @ inv_omega @ Q
    )

    posterior_cov = sigma + middle

    mu = pd.Series(
        posterior_mu,
        index=cov.index,
        name="Retorno Black-Litterman",
    )
    cov_bl = pd.DataFrame(
        posterior_cov,
        index=cov.index,
        columns=cov.columns,
    )

    return mu, cov_bl


def portfolio_metrics(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
):
    ret = float(weights @ mu.values)
    vol = float(np.sqrt(max(weights.T @ cov.values @ weights, 0)))
    sharpe = np.nan if vol == 0 else (ret - risk_free_rate) / vol
    return ret, vol, sharpe


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float,
):
    n = len(mu)

    if max_weight * n < 1.0:
        raise ValueError(
            "El peso máximo por activo es incompatible con el número de activos."
        )

    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w):
        ret, vol, _ = portfolio_metrics(w, mu, cov, risk_free_rate)
        if vol <= 1e-12:
            return 1e6
        return -((ret - risk_free_rate) / vol)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success:
        raise ValueError(f"No convergió la optimización: {result.message}")

    weights = np.clip(result.x, 0, None)
    weights = weights / weights.sum()
    return weights


def simulate_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    n_samples: int = 5000,
):
    n = len(mu)
    weights = np.random.dirichlet(np.ones(n), n_samples)
    rets = weights @ mu.values
    vars_ = np.einsum("ij,jk,ik->i", weights, cov.values, weights)
    vols = np.sqrt(np.maximum(vars_, 0))
    sharpes = np.divide(
        rets - risk_free_rate,
        vols,
        out=np.full_like(rets, np.nan),
        where=vols > 0,
    )

    return pd.DataFrame(
        {
            "Rendimiento": rets,
            "Volatilidad": vols,
            "Sharpe": sharpes,
        }
    )


def portfolio_summary(
    name: str,
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
):
    ret, vol, sharpe = portfolio_metrics(weights, mu, cov, rf)
    return {
        "Portafolio": name,
        "Rendimiento esperado": ret,
        "Volatilidad": vol,
        "Sharpe": sharpe,
    }


def make_excel_output(
    comparison: pd.DataFrame,
    weights_table: pd.DataFrame,
    returns_table: pd.DataFrame,
    views_table: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Comparacion", index=False)
        weights_table.to_excel(writer, sheet_name="Pesos", index=False)
        returns_table.to_excel(writer, sheet_name="Retornos", index=False)
        views_table.to_excel(writer, sheet_name="Views", index=False)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# SIDEBAR: FUENTE Y PARÁMETROS
# =========================================================

st.sidebar.header("1. Fuente de datos")

source = st.sidebar.radio(
    "Origen de precios",
    ["Yahoo Finance", "Excel"],
)

frequency_label = st.sidebar.selectbox(
    "Periodicidad",
    list(TRADING_DAYS.keys()),
    index=0,
)

annualization = TRADING_DAYS[frequency_label]
interval = INTERVALS[frequency_label]

horizon_label = st.sidebar.selectbox(
    "Horizonte histórico",
    list(HORIZONS.keys()),
    index=1,
)

years = HORIZONS[horizon_label]
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=int(365.25 * years))

prices = pd.DataFrame()
tickers = []

if source == "Yahoo Finance":
    ticker_text = st.sidebar.text_input(
        "Tickers (4 a 15)",
        value=", ".join(DEFAULT_TICKERS),
    )
    tickers = [
        x.strip().upper()
        for x in ticker_text.split(",")
        if x.strip()
    ]
    tickers = list(dict.fromkeys(tickers))

    if not 4 <= len(tickers) <= 15:
        st.sidebar.warning("Utiliza entre 4 y 15 tickers.")

    load_data = st.sidebar.button(
        "Descargar precios",
        type="primary",
    )

    if load_data:
        with st.spinner("Descargando precios desde Yahoo Finance..."):
            try:
                prices = download_prices(
                    tuple(tickers),
                    start_date,
                    end_date,
                    interval,
                )
                st.session_state["bl_prices"] = prices
                st.session_state["bl_assets"] = list(prices.columns.astype(str))
            except Exception as exc:
                st.error(f"No fue posible descargar los precios: {exc}")

    if "bl_prices" in st.session_state:
        prices = st.session_state["bl_prices"].copy()

else:
    uploaded = st.sidebar.file_uploader(
        "Archivo Excel de precios",
        type=["xlsx"],
    )
    if uploaded is not None:
        try:
            prices = read_excel_prices(uploaded)
            st.session_state["bl_prices"] = prices
            st.session_state["bl_assets"] = list(prices.columns.astype(str))
        except Exception as exc:
            st.error(f"No fue posible procesar el Excel: {exc}")


st.sidebar.header("2. Parámetros Black-Litterman")

risk_free_rate = (
    st.sidebar.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=-10.0,
        max_value=50.0,
        value=4.0,
        step=0.25,
    )
    / 100.0
)

tau = st.sidebar.slider(
    "Tau (τ)",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark para aversión al riesgo",
    value="SPY",
).strip().upper()

max_weight = (
    st.sidebar.slider(
        "Peso máximo por activo (%)",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
    )
    / 100.0
)

frontier_samples = st.sidebar.slider(
    "Simulaciones de frontera",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000,
)


# =========================================================
# PROCESAMIENTO PRINCIPAL
# =========================================================

if prices.empty:
    st.info(
        "Carga precios desde Yahoo Finance o mediante Excel para configurar "
        "capitalizaciones y views."
    )
    st.stop()

prices = prices.dropna(axis=1, how="all").ffill().dropna(how="any")

if prices.shape[1] < 2:
    st.error("Se requieren al menos dos activos válidos.")
    st.stop()

if len(prices) < 12:
    st.error("Se requieren al menos 12 observaciones históricas.")
    st.stop()

returns, mu_hist, cov = annualized_stats(prices, annualization)
assets = list(cov.columns.astype(str))

st.subheader("Datos utilizados")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Activos", len(assets))
c2.metric("Observaciones", len(prices))
c3.metric("Periodicidad", frequency_label)
c4.metric("Horizonte", horizon_label)

with st.expander("Ver precios históricos"):
    st.dataframe(prices, use_container_width=True)


# =========================================================
# CAPITALIZACIONES Y PRIOR DE MERCADO
# =========================================================

st.subheader("1. Portafolio de equilibrio y capitalizaciones")

if source == "Yahoo Finance":
    auto_caps = download_market_caps(tuple(assets))
else:
    auto_caps = {a: np.nan for a in assets}

default_cap_values = []
valid_caps = [v for v in auto_caps.values() if pd.notna(v) and v > 0]
fallback_cap = float(np.nanmedian(valid_caps)) if valid_caps else 1.0

for asset in assets:
    v = auto_caps.get(asset, np.nan)
    default_cap_values.append(v if pd.notna(v) and v > 0 else fallback_cap)

caps_df = pd.DataFrame(
    {
        "Activo": assets,
        "Capitalización": default_cap_values,
    }
)

st.caption(
    "Revise o modifique las capitalizaciones. En modo Excel se utilizan valores "
    "iguales como punto de partida hasta que se sustituyan por datos de mercado."
)

edited_caps = st.data_editor(
    caps_df,
    use_container_width=True,
    hide_index=True,
    disabled=["Activo"],
    key="market_caps_editor",
)

try:
    market_weights = market_weights_from_caps(
        edited_caps.set_index("Activo")["Capitalización"]
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

if source == "Yahoo Finance":
    try:
        benchmark_returns = download_benchmark_returns(
            benchmark_ticker,
            start_date,
            end_date,
            interval,
        )
    except Exception:
        benchmark_returns = pd.Series(dtype=float)
else:
    benchmark_returns = pd.Series(dtype=float)

delta_auto = infer_risk_aversion(
    benchmark_returns,
    annualization,
    risk_free_rate,
    fallback=2.5,
)

delta = st.number_input(
    "Coeficiente de aversión al riesgo (δ)",
    min_value=0.10,
    max_value=20.0,
    value=float(round(delta_auto, 4)),
    step=0.10,
)

pi = implied_equilibrium_returns(
    cov,
    market_weights,
    delta,
    risk_free_rate,
)

prior_df = pd.DataFrame(
    {
        "Activo": assets,
        "Peso mercado": market_weights.loc[assets].values,
        "Retorno histórico": mu_hist.loc[assets].values,
        "Retorno equilibrio": pi.loc[assets].values,
    }
)

st.dataframe(
    prior_df.style.format(
        {
            "Peso mercado": "{:.2%}",
            "Retorno histórico": "{:.2%}",
            "Retorno equilibrio": "{:.2%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# VIEWS DEL INVERSIONISTA
# =========================================================

st.subheader("2. Views del inversionista")

n_views = st.number_input(
    "Número de views",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
)

view_rows = []

for i in range(int(n_views)):
    st.markdown(f"**View {i + 1}**")
    cols = st.columns([1.2, 1.2, 1.2, 1.4, 1.2])

    view_type = cols[0].selectbox(
        "Tipo",
        ["Absoluta", "Relativa"],
        key=f"view_type_{i}",
    )

    asset_a = cols[1].selectbox(
        "Activo A",
        assets,
        index=min(i, len(assets) - 1),
        key=f"view_asset_a_{i}",
    )

    asset_b = cols[2].selectbox(
        "Activo B",
        assets,
        index=min(i + 1, len(assets) - 1),
        key=f"view_asset_b_{i}",
        disabled=(view_type == "Absoluta"),
    )

    q_value = cols[3].number_input(
        "Retorno / diferencia (%)",
        min_value=-100.0,
        max_value=200.0,
        value=10.0 if view_type == "Absoluta" else 3.0,
        step=0.5,
        key=f"view_q_{i}",
    )

    confidence = cols[4].slider(
        "Confianza (%)",
        min_value=1,
        max_value=99,
        value=70,
        step=1,
        key=f"view_conf_{i}",
    )

    view_rows.append(
        {
            "Tipo": view_type,
            "Activo A": asset_a,
            "Activo B": asset_b,
            "Retorno / diferencia (%)": q_value,
            "Confianza (%)": confidence,
        }
    )

views_df = pd.DataFrame(view_rows)

try:
    P, Q, confidences, view_labels = build_views(
        assets,
        view_rows,
    )

    omega = idzorek_omega(
        cov,
        P,
        confidences,
        tau,
    )

    mu_bl, cov_bl = black_litterman_posterior(
        cov,
        pi,
        P,
        Q,
        omega,
        tau,
    )

except Exception as exc:
    st.error(f"Error al construir las views: {exc}")
    st.stop()

view_display = views_df.copy()
view_display.insert(0, "Descripción", view_labels)
st.dataframe(view_display, use_container_width=True, hide_index=True)


# =========================================================
# OPTIMIZACIÓN Y COMPARACIÓN
# =========================================================

st.subheader("3. Comparación de portafolios")

try:
    w_market = market_weights.loc[assets].values

    w_markowitz = optimize_max_sharpe(
        mu_hist.loc[assets],
        cov.loc[assets, assets],
        risk_free_rate,
        max_weight,
    )

    w_bl = optimize_max_sharpe(
        mu_bl.loc[assets],
        cov_bl.loc[assets, assets],
        risk_free_rate,
        max_weight,
    )

except Exception as exc:
    st.error(f"Error en la optimización: {exc}")
    st.stop()

comparison = pd.DataFrame(
    [
        portfolio_summary(
            "Mercado",
            w_market,
            pi.loc[assets],
            cov.loc[assets, assets],
            risk_free_rate,
        ),
        portfolio_summary(
            "Markowitz",
            w_markowitz,
            mu_hist.loc[assets],
            cov.loc[assets, assets],
            risk_free_rate,
        ),
        portfolio_summary(
            "Black-Litterman",
            w_bl,
            mu_bl.loc[assets],
            cov_bl.loc[assets, assets],
            risk_free_rate,
        ),
    ]
)

st.dataframe(
    comparison.style.format(
        {
            "Rendimiento esperado": "{:.2%}",
            "Volatilidad": "{:.2%}",
            "Sharpe": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

weights_table = pd.DataFrame(
    {
        "Activo": assets,
        "Mercado": w_market,
        "Markowitz": w_markowitz,
        "Black-Litterman": w_bl,
    }
)

fig_weights = px.bar(
    weights_table.melt(
        id_vars="Activo",
        var_name="Portafolio",
        value_name="Peso",
    ),
    x="Activo",
    y="Peso",
    color="Portafolio",
    barmode="group",
    title="Pesos: Mercado vs. Markowitz vs. Black-Litterman",
)
fig_weights.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_weights, use_container_width=True)


# =========================================================
# RETORNOS PRIOR VS POSTERIOR
# =========================================================

returns_table = pd.DataFrame(
    {
        "Activo": assets,
        "Histórico": mu_hist.loc[assets].values,
        "Equilibrio": pi.loc[assets].values,
        "Black-Litterman": mu_bl.loc[assets].values,
    }
)

fig_returns = px.bar(
    returns_table.melt(
        id_vars="Activo",
        var_name="Estimación",
        value_name="Retorno",
    ),
    x="Activo",
    y="Retorno",
    color="Estimación",
    barmode="group",
    title="Retornos: histórico, equilibrio y posterior Black-Litterman",
)
fig_returns.update_yaxes(tickformat=".1%")
st.plotly_chart(fig_returns, use_container_width=True)


# =========================================================
# FRONTERA EFICIENTE
# =========================================================

st.subheader("4. Frontera eficiente")

frontier = simulate_frontier(
    mu_bl.loc[assets],
    cov_bl.loc[assets, assets],
    risk_free_rate,
    frontier_samples,
)

bl_ret, bl_vol, bl_sharpe = portfolio_metrics(
    w_bl,
    mu_bl.loc[assets],
    cov_bl.loc[assets, assets],
    risk_free_rate,
)

mk_ret, mk_vol, mk_sharpe = portfolio_metrics(
    w_markowitz,
    mu_hist.loc[assets],
    cov.loc[assets, assets],
    risk_free_rate,
)

fig_frontier = px.scatter(
    frontier,
    x="Volatilidad",
    y="Rendimiento",
    color="Sharpe",
    title="Frontera eficiente aproximada — Black-Litterman",
)

fig_frontier.add_trace(
    go.Scatter(
        x=[bl_vol],
        y=[bl_ret],
        mode="markers",
        marker=dict(size=18, symbol="star"),
        name="Black-Litterman",
    )
)

fig_frontier.add_trace(
    go.Scatter(
        x=[mk_vol],
        y=[mk_ret],
        mode="markers",
        marker=dict(size=14, symbol="diamond"),
        name="Markowitz",
    )
)

fig_frontier.update_xaxes(tickformat=".1%")
fig_frontier.update_yaxes(tickformat=".1%")
st.plotly_chart(fig_frontier, use_container_width=True)


# =========================================================
# MATRICES DE RIESGO
# =========================================================

st.subheader("5. Matrices de riesgo")

tab1, tab2 = st.tabs(["Correlación", "Covarianza Black-Litterman"])

with tab1:
    corr = returns[assets].corr()
    st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Matriz de correlación",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.dataframe(
        cov_bl.style.format("{:.6f}"),
        use_container_width=True,
    )
    fig_cov = px.imshow(
        cov_bl,
        text_auto=".4f",
        aspect="auto",
        title="Covarianza posterior Black-Litterman",
    )
    st.plotly_chart(fig_cov, use_container_width=True)


# =========================================================
# CONTRIBUCIÓN APROXIMADA AL RIESGO
# =========================================================

st.subheader("6. Contribución al riesgo — Black-Litterman")

sigma_bl = cov_bl.loc[assets, assets].values
port_vol = np.sqrt(w_bl.T @ sigma_bl @ w_bl)

if port_vol > 0:
    marginal = sigma_bl @ w_bl / port_vol
    component = w_bl * marginal
    component_pct = component / component.sum()
else:
    component_pct = np.zeros_like(w_bl)

risk_df = pd.DataFrame(
    {
        "Activo": assets,
        "Peso BL": w_bl,
        "Contribución al riesgo": component_pct,
    }
)

fig_risk = px.bar(
    risk_df,
    x="Activo",
    y="Contribución al riesgo",
    title="Contribución porcentual al riesgo total",
)
fig_risk.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_risk, use_container_width=True)

st.dataframe(
    risk_df.style.format(
        {
            "Peso BL": "{:.2%}",
            "Contribución al riesgo": "{:.2%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# DESCARGAS
# =========================================================

st.subheader("7. Descargas")

excel_bytes = make_excel_output(
    comparison,
    weights_table,
    returns_table,
    view_display,
)

d1, d2 = st.columns(2)

with d1:
    st.download_button(
        "Descargar resultados en Excel",
        data=excel_bytes,
        file_name="resultados_black_litterman.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with d2:
    st.download_button(
        "Descargar pesos Black-Litterman en CSV",
        data=weights_table[["Activo", "Black-Litterman"]].to_csv(index=False),
        file_name="pesos_black_litterman.csv",
        mime="text/csv",
    )


# =========================================================
# INTERPRETACIÓN
# =========================================================

st.subheader("Interpretación ejecutiva")

st.markdown(
    """
- **Mercado:** representa el punto de partida o prior basado en capitalización.
- **Markowitz:** utiliza únicamente rendimientos históricos y covarianzas.
- **Black-Litterman:** parte del equilibrio de mercado e incorpora las views según su confianza.
- **Idzorek:** convierte la confianza de cada view en incertidumbre. Una mayor confianza implica menor incertidumbre y, por tanto, mayor influencia de la view sobre los retornos posteriores.
- **Restricción de peso máximo:** evita concentraciones excesivas en un solo activo.
"""
)
