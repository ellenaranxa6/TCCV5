import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import pandas as pd

# =========================================================
# CONFIG INICIAL
# =========================================================
st.set_page_config(page_title="Manobras - IEEE-123 Bus", layout="wide")
st.title("Manobras - IEEE-123 Bus")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "toposwitch_modo2.db"  # banco novo (modo: vão simples)

# =========================================================
# LOGIN (Página 1)
# =========================================================
USUARIO_OK = "Ellen"
SENHA_OK = "tccifmg"

if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

if not st.session_state["auth_ok"]:
    st.subheader("Login")

    col1, col2 = st.columns([1, 1])
    with col1:
        u = st.text_input("Usuário", value="", key="login_usuario")
    with col2:
        p = st.text_input("Senha", value="", type="password", key="login_senha")

    if st.button("Entrar", key="btn_login"):
        if (u or "").strip() == USUARIO_OK and (p or "").strip() == SENHA_OK:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")
    st.stop()

# =========================================================
# BANCO – AUX
# =========================================================
def get_connection() -> sqlite3.Connection:
    # check_same_thread=False ajuda em alguns cenários do Streamlit
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(show_spinner=False)
def listar_tabelas() -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

@st.cache_data(show_spinner=False)
def carregar_coords() -> Dict[str, Tuple[float, float]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    rows = cur.fetchall()
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}

@st.cache_data(show_spinner=False)
def carregar_topologia() -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT line, from_bus, to_bus, is_switch, norm FROM topology")
    rows = cur.fetchall()
    conn.close()
    topo = []
    for line, f, t, is_sw, norm in rows:
        topo.append(
            dict(
                line=str(line).strip(),
                from_bus=str(f).strip(),
                to_bus=str(t).strip(),
                is_switch=bool(is_sw),
                norm=str(norm).strip() if norm is not None else "",
            )
        )
    return topo

@st.cache_data(show_spinner=False)
def carregar_loads() -> Dict[str, float]:
    # opcional (p/ mostrar carga por barra se quiser no futuro)
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, kw FROM loads")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {str(b).strip(): float(kw) for b, kw in rows}

@st.cache_data(show_spinner=False)
def carregar_spans() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT span_id, line_elem, bus1, bus2 FROM spans ORDER BY span_id",
        conn
    )
    conn.close()
    # normaliza
    for c in ["span_id", "line_elem", "bus1", "bus2"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def carregar_options_por_span(span_id: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT span_id, option_rank, tipo, nf_isol_json, na_elem, nf_block_elem, n_manobras
        FROM maneuver_options
        WHERE span_id = ?
        ORDER BY option_rank
        """,
        conn,
        params=(span_id,)
    )
    conn.close()
    # parsers
    def parse_json_list(s):
        if s is None:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
        return []

    df["nf_isol_list"] = df["nf_isol_json"].apply(parse_json_list)
    for c in ["na_elem", "nf_block_elem", "tipo"]:
        if c in df.columns:
            df[c] = df[c].where(df[c].notna(), None)
    return df

@st.cache_data(show_spinner=False)
def carregar_steps(span_id: str, option_rank: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT span_id, option_rank, step_order, op, element
        FROM maneuver_steps
        WHERE span_id = ? AND option_rank = ?
        ORDER BY step_order
        """,
        conn,
        params=(span_id, int(option_rank))
    )
    conn.close()
    for c in ["op", "element"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def carregar_resumos(span_id: str, option_rank: int) -> Dict[str, Optional[dict]]:
    """
    Busca resumos por opção: carga/barras, tensão, perdas, correntes
    """
    conn = get_connection()

    out = {"load": None, "volt": None, "loss": None, "curr": None}

    # load impact
    try:
        df = pd.read_sql_query(
            """
            SELECT kw_off_base, kw_off_after, rest_kw, n_buses_off, buses_off_json
            FROM option_load_impact
            WHERE span_id = ? AND option_rank = ?
            LIMIT 1
            """,
            conn,
            params=(span_id, int(option_rank))
        )
        if not df.empty:
            r = df.iloc[0].to_dict()
            # parse buses_off_json
            buses = []
            try:
                buses = json.loads(r.get("buses_off_json") or "[]")
            except Exception:
                buses = []
            r["buses_off_list"] = [str(b).strip() for b in buses if str(b).strip()]
            out["load"] = r
    except Exception:
        pass

    # voltage summary
    try:
        df = pd.read_sql_query(
            """
            SELECT n_v_viol, worst_v_bus, worst_vmin_pu, worst_vmax_pu
            FROM option_voltage_summary
            WHERE span_id = ? AND option_rank = ?
            LIMIT 1
            """,
            conn,
            params=(span_id, int(option_rank))
        )
        if not df.empty:
            out["volt"] = df.iloc[0].to_dict()
    except Exception:
        pass

    # losses
    try:
        df = pd.read_sql_query(
            """
            SELECT p_loss_base_kw, q_loss_base_kvar, p_loss_after_kw, q_loss_after_kvar, dp_kw, dp_pct
            FROM option_losses
            WHERE span_id = ? AND option_rank = ?
            LIMIT 1
            """,
            conn,
            params=(span_id, int(option_rank))
        )
        if not df.empty:
            out["loss"] = df.iloc[0].to_dict()
    except Exception:
        pass

    # current summary
    try:
        df = pd.read_sql_query(
            """
            SELECT n_viol_base, n_viol_after, has_new_violation, has_worse_violation,
                   worst_elem, worst_ratio_base, worst_ratio_after
            FROM option_current_summary
            WHERE span_id = ? AND option_rank = ?
            LIMIT 1
            """,
            conn,
            params=(span_id, int(option_rank))
        )
        if not df.empty:
            out["curr"] = df.iloc[0].to_dict()
    except Exception:
        pass

    conn.close()
    return out

# =========================================================
# NORMALIZAÇÕES
# =========================================================
def norm_token(x: str) -> str:
    return str(x).strip().lower().replace("line.", "")

def norm_line_token(x: str) -> str:
    s = norm_token(x)
    if not s:
        return ""
    if s.startswith("l") and s[1:].isdigit():
        return s
    if s.isdigit():
        return f"l{s}"
    return s

def pretty_elem(x: Optional[str]) -> str:
    if not x:
        return "•"
    return norm_line_token(x)

def pretty_nf_list(xs: List[str]) -> str:
    if not xs:
        return "•"
    return ", ".join([pretty_elem(x) for x in xs])

# =========================================================
# STATUS / BADGES
# =========================================================
def compute_status_flags(res: Dict[str, Optional[dict]]) -> Dict[str, bool]:
    """
    "Atenção" se:
      - tensão violada (n_v_viol > 0)
      - has_new_violation = 1
      - has_worse_violation = 1
      - dp_kw > 0 (perdas aumentaram)
    """
    flags = {
        "v_viol": False,
        "new_curr": False,
        "worse_curr": False,
        "loss_up": False,
    }

    v = res.get("volt") or {}
    if v and int(v.get("n_v_viol") or 0) > 0:
        flags["v_viol"] = True

    c = res.get("curr") or {}
    if c:
        flags["new_curr"] = int(c.get("has_new_violation") or 0) == 1
        flags["worse_curr"] = int(c.get("has_worse_violation") or 0) == 1

    l = res.get("loss") or {}
    if l:
        dp_kw = l.get("dp_kw")
        try:
            flags["loss_up"] = (dp_kw is not None) and (float(dp_kw) > 0.0)
        except Exception:
            flags["loss_up"] = False

    return flags

def render_status_badges(flags: Dict[str, bool]) -> str:
    # string curta pro dataframe
    if any(flags.values()):
        itens = []
        if flags["v_viol"]:
            itens.append("⚠️V")
        if flags["new_curr"]:
            itens.append("⚠️I(new)")
        if flags["worse_curr"]:
            itens.append("⚠️I(↑)")
        if flags["loss_up"]:
            itens.append("⚠️ΔP+")
        return " ".join(itens) if itens else "⚠️"
    return "✅ OK"

# =========================================================
# PLOT – MAPA
# =========================================================
def construir_mapa_base(
    coords: Dict[str, Tuple[float, float]],
    topo: List[Dict],
    show_line_labels: bool = True,
) -> go.Figure:
    edge_x, edge_y = [], []
    for el in topo:
        u, v = el["from_bus"], el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="#D3D3D3", width=1),
            hoverinfo="none",
            name="Linhas"
        )
    )

    node_x = [coords[b][0] for b in coords]
    node_y = [coords[b][1] for b in coords]
    node_text = list(coords.keys())

    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=6, color="green"),
            name="Barras",
            hovertemplate="Barra %{text}<extra></extra>",
        )
    )

    if show_line_labels:
        label_x, label_y, label_text = [], [], []
        for el in topo:
            line = el["line"]
            u, v = el["from_bus"], el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                label_x.append((x0 + x1) / 2.0)
                label_y.append((y0 + y1) / 2.0)
                label_text.append(str(line))
        fig.add_trace(
            go.Scatter(
                x=label_x, y=label_y,
                mode="text",
                text=label_text,
                textposition="middle center",
                textfont=dict(color="#555555", size=7),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=650,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

def topo_por_line_dict(topo: List[Dict]) -> Dict[str, Dict]:
    return {norm_line_token(el["line"]): el for el in topo}

def plotar_mapa_span_opcao(
    coords: Dict[str, Tuple[float, float]],
    topo: List[Dict],
    span_bus1: Optional[str],
    span_bus2: Optional[str],
    option_row: Dict,
    buses_off_list: List[str],
) -> go.Figure:
    fig = construir_mapa_base(coords, topo, show_line_labels=True)
    topo_d = topo_por_line_dict(topo)

    # trecho selecionado (bus1-bus2) em preto se coords existir
    if span_bus1 and span_bus2 and (span_bus1 in coords) and (span_bus2 in coords):
        x0, y0 = coords[span_bus1]
        x1, y1 = coords[span_bus2]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="black", width=5),
                name="Trecho selecionado",
                hoverinfo="none",
            )
        )

    # barras desligadas
    all_buses = set(coords.keys())
    buses_off = set([str(b).strip() for b in (buses_off_list or []) if str(b).strip()])
    buses_on = all_buses - buses_off

    off_x = [coords[b][0] for b in buses_off if b in coords]
    off_y = [coords[b][1] for b in buses_off if b in coords]
    if off_x:
        fig.add_trace(
            go.Scatter(
                x=off_x, y=off_y,
                mode="markers",
                marker=dict(size=9, color="red"),
                name="Barras desligadas",
                hoverinfo="skip",
            )
        )

    on_x = [coords[b][0] for b in buses_on if b in coords]
    on_y = [coords[b][1] for b in buses_on if b in coords]
    if on_x:
        fig.add_trace(
            go.Scatter(
                x=on_x, y=on_y,
                mode="markers",
                marker=dict(size=7, color="green"),
                name="Barras energizadas",
                hoverinfo="skip",
            )
        )

    def add_line(ln: Optional[str], color: str, label: str, width: int = 6, dash: Optional[str] = None):
        if not ln:
            return
        key = norm_line_token(ln)
        el = topo_d.get(key)
        if not el:
            return
        u, v = el["from_bus"], el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=width, dash=dash),
                    name=label,
                    hoverinfo="none",
                )
            )

    # NFs isolação (lista)
    for nf in option_row.get("nf_isol_list", []) or []:
        add_line(nf, "red", "NF isolação (abrir)", width=6, dash="dash")

    # NA / NF contenção
    add_line(option_row.get("na_elem"), "cyan", "NA (fechar)", width=7, dash=None)
    add_line(option_row.get("nf_block_elem"), "purple", "NF contenção (abrir)", width=7, dash=None)

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# =========================================================
# UI / APP
# =========================================================
st.sidebar.header("📂 Status do banco")

if not DB_PATH.exists():
    st.sidebar.error(f"Banco `{DB_PATH.name}` não encontrado na pasta do app.")
    st.stop()

tabelas = listar_tabelas()
st.sidebar.write("Banco:", f"`{DB_PATH.name}`")
st.sidebar.write("Tabelas:", ", ".join(tabelas))

coords = carregar_coords()
topo = carregar_topologia()
_ = carregar_loads()
spans_df = carregar_spans()

if spans_df.empty or not coords or not topo:
    st.error("Banco encontrado, mas `spans`, `coords` ou `topology` está vazio.")
    st.stop()

nome_operador = st.text_input("Nome do operador responsável", value="Ellen")
st.info(f"Usuário: **{nome_operador}**")

st.markdown("---")

# Mapa base
st.subheader("Rede IEEE-123 Bus — Visão Geral")
st.plotly_chart(construir_mapa_base(coords, topo, show_line_labels=True), use_container_width=True)

st.markdown("---")
st.subheader("Operação por Vão (linha Lxx)")

# =========================
# Seleção por busca (Lxx)
# =========================
# cria coluna auxiliar "display"
def span_display_row(r):
    sid = str(r["span_id"]).strip()
    b1 = (str(r["bus1"]).strip() if "bus1" in r and pd.notna(r["bus1"]) else "")
    b2 = (str(r["bus2"]).strip() if "bus2" in r and pd.notna(r["bus2"]) else "")
    if b1 and b2 and b1 != "None" and b2 != "None":
        return f"{sid}  ({b1} ↔ {b2})"
    return sid

spans_df = spans_df.copy()
spans_df["display"] = spans_df.apply(span_display_row, axis=1)

# Busca
q = st.text_input("Buscar vão (ex.: L35)", value="")
q_norm = (q or "").strip().upper()

df_f = spans_df
if q_norm:
    df_f = spans_df[spans_df["span_id"].str.upper().str.contains(q_norm, na=False)].copy()

if df_f.empty:
    st.warning("Nenhum vão encontrado para a busca informada.")
    st.stop()

# Selectbox com busca já aplicada
span_sel = st.selectbox(
    "Selecione o vão (Lxx):",
    options=df_f["display"].tolist(),
)

# recupera span_id real
row_sel = df_f[df_f["display"] == span_sel].iloc[0].to_dict()
SPAN_ID = str(row_sel["span_id"]).strip()
SPAN_BUS1 = (str(row_sel.get("bus1", "")).strip() if row_sel.get("bus1") is not None else None)
SPAN_BUS2 = (str(row_sel.get("bus2", "")).strip() if row_sel.get("bus2") is not None else None)

# =========================
# Carregar opções do span_id
# =========================
st.markdown("### Opções de Manobra (Top 3 do banco)")
opt_df = carregar_options_por_span(SPAN_ID)

# força apenas 3 opções (rank 1..3) e ordena
opt_df["option_rank"] = opt_df["option_rank"].astype(int)
opt_df = opt_df.sort_values("option_rank")
opt_df = opt_df[opt_df["option_rank"].isin([1, 2, 3])].copy()

if opt_df.empty:
    st.error(f"Não existem opções cadastradas em maneuver_options para {SPAN_ID}.")
    st.stop()

# monta tabela de resumo (com reports do DB)
summary_rows = []
cache_resumos = {}  # (rank)->dict

for _, o in opt_df.iterrows():
    rank = int(o["option_rank"])
    res = carregar_resumos(SPAN_ID, rank)
    cache_resumos[rank] = res

    flags = compute_status_flags(res)
    status_txt = render_status_badges(flags)

    load = res.get("load") or {}
    volt = res.get("volt") or {}
    loss = res.get("loss") or {}
    curr = res.get("curr") or {}

    kw_base = float(load.get("kw_off_base") or 0.0)
    kw_after = float(load.get("kw_off_after") or 0.0)
    rest_kw = float(load.get("rest_kw") or max(0.0, kw_base - kw_after))
    n_buses_off = int(load.get("n_buses_off") or 0)

    dp_kw = loss.get("dp_kw", None)
    dp_pct = loss.get("dp_pct", None)

    n_v_viol = int(volt.get("n_v_viol") or 0)

    has_new = int(curr.get("has_new_violation") or 0)
    has_worse = int(curr.get("has_worse_violation") or 0)

    summary_rows.append(
        dict(
            Opção=rank,
            Status=status_txt,
            Tipo=str(o.get("tipo") or "").strip(),
            NF_isoladora=pretty_nf_list(o.get("nf_isol_list") or []),
            NA=pretty_elem(o.get("na_elem")),
            NF_contenção=pretty_elem(o.get("nf_block_elem")),
            Manobras=int(o.get("n_manobras") or 0),
            Carga_base_kW=kw_base,
            Carga_final_kW=kw_after,
            Carga_rest_kW=rest_kw,
            Barras_off=n_buses_off,
            Tensão_viol=n_v_viol,
            Nova_I=has_new,
            Piora_I=has_worse,
            ΔP_kW=(None if dp_kw is None else float(dp_kw)),
            ΔP_pct=(None if dp_pct is None else float(dp_pct)),
        )
    )

df_summary = pd.DataFrame(summary_rows)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.caption("Status: ✅ OK | ⚠️V (tensão) | ⚠️I(new) (nova violação) | ⚠️I(↑) (piora) | ⚠️ΔP+ (perdas aumentaram)")

# =========================
# Selecionar opção p/ detalhar
# =========================
if "op_rank" not in st.session_state:
    st.session_state["op_rank"] = int(df_summary["Opção"].min())

colA, colB = st.columns([1, 2])
with colA:
    st.session_state["op_rank"] = st.selectbox(
        "Detalhar opção:",
        options=df_summary["Opção"].tolist(),
        index=df_summary["Opção"].tolist().index(st.session_state["op_rank"]),
        key="select_rank"
    )
with colB:
    st.write(f"**Vão:** `{SPAN_ID}`  |  **Operador:** `{nome_operador}`")

rank_sel = int(st.session_state["op_rank"])
opt_row = opt_df[opt_df["option_rank"] == rank_sel].iloc[0].to_dict()
res_sel = cache_resumos.get(rank_sel) or carregar_resumos(SPAN_ID, rank_sel)
flags_sel = compute_status_flags(res_sel)

# =========================
# Cards de detalhamento
# =========================
load = res_sel.get("load") or {}
volt = res_sel.get("volt") or {}
loss = res_sel.get("loss") or {}
curr = res_sel.get("curr") or {}

buses_off_list = load.get("buses_off_list") or []

st.markdown("### Detalhamento da Opção Selecionada")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Status", render_status_badges(flags_sel))
with c2:
    st.metric("Manobras", int(opt_row.get("n_manobras") or 0))
with c3:
    st.metric("Barras desligadas", int(load.get("n_buses_off") or 0))
with c4:
    st.metric("Tensão (barras em violação)", int(volt.get("n_v_viol") or 0))

st.write(
    f"- **NF Isolação (abrir):** `{[pretty_elem(x) for x in (opt_row.get('nf_isol_list') or [])]}`  \n"
    f"- **NA (fechar):** `{pretty_elem(opt_row.get('na_elem'))}`  \n"
    f"- **NF Contenção (abrir):** `{pretty_elem(opt_row.get('nf_block_elem'))}`"
)

# =========================
# Abas: Relatórios / Passos / Barras OFF
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Relatórios", "🧩 Passos", "🗺️ Mapa", "📌 Barras desligadas"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Impacto de Carga")
        kw_base = float(load.get("kw_off_base") or 0.0)
        kw_after = float(load.get("kw_off_after") or 0.0)
        rest_kw = float(load.get("rest_kw") or max(0.0, kw_base - kw_after))
        st.write(
            f"- **Carga base interrompida (kW):** `{kw_base:.2f}`  \n"
            f"- **Carga final interrompida (kW):** `{kw_after:.2f}`  \n"
            f"- **Carga restabelecida (kW):** `{rest_kw:.2f}`  \n"
            f"- **Nº barras desligadas:** `{int(load.get('n_buses_off') or 0)}`"
        )

    with col2:
        st.subheader("Tensão / Perdas / Corrente")
        n_v = int(volt.get("n_v_viol") or 0)
        worst_bus = volt.get("worst_v_bus", None)
        worst_vmin = volt.get("worst_vmin_pu", None)
        dp_kw = loss.get("dp_kw", None)
        dp_pct = loss.get("dp_pct", None)

        st.write(
            f"- **Tensão:** `{n_v}` barras em violação  \n"
            f"- **Pior barra:** `{worst_bus if worst_bus else '•'}`  \n"
            f"- **Vmin (pu):** `{(float(worst_vmin) if worst_vmin is not None else '•')}`  \n"
            f"- **ΔPerdas P (kW):** `{(float(dp_kw) if dp_kw is not None else '•')}`  \n"
            f"- **ΔPerdas (%):** `{(float(dp_pct) if dp_pct is not None else '•')}`  \n"
            f"- **Nova violação de corrente:** `{int(curr.get('has_new_violation') or 0)}`  \n"
            f"- **Piora de violação:** `{int(curr.get('has_worse_violation') or 0)}`"
        )

        if curr.get("worst_elem"):
            st.caption(
                f"Pior elemento: {pretty_elem(curr.get('worst_elem'))} | "
                f"ratio base={curr.get('worst_ratio_base')} | ratio after={curr.get('worst_ratio_after')}"
            )

with tab2:
    st.subheader("Sequência de Manobra (NF-NA-NF)")
    steps_df = carregar_steps(SPAN_ID, rank_sel)
    if steps_df.empty:
        st.info("Sem passos cadastrados para essa opção (maneuver_steps vazio).")
    else:
        st.dataframe(steps_df[["step_order", "op", "element"]], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Mapa da Manobra")
    st.plotly_chart(
        plotar_mapa_span_opcao(
            coords=coords,
            topo=topo,
            span_bus1=(SPAN_BUS1 if SPAN_BUS1 and SPAN_BUS1 != "None" else None),
            span_bus2=(SPAN_BUS2 if SPAN_BUS2 and SPAN_BUS2 != "None" else None),
            option_row=opt_row,
            buses_off_list=buses_off_list,
        ),
        use_container_width=True
    )

with tab4:
    st.subheader("Barras desligadas (buses_off_json)")
    if not buses_off_list:
        st.info("Lista de barras desligadas não disponível para essa opção.")
    else:
        st.write(buses_off_list)
        st.caption(f"Total: {len(buses_off_list)}")

st.markdown("---")
st.caption("Observação: como sua planilha/base só contém sequências OK, não aplicamos filtro. O status serve como verificação de consistência (tensão/corrente/perdas).")
