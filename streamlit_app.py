import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Manobras - IEEE-123 Bus", layout="wide")
st.title("Manobras - IEEE-123 Bus (Modo 2)")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "toposwitch_modo2.db"

# =========================================================
# LOGIN
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
# DB helpers
# =========================================================
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

@st.cache_data(show_spinner=False)
def listar_tabelas() -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    out = [r[0] for r in cur.fetchall()]
    conn.close()
    return out

@st.cache_data(show_spinner=False)
def has_table(name: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    ok = cur.fetchone() is not None
    conn.close()
    return ok

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
        topo.append(dict(
            line=str(line).strip(),
            from_bus=str(f).strip(),
            to_bus=str(t).strip(),
            is_switch=bool(is_sw),
            norm=str(norm).strip() if norm is not None else "",
        ))
    return topo

def norm_elem_token(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s.startswith("line."):
        s = s.replace("line.", "", 1)
    return s

def parse_json_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [norm_elem_token(x) for x in v if str(x).strip()]
    except Exception:
        pass
    return []

@st.cache_data(show_spinner=False)
def listar_spans() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT span_id, line_elem, bus1, bus2 FROM spans ORDER BY span_id",
        conn
    )
    conn.close()
    return df

@st.cache_data(show_spinner=False)
def carregar_opcoes_span(span_id: str) -> pd.DataFrame:
    """
    Retorna opções (até 3) já com joins de métricas.
    """
    conn = get_connection()
    q = """
    SELECT
        mo.span_id,
        mo.option_rank,
        mo.tipo,
        mo.nf_isol_json,
        mo.na_elem,
        mo.nf_block_elem,
        mo.n_manobras,

        li.kw_off_base,
        li.kw_off_after,
        li.rest_kw,
        li.n_buses_off,
        li.buses_off_json,

        ls.p_loss_base_kw,
        ls.p_loss_after_kw,
        ls.dp_kw,
        ls.dp_pct,

        vs.n_v_viol,
        vs.worst_v_bus,
        vs.worst_vmin_pu,
        vs.worst_vmax_pu,

        cs.has_new_violation,
        cs.has_worse_violation,
        cs.worst_elem,
        cs.worst_ratio_base,
        cs.worst_ratio_after
    FROM maneuver_options mo
    LEFT JOIN option_load_impact li
        ON li.span_id = mo.span_id AND li.option_rank = mo.option_rank
    LEFT JOIN option_losses ls
        ON ls.span_id = mo.span_id AND ls.option_rank = mo.option_rank
    LEFT JOIN option_voltage_summary vs
        ON vs.span_id = mo.span_id AND vs.option_rank = mo.option_rank
    LEFT JOIN option_current_summary cs
        ON cs.span_id = mo.span_id AND cs.option_rank = mo.option_rank
    WHERE mo.span_id = ?
    ORDER BY mo.option_rank
    LIMIT 3
    """
    df = pd.read_sql_query(q, conn, params=(span_id,))
    conn.close()
    return df

@st.cache_data(show_spinner=False)
def carregar_steps(span_id: str, option_rank: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT step_order, op, element
        FROM maneuver_steps
        WHERE span_id=? AND option_rank=?
        ORDER BY step_order
        """,
        conn,
        params=(span_id, int(option_rank))
    )
    conn.close()
    return df

# =========================================================
# MAPA
# =========================================================
def topo_por_line_dict(topo: List[Dict]) -> Dict[str, Dict]:
    return {norm_elem_token(el["line"]): el for el in topo}

def construir_mapa_base(coords: Dict[str, Tuple[float, float]], topo: List[Dict]) -> go.Figure:
    edge_x, edge_y = [], []
    for el in topo:
        u, v = el["from_bus"], el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#D3D3D3", width=1),
        hoverinfo="none",
        name="Linhas"
    ))

    node_x = [coords[b][0] for b in coords]
    node_y = [coords[b][1] for b in coords]
    node_text = list(coords.keys())

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=6, color="green"),
        name="Barras",
        hovertemplate="Barra %{text}<extra></extra>",
    ))

    fig.update_layout(
        height=650,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

def add_line_trace(fig: go.Figure, coords, topo_d, ln_token: str, color: str, label: str, width: int, dash: Optional[str]=None):
    if not ln_token:
        return
    el = topo_d.get(norm_elem_token(ln_token))
    if not el:
        return
    u, v = el["from_bus"], el["to_bus"]
    if u in coords and v in coords:
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=label,
            hoverinfo="none"
        ))

def plotar_mapa(coords, topo, span_line_token: str, nf_list: List[str], na: str, nf_block: str, buses_off: List[str]) -> go.Figure:
    fig = construir_mapa_base(coords, topo)
    topo_d = topo_por_line_dict(topo)

    # destaca o vão selecionado (preto)
    add_line_trace(fig, coords, topo_d, span_line_token, "black", "Vão selecionado", 5, None)

    # barras desligadas (vermelho) e energizadas (verde)
    all_buses = set(coords.keys())
    buses_off_set = set([str(b).strip() for b in (buses_off or [])])
    buses_on = all_buses - buses_off_set

    off_x = [coords[b][0] for b in buses_off_set if b in coords]
    off_y = [coords[b][1] for b in buses_off_set if b in coords]
    if off_x:
        fig.add_trace(go.Scatter(x=off_x, y=off_y, mode="markers",
                                 marker=dict(size=8, color="red"),
                                 name="Barras desligadas", hoverinfo="skip"))

    on_x = [coords[b][0] for b in buses_on if b in coords]
    on_y = [coords[b][1] for b in buses_on if b in coords]
    if on_x:
        fig.add_trace(go.Scatter(x=on_x, y=on_y, mode="markers",
                                 marker=dict(size=7, color="green"),
                                 name="Barras energizadas", hoverinfo="skip"))

    # NFs/NA/NFblock
    for nf in (nf_list or []):
        add_line_trace(fig, coords, topo_d, nf, "red", "NF isolação (abrir)", 6, "dash")

    add_line_trace(fig, coords, topo_d, na, "cyan", "NA (fechar)", 7, None)
    add_line_trace(fig, coords, topo_d, nf_block, "purple", "NF contenção (abrir)", 7, None)

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# =========================================================
# UI
# =========================================================
st.sidebar.header("📂 Status do banco (Modo 2)")

if not DB_PATH.exists():
    st.sidebar.error(f"Banco `{DB_PATH.name}` não encontrado na pasta do app.")
    st.stop()

tabelas = listar_tabelas()
st.sidebar.write("Banco:", f"`{DB_PATH.name}`")
st.sidebar.write("Tabelas:", ", ".join(tabelas))

# valida se tem mapa
if not has_table("coords") or not has_table("topology"):
    st.error(
        "Seu banco novo ainda não tem as tabelas **coords** e **topology**.\n\n"
        "➡️ Rode o script `enrich_db_map.py` para preencher coords/topology e habilitar o mapa."
    )
    st.stop()

coords = carregar_coords()
topo = carregar_topologia()
if not coords or not topo:
    st.error("`coords` ou `topology` está vazio. Rode novamente o enrich_db_map.py (e confira BusCoords.dat).")
    st.stop()

nome_operador = st.text_input("Nome do operador responsável", value="Ellen")
st.info(f"Usuário: **{nome_operador}**")

st.markdown("---")
st.subheader("Rede IEEE-123 Bus Operação")
st.plotly_chart(construir_mapa_base(coords, topo), use_container_width=True)

st.markdown("---")
st.subheader("Operação por Vão (vão simples)")

spans_df = listar_spans()
if spans_df.empty:
    st.error("Tabela `spans` está vazia.")
    st.stop()

# Busca + select filtrado
busca = st.text_input("Buscar vão (ex.: L35)", value="")
spans_f = spans_df.copy()
if busca.strip():
    spans_f = spans_f[spans_f["span_id"].str.contains(busca.strip(), case=False, na=False)]

span_opts = spans_f["span_id"].tolist()
if not span_opts:
    st.warning("Nenhum vão encontrado com esse filtro.")
    st.stop()

span_id = st.selectbox("Selecione o vão (span_id):", options=span_opts, index=0)

# carrega 3 opções
opt_df = carregar_opcoes_span(span_id)
if opt_df.empty:
    st.error("Não há opções (maneuver_options) para esse vão.")
    st.stop()

def status_ok_atencao(row) -> tuple[str, List[str]]:
    reasons = []
    n_v_viol = int(row.get("n_v_viol") or 0)
    if n_v_viol > 0:
        reasons.append("Violação de tensão")

    if int(row.get("has_new_violation") or 0) == 1:
        reasons.append("Nova violação de corrente")

    if int(row.get("has_worse_violation") or 0) == 1:
        reasons.append("Piora de violação de corrente")

    dp_kw = row.get("dp_kw")
    try:
        if dp_kw is not None and float(dp_kw) > 0:
            reasons.append("Δperdas > 0")
    except Exception:
        pass

    if reasons:
        return "⚠️ Atenção", reasons
    return "✅ OK", reasons

# monta tabela resumo
rows = []
for _, r in opt_df.iterrows():
    nf_list = parse_json_list(r.get("nf_isol_json"))
    na = norm_elem_token(r.get("na_elem"))
    nf_block = norm_elem_token(r.get("nf_block_elem"))

    status, reasons = status_ok_atencao(r)

    rows.append({
        "Opção": int(r["option_rank"]),
        "Status": status,
        "Tipo": str(r.get("tipo") or ""),
        "NF_isolação": ",".join(nf_list) if nf_list else "•",
        "NA": na if na else "•",
        "NF_contenção": nf_block if nf_block else "•",
        "Manobras": int(r.get("n_manobras") or 0),

        "kW_base": float(r.get("kw_off_base") or 0.0),
        "kW_final": float(r.get("kw_off_after") or 0.0),
        "kW_rest": float(r.get("rest_kw") or 0.0),
        "Barras_off": int(r.get("n_buses_off") or 0),

        "Perdas_base_kW": float(r.get("p_loss_base_kw") or 0.0),
        "Perdas_final_kW": float(r.get("p_loss_after_kw") or 0.0),
        "Δperdas_kW": float(r.get("dp_kw") or 0.0),

        "V_viol": int(r.get("n_v_viol") or 0),
        "I_nova": int(r.get("has_new_violation") or 0),
        "I_piora": int(r.get("has_worse_violation") or 0),
        "Motivos": "; ".join(reasons) if reasons else "—",
    })

df_show = pd.DataFrame(rows).sort_values("Opção")
st.markdown("### Opções (máx. 3)")
st.dataframe(df_show, use_container_width=True)

# escolha da opção
if "opt_rank" not in st.session_state:
    st.session_state["opt_rank"] = int(df_show["Opção"].iloc[0])

st.session_state["opt_rank"] = st.selectbox(
    "Detalhar opção:",
    options=df_show["Opção"].tolist(),
    index=df_show["Opção"].tolist().index(st.session_state["opt_rank"])
)

# pega linha selecionada
sel = opt_df[opt_df["option_rank"] == st.session_state["opt_rank"]].iloc[0]

nf_list = parse_json_list(sel.get("nf_isol_json"))
na = norm_elem_token(sel.get("na_elem"))
nf_block = norm_elem_token(sel.get("nf_block_elem"))

# span_line_token
span_line_elem = str(spans_df[spans_df["span_id"] == span_id]["line_elem"].iloc[0] or "")
span_line_token = norm_elem_token(span_line_elem)

# buses_off list
buses_off = []
try:
    buses_off = json.loads(sel.get("buses_off_json") or "[]")
except Exception:
    buses_off = []

steps_df = carregar_steps(span_id, int(sel["option_rank"]))
status, reasons = status_ok_atencao(sel)

tab1, tab2, tab3 = st.tabs(["Detalhamento", "Mapa", "Barras desligadas"])

with tab1:
    st.subheader(f"Detalhamento — {span_id} | Opção {int(sel['option_rank'])}")
    st.write(f"- **Status:** {status}")
    if reasons:
        st.write(f"- **Motivos:** {', '.join(reasons)}")

    st.write(
        f"- **NF isolação:** `{nf_list if nf_list else '-'}`\n"
        f"- **NA:** `{na if na else '-'}`\n"
        f"- **NF contenção:** `{nf_block if nf_block else '-'}`\n"
        f"- **Manobras:** `{int(sel.get('n_manobras') or 0)}`\n"
        f"- **Carga base (kW):** `{float(sel.get('kw_off_base') or 0.0):.2f}`\n"
        f"- **Carga final (kW):** `{float(sel.get('kw_off_after') or 0.0):.2f}`\n"
        f"- **Carga restabelecida (kW):** `{float(sel.get('rest_kw') or 0.0):.2f}`\n"
        f"- **Barras desligadas:** `{int(sel.get('n_buses_off') or 0)}`\n"
        f"- **Perdas base (kW):** `{float(sel.get('p_loss_base_kw') or 0.0):.2f}`\n"
        f"- **Perdas final (kW):** `{float(sel.get('p_loss_after_kw') or 0.0):.2f}`\n"
        f"- **Δperdas (kW):** `{float(sel.get('dp_kw') or 0.0):.2f}`\n"
    )

    st.markdown("#### Passo a passo")
    if steps_df.empty:
        st.info("Sem passos cadastrados em `maneuver_steps` para esta opção.")
    else:
        st.dataframe(steps_df, use_container_width=True)

    st.markdown("#### Restrições (resumo)")
    st.write(
        f"- **Violação de tensão (n):** `{int(sel.get('n_v_viol') or 0)}`\n"
        f"- **Pior barra (tensão):** `{sel.get('worst_v_bus')}` | "
        f"vmin `{sel.get('worst_vmin_pu')}` | vmax `{sel.get('worst_vmax_pu')}`\n"
        f"- **Nova violação de corrente:** `{int(sel.get('has_new_violation') or 0)}`\n"
        f"- **Piora de violação de corrente:** `{int(sel.get('has_worse_violation') or 0)}`\n"
        f"- **Pior elemento (corrente):** `{sel.get('worst_elem')}` | "
        f"ratio base `{sel.get('worst_ratio_base')}` | ratio após `{sel.get('worst_ratio_after')}`\n"
    )

with tab2:
    st.subheader("🗺️ Mapa da manobra")
    fig = plotar_mapa(coords, topo, span_line_token, nf_list, na, nf_block, buses_off)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Barras desligadas")
    st.write(buses_off if buses_off else [])

# aba “rejeitadas” (vazia por enquanto)
st.markdown("---")
st.subheader("Manobras rejeitadas")
st.caption("No momento, a tabela `rejected_maneuvers` está vazia (só sequências OK).")
st.info("Quando você quiser, eu adiciono aqui a consulta + filtros por motivo, mas por agora não é necessário.")
