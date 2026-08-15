"""Gera as figuras ilustrativas dos manuscritos (`docs/manuscrito/`):

1. `figura1_esquema_metapaths.png` -- esquema abstrato dos 3 metapaths de
   hipotese da HIN (sem dado real, so ilustra a estrutura).
2. `figura2_caso_socio_comum.png` -- um caso REAL do banco (`data/raw/
   grande_vitoria.db`), **anonimizado antes de desenhar** (nome do socio e
   CNPJs das empresas substituidos por rotulos genericos) -- ilustra o
   mecanismo exato de circularidade discutido na Secao 5.3 dos manuscritos:
   duas empresas que so entram em `y_qualquer` (nao em `y_direto`) porque
   compartilham um socio que foi sancionado como pessoa fisica.

Caso escolhido (verificado contra o banco real em 14/08/2026, ver
`docs/research_plan.md` para a nota de rigor): socio com sancao CEIS
registrada como pessoa fisica, presente como socio em exatamente 2 empresas
da Grande Vitoria (o caso mais simples de visualizar dentre os 6 socios
vinculados que aparecem em mais de uma empresa no banco) -- nenhuma das
duas empresas tem sancao `match_confianca='direto'` propria.

Anonimizacao (decisao de LGPD/etica ja registrada em `docs/research_plan.md`,
secao 8: "tratar com cautela na exposicao de nomes individuais nos
resultados publicados"): nome do socio e CNPJs nunca aparecem na figura --
so atributos agregados nao identificantes (municipio, se tem sancao direta
propria).

Uso:
    uv run python scripts/gerar_figuras_manuscrito.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.data.loaders import GrandeVitoriaLoader

FIGURAS_DIR = Path(__file__).resolve().parent.parent / "docs" / "manuscrito" / "figuras"

# CNPJs do caso real escolhido (ver docstring) -- usados so para reconferir
# contra o banco real antes de desenhar, nunca aparecem na figura.
_CNPJ_EMPRESA_A = "16869491000141"
_CNPJ_EMPRESA_B = "13622530000113"
_NOME_SOCIO_REAL = "GILBERTO MIRANDA LANES"


def _verificar_caso_real(loader: GrandeVitoriaLoader) -> dict:
    """Reconfere o caso contra o banco real antes de desenhar -- nunca
    hardcodar um caso sem verificar que ele ainda bate com os dados atuais.
    """
    sancoes = loader.sancoes_administrativas()
    empresas = loader.empresas()
    socios = loader.socios()

    caso = sancoes[
        (sancoes["match_confianca"] == "socio") & (sancoes["nome_socio_vinculado"] == _NOME_SOCIO_REAL)
    ]
    assert set(caso["cnpj_empresa"]) == {_CNPJ_EMPRESA_A, _CNPJ_EMPRESA_B}, (
        "Caso real mudou ou nao bate mais com o banco atual -- reescolher exemplo."
    )

    municipio_a = empresas.loc[empresas["cnpj"] == _CNPJ_EMPRESA_A, "municipio"].iloc[0]
    municipio_b = empresas.loc[empresas["cnpj"] == _CNPJ_EMPRESA_B, "municipio"].iloc[0]
    diretas_a = sancoes[(sancoes["cnpj_empresa"] == _CNPJ_EMPRESA_A) & (sancoes["match_confianca"] == "direto")]
    diretas_b = sancoes[(sancoes["cnpj_empresa"] == _CNPJ_EMPRESA_B) & (sancoes["match_confianca"] == "direto")]
    n_empresas_do_socio = socios[socios["nome_socio"].str.upper().str.strip() == _NOME_SOCIO_REAL]["cnpj_empresa"].nunique()

    assert len(diretas_a) == 0 and len(diretas_b) == 0, "Uma das empresas tem sancao direta -- exemplo nao serve mais."
    assert n_empresas_do_socio == 2, "Socio passou a aparecer em outro numero de empresas -- reescolher exemplo."

    return {"municipio_a": municipio_a, "municipio_b": municipio_b}


def gerar_figura_esquema_metapaths() -> Path:
    """Figura 1: esquema abstrato dos 3 metapaths de hipotese -- sem dado
    real, so estrutura (nos genericos "Empresa 1"/"Empresa 2")."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metapaths = [
        ("Sócio comum", "Sócio"),
        ("Endereço comum", "Endereço"),
        ("Vínculo político comum", "Vínculo\npolítico"),
    ]
    for ax, (titulo, no_meio) in zip(axes, metapaths, strict=True):
        g = nx.Graph()
        g.add_edges_from([("Empresa 1", no_meio), (no_meio, "Empresa 2")])
        pos = {"Empresa 1": (0, 0), no_meio: (1, 0.6), "Empresa 2": (2, 0)}
        cores = ["#4C72B0", "#DD8452", "#4C72B0"]
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2600, node_color=cores, edgecolors="black")
        nx.draw_networkx_edges(g, pos, ax=ax, width=1.6)
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")
        ax.set_title(titulo, fontsize=10)
        ax.axis("off")
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.5, 1.1)

    fig.suptitle("Metapaths de hipótese da HIN (empresa–X–empresa)", fontsize=12)
    fig.tight_layout()
    out = FIGURAS_DIR / "figura1_esquema_metapaths.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def gerar_figura_caso_socio_comum(municipio_a: str, municipio_b: str) -> Path:
    """Figura 2: caso real anonimizado -- duas empresas que so entram em
    `y_qualquer` via socio comum (nenhuma tem sancao direta propria)."""
    g = nx.Graph()
    empresa_a = f"Empresa A\n({municipio_a})"
    empresa_b = f"Empresa B\n({municipio_b})"
    socio = "Sócio X"
    g.add_edge(empresa_a, socio)
    g.add_edge(socio, empresa_b)
    pos = {empresa_a: (0, 0), socio: (1.3, 0.8), empresa_b: (2.6, 0)}

    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx_nodes(g, pos, nodelist=[empresa_a, empresa_b], ax=ax, node_size=4200, node_color="#4C72B0", edgecolors="black")
    nx.draw_networkx_nodes(g, pos, nodelist=[socio], ax=ax, node_size=4200, node_color="#C44E52", edgecolors="black")
    nx.draw_networkx_edges(g, pos, ax=ax, width=2.0)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9.5, font_color="white", font_weight="bold")

    ax.text(1.3, 1.35, "sancionado como pessoa física (CEIS)", ha="center", fontsize=8.5, style="italic")
    ax.text(
        1.3, -0.55,
        "Empresa A e Empresa B: y_direto = False (nenhuma sancionada diretamente)\n"
        "y_qualquer = True para as duas (só via sócio comum)",
        ha="center", fontsize=9, style="italic",
    )
    ax.set_xlim(-0.9, 3.5)
    ax.set_ylim(-1.0, 1.7)
    ax.axis("off")
    ax.set_title("Caso real (anonimizado): circularidade via sócio comum", fontsize=11)
    fig.tight_layout()
    out = FIGURAS_DIR / "figura2_caso_socio_comum.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    loader = GrandeVitoriaLoader()
    municipios = _verificar_caso_real(loader)

    f1 = gerar_figura_esquema_metapaths()
    print(f"Figura 1 salva em: {f1}")
    f2 = gerar_figura_caso_socio_comum(municipios["municipio_a"], municipios["municipio_b"])
    print(f"Figura 2 salva em: {f2}")


if __name__ == "__main__":
    main()
