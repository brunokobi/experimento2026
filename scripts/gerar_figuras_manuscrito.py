"""Gera as figuras ilustrativas dos manuscritos (`docs/manuscrito/`), em
portugues (pra `dissertacao_pt.md`) e em ingles (pra `paper_en.md` -- achado
real ao compilar o preprint do arXiv em 15/08/2026: as duas versoes do
manuscrito referenciavam o MESMO arquivo de figura, com texto em portugues
dentro da imagem, o que ficava incorreto no artigo em ingles). Nomes de
arquivo numerados pela ordem em que cada figura aparece no texto (Secao 3.2
antes da 3.3), nao pela ordem em que sao geradas neste script, com sufixo
de idioma:

1. `figura1_caso_socio_comum_{pt,en}.png` -- um caso REAL do banco
   (`data/raw/grande_vitoria.db`), **anonimizado antes de desenhar** (nome
   do socio e CNPJs das empresas substituidos por rotulos genericos) --
   ilustra o mecanismo exato de circularidade discutido na Secao 5.3 dos
   manuscritos: duas empresas que so entram em `y_qualquer` (nao em
   `y_direto`) porque compartilham um socio que foi sancionado como pessoa
   fisica. Aparece na Secao 3.2 dos manuscritos.
2. `figura2_esquema_metapaths_{pt,en}.png` -- esquema abstrato dos 3
   metapaths de hipotese da HIN (sem dado real, so ilustra a estrutura).
   Aparece na Secao 3.3, depois da Figura 1.

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

_TEXTOS = {
    "pt": {
        "socio": "Sócio X",
        "sancionado": "sancionado como pessoa física (CEIS)",
        "legenda_caso": (
            "Empresa A e Empresa B: y_direto = False (nenhuma sancionada diretamente)\n"
            "y_qualquer = True para as duas (só via sócio comum)"
        ),
        "titulo_caso": "Caso real (anonimizado): circularidade via sócio comum",
        "metapaths": [
            ("Sócio comum", "Sócio"),
            ("Endereço comum", "Endereço"),
            ("Vínculo político comum", "Vínculo\npolítico"),
        ],
        "titulo_esquema": "Metapaths de hipótese da HIN (empresa–X–empresa)",
        "empresa": "Empresa",
    },
    "en": {
        "socio": "Partner X",
        "sancionado": "personally sanctioned (CEIS)",
        "legenda_caso": (
            "Company A and Company B: y_direto = False (neither directly sanctioned)\n"
            "y_qualquer = True for both (only via shared partner)"
        ),
        "titulo_caso": "Real case (anonymized): shared-partner circularity",
        "metapaths": [
            ("Shared partner", "Partner"),
            ("Shared address", "Address"),
            ("Shared political\nconnection", "Political\nconnection"),
        ],
        "titulo_esquema": "HIN hypothesis metapaths (company–X–company)",
        "empresa": "Company",
    },
}


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


def gerar_figura_esquema_metapaths(lang: str) -> Path:
    """Figura 2: esquema abstrato dos 3 metapaths de hipotese -- sem dado
    real, so estrutura (nos genericos "Empresa 1"/"Empresa 2" ou "Company
    1"/"Company 2"). Numerada depois da Figura 1 porque aparece mais tarde
    no texto dos manuscritos (Secao 3.3, depois do caso real da Secao 3.2)."""
    t = _TEXTOS[lang]
    empresa_1, empresa_2 = f"{t['empresa']} 1", f"{t['empresa']} 2"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (titulo, no_meio) in zip(axes, t["metapaths"], strict=True):
        g = nx.Graph()
        g.add_edges_from([(empresa_1, no_meio), (no_meio, empresa_2)])
        pos = {empresa_1: (0, 0), no_meio: (1, 0.6), empresa_2: (2, 0)}
        cores = ["#4C72B0", "#DD8452", "#4C72B0"]
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2600, node_color=cores, edgecolors="black")
        nx.draw_networkx_edges(g, pos, ax=ax, width=1.6)
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_color="white", font_weight="bold")
        ax.set_title(titulo, fontsize=10)
        ax.axis("off")
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.5, 1.1)

    fig.suptitle(t["titulo_esquema"], fontsize=12)
    fig.tight_layout()
    out = FIGURAS_DIR / f"figura2_esquema_metapaths_{lang}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def gerar_figura_caso_socio_comum(municipio_a: str, municipio_b: str, lang: str) -> Path:
    """Figura 1: caso real anonimizado -- duas empresas que so entram em
    `y_qualquer` via socio comum (nenhuma tem sancao direta propria).
    Numerada como a primeira figura porque aparece na Secao 3.2, antes do
    esquema abstrato da Secao 3.3."""
    t = _TEXTOS[lang]
    empresa_a_label = "Empresa A" if lang == "pt" else "Company A"
    empresa_b_label = "Empresa B" if lang == "pt" else "Company B"
    g = nx.Graph()
    empresa_a = f"{empresa_a_label}\n({municipio_a})"
    empresa_b = f"{empresa_b_label}\n({municipio_b})"
    socio = t["socio"]
    g.add_edge(empresa_a, socio)
    g.add_edge(socio, empresa_b)
    pos = {empresa_a: (0, 0), socio: (1.3, 0.8), empresa_b: (2.6, 0)}

    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx_nodes(g, pos, nodelist=[empresa_a, empresa_b], ax=ax, node_size=4200, node_color="#4C72B0", edgecolors="black")
    nx.draw_networkx_nodes(g, pos, nodelist=[socio], ax=ax, node_size=4200, node_color="#C44E52", edgecolors="black")
    nx.draw_networkx_edges(g, pos, ax=ax, width=2.0)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9.5, font_color="white", font_weight="bold")

    ax.text(1.3, 1.35, t["sancionado"], ha="center", fontsize=8.5, style="italic")
    ax.text(1.3, -0.55, t["legenda_caso"], ha="center", fontsize=9, style="italic")
    ax.set_xlim(-0.9, 3.5)
    ax.set_ylim(-1.0, 1.7)
    ax.axis("off")
    ax.set_title(t["titulo_caso"], fontsize=11)
    fig.tight_layout()
    out = FIGURAS_DIR / f"figura1_caso_socio_comum_{lang}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    loader = GrandeVitoriaLoader()
    municipios = _verificar_caso_real(loader)

    for lang in ("pt", "en"):
        f1 = gerar_figura_caso_socio_comum(municipios["municipio_a"], municipios["municipio_b"], lang)
        print(f"Figura 1 ({lang}) salva em: {f1}")
        f2 = gerar_figura_esquema_metapaths(lang)
        print(f"Figura 2 ({lang}) salva em: {f2}")


if __name__ == "__main__":
    main()
