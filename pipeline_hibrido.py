import pandas as pd
from pyswip import Prolog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CSV_PATH = "dados_financeiros.csv"
PROLOG_PATH = "rede_social.pl"

# ============================================================
# 1. Inicializar Prolog e carregar base de conhecimento
# ============================================================

prolog = Prolog()
prolog.consult(PROLOG_PATH)

# ============================================================
# 2. Descobrir todos os inadimplentes diretamente do Prolog
# ============================================================

def obter_todos_inadimplentes():
    """Consulta o Prolog e retorna lista de todos os inadimplentes."""
    resultados = list(prolog.query("inadimplente(X)"))
    return [r["X"] for r in resultados]

INADIMPLENTES = obter_todos_inadimplentes()

# ============================================================
# 3. Função de risco relacional (contra TODOS os inadimplentes)
# ============================================================

def obter_grau_minimo(nome, inadimplentes):
    """
    Para cada inadimplente na base, busca o menor grau de conexão
    com `nome`. Retorna o menor grau encontrado entre todos.
    Retorna 999 se não houver nenhuma conexão.
    """
    menor_grau_global = 999

    for alvo in inadimplentes:
        if nome == alvo:
            # O próprio inadimplente: grau 0
            return 0
        query = list(prolog.query(f"risco_conexao({nome}, {alvo}, Grau)"))
        if query:
            menor_grau = min(r["Grau"] for r in query)
            if menor_grau < menor_grau_global:
                menor_grau_global = menor_grau

    return menor_grau_global

def obter_detalhes_risco(nome, inadimplentes):
    """
    Retorna um dict com o grau mínimo para cada inadimplente,
    além do grau global mínimo.
    """
    detalhes = {}
    for alvo in inadimplentes:
        if nome == alvo:
            detalhes[alvo] = 0
            continue
        query = list(prolog.query(f"risco_conexao({nome}, {alvo}, Grau)"))
        if query:
            detalhes[alvo] = min(r["Grau"] for r in query)
        else:
            detalhes[alvo] = 999

    grau_minimo = min(detalhes.values()) if detalhes else 999
    return grau_minimo, detalhes

# ============================================================
# 4. Carregar dados e treinar modelo
# ============================================================

df = pd.read_csv(CSV_PATH)

# Feature relacional: menor grau contra qualquer inadimplente
df["grau_risco_rede"] = df["cliente_id"].apply(
    lambda nome: obter_grau_minimo(nome, INADIMPLENTES)
)

FEATURES = ["renda_mensal", "score_classico", "grau_risco_rede"]
X = df[FEATURES]
y = df["inadimplente_historico"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_scaled, y)

# ============================================================
# 5. Função principal de análise para qualquer cliente
# ============================================================

def analisar_cliente(nome):
    """
    Analisa o risco de crédito de um cliente pelo nome.
    Funciona para clientes presentes no CSV ou apenas na rede social.
    """
    nome = nome.strip().lower()
    print(f"\n{'='*58}")
    print(f"  ANÁLISE DE RISCO RELACIONAL — cliente: {nome.upper()}")
    print(f"{'='*58}")

    # --- Dados financeiros (se existirem no CSV) ---
    linha = df[df["cliente_id"] == nome]
    tem_dados_financeiros = not linha.empty

    if tem_dados_financeiros:
        renda    = linha["renda_mensal"].values[0]
        score    = linha["score_classico"].values[0]
        historico = linha["inadimplente_historico"].values[0]
        grau_risco, detalhes = obter_detalhes_risco(nome, INADIMPLENTES)

        entrada = pd.DataFrame([[renda, score, grau_risco]], columns=FEATURES)
        entrada_scaled = scaler.transform(entrada)
        prob = modelo.predict_proba(entrada_scaled)[0][1]

        print(f"\n  Dados financeiros encontrados no CSV:")
        print(f"    Renda mensal       : R$ {renda:,.2f}")
        print(f"    Score clássico     : {score}")
        print(f"    Inadimplente hist. : {'SIM' if historico else 'NÃO'}")
    else:
        # Sem dados no CSV: usa medianas do dataset para simulação
        renda  = df["renda_mensal"].median()
        score  = df["score_classico"].median()
        grau_risco, detalhes = obter_detalhes_risco(nome, INADIMPLENTES)

        entrada = pd.DataFrame([[renda, score, grau_risco]], columns=FEATURES)
        entrada_scaled = scaler.transform(entrada)
        prob = modelo.predict_proba(entrada_scaled)[0][1]

        print(f"\n  ⚠  Cliente não encontrado no CSV.")
        print(f"     Probabilidade estimada com medianas do dataset")
        print(f"     (renda=R${renda:,.0f}, score={score:.0f}) + grau relacional real.")

    # --- Grau de proximidade com cada inadimplente ---
    print(f"\n  Proximidade com inadimplentes conhecidos:")
    print(f"  {'Inadimplente':<14} {'Grau':>6}  {'Interpretação'}")
    print(f"  {'-'*14} {'-'*6}  {'-'*28}")

    for inadim, grau in sorted(detalhes.items(), key=lambda x: x[1]):
        if grau == 0:
            interp = "É o próprio inadimplente"
        elif grau == 1:
            interp = "Conexão DIRETA ⚠"
        elif grau == 2:
            interp = "2 saltos (alto risco)"
        elif grau == 3:
            interp = "3 saltos (risco moderado)"
        elif grau <= 5:
            interp = f"{grau} saltos (risco baixo)"
        else:
            interp = "Sem conexão detectada"
        print(f"  {inadim:<14} {grau:>6}  {interp}")

    print(f"\n  Grau mínimo global       : {grau_risco}")

    # --- Probabilidade e veredicto ---
    print(f"\n  Probabilidade de inadimplência : {prob:.1%}")

    if prob >= 0.60:
        risco_label = "🔴 ALTO"
    elif prob >= 0.35:
        risco_label = "🟡 MODERADO"
    else:
        risco_label = "🟢 BAIXO"

    print(f"  Classificação de risco         : {risco_label}")

    # --- Saída no estilo ProbLog ---
    inadim_mais_proximo = min(detalhes, key=detalhes.get)
    grau_mais_proximo   = detalhes[inadim_mais_proximo]
    print(f"\n  Saída relacional-estatística (estilo ProbLog):")
    print(f"  {prob:.2f} :: risco({nome}) :-")
    print(f"      conectado_a({nome}, {inadim_mais_proximo}, grau={grau_mais_proximo}).")
    print(f"{'='*58}\n")

# ============================================================
# 6. Interface de entrada genérica
# ============================================================

def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   PIPELINE HÍBRIDO — Análise de Risco de Crédito    ║")
    print("║   (Regressão Logística + Lógica Relacional Prolog)  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Inadimplentes na base Prolog: {', '.join(INADIMPLENTES)}")
    print(f"  Clientes com dados financeiros: {', '.join(df['cliente_id'].tolist())}")

    # Aceita nomes via argumento de linha de comando OU modo interativo
    if len(sys.argv) > 1:
        nomes = sys.argv[1:]
    else:
        print("\n  Digite os nomes para análise separados por vírgula")
        print("  (ou pressione Enter para analisar todos do CSV):")
        entrada = input("  > ").strip()
        if entrada:
            nomes = [n.strip() for n in entrada.split(",") if n.strip()]
        else:
            nomes = df["cliente_id"].tolist()

    for nome in nomes:
        analisar_cliente(nome)

if __name__ == "__main__":
    main()