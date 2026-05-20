% ==========================================
% 1. BASE DE FATOS: TRANSAÇÕES (CONEXÕES)
% Formato: transacao_entre(Origem, Destino, Valor).
% ==========================================

% Relações originais da base de dados
transacao_entre(joao, ana, 1500).
transacao_entre(ana, carlos, 800). 
transacao_entre(carlos, daniel, 50).
transacao_entre(joao, marcos, 3000).
transacao_entre(marcos, elena, 1200).
transacao_entre(elena, roberto, 450).
transacao_entre(roberto, carlos, 250).
transacao_entre(ana, bia, 600).
transacao_entre(bia, felipe, 900).
transacao_entre(felipe, daniel, 100).
transacao_entre(lucas, elena, 2000).
transacao_entre(lucas, thiago, 500).
transacao_entre(thiago, marta, 750).
transacao_entre(marta, felipe, 300).
transacao_entre(bruno, joao, 2100).
transacao_entre(julia, roberto, 1100).
transacao_entre(carlos, fernanda, 400).
transacao_entre(fernanda, bia, 950).
transacao_entre(marcos, thiago, 1300).
transacao_entre(bruno, julia, 850).
transacao_entre(marta, daniel, 150).

% ==========================================
% 2. BASE DE FATOS: HISTÓRICO DE INADIMPLÊNCIA
% Formato: inadimplente(Nome).
% ==========================================

% Inadimplente original
inadimplente(daniel).
inadimplente(roberto).
inadimplente(felipe).


% ==========================================
% 3. REGRAS: PROPAGAÇÃO DE RISCO RECURSIVA (CORRIGIDA)
% ==========================================

% Regra principal de ancoragem: inicializa a travessia rastreando o nó de Origem.
risco_conexao(X, Y, Grau) :-
    risco_conexao_aux(X, Y, [X], Grau).

% Caso Base (Grau 1): Avaliação de conexão direta.
risco_conexao_aux(X, Y, _, 1) :- 
    (transacao_entre(X, Y, _) ; transacao_entre(Y, X, _)).

% Caso Recursivo (Grau > 1): Expansão para a vizinhança isolando nós previamente mapeados.
risco_conexao_aux(X, Y, Visitados, Grau) :-
    (transacao_entre(X, Z, _) ; transacao_entre(Z, X, _)),
    \+ member(Z, Visitados), 
    risco_conexao_aux(Z, Y, [Z|Visitados], GrauAnterior),
    Grau is GrauAnterior + 1.