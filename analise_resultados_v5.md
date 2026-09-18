# Análise dos resultados da execução completa (v5)

Base: `experimentos/results/` — 4 backbones, Abordagem 1 e 2, LOMDO, grid search 3×3 e o experimento de explicabilidade comum. Tudo rodou; nada ficou pendente.

---

## 1. O problema estrutural que explica quase todo o resto

O `split_summary.json` mostra uma distribuição de classes muito desigual entre as partições, porque o split é por **data de captura** e as datas não têm a mesma composição:

| partição | healthy | rust | % rust | n |
|---|---|---|---|---|
| treino | 719 | 1424 | 66,5% | 2143 |
| **validação** | **567** | **48** | **7,8%** | 615 |
| teste (ASDID) | 464 | 310 | 40,1% | 774 |
| Embrapa (OOD) | 9 | 65 | 87,8% | 74 |

O *early stopping* e o `ReduceLROnPlateau` monitoram **F1 da classe rust na validação — que tem só 48 exemplos positivos**. Cada positivo vale ~2 pontos percentuais de recall. Uma variação mínima na trajetória de treino muda a época escolhida, e a época escolhida muda muito o resultado final.

Isso não é teoria: é o que se vê nos dados.

- **A ResNet-50 mudou completamente entre a execução anterior e esta.** Antes (no que está escrito no docx): F1 ASDID 0,954. Agora: **0,798**, com recall despencando para 0,665 — deixa passar 104 de 310 imagens doentes. O código é o mesmo; o que mudou foi a trajetória do RNG (antes ela rodava depois da ResNet-18 nas células manuais, agora `train_backbone()` reseta a semente antes de construir o modelo).
- **A VGG-16 colapsa em 2 dos 4 folds do LOMDO** (detalhe na seção 3).

Ou seja: a instabilidade não é ruído de terceira casa decimal, é da ordem de 0,15 no F1. **Esse é o ponto mais atacável do trabalho numa banca** e precisa ser tratado explicitamente — seja corrigindo, seja assumindo por escrito.

**Recomendação concreta:** trocar o critério de *early stopping* de F1 para **AUC-ROC de validação**. AUC não depende de limiar e é bem mais estável com poucos positivos. Alternativamente, refazer o split garantindo estratificação mínima por classe dentro do agrupamento por data. A primeira opção é uma linha de código (`CONFIG["MONITOR"] = "auc"`, se o `run_epoch` já calcular AUC) e não invalida o desenho por grupo.

### Um aviso sobre o teste OOD

A Embrapa tem **9 imagens healthy**. Todos os modelos acertam as 9 (especificidade = 1,000), então a acurácia balanceada OOD é sempre `(1,000 + recall)/2` e a precisão é sempre 1,000. Isso infla a leitura: um único erro numa dessas 9 imagens derrubaria a especificidade para 0,889. **Qualquer afirmação sobre a classe saudável fora do domínio está apoiada em 9 imagens** e precisa desse ressalva no texto.

---

## 2. Abordagem 1 — fine-tuning ponta a ponta

Formato: ASDID / Embrapa (gap).

| métrica | ResNet-18 | ResNet-50 | VGG-16 | GoogLeNet |
|---|---|---|---|---|
| acurácia balanceada | 0,956 / 0,946 | 0,832 / 0,915 | **0,987** / 0,938 | 0,954 / 0,915 |
| F1 (rust) | 0,954 / **0,943** | 0,798 / 0,908 | **0,987** / 0,934 | 0,951 / 0,908 |
| recall (rust) | 0,913 / 0,892 | 0,665 / 0,831 | **0,974** / 0,877 | 0,913 / 0,831 |
| precisão (rust) | 1,000 / 1,000 | 1,000 / 1,000 | 1,000 / 1,000 | 0,993 / 1,000 |
| AUC-ROC | 0,999 / 0,978 | 0,996 / **0,988** | 0,999 / 0,926 | 0,989 / 0,966 |

**A ResNet-18 reproduziu exatamente os números que já estão escritos no docx** (0,956 / 0,954 / 0,999 e 0,946 / 0,943 / 0,978). Só ela.

Dois padrões merecem comentário no texto:

**Precisão 1,000 em todos os modelos, nos dois domínios.** Nenhum modelo classifica folha saudável como doente. Todos os erros são falsos negativos. Isso é operacionalmente relevante — num sistema de apoio à decisão em campo, errar para o lado de não alarmar é o pior perfil de erro, porque a ferrugem asiática é de progressão rápida. Vale dizer isso.

**A VGG-16 tem o melhor número in-distribution e o pior AUC OOD (0,926) e o maior gap de acurácia (+0,098).** Ou seja: o melhor resultado no teste controlado é também o que menos se sustenta fora do domínio.

---

## 3. LOMDO — o achado mais importante desta rodada

| backbone | F1 médio | desvio | folds problemáticos |
|---|---|---|---|
| ResNet-18 | **0,985** | 0,019 | — |
| ResNet-50 | 0,984 | 0,023 | — |
| GoogLeNet | 0,973 | 0,042 | — |
| **VGG-16** | **0,757** | **0,287** | 2 de 4 |

A VGG-16 quebra em dois folds:

- `2020:10:04`: balAcc = 0,500, recall = 1,000, precisão = 0,223, **AUC = 0,333**
- `2021:10:15`: balAcc = 0,500, recall = 1,000, precisão = 0,572, AUC = 0,500

Recall 1,000 com precisão igual à taxa base significa **predizer rust para tudo**. E AUC 0,333 é *pior que o acaso* — o modelo ordenou as amostras ao contrário. A VGG-16 simplesmente não treinou nesses folds.

**Este é o argumento central que a validação cruzada habilita:** a VGG-16 tem o melhor resultado no split único (F1 0,987) e o pior comportamento sob reamostragem por data. O número do split único é sorte de partição, não capacidade do modelo. Sem o LOMDO, a conclusão natural do trabalho teria sido "a VGG-16 é a melhor arquitetura" — e estaria errada.

**A ResNet-18 é a única arquitetura boa nos quatro eixos simultaneamente**: forte no controlado, a melhor no OOD por F1, estável no LOMDO e (seção 5) com a explicação mais fiel no ASDID. É a escolha defensável como modelo principal.

---

## 4. Abordagem 2 — extrator congelado + classificadores clássicos

**14 das 20 combinações colapsam** no teste OOD (acurácia balanceada ≤ 0,51), contra 3 de 5 na versão anterior que só tinha ResNet-18/50.

O padrão do colapso é sempre o mesmo: recall = 1,000, balAcc = 0,500, **F1 = 0,935**. Predizem rust para tudo. E aqui está a armadilha: como a Embrapa tem 65 rust e 9 healthy, prever tudo como rust dá F1 = 0,935 — que **parece um resultado excelente**. Só a acurácia balanceada revela o colapso.

> Isso é um argumento metodológico forte para o TCC, independente do resultado: em avaliação com classes desbalanceadas, F1 da classe positiva pode mascarar completamente a ausência de discriminação. É por isso que a acurácia balanceada e a AUC foram adotadas como métricas primárias.

A VGG-16 é a pior base de features: **os cinco classificadores colapsam**.

### Correção necessária no que já está escrito

O texto atual afirma que só o MLP mantém sinal discriminativo real fora do domínio, e trata a Regressão Logística como "ainda pior". **Isso está errado.** A RL sobre ResNet-18 tem recall 0,246 mas **AUC = 0,978**; sobre GoogLeNet, recall 0,108 e **AUC = 0,986**. AUC é independente de limiar. As features separam as classes muito bem — o que falha é o **limiar de decisão em 0,5**, que ficou calibrado para a distribuição do ASDID e não transfere para a Embrapa.

Diagnóstico correto: **não é ausência de sinal, é descalibração de limiar sob mudança de domínio.** É uma afirmação mais precisa, mais defensável e mais interessante — e sugere naturalmente o trabalho futuro (recalibração de limiar no domínio alvo).

O MLP sobre ResNet-18 continua sendo o único que mantém desempenho utilizável sem recalibração (balAcc 0,883).

---

## 5. Grid search 3×3 — ResNet-18

| config | LR | weight decay | F1 fold 1 | F1 fold 2 | média |
|---|---|---|---|---|---|
| melhor | 3e-4 | 1e-3 | 0,977 | 1,000 | 0,9884 |
| | 3e-4 | 1e-4 | 0,994 | 0,969 | 0,9816 |
| | 3e-3 | 1e-3 | 0,959 | 1,000 | 0,9793 |
| **default** | **1e-3** | **1e-4** | 0,958 | 0,977 | 0,9672 |
| pior | 3e-3 | 1e-4 | 0,903 | 0,976 | 0,9396 |

Amplitude da grade inteira: 0,9396 a 0,9884 (spread 0,049). Delta melhor − default = **+0,021**.

O script imprime "delta relevante" porque passou do corte de 0,01, mas **a leitura honesta é a oposta**: com apenas 2 folds, a variação *dentro* de uma mesma configuração entre folds (0,958 → 0,977 no default; 0,977 → 1,000 no melhor) é da mesma ordem do delta *entre* configurações. Não há como separar sinal de ruído com n=2.

**Conclusão a escrever: não há evidência de que o ajuste de hiperparâmetros melhore o modelo nesta faixa; o valor default está dentro do intervalo de variação amostral do melhor candidato.** Isso é um resultado legítimo e conveniente — justifica por escrito não retreinar tudo. Há uma tendência fraca a favor de LR menor (3e-4 aparece em 3 dos 4 melhores), que pode ser mencionada como indicação sem ser tratada como conclusão.

---

## 6. Explicabilidade comum — o que funcionou, o que quebrou

### 6.1 A fração de fundo falhou por completo — bug meu

Os 256 registros têm `fracao_fundo = NaN`. **Todas as máscaras foram descartadas.** Duas causas, uma dentro da outra:

1. **Bug na implementação de Otsu.** `np.nan_to_num` converte `+inf` no maior float finito, não em zero. Na última faixa do histograma o denominador é zero → `inf` → o `argmax` escolhia sempre a última faixa. O limiar saía praticamente no máximo do ExG (0,437 num intervalo que termina em 0,439), selecionando quase nenhum pixel. Corrigido ignorando faixas com peso zero.
2. **Mesmo com o Otsu corrigido, o ExG não serve para este dataset.** As imagens da Embrapa são folhas sobre um **fundo padronizado impresso** — grade de caracteres alfanuméricos mais uma cartela de calibração de cores. Essa cartela **contém um adesivo verde mais saturado que a própria folha**, que costuma estar marrom/oliva. O índice de excesso de verde aponta para o adesivo, não para a folha. Testei também uma máscara por textura (a grade tem variância local alta, a folha é lisa): também falha, porque o interior de cada célula impressa é liso.

O teste sintético que rodei antes passou porque usei uma folha verde idealizada sobre fundo escuro. Era limpo demais para pegar qualquer um dos dois problemas.

### 6.2 O substituto — e ele já está calculado, custo zero de GPU

Existe uma medida de "Clever Hans" que **não precisa de segmentação nenhuma** e já está nas curvas salvas: a probabilidade que o modelo atribui a rust quando **não há conteúdo** — imagem totalmente borrada (passo 0 da curva de insertion) e imagem totalmente apagada (passo final da curva de deletion).

| modelo | p(rust) imagem intacta | p(borrada) | p(apagada) |
|---|---|---|---|
| **VGG-16 E2E** | 0,371 | **0,000** | **0,001** |
| ResNet-18 E2E | 0,369 | 0,166 | 0,288 |
| ResNet-50 E2E | 0,294 | 0,437 | 0,321 |
| **GoogLeNet E2E** | 0,277 | **0,550** | **0,599** |
| ResNet-18 congelado (SVM) | 0,835 | 0,801 | 0,529 |
| ResNet-50 congelado (RL) | 0,097 | 0,954 | 0,031 |

Leitura direta e forte:

- A **VGG-16 E2E é a única que só responde a conteúdo real**: numa imagem borrada ou apagada, p(rust) = 0.
- A **GoogLeNet E2E prediz rust com 60% de confiança numa imagem em branco.** Tem viés de classe embutido, independente da imagem.
- O **colapso da Abordagem 2 fica medido, não só descrito**: a ResNet-18 congelada dá p = 0,835 na imagem real e **0,801 na imagem borrada** — apagar tudo o que a explicação aponta como importante derruba só de 0,835 para 0,529. A predição praticamente não depende da imagem. A ResNet-50 congelada é ainda mais absurda: 0,097 na imagem real e **0,954 na borrada**.

Essa tabela substitui a fração de fundo com vantagem: é model-agnostic, não depende de heurística de segmentação, e ataca a mesma pergunta de forma mais direta.

### 6.3 As métricas de fidelidade brutas estão confundidas — precisam ser renormalizadas

O `explicabilidade_fidelidade.png` **não deve ir para o TCC como está**. A deletion/insertion AUC bruta escala com a confiança base do modelo. A GoogLeNet congelada aparece com deletion AUC = 0,011, o que parece fidelidade excelente — quando na verdade ela emite p ≈ 0,054 em tudo e não detecta rust nenhum na Embrapa (0 de 16 imagens previstas como rust). AUC baixa por ausência de sinal, não por explicação fiel.

Recalculei restringindo às imagens que o modelo de fato prevê como rust (p ≥ 0,5) e normalizando pela confiança base:

**ASDID** (fidelidade líquida = insertion − deletion normalizadas, maior é melhor):

| modelo | n | del | ins | líquida |
|---|---|---|---|---|
| **ResNet-18 E2E** | 8 | 0,393 | 0,876 | **+0,482** |
| VGG-16 E2E | 8 | 0,418 | 0,824 | +0,406 |
| ResNet-50 E2E | 5 | 0,628 | 0,960 | +0,331 |
| GoogLeNet congelado | 9 | 0,594 | 0,735 | +0,141 |
| GoogLeNet E2E | 8 | 0,786 | 0,870 | +0,084 |
| ResNet-18 congelado | 8 | 0,800 | 0,875 | +0,075 |
| ResNet-50 congelado | 8 | 0,787 | 0,810 | +0,024 |
| VGG-16 congelado | 8 | 0,814 | 0,647 | −0,167 |

**Resultado limpo: as quatro Abordagens 1 ficam acima das Abordagens 2 correspondentes** (exceto GoogLeNet, cuja E2E é fraca). O fine-tuning ponta a ponta não só acerta mais — ele acerta por razões que sobrevivem ao teste de fidelidade. É um eixo de evidência independente do desempenho, e corrobora o pivô metodológico do trabalho.

Na Embrapa os n caem muito (a maioria dos modelos não prevê rust com confiança fora do domínio), então a tabela OOD é sugestiva, não conclusiva — a VGG-16 E2E lidera com +0,778 sobre n=6. **Vale reportar com o n explícito e sem tirar conclusão forte.**

### 6.4 A figura qualitativa é boa e sustenta o argumento

Nas duas primeiras linhas (folhas doentes, p entre 0,88 e 1,00), o calor está **sobre a folha** nas quatro arquiteturas, não sobre a grade impressa do fundo. Isso é evidência visual direta contra o efeito Clever Hans, e é forte justamente porque o fundo da Embrapa é altamente estruturado e distinto do ASDID — se os modelos estivessem lendo protocolo de aquisição, o calor estaria lá.

Duas ressalvas a declarar:

- Com `patch=32` numa imagem de 224 px e a folha ocupando ~60 px, a resolução **não permite afirmar que o modelo olha para as pústulas** — só que olha para a folha. A afirmação precisa parar aí.
- A terceira linha é uma folha rotulada como rust que **as quatro arquiteturas classificam como saudável** (p entre 0,00 e 0,04). Como não há predição positiva, o mapa de oclusão ali é ruído amplificado pela normalização — o calor espalhado no fundo **não** significa Clever Hans. Ou se remove essa linha da figura, ou se rotula explicitamente como falso negativo comum às quatro arquiteturas. Vale checar essa imagem: pode ser um erro de rótulo no PDDB.

---

## 7. O que muda no TCC

**Obrigatório:**

1. Substituir todos os números de ResNet-50 (Resumo, tabelas, discussão). Os antigos não são mais reprodutíveis pelo código entregue.
2. Reescrever o diagnóstico da Regressão Logística na Abordagem 2: descalibração de limiar, não ausência de sinal. Citar a AUC.
3. Acrescentar a seção de Limitações o desbalanceamento da partição de validação (48 positivos) e sua consequência sobre a estabilidade do *early stopping*.
4. Acrescentar a ressalva das 9 imagens healthy da Embrapa onde houver afirmação sobre generalização na classe saudável.
5. Não usar `explicabilidade_fidelidade.png` como está — refazer com a normalização da seção 6.3.
6. Remover a fração de fundo do texto, ou substituí-la pela tabela de resposta a conteúdo nulo (seção 6.2).

**Ganhos novos a explorar:**

7. A instabilidade da VGG-16 no LOMDO é o melhor argumento do trabalho para justificar validação cruzada por data. Merece parágrafo próprio na Discussão.
8. A tabela de fidelidade no ASDID dá base quantitativa para preferir Abordagem 1 sobre Abordagem 2, além do desempenho.
9. O grid search justifica por escrito não ter feito busca extensiva.
10. A escolha da ResNet-18 como modelo principal agora se sustenta em quatro eixos independentes, não em uma casa decimal de F1.

**Decisão que ainda depende de você:** rodar de novo com `MONITOR="auc"` para estabilizar a seleção de época (custo: reexecutar Abordagem 1 e LOMDO dos 4 backbones), ou manter os números atuais e assumir a limitação por escrito. A primeira opção provavelmente conserta a ResNet-50 e a VGG-16 no LOMDO; a segunda é honesta e não custa nada, mas deixa em aberto por que uma arquitetura padrão como a ResNet-50 teve recall 0,665.
