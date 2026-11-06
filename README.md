# Saúde mental no setor de tecnologia — classificação supervisionada (UFF)

Projeto de aplicação da disciplina **Introdução ao Aprendizado de Máquina para Saúde**, Instituto de Computação, Universidade Federal Fluminense — 2026.2.

**Autor:** Daniel Fontoura  
**Dataset:** [OSMI Mental Health in Tech Survey 2014 — Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)  
**Documentação original:** [Open Sourcing Mental Illness (OSMI)](https://www.osmihelp.org/research.html)

> **Alcance:** classificamos **retrospectivamente** se a pessoa respondeu que **já procurou** tratamento (`treatment`). O projeto **não** detecta transtornos mentais, não prevê necessariamente busca futura por atendimento e **não deve ser usado** para classificar trabalhadores ou tomar decisões individuais de RH.

## 1. Motivação e objetivo

Pergunta principal: **em que medida características e percepções do ambiente de trabalho estão associadas à busca autorrelatada por tratamento de saúde mental entre os respondentes da OSMI 2014?**

Como experimento complementar, investigamos quanto cresce a capacidade discriminativa quando introduzimos variáveis pessoais e uma pergunta condicional sobre interferência percebida da saúde mental no trabalho. O objetivo científico é comparar a validade das informações usadas, não perseguir a maior métrica sem examinar suas limitações.

## 2. Descrição do dataset

Pesquisa voluntária de 2014 sobre atitudes e experiências de saúde mental no setor de tecnologia:

| Característica | Valor |
| --- | ---: |
| Observações | **1.259** |
| Colunas originais | **27** |
| `treatment=Yes` | **637 (50,6%)** |
| `treatment=No` | **622 (49,4%)** |
| Valores anômalos de idade | **8** |
| Respostas não preenchidas em `work_interfere` | **264** |
| Formas textuais distintas em `Gender` | **49** |

Fonte: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey. O Kaggle informa licença **CC BY-SA 4.0** para a versão hospedada. Consulte as condições de atribuição antes de redistribuir a base. O repositório **não versiona o CSV bruto**, pois a coluna `comments` contém relatos livres potencialmente sensíveis. Para executar localmente, baixe `survey.csv` e guarde em `data/survey.csv`.

### Preparação e pré-processamento

- `treatment` convertido em `Yes=1` e `No=0`, sem excluir respondentes.
- Idades fora da faixa *pré-especificada* de 18 a 90 anos marcadas como ausentes (**8 casos**); a imputação pela mediana ocorre **dentro do pipeline**, ajustada apenas no treino de cada fold.
- `Gender` livre agrupado com mapeamento conservador: masculino, feminino e outra resposta/ambígua; respostas ambíguas não são adivinhadas. Atributo somente do cenário ampliado.
- Para categóricas, ausências são uma categoria explícita `Não informado`, seguida de One-Hot Encoding com tolerância a categorias novas.
- Campos fora do modelo: `Timestamp`, `Country` (caracterização de subgrupos), `state`, `comments` e `Gender` textual bruto. O notebook nunca lista comentários individuais.
- `work_interfere` **não é imputada como a resposta mais comum**; sua ausência é informativa e é explicitamente investigada na ablação.

### Variáveis de cada experimento

**A — Organizacional (18):** `self_employed`, `no_employees`, `remote_work`, `tech_company`, `benefits`, `care_options`, `wellness_program`, `seek_help`, `anonymity`, `leave`, `mental_health_consequence`, `phys_health_consequence`, `coworkers`, `supervisor`, `mental_health_interview`, `phys_health_interview`, `mental_vs_physical`, `obs_consequence`.

**B — Ampliado (22):** as 18 acima **mais** `Age`, `Gender_group`, `family_history`, `work_interfere`.

**Ablação:** B sem `work_interfere`, mantendo todos os demais preditores.

> Parte das 18 variáveis organizacionais são **percepções e atitudes dos respondentes**, não indicadores auditados das empresas. O desenho transversal não assegura que esses preditores sejam anteriores à busca por tratamento.

## 3. Modelo de aprendizado de máquina

Formalização: cada respondente tem atributos $x_i$ e resposta $y_i\in\{0,1\}$. O algoritmo estima $\hat p_i=P(y_i=1\mid x_i)$ e define $\hat y_i=\mathbb{1}[\hat p_i\geq0,5]$.

Na regressão logística, $\hat p_i = 1/(1+e^{-(\beta_0+\beta^\top x_i)})$. São comparadas **Regressão Logística**, **Árvore de Decisão** e **Random Forest**, além do *DummyClassifier* de referência. As árvores usam partições por impureza; a floresta agrega múltiplas árvores. Hiperparâmetros e grades estão no notebook e em `src/modeling.py`.

### Validação

- `train_test_split`: **80% treino (1.007, sendo 509 positivos)** e **20% teste (252, sendo 128 positivos)**, `stratify=y`, `random_state=42`.
- Dentro dos **1.007** de treino: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` e `GridSearchCV(scoring='roc_auc')` para ajuste de hiperparâmetros.
- Transformações e imputações **ajustadas dentro dos folds**; jamais no dataset inteiro antes da CV.
- Apenas **após a escolha por AUROC de validação cruzada** o conjunto de teste é utilizado para avaliação final. Limiar de classificação **0,5**, definido previamente.
- Curvas de aprendizado (AUROC em função da quantidade de exemplos de treino) são exibidas como acompanhamento de treinamento. **Não são iterações de épocas de rede neural**.

### Medidas de desempenho

- **AUROC:** discriminação por diferentes limiares (principal para seleção).
- **AUPRC:** área sob a curva precisão-revocação.
- **Precisão:** $VP/(VP+FP)$.
- **Recall/sensibilidade:** $VP/(VP+FN)$.
- **F1:** média harmônica de precisão e recall.
- **Acurácia:** $(VP+VN)/N$.
- **Matriz de confusão:** análise explícita de VP, VN, FP e FN.
- **Intervalos de confiança exploratórios de 95% da AUROC:** 1.000 reamostragens bootstrap do teste; diferença B−A por bootstrap pareado.

## 4. Avaliação — resultados efetivamente obtidos

Modelos selecionados pela **AUROC média em validação cruzada (apenas treino)**: Regressão Logística nos dois cenários.

| Cenário / modelo escolhido | AUROC (teste) | AUPRC | Acurácia | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A — Organizacional / Regressão Logística | **0,698** | 0,731 | 0,663 | 0,675 | 0,648 | 0,661 |
| B — Ampliado / Regressão Logística | **0,898** | 0,885 | 0,837 | 0,832 | 0,852 | 0,842 |
| Ablação B sem `work_interfere` / Regressão Logística | **0,758** | 0,765 | 0,714 | 0,733 | 0,688 | 0,710 |

**IC95% bootstrap AUROC:** A = [0,629; 0,763]; B = [0,858; 0,933]. Para a diferença **B−A**, estimativa 0,200 e intervalo [0,134; 0,263]. São intervalos exploratórios sobre o único hold-out e **não** incluem toda a incerteza de seleção e treinamento.

No teste, a matriz A apresentou **TN=84, FP=40, FN=45, TP=83**. Para B: **TN=102, FP=22, FN=19, TP=109**. A tabela integral para todos os seis modelos está em `resultados/metricas_modelos.csv`.

**Resultado crítico:** das **264 ausências** de `work_interfere`, **260** coincidem com `treatment=No` e **4** com `Yes`. A ablação mostra que essa resposta condicional domina grande parte do ganho do modelo B. É um **atalho associativo retrospectivo**, não demonstra eficácia clínica nem estabelece ligação causal entre ambiente e tratamento.

### Gráficos gerados pelo notebook

Arquivos em `artigo/figuras/`:

- `01_eda_dataset.png`: classes e ausências;
- `02_modelos_roc.png`: comparação de AUROC e curvas ROC;
- `03_matrizes_confusao.png`: erros no hold-out;
- `04_curvas_aprendizado.png`: tamanho de treino vs. AUROC de treino e CV;
- `05_importancia_permutacao.png`: importância exploratória por coluna original.
- `06_evolucao_numero_arvores.png`: AUROC de treino/validação interna conforme aumenta o número de árvores na Random Forest (não são épocas).

## 5. Execução e entrega no Google Classroom

No terminal do VS Code, na **raiz do repositório**:

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Copie o survey.csv baixado do Kaggle para data/survey.csv
jupyter notebook Projeto_Saude_Mental_ML.ipynb
```

Clique em **Run All** no notebook (ou execute em VS Code com kernel do ambiente). Os resultados são gravados em `resultados/`. Se estiver usando este pacote de trabalho em ZIP, já há um arquivo CSV local incluído para teste; **não o submeta ao GitHub**: o `.gitignore` o exclui.

Para gerar novamente as tabelas do artigo após executar o notebook:

```bash
python artigo/atualizar_resultados.py
```

O artigo foi elaborado com o **template LaTeX da disciplina**, em `artigo/main.tex`, com citações no `artigo/referencias.bib` e figuras reais. Abra a pasta `artigo` em um projeto Overleaf (Upload Project) e compile com **pdfLaTeX + BibTeX**; o PDF de referência já está disponível localmente em `artigo/main.pdf`.

A instrução do professor solicita no **Google Classroom o link de um repositório público do GitHub** que contenha **README.md e o notebook `.ipynb` executável**. Procedimento: publicar o projeto, conferir o link numa janela anônima e entregá-lo pelo Classroom. O artigo pode seguir o fluxo de envio específico informado pelo professor; a orientação do Classroom anexada não determina por si só que o PDF do artigo substitua o repositório.

## 6. Conclusão e limitações

Há associação preditiva moderada entre respostas organizacionais e histórico autorrelatado de busca por tratamento na amostra OSMI 2014. Informações individuais elevam a discriminação, mas a variável condicional `work_interfere` responde pela maior parte do salto. O estudo é transversal e voluntário, não representativo da população de TI, não dispõe de verificação de duplicatas ou validação externa e não estima diagnóstico, riscos futuros ou efeitos de intervenções. O melhor uso desta pesquisa é **discussão acadêmica sobre preditores, métricas, potencial vazamento semântico e ética**, ou estudos agregados sobre barreiras institucionais com consentimento e governança.

## Referências essenciais

- Open Sourcing Mental Illness. *Mental Health in Tech Survey* (2014). https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey
- McGrath et al. (2023). *Disclosure of Mental Health Problems or Suicidality at Work: A Systematic Review*. https://doi.org/10.3390/ijerph20085548
- Collins et al. (2024). *TRIPOD+AI Statement*. https://doi.org/10.1136/bmj-2023-078378
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. https://jmlr.org/papers/v12/pedregosa11a.html
- Bibliografia completa em `artigo/referencias.bib`.