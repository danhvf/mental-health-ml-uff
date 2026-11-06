"""Atualiza tabelas/macros LaTeX a partir do notebook executado."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artigo'
d=json.loads((ROOT/'resultados'/'resultados.json').read_text(encoding='utf-8'))
models=pd.read_csv(ROOT/'resultados'/'metricas_modelos.csv')
def fmt(n): return f'{n:.3f}'.replace('.',',')
def latex_escape(s): return s.replace('—','--').replace('_',r'\_').replace('Ã','Ã')
out=[]
out += [r'\begin{table}[htbp]',r'\centering',r'\caption{Desempenho dos algoritmos no conjunto de teste ($n=252$).}',r'\label{tab:metricas}',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}llccccc@{}}',r'\toprule',r'\textbf{Grupo} & \textbf{Algoritmo} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{Acurácia} & \textbf{Recall} & \textbf{F1}\\',r'\midrule']
for _,r in models.iterrows():
 group='A' if r['Experimento'].startswith('A') else 'B'
 name={'Regressão logística':'Regressão logística','Árvore de decisão':'Árvore de decisão','Random Forest':'Random Forest'}[r['Modelo']]
 out.append(f"{group} & {name} & {fmt(r['AUROC'])} & {fmt(r['AUPRC'])} & {fmt(r['Acurácia'])} & {fmt(r['Recall'])} & {fmt(r['F1'])}"+r'\\')
 if name=='Random Forest' and group=='A':out.append(r'\midrule')
out += [r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}{\footnotesize\color{cinzasec}Fonte: elaboração própria com base na OSMI 2014. Limiar 0,5; seleção por AUROC na validação cruzada.}',r'\end{table}']
(ART/'tabela_resultados.tex').write_text('\n'.join(out)+'\n',encoding='utf-8')
a=d['best_A']['metrics'];b=d['best_B']['metrics'];abl=d['ablation_no_work_interfere_logistic']
macros={
 'aucA':a['AUROC'],'aucB':b['AUROC'],'fA':a['F1'],'fB':b['F1'],
 'recallA':a['Recall'],'recallB':b['Recall'],'aucAbl':abl['AUROC'],
 'deltaAUC':b['AUROC']-a['AUROC'], 'aucAlo':d['best_A']['auc_ci95'][0],
 'aucAhi':d['best_A']['auc_ci95'][1], 'aucBlo':d['best_B']['auc_ci95'][0],
 'aucBhi':d['best_B']['auc_ci95'][1], 'diffLo':d['auc_diff_b_minus_a_ci95'][0],
 'diffHi':d['auc_diff_b_minus_a_ci95'][1],
}
with open(ART/'numeros_resultados.tex','w',encoding='utf-8') as f:
 for k,v in macros.items():f.write('\\newcommand{\\'+k+'}{'+fmt(v)+'}\n')
print('Tabelas/macros LaTeX atualizadas.')
