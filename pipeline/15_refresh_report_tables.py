#!/usr/bin/env python3
"""Refresh marked report tables from corrected output files; narrative stays editorial.
Run after 02, 05, 05b, 07, 10 and 12, before executing/exporting the report notebook.
"""
from pathlib import Path
import itertools,json,re
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'
LABELS={'garch':'GARCH(1,1)','bootstrap':'Window bootstrap','lstm':'LSTM GAN','gan':'MLP GAN','wf':'Historical validation'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def fmt(value,n=3):
    return '—' if value is None else f'{value:.{n}f}'

def build_tables():
    fidelity=json.loads((OUT/'fidelity_metrics_summary.json').read_text())['metrics']
    rank=json.loads((OUT/'rq2_distinct_rules.json').read_text())['generators']
    paired=json.loads((OUT/'paired_rank_tests.json').read_text())
    tables={}
    tables['4.1']=table(['Source','Daily σ','Return ACF','Squared-return ACF','Excess kurtosis','Skewness','W₁'],[[r['source'],fmt(r['daily_std'],5),fmt(r['acf_returns_lags1_20']),fmt(r['acf_squared_lags1_20']),fmt(r['excess_kurtosis'],2),fmt(r['skewness'],2),fmt(r['wasserstein_vs_real'],6)] for r in fidelity])
    tables['4.2']=table(['Method','Spearman ρ','95% Fisher-z CI','p-value','n'],[[LABELS[k],fmt(r['spearman_rho']),f"[{fmt(r['fisher_ci'][0])}, {fmt(r['fisher_ci'][1])}]",'<0.001' if r['spearman_p']<.001 else fmt(r['spearman_p']),r['n']] for k,r in rank.items()])
    tables['4.3']=table(['Method','Endorsed','Above threshold on real data','FDR','Mean optimism','Selection regret'],[[LABELS[k],r['n_endorsed'],r['n_endorsed_work'],'—' if r['fdr'] is None else f"{100*r['fdr']:.0f}%",fmt(r['optimism_mean'],4),fmt(r['selection_regret'],4)] for k,r in rank.items()])
    tables['4.4']=table(['Method','ρ on common rules','95% percentile interval'],[[LABELS[k],fmt(paired['point_rho'][k]),f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]"] for k,r in paired['rho'].items()])
    tables['4.5']=table(['Difference in ρ','Mean','95% percentile interval','Resamples ≤ 0'],[[f"{LABELS[key.split('-')[0]]} − {LABELS[key.split('-')[1]]}",fmt(r['mean']),f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]",f"{100*r['share_le_zero']:.1f}%"] for key,r in paired['differences'].items()])
    a=pd.read_csv(OUT/'evaluation_results.csv'); b=pd.read_csv(OUT/'evaluation_results_lstm.csv'); wf=pd.read_csv(OUT/'walkforward_comparison.csv').rename(columns={'era_a_sharpe':'wf_sharpe'})
    df=a.merge(b[['strategy_name','lstm_sharpe']],on='strategy_name',validate='one_to_one').merge(wf[['strategy_name','wf_sharpe']],on='strategy_name',validate='one_to_one')
    df['family']=df['strategy_name'].str.split('_').str[0]
    rows=[]
    for family,label in [('MA','MA crossover'),('Momentum','Momentum'),('RSI','RSI-style')]:
        group=df[df.family==family]; n=int(group.real_sharpe.notna().sum())
        values=[]
        for k in LABELS:
            valid=group[['real_sharpe',f'{k}_sharpe']].dropna()
            values.append(fmt(stats.spearmanr(valid.iloc[:,0],valid.iloc[:,1])[0]) if len(valid)>=10 else '—')
        rows.append([label,f'{n}/{len(group)}']+values)
    tables['4.6']=table(['Rule family','Real defined / specified']+list(LABELS.values()),rows)
    return tables

def main():
    path=ROOT/'DSM500_CW2_Report_Final.ipynb'
    nb=json.loads(path.read_text()); tables=build_tables(); found=set()
    for c in nb['cells']:
        if c['cell_type']!='markdown':continue
        s=''.join(c['source'])
        for name,body in tables.items():
            pattern=rf'<!-- TABLE_{re.escape(name)}_START -->.*?<!-- TABLE_{re.escape(name)}_END -->'
            if re.search(pattern,s,re.S):
                s=re.sub(pattern,f'<!-- TABLE_{name}_START -->\n{body}\n<!-- TABLE_{name}_END -->',s,flags=re.S);found.add(name)
        c['source']=s.splitlines(keepends=True)
    assert found==set(tables),(found,set(tables))
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print('Refreshed tables:',', '.join(sorted(found)))

if __name__=='__main__':main()
