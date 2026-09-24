import json, glob, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

SUR,TXT,MUT,GRID='#fcfcfb','#0b0b0b','#52514e','#e6e5e0'
C_OURS,C_BASE,C_OFF,C_DATA='#2a78d6','#eb6834','#4a3aa7','#eda100'

def curve(sacred, div=6, w=4):
    m=json.load(open(sacred+'/metrics.json'))
    s=np.array(m['test_return_mean']['steps'])/1e6; v=np.array(m['test_return_mean']['values'])/div
    return s, v, s[w-1:], np.convolve(v,np.ones(w)/w,mode='valid')

def by_name(name, env='mamujoco-HalfCheetah-6x1'):
    for d in glob.glob(f'results/sacred/mafpo_gauss/{env}/[0-9]*'):
        try:
            if json.load(open(d+'/config.json'))['name']==name: return d
        except Exception: pass

s1,v1,t1,m1 = curve('results/sacred/mafpo_gauss/mamujoco-HalfCheetah-6x1/1')
b=by_name('mappo_gauss_hc6x1_10M'); s2,v2,t2,m2 = curve(b) if b else (None,)*4
ev=sorted(glob.glob('/workspace/mac-flow-exp/MAC-Flow-baseline/6halfcheetah_Expert_alpha3_500k/sd000_*/eval.csv'))
rows=list(csv.DictReader(open(ev[-1]))); mac=float(rows[-1]['evaluation/mean_episode_return']); mac_steps=int(rows[-1]['step'])

fig,ax=plt.subplots(figsize=(9.5,5.4),facecolor=SUR)
ax.set_facecolor(SUR); [ax.spines[k].set_visible(False) for k in ('top','right')]
ax.spines['left'].set_color('#d5d4cf'); ax.spines['bottom'].set_color('#d5d4cf')
ax.grid(axis='y',color=GRID,lw=0.8); ax.tick_params(colors=MUT)

# offline / dataset references first (recessive, behind the curves)
refs=[(2785,'Expert dataset mean',C_DATA,'--',0.55),
      (3866,'Expert dataset best episode',C_DATA,':',0.55),
      (mac, f'MAC-Flow (offline, {mac_steps//1000}k grad steps)',C_OFF,'-',9.0)]
for y,lab,c,ls,xl in refs:
    ax.axhline(y,color=c,ls=ls,lw=1.8,alpha=0.95)
    ax.text(xl,y+90,f'{lab}  {y:,.0f}',color=TXT,fontsize=9,va='bottom',
            ha='right' if xl>5 else 'left')

if s2 is not None:
    ax.plot(s2,v2,color=C_BASE,lw=0.9,alpha=0.28)
    ax.plot(t2,m2,color=C_BASE,lw=2.2,label=f'MAPPO-Gauss (MLP head) — online, at {s2[-1]:.1f}M')
ax.plot(s1,v1,color=C_OURS,lw=0.9,alpha=0.28)
ax.plot(t1,m1,color=C_OURS,lw=2.2,label='MAFPO-Gauss (flow) — online, ours')

ax.set_xlim(0,10.2); ax.set_ylim(-250,6900)
ax.set_xlabel('environment steps (M)',color=MUT)
ax.set_ylabel('test return  (team reward per episode)',color=MUT)
ax.set_title('HalfCheetah-6x1 — online flow policy vs online Gaussian policy vs offline MAC-Flow',
             color=TXT,fontsize=12,loc='left')
ax.legend(frameon=False,loc='upper left',fontsize=9.5)
fig.text(0.012,0.015,
 'Offline methods consume no environment steps, so they appear as horizontal references. Returns are per-episode team reward '
 '(agent-mean); our logged 6-agent sum is divided by 6.\nMAC-Flow reproduced locally on the OMIGA Expert vault (paper reports 4,650; '
 'our eval env rebuilds the OMIGA observation convention on gymnasium MaMuJoCo + MuJoCo 3.13). Single seed.',
 color=MUT,fontsize=7.8)
plt.tight_layout(rect=(0,0.075,1,1)); out='figs/hc6x1_vs_macflow.png'; plt.savefig(out,dpi=160)
print(out, f'| ours final {m1[-1]:.0f} | base {m2[-1]:.0f}@{s2[-1]:.2f}M | macflow {mac:.0f}')
