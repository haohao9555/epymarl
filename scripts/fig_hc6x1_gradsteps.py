import json, glob, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
SUR,TXT,MUT,GRID='#fcfcfb','#0b0b0b','#52514e','#e6e5e0'
C_OURS,C_BASE,C_MAC,C_DATA='#2a78d6','#eb6834','#1baf7a','#eda100'

# our runs: one PPO update per 8 episodes (8 envs x 1000 steps = 8000 env steps),
# each update = 4 epochs x ceil(8000/512)=16 minibatches = 64 actor gradient steps.
GRAD_PER_ENVSTEP = 64/8000
def ours(sacred, div=6, w=4):
    m=json.load(open(sacred+'/metrics.json'))
    s=np.array(m['test_return_mean']['steps'])*GRAD_PER_ENVSTEP
    v=np.array(m['test_return_mean']['values'])/div
    return s, v, s[w-1:], np.convolve(v,np.ones(w)/w,mode='valid'), m['test_return_mean']['steps'][-1]
def by_name(n,env='mamujoco-HalfCheetah-6x1'):
    for d in glob.glob(f'results/sacred/mafpo_gauss/{env}/[0-9]*'):
        try:
            if json.load(open(d+'/config.json'))['name']==n: return d
        except Exception: pass

s1,v1,t1,m1,e1 = ours('results/sacred/mafpo_gauss/mamujoco-HalfCheetah-6x1/1')
s2,v2,t2,m2,e2 = ours(by_name('mappo_gauss_hc6x1_10M'))
rows=list(csv.DictReader(open(sorted(glob.glob('/workspace/mac-flow-exp/MAC-Flow-baseline/6halfcheetah_Expert_alpha3_500k/sd000_*/eval.csv'))[-1])))
sm=np.array([int(r['step']) for r in rows]); vm=np.array([float(r['evaluation/mean_episode_return']) for r in rows])

fig,ax=plt.subplots(figsize=(9.8,5.6),facecolor=SUR)
ax.set_facecolor(SUR); [ax.spines[k].set_visible(False) for k in ('top','right')]
ax.spines['left'].set_color('#d5d4cf'); ax.spines['bottom'].set_color('#d5d4cf')
ax.grid(axis='y',color=GRID,lw=0.8); ax.grid(axis='x',color=GRID,lw=0.6,alpha=0.6); ax.tick_params(colors=MUT)
ax.set_xscale('log')

for y,lab,ls in [(2785,'Expert dataset mean  2,785','--'),(3866,'Expert dataset best episode  3,866',':')]:
    ax.axhline(y,color=C_DATA,ls=ls,lw=1.6,alpha=0.9)
    ax.text(1.15e3,y+95,lab,color=TXT,fontsize=8.5,va='bottom')

ax.plot(sm,vm,color=C_MAC,lw=2.2,marker='o',ms=4,label='MAC-Flow (offline, 0 env steps)')
ax.plot(s2,v2,color=C_BASE,lw=0.9,alpha=0.28)
ax.plot(t2,m2,color=C_BASE,lw=2.2,label=f'MAPPO-Gauss, MLP head (online, {e2/1e6:.1f}M env steps so far)')
ax.plot(s1,v1,color=C_OURS,lw=0.9,alpha=0.28)
ax.plot(t1,m1,color=C_OURS,lw=2.2,label=f'MAFPO-Gauss, flow (online, {e1/1e6:.1f}M env steps) — ours')
for x,y,c,lab in [(sm[-1],vm[-1],C_MAC,f'{vm[-1]:,.0f}'),(s2[-1],v2[-1],C_BASE,f'{v2[-1]:,.0f}'),(s1[-1],v1[-1],C_OURS,f'{v1[-1]:,.0f}')]:
    ax.plot([x],[y],'o',color=c,ms=6,mec=SUR,mew=1.6); ax.annotate(lab,(x,y),textcoords='offset points',xytext=(8,-2),color=TXT,fontsize=9)

ax.set_xlim(1e3,7e5); ax.set_ylim(-250,7000)
ax.set_xlabel('gradient steps  (log scale)',color=MUT)
ax.set_ylabel('test return  (team reward per episode)',color=MUT)
ax.set_title('HalfCheetah-6x1 — same optimisation budget, different data source',color=TXT,fontsize=12,loc='left')
ax.legend(frameon=False,loc='lower right',fontsize=9.5)
fig.text(0.012,0.015,
 'Shared x-axis is gradient steps, the only quantity all three spend (MAC-Flow performs zero environment interaction, so an env-step axis '
 'cannot show it).\nOnline runs additionally consume environment steps, given in the legend; one PPO update = 8,000 env steps = 64 actor '
 'gradient steps. MAC-Flow is trained on the OMIGA Expert vault (1M transitions).\nReturns are per-episode team reward (agent-mean); our logged '
 '6-agent sum is divided by 6. Single seed.',color=MUT,fontsize=7.6)
plt.tight_layout(rect=(0,0.085,1,1)); out='figs/hc6x1_gradsteps.png'; plt.savefig(out,dpi=160)
print(out,f'| ours {v1[-1]:.0f}@{s1[-1]:.0f} grad | base {v2[-1]:.0f}@{s2[-1]:.0f} | mac {vm[-1]:.0f}@{sm[-1]}')
