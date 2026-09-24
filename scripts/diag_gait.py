"""Load a mafpo_gauss checkpoint, roll it out deterministically and report what the gait
actually is: forward velocity, control cost, torso pitch (flip detection), and a few frames."""
import sys, os, json, types, numpy as np, torch as th
sys.path.insert(0,'src')
from modules.agents import REGISTRY as agent_REGISTRY
from components.obs_normalizer import ObsNormalizer
from gymnasium_robotics import mamujoco_v1

ckpt, sacred, n_ep = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 3
cfg=json.load(open(sacred+'/config.json')); args=types.SimpleNamespace(**cfg)
key=cfg['env_args']['key']; scen,conf=key.split('-')[1], key.split('-')[2]
env=mamujoco_v1.parallel_env(scen, conf, agent_obsk=cfg['env_args'].get('agent_obsk',1))
env.reset(seed=0); agents=list(env.possible_agents)
N=len(agents); A=env.action_space(agents[0]).shape[0]; O=max(env.observation_space(a).shape[0] for a in agents)  # gymma pads shorter obs with zeros
args.n_agents, args.n_actions = N, A
scheme={'obs':{'vshape':O},'state':{'vshape':N*O}}
norm=ObsNormalizer(scheme,args,'cpu'); norm.load_state_dict(th.load(ckpt+'/obs_norm.th',map_location='cpu'))
inp=O+(N if args.obs_agent_id else 0)
actor=agent_REGISTRY[args.agent](inp,args); actor.load_state_dict(th.load(ckpt+'/agent.th',map_location='cpu')); actor.eval()
low=np.array(env.action_space(agents[0]).low); high=np.array(env.action_space(agents[0]).high)

for ep in range(n_ep):
    obs,_=env.reset(seed=100+ep); h=actor.init_hidden().expand(N,-1).contiguous()
    vx=[]; ctrl=[]; R=0.0; qpos_pitch=[]
    for t in range(1000):
        x=th.tensor(np.stack([np.pad(obs[a],(0,O-len(obs[a]))) for a in agents]),dtype=th.float32)
        x=norm.normalize_obs(x)
        if args.obs_agent_id: x=th.cat([x,th.eye(N)],dim=-1)
        with th.no_grad():
            h=actor.encode(x,h); mu=actor.mean(h,th.zeros(N,A)); act=th.sigmoid(mu).numpy()
        real={a: (low+act[i]*(high-low)).astype('float32') for i,a in enumerate(agents)}
        obs,rew,term,trunc,info=env.step(real)
        r=float(list(rew.values())[0]); R+=r
        i0=info[agents[0]]; vx.append(i0.get('x_velocity',np.nan)); ctrl.append(-i0.get('reward_ctrl',np.nan))
        qpos_pitch.append(float(env.single_agent_env.unwrapped.data.qpos[2]))
        if all(term.values()) or all(trunc.values()): break
    vx=np.array(vx); ctrl=np.array(ctrl); p=np.array(qpos_pitch)
    print(f"ep{ep}: steps={len(vx)} return={R:8.1f} | x_vel mean={np.nanmean(vx):6.2f} max={np.nanmax(vx):6.2f} "
          f"neg%={100*np.mean(vx<0):4.1f} | ctrl_cost mean={np.nanmean(ctrl):5.3f} "
          f"| torso pitch mean={p.mean():6.2f} |pitch|>pi frac={np.mean(np.abs(p)>np.pi)*100:4.1f}%")
