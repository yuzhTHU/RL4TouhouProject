import torch
from environment.EoSD import EoSD
from model.sac import SAC
from utils.utils import set_seed, get_args
from tqdm import tqdm

ARGS = get_args()
set_seed(ARGS.SEED)

model = SAC(ARGS)
env = EoSD()
env.init()
for episode in tqdm(range(200)):
    state = env.start()
    for step in tqdm(range(20000)):
        action = model(state)
        state_, reward, done = env.step(action)
        reward = torch.tensor([[reward]]).to(ARGS.DEVICE)
        done = torch.tensor([[done]]).to(ARGS.DEVICE)
        tqdm.write(f"state={state[0, -5:]}, action={''.join('1' if a else '0' for a in action.ravel())}, reward={reward}, done={done}")
        if ARGS.TRAIN:
            model.store_transition(state, action, reward, state_, done)
            model.learn()
        if done:
            break
        state = state_
    env.exit()