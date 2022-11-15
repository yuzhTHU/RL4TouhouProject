import torch
from environment.EoSD import EoSD
from model.sac import SAC
from utils.utils import set_seed, get_args, get_time
from tqdm import tqdm

ARGS = get_args()
set_seed(ARGS.SEED)

model = SAC(ARGS)
env = EoSD()
env.init()
for episode in tqdm(range(1, ARGS.EPISODE+1)):
    state = env.start()
    for _ in tqdm(range(ARGS.MAX_EP)):
        action = model(state)
        state_, reward, done = env.step(action)
        tqdm.write(f"state={state[0, -5:]}, action={'&'.join([k for k, v in env._get_key_dict(action).items() if v])}, reward={reward}, done={done}")
        reward = torch.tensor([[reward]]).to(ARGS.DEVICE)
        done = torch.tensor([[done]]).to(ARGS.DEVICE)
        if ARGS.TRAIN:
            model.store_transition(state, action, reward, state_, done)
            model.learn()
        if episode % 100 == 0:
            torch.save(model.state_dict(), f'checkpoint/sac/{get_time()}.bin')
        if done:
            break
        state = state_
    env.exit()
