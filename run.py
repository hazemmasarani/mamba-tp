import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.modeling_mamba import MambaForCausalLM
from transformers import AutoConfig

mamba_model = "state-spaces/mamba-2.8b-hf"

def shard_tensor(tensor, rank, world_size, dim=0):
    chunks = torch.tensor_split(tensor, world_size, dim=dim)
    return chunks[rank].contiguous()

def calc_intermediate_sizes(total_size, num_parts):

    base = total_size // num_parts
    remainder = total_size % num_parts

    sizes = []
    for i in range(num_parts):
        if i < remainder:
            sizes.append(base + 1)
        else:
            sizes.append(base)
    
    return sizes

def split_state_dict(state_dict, rank, world_size):
    for key in list(state_dict.keys()):
        if "mixer.in_proj" in key:

            total_dim = state_dict[key].shape[0]
            intermediate_size = total_dim // 2

            proj1 = state_dict[key][:intermediate_size]
            proj2 = state_dict[key][intermediate_size:]

            proj1_shard = shard_tensor(proj1, rank, world_size, dim=0)
            proj2_shard = shard_tensor(proj2, rank, world_size, dim=0)

            state_dict[key] = torch.cat([proj1_shard, proj2_shard], dim=0)
        elif "mixer.A_log" in key or "mixer.D" in key or "mixer.conv1d" in key or "mixer.dt_proj" in key:
            state_dict[key] = shard_tensor(state_dict[key], rank, world_size, dim=0)  # split along the intermediate dimension
        elif "mixer.x_proj" in key or "mixer.out_proj" in key:
            state_dict[key] = shard_tensor(state_dict[key], rank, world_size, dim=-1)  # split along the intermediate dimension
    return state_dict

def worker(rank, world_size, device_map):

    device = torch.device(f"cuda:{device_map[rank]}")
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',  # single-node example
        world_size = world_size,
        rank=rank,
        device_id = device
    )

    config = AutoConfig.from_pretrained(mamba_model)

    config.rank = rank
    config.world_size = world_size
    config.local_intermediate_size = calc_intermediate_sizes(config.intermediate_size , config.world_size)[rank]

    model = MambaForCausalLM(config).to(f"cuda:{device_map[rank]}")

    state_dict = torch.load("model/state_dict.pt")

    state_dict = split_state_dict(state_dict, rank, world_size)

    model.load_state_dict(state_dict)

    input_ids = torch.load("input/input_ids.pt").to(f"cuda:{device_map[rank]}")

    batch_size, seq_len = input_ids.shape

    out = model(input_ids)

    if rank == 0:
        torch.save(out.logits, f"results/logits_dist_{config.world_size}.pt")

    print(out.logits.shape)


if __name__ == "__main__":

    world_size = 4

    device_map = {
        0:0,
        1:1,
        2:2,
        3:3
    }

    process = []

    for i in range(world_size):
        p = mp.Process(target=worker, args=(i, world_size, device_map))
        p.start()
        process.append(p)

    for p in process:
        p.join()
    
