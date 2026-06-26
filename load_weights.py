import torch

from model.modeling_mamba import MambaForCausalLM

model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf").cpu()

state_dict = model.state_dict()

torch.save(state_dict, "model/state_dict.pt")
