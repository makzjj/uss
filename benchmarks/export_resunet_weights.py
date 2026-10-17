"""Run ONCE on the GPU server (inside the USS container, where Lightning is installed).
 
Extracts the ResUNet30 weights from the Lightning .ckpt into a plain PyTorch
state_dict, so the Pi only needs torch + einops + torchlibrosa to rebuild it.
"""
from pathlib import Path
 
import torch
 
CKPT = Path.home() / ".cache/uss/checkpoints/ss_model=resunet30,querynet=at_soft,data=full,devices=8,step=1000000.ckpt"
OUT = Path("resunet30_at_soft_fp32.pth")
 
ckpt = torch.load(CKPT, map_location="cpu")          # torch 2.1 on the server: full unpickle is fine
state = ckpt["state_dict"]
 
# LitSeparation stores the separator as self.ss_model -> keys look like "ss_model.base...."
ss_state = {k[len("ss_model."):]: v for k, v in state.items() if k.startswith("ss_model.")}
print(f"{len(ss_state)} ResUNet30 tensors, "
      f"{sum(v.numel() for v in ss_state.values()) / 1e6:.1f} M parameters")
 
torch.save(ss_state, OUT)
print(f"Saved -> {OUT.resolve()}")