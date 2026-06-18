# test Code
import torch
from tqdm import tqdm

restormer = Restormer(in_channels=1, out_channels=1, dim=48, num_blocks=4, num_heads=8).cuda()
swinir = SwinIR().cuda()

restormer.eval()
swinir.eval()

print("Starting dry run")

with torch.no_grad():
    for batch_idx, (ir_batch, rgb_target) in enumerate(tqdm(train_loader, desc="Enhancing Batches")):

        ir_batch = ir_batch.cuda()
        clean_ir_batch = restormer(ir_batch)
        enhanced_ir_batch = swinir(clean_ir_batch)

        del ir_batch
        del clean_ir_batch
        del enhanced_ir_batch
        torch.cuda.empty_cache()

print("\ndata flow between restormer and swinIR is established")