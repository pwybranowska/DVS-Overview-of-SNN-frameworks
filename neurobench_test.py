import torch
from torch.utils.data import DataLoader
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)
from gen1_snntorch_snn import (
    SNNTorchSpikingSSD,
    GEN1Dataset,
    CONFIG
    )

def neurobench_collate(batch):
    cubes, targets, _stems = zip(*batch)
    return torch.stack(cubes, 0), list(targets)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # model init
    ckpt = torch.load("snntorch_gen1_best.pt", map_location="cpu")
    model = SNNTorchSpikingSSD()
    model.load_state_dict(ckpt["model"])
    model.eval()

    # dataloader
    val_dataset = GEN1Dataset("val")
    neurobench_val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=neurobench_collate,
        pin_memory=(device.type == "cuda"),
    )

    wrapped = TorchModel(model)

    benchmark = Benchmark(
        model=wrapped,
        dataloader=neurobench_val_loader,
        preprocessors=[],
        postprocessors=[],
        metric_list=[[Footprint, ConnectionSparsity], 
                    [ActivationSparsity, SynapticOperations]] # metrics split worload/static!!!!
    )
    results = benchmark.run()
    print(results)

