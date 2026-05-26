import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
)
from neurobench.metrics.static import (
    Footprint,
    ParameterCount,
    ConnectionSparsity
)
from snnTorch.gen1_snntorch_snn import (
    SNNTorchSpikingSSD,
    GEN1Dataset
    )
from Norse.gen1_norse_snn import (
    NorseSpikingSSD
)
from SpikingJelly.gen1_spikingjelly_snn import (
    SpikingJellySSD
)

from snntorch import Leaky as snnTorch_Leaky
from norse.torch.module.lif import LIF as Norse_LIF
from spikingjelly.activation_based.neuron import BaseNode as SpikingJelly_BaseNode
from spikingjelly.activation_based.neuron import LIFNode as SpikingJelly_LIFNode

CONFIG = {
    'batch_size' : 4,
    'num_workers': 2
}

def neurobench_collate(batch):
    cubes, targets, _stems = zip(*batch)
    return torch.stack(cubes, 0), list(targets)

class NorseLIFWrapper(nn.Module):
    def __init__(self, lif):
        super().__init__()
        self.lif = lif

    @property
    def p(self):
        return self.lif.p

    def forward(self, x):
        spikes, _ = self.lif(x)
        return spikes

def patch_norse_model(model):
    for name, module in model.named_children():
        if isinstance(module, Norse_LIF):
            setattr(model, name, NorseLIFWrapper(module))
        else:
            patch_norse_model(module)  # recurse into ConvSpikeBlocks
    return model


models_to_test = [
    ('Norse',       NorseSpikingSSD,    r'Norse\norse_gen1_best.pt',                NorseLIFWrapper),
    ('snnTorch',    SNNTorchSpikingSSD, r'snnTorch\snntorch_gen1_best.pt',          snnTorch_Leaky),
    ('SpikingJelly',SpikingJellySSD,    r'SpikingJelly\spikingjelly_gen1_best.pt',  SpikingJelly_BaseNode)
]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    all_results = {}
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

    for model_name, model_cls, ckpt_path, neuron_cls in models_to_test:
    # model init
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = model_cls()
        model.load_state_dict(ckpt["model"])
        model.eval()

        if model_name == 'Norse':
            model = patch_norse_model(model)
            wrapped = TorchModel(model)
            wrapped.add_activation_module(NorseLIFWrapper)
        else:
            wrapped = TorchModel(model)
            wrapped.add_activation_module(neuron_cls)

        benchmark = Benchmark(
            model=wrapped,
            dataloader=neurobench_val_loader,
            preprocessors=[],
            postprocessors=[],
            # metrics split static/workload
            metric_list=[[Footprint, ParameterCount, ConnectionSparsity],           # model size in bytes, fraction odf zero weights
                        [ActivationSparsity, SynapticOperations]]   # fraction of zero activations, average no. synaptic operations
        )
        all_results[model_name] = benchmark.run()
        print(f'\n{model_name}:\n {all_results[model_name]}\n')

