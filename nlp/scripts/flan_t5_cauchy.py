from _common import *

from scripts._common import DictConfig

log = logging.getLogger(__name__)

from collections import defaultdict

import lightning as L
import lightning.pytorch as pl
from flan_t5_checkpoint_path import finetuned_model_path
from flan_t5_individuals import Program as _Program
from flan_t5_individuals import metric_func
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator

from datasets import DatasetDict, load_dataset, load_from_disk
from src.tasks.arithmetic import state_dict_cauchy, state_dict_sub
from src.utils import num_devices

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Program(_Program):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if hasattr(cfg, "seed") and cfg.seed is not None:
            log.info(f"set seed to {cfg.seed}")
            L.seed_everything(cfg.seed)

        if cfg.peft.peft_config is None:
            self.results_dir = RESULTS_DIR / cfg.model.name
        else:
            self.results_dir = RESULTS_DIR / (cfg.model.name + "_" + cfg.peft.name)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.results_dir / "averaging.csv"

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()
        
        # weight for \theta_m
        self.weight = 1.0
        self.iters = 50
        self.tol = 1e-5
        self.eps = 1e-2
        self.scale_radius = 10000
        self.init_zero=False
        self.load = False
        self.common_space=False

    def run(self):
        self.load_models()
        self.load_datasets()

        self.eval_cauchy()

    def eval_cauchy(self):
        name = f"Cauchy_iters{self.iters}_eps{self.eps}_scaleRadius{self.scale_radius}"
        name = name + "_subspace" if self.common_space else name
        name = name + "_init_zero" if self.init_zero else name
        
        if self.load:
            # Load existing
            moo_nash_task_vector = torch.load(f"cache/logs/GLUE/large/" + name + ".pt")
        
        else:
            import time
            start = time.time()
            moo_nash_task_vector = state_dict_cauchy(self.task_vectors, common_space=self.common_space, eps=self.eps, 
                                                       max_iters=self.iters, scale_radius=self.scale_radius,
                                                       init_zero=self.init_zero
                                                       )
            end = time.time()
            print("Elasped time:", end - start)
            # Save
            # torch.save(moo_nash_task_vector, f"cache/logs/GLUE/large/" + name + ".pt")
        
        model = deepcopy(self.pretrained_model)
        for n, p in moo_nash_task_vector.items():
            model.state_dict()[n] += self.weight * moo_nash_task_vector[n]
        model = self.fabric.setup_module(model)
        _results = self.eval_model_on_datasets(model)
        results = {dataset_name: [score] for dataset_name, score in _results.items()}
        print(df := pd.DataFrame(results))
        df.to_csv(self.results_path, index=False)


@hydra.main(str(CONFIG_DIR), "flan_t5_default", None)
def main(cfg: DictConfig):
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
