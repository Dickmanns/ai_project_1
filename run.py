import hydra
from omegaconf import DictConfig
from project.main import model
import os

from hydra.utils import get_original_cwd, to_absolute_path

@hydra.main(config_path="src/conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    m = model()
    m.train_model()
    trained_model = m.load_model()
    m.test_model_selfData(trained_model)

    # print(f"Working directory : {os.getcwd()}")
    # print(
    #     f"Output directory  : {get_original_cwd()}"
    # )

if __name__ == '__main__':
    main()
