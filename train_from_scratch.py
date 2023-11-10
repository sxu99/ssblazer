import argparse
from ssblazer.my_model import SSBlazer
from ssblazer.dataloader import DatasetFromCSV,DataModule
from utils import *
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint,TQDMProgressBar
import lightning.pytorch as pl


def train(train_path, test_path, model_dir):
    n_workers = get_n_workers()
    n_gpus = get_n_gpus()
    batch_size = 2048
    epochs = 50
    warmup_epochs = 5
    precision = 16
    lr = 0.001

    if n_gpus == 0:
        raise Exception("No GPU found")

    params = {
        "batchsize": batch_size,
        "max_epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "compute_precision": precision,
        "n_gpus": n_gpus,
        "lr": lr,
    }
    for key, value in params.items():
        logging.info("{} = {}".format(key, value))

    train_dataset = DatasetFromCSV(train_path)
    test_dataset = DatasetFromCSV(test_path)

    train_loader = DataModule(batch_size, n_workers, train_dataset).train_dataloader()
    test_loader = DataModule(batch_size, n_workers, test_dataset).test_dataloader()

    warmup_iters = int(len(train_loader) / n_gpus) * warmup_epochs
    max_iters = int(len(train_loader) / n_gpus) * epochs

    params["warmup_iters"] = warmup_iters
    params["max_iters"] = max_iters

    model = SSBlazer(warmup=warmup_iters, max_epochs=max_iters, lr=lr)

    callbacks = [TQDMProgressBar(refresh_rate=int((len(train_loader) / n_gpus) / 10))]
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(ModelCheckpoint(dirpath=model_dir, save_top_k=-1, save_last=True))

    trainer = pl.Trainer(
        accelerator="auto",
        logger=None,
        devices="auto",
        num_nodes=1,
        precision=precision,
        max_epochs=epochs,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        gradient_clip_val=1,
    )
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, required=True, help="Path to the training data"
    )
    parser.add_argument("--test", type=str, required=True, help="Path to the test data")
    args = parser.parse_args()

    set_root_logging()
    set_logging("lightning.pytorch")
    set_logging("torch")

    model_dir = "./models"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    train(args.train, args.test, model_dir)
