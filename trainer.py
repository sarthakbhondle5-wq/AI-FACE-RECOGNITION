import os
import shutil
import subprocess
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from base.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Class Trainer wrap PyTorch Lightning.
    Quan ly checkpoint, log TensorBoard va Early Stopping.
    """
    def __init__(self, model, data_loader, config):
        super().__init__(model, data_loader, config)

        run_name = self.config.get("name", "FaceRecognition_Project")
        ckpt_dir = os.path.join(self.save_dir, run_name, "latest")
        log_run_dir = os.path.join(self.log_dir, run_name)
        log_version_dir = os.path.join(log_run_dir, "current")

        # xoa cac run name khac de tiet kiem dia
        if os.path.exists(self.save_dir):
            for entry in os.listdir(self.save_dir):
                entry_path = os.path.join(self.save_dir, entry)
                if os.path.isdir(entry_path) and entry not in [run_name, "predictions"]:
                    try:
                        shutil.rmtree(entry_path)
                    except Exception:
                        pass

        if os.path.exists(self.log_dir):
            for entry in os.listdir(self.log_dir):
                entry_path = os.path.join(self.log_dir, entry)
                if os.path.isdir(entry_path) and entry != run_name:
                    try:
                        shutil.rmtree(entry_path)
                    except Exception:
                        pass

        # xoa cac checkpoint cu cua run hien tai de tiet kiem dia
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        # xoa log cu cua run hien tai de TensorBoard sach se
        if os.path.exists(log_run_dir):
            for entry in os.listdir(log_run_dir):
                if entry != "current" and entry != "latest":
                    entry_path = os.path.join(log_run_dir, entry)
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    elif os.path.isfile(entry_path):
                        os.remove(entry_path)
                    
        if os.path.exists(log_version_dir):
            shutil.rmtree(log_version_dir)

        # cau hinh TensorBoard Logger
        self.tb_logger = TensorBoardLogger(
            save_dir=self.log_dir,
            name=run_name,
            version="current"
        )

        # ModelCheckpoint callback luu lai 1 model tot nhat theo val_loss
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-checkpoint-epoch={epoch:02d}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False
        )

        # Early Stopping callback
        trainer_cfg = self.config.get("trainer", {})
        early_stop_patience = trainer_cfg.get("early_stop", 10)
        self.early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            mode="min",
            verbose=True
        )

        self.progress_bar = RichProgressBar()

        accelerator = trainer_cfg.get("accelerator", "auto")
        devices = trainer_cfg.get("devices", 1)
        precision = trainer_cfg.get("precision", "32-true")

        print(f"[Trainer] Dang khoi tao PyTorch Lightning Trainer...")
        print(f"[Trainer] Accelerator: {accelerator} | Devices: {devices} | Precision: {precision}")
        print(f"[Trainer] Early Stopping patience: {early_stop_patience} epochs")

        # khoi tao PyTorch Lightning Trainer
        self.pl_trainer = pl.Trainer(
            max_epochs=trainer_cfg.get("epochs", 50),
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=[self.checkpoint_callback, self.early_stop_callback, self.progress_bar],
            logger=self.tb_logger,
            log_every_n_steps=trainer_cfg.get("log_step", 5),
            check_val_every_n_epoch=trainer_cfg.get("val_epoch_interval", 1),
            default_root_dir=self.save_dir
        )

        # chay TensorBoard ngam
        self._launch_tensorboard(log_version_dir)

    def _launch_tensorboard(self, logdir):
        # tat cac process TensorBoard dang chay tren cong 6006 de tranh loi
        try:
            subprocess.run(
                ["powershell", "-Command",
                 "Get-NetTCPConnection -LocalPort 6006 -ErrorAction SilentlyContinue | "
                 "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
                capture_output=True
            )
        except Exception:
            pass

        tb_logdir = os.path.join(self.log_dir, self.config.get("name", "FaceRecognition_Project"), "current")

        try:
            subprocess.Popen(
                ["tensorboard", f"--logdir={tb_logdir}", "--port=6006", "--reload_interval=5"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            print(f"[TensorBoard] Da khoi dong tai http://localhost:6006")
        except Exception as e:
            print(f"[TensorBoard] Khong tu dong khoi dong duoc: {e}")
            print(f"[TensorBoard] Hay chay thu cong: tensorboard --logdir={tb_logdir} --port=6006")

    def train(self):
        print(f"[Trainer] Bat dau train...")
        self.pl_trainer.fit(self.model, datamodule=self.data_loader)
        print(f"[Trainer] Train hoan thanh!")

        best_path = self.checkpoint_callback.best_model_path
        print(f"[Trainer] Checkpoint tot nhat luu tai: {best_path}")
        return best_path
