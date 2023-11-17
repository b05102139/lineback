import requests
import matplotlib.pyplot as plt
import io
from copy import deepcopy
import lightning as pl
from lightning.pytorch.callbacks import Callback

class lineback(Callback):
    def __init__(self, token: str, process_name: str):
        self.token = token
        self.process_name = process_name
        self.url = "https://notify-api.line.me/api/notify"
        self.headers = {"Authorization": "Bearer " + token}
        self.metrics_history = {}
        self.epoch_history = []

    def _send_message(self, msg, img=None):
        if img is None:
            payload = {'message': msg}
            r = requests.post(self.url, headers=self.headers, data=payload)
        elif img is not None:
            payload = {'message': msg}
            img.seek(0)
            files = {'imageFile': img}
            r = requests.post(self.url, headers=self.headers, data=payload, files=files)
        return r.status_code

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._send_message(f'{self.process_name} has started training.')

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._send_message(f'{self.process_name} has finished training.')
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._send_message(f'Epoch {trainer.current_epoch} has started.')
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics, meta = self._collect_metrics()
        contents = [f"Epoch {trainer.current_epoch} has ended.",
            f"Metrics from {self.process_name} at step {meta['step']} (epoch {meta['epoch']}):"
        ]
        contents += [
            f"{metric_name}: {metric_value:.6f}"
            for metric_name, metric_value in metrics.items()
            if metric_name != "epoch"
        ]
        text = '\n'.join(contents)
        for metric_name in self.metrics_history.keys():
            img_bytes = self._metric_plot(self.epoch_history, self.metrics_history[metric_name])
            self._send_message(text, img=img_bytes)

    def _collect_metrics(self):
        ckpt_name_metrics = deepcopy(trainer.logged_metrics)
        meta = {"step": trainer.global_step, "epoch": trainer.current_epoch}
        self.epoch_history.append(trainer.current_epoch)
        for metric_name, metric_value in ckpt_name_metrics.items():
            metric_value = metric_value.item()
            try:
                past_values = self.metrics_history[metric_name]
                self.metrics_history[metric_name] = past_values + [metric_value]
            except KeyError:
                self.metrics_history[metric_name] = [metric_value]
        return ckpt_name_metrics, meta
    
    def _metric_plot(self, step_values: list, value_records: list):
        plt.plot(step_values, value_records)
        buf = io.BytesIO()
        plt.savefig(buf, format="jpg")
        return buf
