"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        evaluate_only = cfg.run_cfg.evaluate

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets(evaluate_only=evaluate_only)

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        # print(f"[valid_step] Sample keys: {list(samples.keys())}")
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        # print(f"[evaluation] Sample keys: {list(next(iter(data_loader)).keys())}")
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        # Opt-in cache optimization for NIGHT26 graph experiments only.
        # 1 preserves the historical per-microbatch behavior. 0 disables
        # empty_cache; PyTorch's caching allocator releases unused blocks
        # under memory pressure. Does not change any model tensor or RNG.
        cache_interval = int(os.environ.get("NO31_CORA_EMPTY_CACHE_INTERVAL", "1"))
        if cache_interval < 0:
            raise ValueError("NO31_CORA_EMPTY_CACHE_INTERVAL must be >= 0")
        if cache_interval != 1:
            print(f"[N28-CACHE-CONTROL] empty_cache_interval={cache_interval}",
                  flush=True)
        # graphlora-train-controls-v1: valid accumulation factor.
        accum_grad_iters = int(accum_grad_iters)
        if accum_grad_iters < 1:
            raise ValueError("accum_grad_iters must be >= 1")

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            # Average each accumulation window, including the final shorter one.
            # graphlora-train-controls-v1: no extra optimizer step is introduced.
            window_size = min(
                accum_grad_iters,
                iters_per_epoch - (i // accum_grad_iters) * accum_grad_iters,
            )
            backward_loss = loss / window_size
            if use_amp:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            
            

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0 or (i + 1) == iters_per_epoch:
                if use_amp:
                    # A graph-gradient backward hook runs on SCALED grads.
                    # Only GradScaler.step can distinguish a recoverable AMP
                    # overflow and skip the optimizer update.  Count actual
                    # skipped updates in NIGHT24 without altering baselines.
                    n24_graph = getattr(model, "graph_joint", None)
                    scale_before = scaler.get_scale() if n24_graph is not None else None
                    # optional global grad-norm clipping (stabilization S1);
                    # read from the model so any task can opt in via config
                    gc = float(getattr(model, "grad_clip", 0.0) or 0.0)
                    if gc > 0:
                        scaler.unscale_(optimizer)
                        params = [p for g in optimizer.param_groups
                                  for p in g["params"] if p.grad is not None]
                        torch.nn.utils.clip_grad_norm_(params, gc)
                    scaler.step(optimizer)
                    scaler.update()
                    if n24_graph is not None:
                        scale_after = scaler.get_scale()
                        if scale_after < scale_before:
                            n24_graph._n24_amp_skip_total += 1
                            n24_graph._n24_amp_skip_streak += 1
                            print(
                                f"[N24-AMP-OVERFLOW-SKIP] step={i} "
                                f"scale={scale_before:g}->{scale_after:g} "
                                f"total={n24_graph._n24_amp_skip_total} "
                                f"streak={n24_graph._n24_amp_skip_streak}",
                                flush=True,
                            )
                            if n24_graph._n24_amp_skip_streak >= 24:
                                raise FloatingPointError(
                                    "N24: 24 consecutive AMP optimizer-step "
                                    "skips; nonfinite gradients did not recover "
                                    "as GradScaler reduced its scale."
                                )
                        else:
                            n24_graph._n24_amp_skip_streak = 0
                            n24_graph._n24_amp_success_total += 1
                        # Do not spend the full 17k-batch epoch on a
                        # persistent, early CoRA->graph NaN regression.
                        # Leave stage probes untouched for diagnosis.
                        if (n24_graph._grad_audit_calls <= 64
                                and n24_graph._n24_bad_gradient_events >= 8):
                            raise FloatingPointError(
                                "N24 diagnostic abort: >=8 nonfinite graph "
                                "backwards within first 64 microbatches; "
                                "inspect [N24-GRAD-STAGE] to find the first "
                                "nonfinite CoRA/QFormer/graph boundary."
                            )
                else:
                    gc = float(getattr(model, "grad_clip", 0.0) or 0.0)
                    if gc > 0:
                        params = [p for g in optimizer.param_groups
                                  for p in g["params"] if p.grad is not None]
                        torch.nn.utils.clip_grad_norm_(params, gc)
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # optional mid-epoch evaluation hook (set by the runner via
            # run.mid_epoch_eval_iters): runs a validation pass at specific
            # intra-epoch steps. The hook snapshots/restores every RNG state
            # itself, so the training trajectory is untouched (bitwise).
            _mid_hook = getattr(self, "mid_epoch_hook", None)
            if _mid_hook is not None:
                _mid_hook(cur_epoch=inner_epoch, cur_step=i, model=model)
            if cache_interval > 0 and (i + 1) % cache_interval == 0:
                torch.cuda.empty_cache()

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
