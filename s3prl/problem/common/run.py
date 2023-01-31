"""
The backbone run procedure for the common train/valid/test

Authors
  * Leo 2022
"""

import logging
import pickle
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml

from s3prl.problem.base import Problem
from s3prl.task.utterance_classification_task import UtteranceClassificationTask

logger = logging.getLogger(__name__)

__all__ = ["Common"]


class Common(Problem):
    def run(
        self,
        target_dir: str,
        cache_dir: str = None,
        remove_all_cache: bool = False,
        start: int = 0,
        stop: int = None,
        num_workers: int = 6,
        eval_batch: int = -1,
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
        test_ckpt_dir: str = None,
        prepare_data: dict = None,
        build_encoder: dict = None,
        build_dataset: dict = None,
        build_batch_sampler: dict = None,
        build_collate_fn: dict = None,
        build_upstream: dict = None,
        build_featurizer: dict = None,
        build_downstream: dict = None,
        build_model: dict = None,
        build_task: dict = None,
        build_optimizer: dict = None,
        build_scheduler: dict = None,
        save_model: dict = None,
        save_task: dict = None,
        train: dict = None,
        evaluate: dict = None,
    ):
        """
        ========  ====================
        stage     description
        ========  ====================
        0         Parse the corpus and save the metadata file (waveform path, label...)
        1         Build the encoder to encode the labels
        2         Train the model
        3         Evaluate the model on multiple test sets
        ========  ====================

        Args:
            target_dir (str):
                The directory that stores the script result.
            cache_dir (str):
                The directory that caches the processed data.
                Default: /home/user/.cache/s3prl/data
            remove_all_cache (bool):
                Whether to remove all the cache stored under `cache_dir`.
                Default: False
            start (int):
                The starting stage of the problem script.
                Default: 0
            stop (int):
                The stoping stage of the problem script, set `None` to reach the final stage.
                Default: None
            num_workers (int): num_workers for all the torch DataLoder
            eval_batch (int):
                During evaluation (valid or test), limit the number of batch.
                This is helpful for the fast development to check everything won't crash.
                If is -1, disable this feature and evaluate the entire epoch.
                Default: -1
            device (str):
                The device type for all torch-related operation: "cpu" or "cuda"
                Default: "cuda"
            world_size (int):
                How many processes are running this script simultaneously (in parallel).
                Usually this is just 1, however if you are runnig distributed training,
                this should be > 1.
                Default: 1
            rank (int):
                When distributed training, world_size > 1. Take :code:`world_size == 8` for
                example, this means 8 processes (8 GPUs) are runing in parallel. The script
                needs to know which process among 8 processes it is. In this case, :code:`rank`
                can range from 0~7. All the 8 processes have the same :code:`world_size` but
                different :code:`rank` (process id).
            test_ckpt_dir (str):
                Specify the checkpoint path for testing. If not, use the validation best
                checkpoint under the given :code:`target_dir` directory.
            **kwds:
                The other arguments like :code:`prepare_data` and :code:`build_model` are
                method specific-arguments for methods like :obj:`prepare_data` and
                :obj:`build_model`, and will not be used in the core :obj:`run` logic.
                See the specific method documentation for their supported arguments and
                meaning
        """

        yaml_path = Path(target_dir) / "configs" / f"{self._get_time_tag()}.yaml"
        yaml_path.parent.mkdir(exist_ok=True, parents=True)
        with yaml_path.open("w") as f:
            yaml.safe_dump(self._get_current_arguments(), f)

        cache_dir: str = cache_dir or Path.home() / ".cache" / "s3prl" / "data"
        prepare_data: dict = prepare_data or {}
        build_encoder: dict = build_encoder or {}
        build_dataset: dict = build_dataset or {}
        build_batch_sampler: dict = build_batch_sampler or {}
        build_collate_fn: dict = build_collate_fn or {}
        build_upstream: dict = build_upstream or {}
        build_featurizer: dict = build_featurizer or {}
        build_downstream: dict = build_downstream or {}
        build_model: dict = build_model or {}
        build_task: dict = build_task or {}
        build_optimizer: dict = build_optimizer or {}
        build_scheduler: dict = build_scheduler or {}
        save_model: dict = save_model or {}
        save_task: dict = save_task or {}
        train: dict = train or {}
        evaluate = evaluate or {}

        target_dir: Path = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        if remove_all_cache:
            shutil.rmtree(cache_dir)

        stage_id = 0
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: prepare data")
            train_csv, valid_csv, test_csvs = self.prepare_data(
                prepare_data, target_dir, cache_dir, get_path_only=False
            )

        train_csv, valid_csv, test_csvs = self.prepare_data(
            prepare_data, target_dir, cache_dir, get_path_only=True
        )

        def check_fn():
            assert Path(train_csv).is_file() and Path(valid_csv).is_file()
            for test_csv in test_csvs:
                assert Path(test_csv).is_file()

        self._stage_check(stage_id, stop, check_fn)

        stage_id = 1
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: build encoder")
            encoder_path = self.build_encoder(
                build_encoder,
                target_dir,
                cache_dir,
                train_csv,
                valid_csv,
                test_csvs,
                get_path_only=False,
            )

        encoder_path = self.build_encoder(
            build_encoder,
            target_dir,
            cache_dir,
            train_csv,
            valid_csv,
            test_csvs,
            get_path_only=True,
        )

        def check_fn():
            assert Path(encoder_path).is_file()

        self._stage_check(stage_id, stop, check_fn)

        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        model_output_size = len(encoder)
        model = self.build_model(
            build_model,
            model_output_size,
            build_upstream,
            build_featurizer,
            build_downstream,
        )
        frame_shift = model.downsample_rate

        stage_id = 2
        train_dir = target_dir / "train"
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: Train Model")
            train_ds, train_bs = self._build_dataset_and_sampler(
                target_dir,
                cache_dir,
                "train",
                train_csv,
                encoder_path,
                frame_shift,
                build_dataset,
                build_batch_sampler,
            )
            valid_ds, valid_bs = self._build_dataset_and_sampler(
                target_dir,
                cache_dir,
                "valid",
                valid_csv,
                encoder_path,
                frame_shift,
                build_dataset,
                build_batch_sampler,
            )

            with Path(encoder_path).open("rb") as f:
                encoder = pickle.load(f)

            build_model_all_args = dict(
                build_model=build_model,
                model_output_size=len(encoder),
                build_upstream=build_upstream,
                build_featurizer=build_featurizer,
                build_downstream=build_downstream,
            )
            build_task_all_args_except_model = dict(
                build_task=build_task,
                encoder=encoder,
                valid_df=pd.read_csv(valid_csv),
            )

            self.train(
                train,
                train_dir,
                build_model_all_args,
                build_task_all_args_except_model,
                save_model,
                save_task,
                build_optimizer,
                build_scheduler,
                evaluate,
                train_ds,
                train_bs,
                self.build_collate_fn(build_collate_fn, "train"),
                valid_ds,
                valid_bs,
                self.build_collate_fn(build_collate_fn, "valid"),
                device=device,
                eval_batch=eval_batch,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
            )

            def check_fn():
                assert (train_dir / "valid_best").is_dir()

            self._stage_check(stage_id, stop, check_fn)

        stage_id = 3
        if start <= stage_id:
            test_ckpt_dir: Path = Path(
                test_ckpt_dir or target_dir / "train" / "valid_best"
            )
            assert test_ckpt_dir.is_dir()
            logger.info(f"Stage {stage_id}: Test model: {test_ckpt_dir}")
            for test_idx, test_csv in enumerate(test_csvs):
                test_name = Path(test_csv).stem
                test_dir: Path = (
                    target_dir
                    / "evaluate"
                    / test_ckpt_dir.relative_to(train_dir).as_posix().replace("/", "-")
                    / test_name
                )
                test_dir.mkdir(exist_ok=True, parents=True)

                logger.info(f"Stage {stage_id}.{test_idx}: Test model on {test_csv}")
                test_ds, test_bs = self._build_dataset_and_sampler(
                    target_dir,
                    cache_dir,
                    "test",
                    test_csv,
                    encoder_path,
                    frame_shift,
                    build_dataset,
                    build_batch_sampler,
                )

                _, valid_best_task = self.load_model_and_task(
                    test_ckpt_dir, task_overrides={"test_df": pd.read_csv(test_csv)}
                )
                logs = self.evaluate(
                    evaluate,
                    "test",
                    valid_best_task,
                    test_ds,
                    test_bs,
                    self.build_collate_fn(build_collate_fn, "test"),
                    eval_batch,
                    test_dir,
                    device,
                    num_workers,
                )
                test_metrics = {name: float(value) for name, value in logs.items()}
                logger.info(f"test results: {test_metrics}")
                with (test_dir / f"result.yaml").open("w") as f:
                    yaml.safe_dump(test_metrics, f)

    def _build_dataset_and_sampler(
        self,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        encoder_path: str,
        frame_shift: int,
        build_dataset: dict,
        build_batch_sampler: dict,
    ):
        logger.info(f"Build {mode} dataset")
        dataset = self.build_dataset(
            build_dataset,
            target_dir,
            cache_dir,
            mode,
            data_csv,
            encoder_path,
            frame_shift,
        )
        logger.info(f"Build {mode} batch sampler")
        batch_sampler = self.build_batch_sampler(
            build_batch_sampler,
            target_dir,
            cache_dir,
            mode,
            data_csv,
            dataset,
        )
        return dataset, batch_sampler

    def build_task(
        self,
        build_task: dict,
        model: torch.nn.Module,
        encoder,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
    ):
        """
        Build the task, which defines the logics for every train/valid/test forward step for the :code:`model`,
        and the logics for how to reduce all the batch results from multiple train/valid/test steps into metrics

        By default build :obj:`UtteranceClassificationTask`

        Args:
            build_task (dict): same in :obj:`default_config`, no argument supported for now
            model (torch.nn.Module): the model built by :obj:`build_model`
            encoder: the encoder built by :obj:`build_encoder`

        Returns:
            Task
        """
        task = UtteranceClassificationTask(model, encoder)
        return task
