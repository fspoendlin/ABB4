"""Renumber predicted antibody structures to IMGT numbering using ANARCII."""

import argparse
from glob import glob
import multiprocessing as mp
import os
import sys
from typing import Iterable
import queue

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renumber predicted antibody structures to IMGT numbering using ANARCII"
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path containing prediction subdirectories with PDB files",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output root for renumbered PDB files; defaults to <input_dir>_imgt per input directory",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for ANARCII")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (example: 0,1,2,3). Defaults to CUDA_VISIBLE_DEVICES or GPU 0.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Worker processes. Default: auto (GPUs count in GPU mode, else min(cpu_count, 8)).",
    )
    parser.add_argument(
        "--anarcii_batch_size",
        type=int,
        default=128,
        help="ANARCII internal batch size for sequence processing",
    )
    parser.add_argument(
        "--task_batch_size",
        type=int,
        default=16,
        help="Number of PDB files sent per queue task",
    )
    parser.add_argument(
        "--ncpu_per_worker",
        type=int,
        default=4,
        help="CPU threads assigned to ANARCII per worker",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1000,
        help="Progress logging frequency (files) in non-interactive mode",
    )
    parser.add_argument(
        "--scan_log_every",
        type=int,
        default=200,
        help="Progress logging frequency (directories) during input scan",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="Use tqdm progress bar (disabled unless this flag is provided)",
    )
    return parser.parse_args()


def parse_gpu_ids(gpu_ids_arg: str | None) -> list[int]:
    if gpu_ids_arg:
        return [int(item.strip()) for item in gpu_ids_arg.split(",") if item.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [int(item.strip()) for item in visible.split(",") if item.strip()]

    return [0]


def chunked(items: list[tuple[str, str, str]], size: int) -> Iterable[list[tuple[str, str, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_tasks(
    pred_path: str,
    out_path: str | None,
    scan_log_every: int,
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    tasks: list[tuple[str, str, str]] = []
    out_dirs_by_input_dir: dict[str, str] = {}

    pred_dirs = sorted([d for d in glob(f"{pred_path}/*") if os.path.isdir(d)])
    total_dirs = len(pred_dirs)
    for idx, pred_dir in enumerate(pred_dirs, 1):
        if out_path:
            out_dir = os.path.join(out_path, os.path.basename(pred_dir))
        else:
            out_dir = f"{pred_dir}_imgt"

        out_dirs_by_input_dir[pred_dir] = out_dir

        pdbs = sorted(glob(f"{pred_dir}/*.pdb"))
        for pdb in pdbs:
            out_stem = os.path.join(out_dir, os.path.splitext(os.path.basename(pdb))[0])
            tasks.append((pdb, out_stem, pred_dir))

        if idx % max(1, scan_log_every) == 0 or idx == total_dirs:
            print(
                f"Scan progress: {idx}/{total_dirs} directories, {len(tasks)} PDB tasks discovered",
                flush=True,
            )

    return tasks, out_dirs_by_input_dir


def process_single_worker(
    tasks: list[tuple[str, str, str]],
    use_gpu: bool,
    gpu_id: int | None,
    anarcii_batch_size: int,
    ncpu_per_worker: int,
    show_tqdm: bool,
):
    from anarcii import Anarcii

    if use_gpu and gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = Anarcii(
        seq_type="antibody",
        batch_size=anarcii_batch_size,
        cpu=not use_gpu,
        ncpu=ncpu_per_worker,
        mode="accuracy",
        verbose=False,
    )

    iterator = tqdm(tasks, total=len(tasks), desc="Renumbering PDBs") if show_tqdm else tasks
    for pdb, out_stem, pred_dir in iterator:
        try:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            model.number(pdb, pdb_out_stem=out_stem)
            yield True, pred_dir, pdb, ""
        except Exception as exc:
            yield False, pred_dir, pdb, str(exc)


def worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    use_gpu: bool,
    gpu_id: int | None,
    anarcii_batch_size: int,
    ncpu_per_worker: int,
):
    from anarcii import Anarcii

    if use_gpu and gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = Anarcii(
        seq_type="antibody",
        batch_size=anarcii_batch_size,
        cpu=not use_gpu,
        ncpu=ncpu_per_worker,
        mode="accuracy",
        verbose=False,
    )

    while True:
        batch = task_queue.get()
        if batch is None:
            break

        for pdb, out_stem, pred_dir in batch:
            try:
                os.makedirs(os.path.dirname(out_stem), exist_ok=True)
                model.number(pdb, pdb_out_stem=out_stem)
                result_queue.put((True, pred_dir, pdb, ""))
            except Exception as exc:
                result_queue.put((False, pred_dir, pdb, str(exc)))


def run_parallel(
    tasks: list[tuple[str, str, str]],
    worker_gpus: list[int | None],
    use_gpu: bool,
    anarcii_batch_size: int,
    task_batch_size: int,
    ncpu_per_worker: int,
):
    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue(maxsize=max(2, len(worker_gpus) * 2))
    result_queue: mp.Queue = ctx.Queue()

    workers: list[mp.Process] = []
    for gpu_id in worker_gpus:
        proc = ctx.Process(
            target=worker_loop,
            args=(
                task_queue,
                result_queue,
                use_gpu,
                gpu_id,
                anarcii_batch_size,
                ncpu_per_worker,
            ),
        )
        proc.start()
        workers.append(proc)

    for task_batch in chunked(tasks, max(1, task_batch_size)):
        task_queue.put(task_batch)

    for _ in workers:
        task_queue.put(None)

    total = len(tasks)
    received = 0
    try:
        while received < total:
            try:
                item = result_queue.get(timeout=30)
            except queue.Empty:
                failed_workers = [
                    proc for proc in workers if proc.exitcode is not None and proc.exitcode != 0
                ]
                if failed_workers:
                    raise RuntimeError(
                        "One or more renumbering workers exited unexpectedly. "
                        "Check SLURM error logs for worker traceback details."
                    )
                continue

            received += 1
            yield item
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join()


def main() -> None:
    args = parse_args()

    print("Starting input scan...", flush=True)
    tasks, out_dirs_by_input_dir = build_tasks(
        args.pred_path,
        args.out_path,
        args.scan_log_every,
    )
    total = len(tasks)
    print(f"Found {len(out_dirs_by_input_dir)} directories and {total} PDB files to process.")
    if total == 0:
        return

    if args.gpu:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if not gpu_ids:
            raise ValueError("No GPU IDs were resolved. Use --gpu_ids to specify at least one GPU.")

        if args.num_workers > 0:
            num_workers = min(args.num_workers, len(gpu_ids))
        else:
            num_workers = len(gpu_ids)

        worker_gpus: list[int | None] = gpu_ids[:num_workers]
        print(f"Using GPU mode with {num_workers} workers on GPUs: {worker_gpus}")
    else:
        if args.num_workers > 0:
            num_workers = args.num_workers
        else:
            num_workers = max(1, min(os.cpu_count() or 1, 8))

        worker_gpus = [None] * num_workers
        print(f"Using CPU mode with {num_workers} workers")

    success_by_dir = {pred_dir: 0 for pred_dir in out_dirs_by_input_dir}
    total_by_dir = {pred_dir: 0 for pred_dir in out_dirs_by_input_dir}
    for _, _, pred_dir in tasks:
        total_by_dir[pred_dir] += 1

    processed = 0
    failed = 0

    if num_workers == 1:
        single_gpu = worker_gpus[0] if args.gpu else None
        result_iter = process_single_worker(
            tasks=tasks,
            use_gpu=args.gpu,
            gpu_id=single_gpu,
            anarcii_batch_size=args.anarcii_batch_size,
            ncpu_per_worker=args.ncpu_per_worker,
            show_tqdm=False,
        )
    else:
        result_iter = run_parallel(
            tasks=tasks,
            worker_gpus=worker_gpus,
            use_gpu=args.gpu,
            anarcii_batch_size=args.anarcii_batch_size,
            task_batch_size=args.task_batch_size,
            ncpu_per_worker=args.ncpu_per_worker,
        )

    progress_bar = tqdm(total=total, desc="Renumbering PDBs") if args.use_tqdm else None

    for ok, pred_dir, pdb, err in result_iter:
        processed += 1
        if ok:
            success_by_dir[pred_dir] += 1
        else:
            failed += 1
            print(f"Error processing {pdb}: {err}")

        if progress_bar is not None:
            progress_bar.update(1)
        elif processed % max(1, args.log_every) == 0 or processed == total:
            print(f"Progress: {processed}/{total} processed, failures: {failed}")

    if progress_bar is not None:
        progress_bar.close()

    for pred_dir, out_dir in out_dirs_by_input_dir.items():
        print(
            f"Renumbered {success_by_dir[pred_dir]}/{total_by_dir[pred_dir]} PDB files in {pred_dir} and saved to {out_dir}"
        )

    print(f"Completed {processed} files with {failed} failures.")


if __name__ == "__main__":
    main()
