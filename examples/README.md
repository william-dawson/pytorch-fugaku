# Examples

Trainers on synthetic tokens that print one JSON line per event (`setup`, `phase`, `step`, `fom`).
In the image they are in `/opt/pytorch-fugaku/examples`. Launch every rank through `pf-run`
(see the repository README and the fugaku-pytorch skill).

- `llama_train.py`: transformers Llama / GPT-2 with DeepSpeed ZeRO-3. For models up to a few
  billion parameters. `--loss-chunk 256` (torch_fugaku.nn) is needed for 4 ranks per node at
  seq 2048; `--zero-init` builds the model already partitioned.
- `tp_train.py`: Llama with tensor parallel inside the node x FSDP2 across nodes
  (torch_fugaku.parallel). For 10B+ models at 4 ranks per node; `--tp 4 --sp`.
  `--check` compares a tiny model step by step against an unsharded copy.

DeepSpeed and transformers are not in the image; install them where the image's Python sees
them (for example a directory on `PYTHONPATH`).
