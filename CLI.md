# CLI workflow

Read the [command manual](src/tmt/commands.md) or use `tmt help` for full command details.
Command sections: [init](src/tmt/commands.md#init), [train](src/tmt/commands.md#train), [sweep](src/tmt/commands.md#sweep),
[evaluate](src/tmt/commands.md#evaluate), [benchmark](src/tmt/commands.md#benchmark), [generate](src/tmt/commands.md#generate), and [help](src/tmt/commands.md#help).

## Installation and PATH

Use Python 3.12 or 3.13. Model commands and `init` require a functional MLX runtime.
From the repository root, install the package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
tmt --help
tmt --version
python -m tmt --help
```

This regular installation copies the package into the virtual environment.
For development, use `python -m pip install -e .` instead.
After source edits, repeat a regular installation before you verify those edits.
Run the tests with `python -m unittest discover -s test -v`.

Activation adds `.venv/bin` to PATH for one terminal.
For new terminals, you can add this optional line to your shell configuration:

```bash
export PATH="/absolute/path/to/main-tmt/.venv/bin:$PATH"
```

Replace the example path with your environment's absolute path.
The whole venv on PATH also selects that environment's Python and other executables.
TMT does not modify shell files or require a global installation.

`--version` reads the installed `test-model-thing` distribution version.
It includes `vcs_info.commit_id` from `direct_url.json` when present.
Otherwise it prints `commit unknown`. It does not query Git in the current folder.
Help and version require no MLX import.

## Package and workspace

The package files are `src/tmt/{__init__.py,__main__.py,cli.py,main.py,benchmark.py}`.
`main.py` owns model arithmetic and constructor defaults. `cli.py` owns commands and the strict checkpoint loader.
`benchmark.py` owns frozen byte scores and the CoLA classifier head. Tests stay in `test/`.

The tracked examples are [model.json](examples/model.json), [tiny-model.json](examples/tiny-model.json), and [grid.json](examples/grid.json).
Installed commands do not need checkout examples.


All relative paths use the current folder. TMT does not search parent folders.
Bare `tmt init` creates `model.json`, `grid.json`, `run.example.json`, `data/train/`, `data/dev/`, and `runs/`.
`model.json` contains every constructor default. A user JSON can override a subset of these values.

The workspace grid uses dimensions 4 and 8, one layer, spread 4, and seeds 11 and 22.

`run.example.json` is a template, not a run record. Its null timestamp is a placeholder.

`init --output PATH` writes only a model JSON. Init preserves current files and can fail on a repeated invocation.

## Complete small workflow

Use a new workspace folder. These commands select a tiny model and raw-byte input:

```bash
tmt init
printf '{"dim":4,"layers":1,"spread":4}\n' > model.json
printf abcd > data/train/tiny.bin
printf wxyz > data/dev/tiny.bin
tmt train --data 'data/train/*' --run first --updates 3
tmt train runs/first/model.safetensors --data 'data/train/*' --resume --updates 2
tmt evaluate runs/first/model.safetensors --data 'data/dev/*' --windows 1 2 --output score.json
tmt generate runs/first/model.safetensors --prompt 'The ' --output sample.bin --bytes 4
tmt benchmark runs/first/model.safetensors
tmt sweep --grid grid.json --data 'data/train/*' --development 'data/dev/*' --output runs/grid --updates 2
```

Train and sweep default to `model.json`. An explicit `--config PATH` overrides it for a fresh run.
`--run NAME` creates `runs/NAME/model.safetensors` in a new folder.
With neither a name nor a checkpoint, train uses a UTC timestamp with microseconds as the run folder name.

For an explicit path, use `tmt train other/model.safetensors --data 'data/train/*' --updates 3`.
A positional checkpoint excludes `--run`. Resume requires that positional checkpoint and forbids `--config` and `--run`.

Resume restores model and optimizer tensors, then adds the requested updates with a new data pass.
It resets internal state and RTU traces at the first document. It does not restore the file cursor or RNG state.

The trainer writes `manifest/run.json` beside the checkpoint at start and exit.
The record contains model `config`, invocation `seed`, UTC `started_at`, `status`, and `last_checkpoint`.
Status is `running`, `complete`, `failed`, or `interrupted`. A resume replaces the record for the invocation.

The trainer writes one JSON object per line to `manifest/loss.jsonl`.
Use `--log-every N` to set the update interval. Its default is `100` updates.
Each row records `completed_updates`, `objective`, and the latest update's `loss`.

After at least one update, the trainer writes a final row at exit if the last update was outside the interval.
The trainer does not duplicate an interval row when the run ends on that interval.
Each invocation replaces this file. This also applies to a resume.

The record also contains `data_glob`, absolute `data_files` in shuffled pass order, requested `updates`, `log_every`, and `objective`.
The objective is `tmt` for the full loss or `ce_only` for next-byte cross entropy.
The file list includes all matched paths. The update budget can stop a pass before the trainer reads every file.
Paths identify the file selection but do not prove that file contents stay the same.

`resume` identifies a resume invocation. `initial_optimizer_step` is the saved optimizer step at resume start, or zero for a fresh run.
`updates` is the budget for this invocation, with additional updates on resume.

`completed_updates` is zero at start and counts completed updates at exit, after success, failure, or interrupt.
This count includes work after the last checkpoint. It does not imply that the trainer saved all completed updates.
See the [manifest fields](src/tmt/commands.md#train) for details.

The trainer prints completed updates, objective, and loss at each interval. It always prints the final loss.

It saves every 500 updates and at normal completion. It prints save feedback only after success.
Ctrl-C does not save. Keep the last successful checkpoint. Direct writes can leave partial files after a process failure.

## Scores and limits

Each file in a quoted glob is one raw-byte document. TMT adds no separators and does not decode input.
The trainer shuffles sorted file paths once per invocation. Each adjacent byte pair gives one optimizer update.
The trainer resets internal state and RTU traces at each document. Empty files give no updates.

The default objective combines variance, latent-space prediction, next-byte cross entropy, and stop loss.
`--ce-only` selects next-byte cross entropy alone.

Evaluation uses frozen weights, no samples, and the first byte as context. It restores internal state and RTU traces after success or failure.
The default input budget is 8192 bytes across sorted files. Each context window counts input bytes before the target.

Sweep requires a new or empty output folder. It rejects a nonempty folder without changes.
Sweep appends one row per completed candidate to `results.jsonl`. It selects the lowest development BPB, with the first candidate as tie winner.
Candidates use `run-NNNN/` folders with `model.safetensors`, `manifest/model.json`, `manifest/run.json`, and `manifest/loss.jsonl`.

The byte-only benchmark always uses 393 input bytes, 390 targets, and 104 common targets with windows 1, 8, 32, and 128.
For optional CoLA, use `tmt benchmark runs/first/model.safetensors --cola-data cola.tsv --epochs 1 --split 0.5`.
`python -m tmt.benchmark` accepts the same benchmark options.
CoLA uses four-column TSV rows, a contiguous split, and head-only updates. It reports MCC times 100.

Generation encodes its nonempty prompt as UTF-8 and writes raw bytes without the prompt.
It prints byte progress at each `--log-every` interval. The default interval is `1000` bytes.

Checkpoints require complete model and optimizer tensors plus version-1 model metadata. The loader checks names, shapes, and dtypes.
Keep output paths apart from input paths. The CLI does not check path collisions or train/development overlap.

Use a positive value for `--updates`, each value in `--windows`, and `--log-every`.
Generation accepts `--bytes 0` and writes an empty file. It rejects negative byte counts.
Whole-file reads can require memory equal to the largest file.

Keep objective loss, frozen BPB, and CoLA MCC separate. Tiny fixtures do not establish general model quality.

File errors and invalid values print a short error to stderr and exit with code `1`. Argument errors exit with code `2`.
An unmatched train glob fails before the CLI creates a named or timestamp run folder.
