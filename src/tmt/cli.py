import argparse, glob, inspect, itertools, json, math, os, random, sys
from datetime import datetime, timezone
from pathlib import Path


def load_config(path=None):
    from tmt.main import Model

    overrides = {}
    if path is not None:
        with open(path) as file:
            overrides = json.load(file)

    if not isinstance(overrides, dict):
        raise ValueError('model config must be a JSON object')
    try:
        settings = inspect.signature(Model).bind(**overrides)
    except TypeError as error:
        raise ValueError(f'invalid model config: {error}') from error
    settings.apply_defaults()
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in settings.arguments.items()
    }


def new_model(settings, seed):
    import random
    import mlx.core as mx
    from tmt.main import Model

    random.seed(seed)
    mx.random.seed(seed)
    model = Model(**settings)
    mx.eval(model.parameters())
    return model


def _validate_metadata(raw):
    if not isinstance(raw, str):
        raise ValueError("checkpoint __metadata__.tmt must be a string")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"checkpoint tmt metadata is invalid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {'version', 'model'}:
        raise ValueError("checkpoint tmt metadata must contain exactly 'version' and 'model'")
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('checkpoint.version must be integer 1')

    from tmt.main import Model
    parameters = inspect.signature(Model).parameters
    settings = value['model']
    if not isinstance(settings, dict):
        raise ValueError('checkpoint.model must be an object')
    missing = sorted(set(parameters) - set(settings))
    extra = sorted(set(settings) - set(parameters))
    if missing:
        raise ValueError(f'checkpoint.model is missing field {missing[0]!r}')
    if extra:
        raise ValueError(f'checkpoint.model has unknown field {extra[0]!r}')

    for name in ('dim', 'layers', 'spread'):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f'checkpoint.model.{name} must be a positive integer')
    for name, minimum, strict in (('temp', 0, False), ('rate', 0, True)):
        number = settings[name]
        try:
            finite = not isinstance(number, bool) and isinstance(number, (int, float)) and math.isfinite(number)
        except OverflowError:
            finite = False
        if not finite or (number <= minimum if strict else number < minimum):
            rule = 'a finite positive number' if strict else 'a finite nonnegative number'
            raise ValueError(f'checkpoint.model.{name} must be {rule}')
    bound = settings['bound']
    if (not isinstance(bound, list) or len(bound) != 2
            or any(type(value) is not int for value in bound)
            or not 0 <= bound[0] < bound[1]):
        raise ValueError('checkpoint.model.bound must be two integers with 0 <= start < end')

    for name, item in settings.items():
        try:
            json.dumps(item, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f'checkpoint.model.{name} must contain finite JSON values') from error

    for name, parameter in parameters.items():
        if name in {'dim', 'layers', 'spread', 'temp', 'rate', 'bound'}:
            continue
        default = parameter.default
        if default is inspect.Parameter.empty:
            continue
        item = settings[name]
        if isinstance(default, float):
            try:
                valid = not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item)
            except OverflowError:
                valid = False
        elif isinstance(default, tuple):
            valid = isinstance(item, list)
        else:
            valid = type(item) is type(default)
        if not valid:
            raise ValueError(f'checkpoint.model.{name} has an invalid JSON type')
    return settings


def _read_header(path):
    try:
        with open(path, 'rb') as source:
            size = os.fstat(source.fileno()).st_size
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise ValueError('invalid checkpoint: truncated Safetensors header length')
            length = int.from_bytes(length_bytes, 'little')
            if length == 0 or length > size - 8:
                raise ValueError('invalid checkpoint: invalid Safetensors header length')
            raw = source.read(length)
            if len(raw) != length:
                raise ValueError('invalid checkpoint: truncated Safetensors header')
        envelope = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid checkpoint: invalid Safetensors header: {error}') from error
    if not isinstance(envelope, dict) or not envelope:
        raise ValueError('invalid checkpoint: Safetensors header must be a nonempty object')
    metadata = envelope.get('__metadata__')
    if (not isinstance(metadata, dict) or set(metadata) != {'tmt'}
            or not isinstance(metadata.get('tmt'), str)):
        raise ValueError("invalid checkpoint: __metadata__ must contain exactly string 'tmt'")
    if not any(name != '__metadata__' for name in envelope):
        raise ValueError('invalid checkpoint: checkpoint contains no tensors')
    return envelope


def _flatten(tree, prefix):
    import mlx.utils as util

    return {f'{prefix}.{name}': value for name, value in util.tree_flatten(tree)}


def _validate_arrays(actual, expected):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f'checkpoint is missing tensor {missing[0]!r}')
    if extra:
        raise ValueError(f'checkpoint has unexpected tensor {extra[0]!r}')
    for name, template in expected.items():
        array = actual[name]
        label = 'model' if name.startswith('m.') else 'optimizer'
        tensor = name[2:]
        if tuple(array.shape) != tuple(template.shape):
            raise ValueError(
                f'{label} tensor {tensor!r} has shape {tuple(array.shape)}, expected {tuple(template.shape)}'
            )
        if array.dtype != template.dtype:
            raise ValueError(
                f'{label} tensor {tensor!r} has dtype {array.dtype}, expected {template.dtype}'
            )


def save_checkpoint(path, model, settings):
    import mlx.core as mx

    resolved = {name: list(value) if isinstance(value, tuple) else value
                for name, value in settings.items()}
    raw = json.dumps({'version': 1, 'model': resolved})
    _validate_metadata(raw)
    model.optimizer.init(model.trainable_parameters())
    tensors = _flatten(model.parameters(), 'm')
    tensors.update(_flatten(model.optimizer.state, 'o'))
    mx.eval(model.parameters(), model.optimizer.state)
    mx.save_safetensors(path, tensors, metadata={'tmt': raw})


def load_model(path, seed=11):
    import mlx.core as mx
    import mlx.utils as util

    envelope = _read_header(path)
    settings = _validate_metadata(envelope['__metadata__']['tmt'])
    model = new_model(settings, seed)
    model.optimizer.init(model.trainable_parameters())
    expected = _flatten(model.parameters(), 'm')
    expected.update(_flatten(model.optimizer.state, 'o'))
    try:
        tensors = mx.load(path)
    except Exception as error:
        raise ValueError(f'invalid checkpoint: {error}') from error
    _validate_arrays(tensors, expected)

    parameters = {name[2:]: value for name, value in tensors.items() if name.startswith('m.')}
    optimizer = {name[2:]: value for name, value in tensors.items() if name.startswith('o.')}
    model.update(util.tree_unflatten(list(parameters.items())))
    model.optimizer.state = util.tree_unflatten(list(optimizer.items()))
    model.compiled = None
    mx.eval(model.parameters(), model.optimizer.state)
    return model, settings


def documents(pattern, max_bytes=None):
    """Yield sorted raw files with an optional prefix across all file bytes."""
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f'no files matched {pattern!r}')
    remaining = max_bytes
    for path in files:
        if remaining == 0:
            break
        data = Path(path).read_bytes()
        if remaining is not None:
            data = data[:remaining]
            remaining -= len(data)
        yield data


def _write_manifest(path, record):
    with open(path, 'w') as file:
        json.dump(record, file, indent=2)
        file.write('\n')


def train(settings, path, pattern, updates=1000, seed=11, ce_only=False, resume=False):
    """Train on adjacent raw-byte pairs and record the run manifest."""
    import mlx.core as mx

    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f'no files matched {pattern!r}')
    if resume:
        model, settings = load_model(path, seed)
    else:
        model = new_model(settings, seed)
    random.shuffle(files)

    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = checkpoint.parent / 'manifest'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / 'run.json'
    record = {
        'config': settings,
        'seed': seed,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'last_checkpoint': str(checkpoint.resolve()) if resume else None,
    }
    def consume(byte):
        model.step(mx.array(byte), frozen=True)
        mx.eval(*[layer.states for layer in model.blocks])

    def save():
        save_checkpoint(checkpoint, model, settings)
        record['last_checkpoint'] = str(checkpoint.resolve())
        print(f'saved checkpoint: {checkpoint}, updates: {completed}')

    completed = 0
    last_loss = None
    status = 'failed'
    try:
        _write_manifest(manifest, record)
        print(f'run folder: {checkpoint.parent}, seed: {seed}, config: {json.dumps(settings)}')
        while completed < updates:
            before_pass = completed
            for filename in files:
                data = Path(filename).read_bytes()
                if not data:
                    continue
                model.reset()
                if len(data) == 1:
                    consume(data[0])
                    continue
                for index in range(len(data) - 1):
                    end = index == len(data) - 2
                    _, _, last_loss = model.train_step(
                        data[index], data[index + 1], end, ce_only
                    )
                    completed += 1
                    if end:
                        consume(data[-1])
                    if completed % 500 == 0:
                        save()
                    if completed >= updates:
                        break
                if completed >= updates:
                    break
            if completed == before_pass:
                raise ValueError('no training targets found in data pass')
        save()
        status = 'complete'
    except KeyboardInterrupt:
        status = 'interrupted'
        raise
    finally:
        record['status'] = status
        _write_manifest(manifest, record)

    loss = 'none' if last_loss is None else f'{float(last_loss.item()):.6f}'
    print(f'updates: {completed}, loss: {loss}, checkpoint: {checkpoint}')
    return model


def evaluate_command(args):
    from tmt.benchmark import evaluate as score

    model, _ = load_model(args.checkpoint)
    data = list(documents(args.data, args.max_bytes))
    result = score(model, data, args.windows)
    print(f"input bytes: {result['input_bytes']}, targets: {result['targets']}, BPB: {result['bpb']}")
    print(f"common targets: {result['common_targets']}, common BPB: {result['common_bpb']}")
    for window in result['windows']:
        print(f"window {window['window']}: BPB {window['bpb']}, common BPB {window['common_bpb']}, delta {window['delta_bpb']}")
    if args.output is not None:
        with open(args.output, 'w') as file:
            json.dump(result, file, indent=2)
            file.write('\n')


def generate(checkpoint, prompt, output, seed=11, count=512, temperature=None):
    """Replay a UTF-8 prompt and write a fixed count of sampled bytes."""
    if not prompt:
        raise ValueError('generation needs a nonempty prompt')

    import mlx.core as mx

    model, _ = load_model(checkpoint, seed)
    mx.eval(model.parameters())
    model.reset()
    if temperature is not None:
        model.temp = temperature
    mx.random.seed(seed)

    prompt = prompt.encode('utf-8')
    for byte in prompt:
        _, (logits, _) = model.step(mx.array(byte), frozen=True)
    generated = bytearray()
    for _ in range(count):
        byte = int(model.sample(logits).item())
        generated.append(byte)
        _, (logits, _) = model.step(mx.array(byte), frozen=True)

    with open(output, 'wb') as file:
        file.write(generated)
    print(f'Generated {len(generated)} bytes to {output}')
    return bytes(generated)


def sweep(grid_path, pattern, development, output, config='model.json', updates=1000,
          seed=11, ce_only=False, max_bytes=8192):
    """Run each model and seed pair in JSON product order."""
    from tmt.benchmark import evaluate

    settings = load_config(config)
    with open(grid_path) as file:
        grid = json.load(file)
    keys = list(grid)
    development_data = list(documents(development, max_bytes))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'sweep output folder {output} is not empty; choose a new output folder'
        )
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / 'results.jsonl'
    rows = []

    for index, values in enumerate(itertools.product(*(grid[key] for key in keys)), 1):
        overrides = dict(zip(keys, values))
        run_seed = overrides.pop('seed', seed)
        run_settings = dict(settings)
        run_settings.update(overrides)
        run_path = output / f'run-{index:04d}'
        run_path.mkdir(parents=True, exist_ok=True)
        manifest_dir = run_path / 'manifest'
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_dir / 'model.json', 'w') as file:
            json.dump(run_settings, file, indent=2)
            file.write('\n')
        checkpoint = run_path / 'model.safetensors'
        model = train(run_settings, checkpoint, pattern, updates, run_seed,
                      ce_only, resume=False)
        score = evaluate(model, development_data)
        row = {
            'run': index,
            'settings': run_settings,
            'seed': run_seed,
            'updates': updates,
            'objective': 'ce_only' if ce_only else 'tmt',
            'data_glob': pattern,
            'development_glob': development,
            'development_max_bytes': max_bytes,
            'development_targets': score['targets'],
            'development_bpb': score['bpb'],
            'checkpoint': str(checkpoint.resolve()),
        }
        with open(results_path, 'a') as file:
            file.write(json.dumps(row) + '\n')
        rows.append(row)

    best = min(rows, key=lambda row: row['development_bpb'])
    print(f'best row: {json.dumps(best)}')
    return best


def init_config(path=None):
    settings = load_config()
    templates = {path: settings} if path is not None else {
        'model.json': settings,
        'grid.json': {'dim': [4, 8], 'layers': [1], 'spread': [4], 'seed': [11, 22]},
        'run.example.json': {'config': settings, 'seed': 11, 'started_at': None,
                             'status': 'running', 'last_checkpoint': None},
    }
    for destination, value in templates.items():
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, 'x') as file:
            json.dump(value, file, indent=2)
            file.write('\n')
        print(f'Wrote template to {destination}')
    if path is None:
        for directory in ('data/train', 'data/dev', 'runs'):
            Path(directory).mkdir(parents=True, exist_ok=True)


def version_text():
    from importlib.metadata import distribution

    installed = distribution('test-model-thing')
    direct_url = installed.read_text('direct_url.json')
    commit = None
    if direct_url:
        commit = json.loads(direct_url).get('vcs_info', {}).get('commit_id')
    return f'tmt {installed.version}, commit {commit or "unknown"}'


def command_manual(topic=None):
    from importlib.resources import files

    manual = files('tmt').joinpath('commands.md').read_text(encoding='utf-8')
    if topic is None:
        return manual
    if topic not in ('init', 'train', 'sweep', 'evaluate', 'benchmark', 'generate', 'help'):
        raise ValueError(f'unknown help topic: {topic}')
    heading = f'## {topic}\n'
    section = manual.split(heading, 1)[1].split('\n## ', 1)[0]
    return heading + section.rstrip() + '\n'


def build_parser():
    parser = argparse.ArgumentParser(prog='tmt', description='TMT experiment commands', epilog='Use tmt help for the complete command manual.')
    parser.add_argument('--version', action='version', version=version_text())
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='write default model settings')
    init.add_argument('--output', help='write only model JSON to PATH; default: initialize current folder')
    training = commands.add_parser('train', help='train a model on byte files')
    training.add_argument('checkpoint', nargs='?', help='explicit checkpoint path; required with --resume')
    training.add_argument('--run', help='new runs/NAME folder; default: UTC timestamp with microseconds; excludes checkpoint and --resume')
    training.add_argument('--data', required=True, help='quoted raw-file glob, one document per file')
    settings = training.add_mutually_exclusive_group()
    settings.add_argument('--config', help='fresh model JSON, default: model.json in current folder; excludes --resume')
    settings.add_argument('--resume', action='store_true', help='load checkpoint settings and add --updates with a new data pass; no exact cursor/RNG resume')
    training.add_argument('--updates', type=int, default=1000, help='additional target-byte optimizer updates, default: 1000')
    training.add_argument('--seed', type=int, default=11, help='Python and MLX seed, default: 11')
    training.add_argument('--ce-only', action='store_true', help='next-byte cross entropy only; default: full TMT objective')
    evaluation = commands.add_parser('evaluate', help='score next-byte predictions')
    evaluation.add_argument('checkpoint', help='strict model and optimizer checkpoint')
    evaluation.add_argument('--data', required=True, help='quoted raw-file glob, one document per file')
    evaluation.add_argument('--max-bytes', type=int, default=8192, help='global input-byte prefix across sorted files, default: 8192')
    evaluation.add_argument('--windows', type=int, nargs='+', default=[], help='context lengths in input bytes; default: no windows')
    evaluation.add_argument('--output', help='optional JSON score file; default: terminal only')
    benchmark = commands.add_parser('benchmark', help='run byte and CoLA benchmarks')
    benchmark.add_argument('checkpoint', help='strict checkpoint; fixed byte suite always runs')
    benchmark.add_argument('--epochs', type=int, default=1, help='CoLA classifier epochs, default: 1')
    benchmark.add_argument('--split', type=float, default=0.5, help='contiguous CoLA train fraction, default: 0.5')
    benchmark.add_argument('--cola-data', help='optional four-column CoLA TSV; default: byte suite only')
    benchmark.add_argument('--seed', type=int, default=11, help='model load and CoLA head seed, default: 11')
    generation = commands.add_parser('generate', help='generate a fixed number of bytes')
    generation.add_argument('checkpoint', help='strict checkpoint for frozen generation')
    generation.add_argument('--prompt', required=True, help='nonempty text encoded as UTF-8')
    generation.add_argument('--output', required=True, help='raw-byte output file without the prompt')
    generation.add_argument('--seed', type=int, default=11, help='sample seed, default: 11')
    generation.add_argument('--bytes', type=int, default=512, help='number of output bytes, default: 512')
    generation.add_argument('--temperature', type=float, help='sampler temperature; default: checkpoint value; zero still samples')
    sweeping = commands.add_parser('sweep', help='run a sequential model grid')
    sweeping.add_argument('--grid', required=True, help='JSON arrays of model values or seed')
    sweeping.add_argument('--data', required=True, help='quoted raw-file train glob')
    sweeping.add_argument('--development', required=True, help='quoted raw-file development glob')
    sweeping.add_argument('--output', required=True, help='sweep folder for run-NNNN and results.jsonl')
    sweeping.add_argument('--config', default='model.json', help='base model JSON, default: model.json in current folder')
    sweeping.add_argument('--updates', type=int, default=1000, help='target-byte optimizer updates per candidate, default: 1000')
    sweeping.add_argument('--seed', type=int, default=11, help='seed when absent from grid, default: 11')
    sweeping.add_argument('--ce-only', action='store_true', help='next-byte cross entropy only; default: full TMT objective')
    sweeping.add_argument('--max-bytes', type=int, default=8192, help='global development input-byte prefix, default: 8192')
    commands.add_parser('help', help='read the bundled command manual or one topic')
    return parser


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    help_flags = ('-h', '--help')
    if argv and (argv[0] == 'help' or (len(argv) > 1 and any(flag in argv for flag in help_flags))):
        if argv[0] == 'help':
            if any(flag in argv for flag in help_flags):
                topic = 'help'
            else:
                if len(argv) > 2:
                    parser.error('help accepts at most one topic')
                topic = argv[1] if len(argv) == 2 else None
        else:
            topic = argv[1] if argv[0] in help_flags else argv[0]
        try:
            print(command_manual(topic), end='')
        except ValueError as error:
            parser.error(str(error))
        return
    args = parser.parse_args(argv)
    if args.command == 'init':
        init_config(args.output)
    elif args.command == 'train':
        if args.checkpoint is not None and args.run is not None:
            parser.error('checkpoint and --run cannot be used together')
        if args.resume and (args.checkpoint is None or args.run is not None):
            parser.error('--resume requires an explicit checkpoint and forbids --run')
        if not glob.glob(args.data, recursive=True):
            raise FileNotFoundError(f'no files matched {args.data!r}')
        settings = None if args.resume else load_config(args.config or 'model.json')
        if args.checkpoint is None:
            name = args.run or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            folder = Path('runs') / name
            folder.mkdir(parents=True, exist_ok=False)
            args.checkpoint = folder / 'model.safetensors'
        train(settings, args.checkpoint, args.data, args.updates, args.seed, args.ce_only, args.resume)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'benchmark':
        model, _ = load_model(args.checkpoint, args.seed)
        from tmt.benchmark import run
        run(args.checkpoint, args.epochs, args.split, args.cola_data, model=model, seed=args.seed)
    elif args.command == 'generate':
        generate(args.checkpoint, args.prompt, args.output, args.seed, args.bytes, args.temperature)
    elif args.command == 'sweep':
        sweep(args.grid, args.data, args.development, args.output, args.config,
              args.updates, args.seed, args.ce_only, args.max_bytes)


def main(argv=None):
    try:
        return _main(argv)
    except (OSError, ValueError) as error:
        print(f'tmt: error: {error}', file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
