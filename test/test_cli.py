import contextlib, inspect, json, math, random, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import Mock, patch


class CliTests(unittest.TestCase):
    manifest_fields = {'config', 'seed', 'data_glob', 'data_files', 'updates', 'objective',
                       'completed_updates', 'resume', 'initial_optimizer_step',
                       'started_at', 'status', 'last_checkpoint'}

    def test_version(self):
        from io import StringIO
        from tmt.cli import main

        installed = Mock(version='0.1.0')
        for metadata, commit in ((None, 'unknown'), ('{"vcs_info":{"commit_id":"abc123"}}', 'abc123')):
            installed.read_text.return_value = metadata
            stdout = StringIO()
            with patch('importlib.metadata.distribution', return_value=installed), contextlib.redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as exit:
                    main(['--version'])
            self.assertEqual(exit.exception.code, 0)
            self.assertEqual(stdout.getvalue(), f'tmt 0.1.0, commit {commit}\n')
        code = """
import builtins, runpy
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'tmt.main' or name == 'mlx' or name.startswith('mlx.'):
        raise ImportError('model import forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_module('tmt', run_name='__main__', alter_sys=True)
"""
        for args in (['--version'], ['--help']):
            result = subprocess.run([sys.executable, '-c', code, *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_command_manual(self):
        from importlib.resources import files
        from tmt.cli import command_manual

        manual = files('tmt').joinpath('commands.md').read_text(encoding='utf-8')
        self.assertEqual(command_manual(), manual)
        script = """
import builtins, runpy
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'tmt.main' or name == 'mlx' or name.startswith('mlx.'):
        raise ImportError('model import forbidden in help')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_module('tmt', run_name='__main__', alter_sys=True)
"""
        with tempfile.TemporaryDirectory() as directory:
            def run(args):
                return subprocess.run([sys.executable, '-c', script, *args], cwd=directory,
                                      capture_output=True, text=True, timeout=10)
            for topic in ('init', 'train', 'sweep', 'evaluate', 'benchmark', 'generate', 'help'):
                expected = '## ' + topic + '\n' + manual.split('## ' + topic + '\n', 1)[1].split('\n## ', 1)[0].rstrip() + '\n'
                self.assertEqual(command_manual(topic), expected)
                for args in (['help', topic], ['--help', topic], ['-h', topic], [topic, '--help'],
                             [topic, '-h'], [topic, '--data', 'missing/*', '--help']):
                    with self.subTest(args=args):
                        result = run(args)
                        self.assertEqual((result.returncode, result.stdout), (0, expected), result.stderr)
            self.assertEqual(run(['help']).stdout, manual)
            for args in (['--help'], ['-h']):
                result = run(args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('tmt help', result.stdout)
                self.assertNotIn('## init', result.stdout)
            for args in (['help', 'nonexistent'], ['--help', 'nonexistent'], ['-h', 'nonexistent'], ['nonexistent', '--help']):
                result = run(args)
                self.assertEqual(result.returncode, 2)
                self.assertIn('unknown help topic', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_installed_workflow(self):
        import os, sysconfig
        from tmt.cli import load_model

        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['PATH'] = sysconfig.get_path('scripts') + os.pathsep + env.get('PATH', '')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(*args):
                result = subprocess.run(['tmt', *args], cwd=root, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout
            self.assertIn('## init', run('help', 'init'))
            run('init')
            (root / 'model.json').write_text('{"dim":4,"layers":1,"spread":4}')
            (root / 'data/train/tiny.bin').write_bytes(b'abcd')
            run('train', '--data', 'data/train/*', '--run', 'first', '--updates', '3')
            checkpoint = root / 'runs/first/model.safetensors'
            model, settings = load_model(checkpoint)
            self.assertEqual((settings['dim'], settings['layers'], settings['spread'], model.optimizer.state['step'].item()), (4, 1, 4, 3))
            manifest = checkpoint.parent / 'manifest/run.json'
            fresh = json.loads(manifest.read_text())
            self.assertEqual((fresh['data_glob'], fresh['data_files'], fresh['updates'], fresh['completed_updates']),
                             ('data/train/*', [str((root / 'data/train/tiny.bin').resolve())], 3, 3))
            self.assertEqual((fresh['objective'], fresh['resume'], fresh['initial_optimizer_step']), ('tmt', False, 0))
            (root / 'data/resume').mkdir()
            (root / 'data/resume/next.bin').write_bytes(b'wxyz')
            run('train', str(checkpoint), '--data', 'data/resume/*', '--resume', '--updates', '2', '--seed', '99', '--ce-only')
            self.assertEqual(load_model(checkpoint)[0].optimizer.state['step'].item(), 5)
            record = json.loads(manifest.read_text())
            self.assertEqual(set(record), self.manifest_fields)
            self.assertEqual((record['status'], record['config'], record['last_checkpoint']), ('complete', settings, str(checkpoint.resolve())))
            self.assertEqual((record['data_glob'], record['data_files'], record['updates'], record['completed_updates']),
                             ('data/resume/*', [str((root / 'data/resume/next.bin').resolve())], 2, 2))
            self.assertEqual((record['seed'], record['objective'], record['resume'], record['initial_optimizer_step']),
                             (99, 'ce_only', True, 3))
            run('evaluate', str(checkpoint), '--data', 'data/train/*', '--windows', '1', '2', '--output', 'score.json')
            self.assertEqual(json.loads((root / 'score.json').read_text())['targets'], 3)
            run('generate', str(checkpoint), '--prompt', 'The ', '--output', 'sample.bin', '--bytes', '4')
            self.assertEqual(len((root / 'sample.bin').read_bytes()), 4)
            self.assertIn('390 targets', run('benchmark', str(checkpoint)))

    def test_installed_failure_paths(self):
        import os, sysconfig

        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        command = str(Path(sysconfig.get_path('scripts')) / 'tmt')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(*args, success=False):
                result = subprocess.run([command, *args], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0 if success else 1, result.stderr)
                self.assertNotIn('Traceback', result.stderr)
                return result

            # Missing data must fail before config loading or run-folder creation.
            for extra in (['--run', 'absent'], []):
                result = run('train', '--data', 'missing/*', *extra)
                self.assertIn('no files matched', result.stderr)
                self.assertEqual(list(root.iterdir()), [])

            (root / 'model.json').write_text(json.dumps(self._tiny_settings()))
            (root / 'grid.json').write_text('{"seed": [11, 22]}')
            (root / 'train.bin').write_bytes(b'abc')
            (root / 'dev.bin').write_bytes(b'xyz')
            args = ['sweep', '--grid', 'grid.json', '--data', 'train.bin',
                    '--development', 'dev.bin', '--output', 'sweep', '--updates', '2']
            (root / 'sweep').mkdir()  # An existing empty folder is valid.
            run(*args, success=True)
            before = {p: p.read_bytes() for p in (root / 'sweep').rglob('*') if p.is_file()}
            result = run(*args[:-1], '1')
            self.assertIn('not empty', result.stderr)
            self.assertEqual(before, {p: p.read_bytes() for p in (root / 'sweep').rglob('*') if p.is_file()})
            checkpoint = root / 'sweep/run-0001/model.safetensors'
            for contents in ('{', '[]', '{"unknown": 1}'):
                (root / 'bad.json').write_text(contents)
                run('train', '--data', 'train.bin', '--config', 'bad.json', '--run', 'bad')
                self.assertFalse((root / 'runs/bad').exists())
            run('generate', 'missing.safetensors', '--prompt', 'x', '--output', 'sample.bin')
            (root / 'invalid.safetensors').write_bytes(b'invalid')
            run('evaluate', 'invalid.safetensors', '--data', 'dev.bin')
            (root / 'short.bin').write_bytes(b'x')
            run('train', '--data', 'short.bin', '--run', 'no-targets', '--updates', '1')
            record = json.loads((root / 'runs/no-targets/manifest/run.json').read_text())
            self.assertEqual((record['status'], record['last_checkpoint']), ('failed', None))
            self.assertEqual((record['updates'], record['completed_updates'], record['initial_optimizer_step']), (1, 0, 0))
            result = run('evaluate', str(checkpoint), '--data', 'short.bin')
            self.assertIn('at least one target', result.stderr)

    def test_init(self):
        from tmt.cli import load_config, main
        from tmt.main import Model

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            main(['init'])
            saved = json.loads((Path(directory) / 'model.json').read_text())
            self.assertEqual(saved, load_config())
            self.assertEqual(set(saved), set(inspect.signature(Model).parameters))
            templates = {Path(directory) / name for name in ('model.json', 'grid.json', 'run.example.json')}
            self.assertEqual(set(Path(directory).rglob('*.json')), templates)
            self.assertTrue(all(Path(name).is_dir() for name in ('data/train', 'data/dev', 'runs')))
            self.assertEqual(json.loads(Path('grid.json').read_text()),
                             {'dim': [4, 8], 'layers': [1], 'spread': [4], 'seed': [11, 22]})
            template = json.loads(Path('run.example.json').read_text())
            self.assertEqual(set(template), self.manifest_fields)
            self.assertEqual((template['config'], template['seed'], template['data_glob'], template['data_files']),
                             (saved, 11, 'data/train/*', []))
            self.assertEqual((template['updates'], template['objective'], template['completed_updates']), (1000, 'tmt', 0))
            self.assertEqual((template['resume'], template['initial_optimizer_step'], template['started_at'], template['last_checkpoint']),
                             (False, 0, None, None))

            explicit = Path(directory) / 'custom' / 'settings.json'
            main(['init', '--output', str(explicit)])
            self.assertEqual(json.loads(explicit.read_text()), saved)
            self.assertEqual(set(Path(directory).rglob('*.json')),
                             templates | {explicit})

        self.assertEqual(load_config(repo / 'examples' / 'model.json'), load_config())
        tiny = load_config(repo / 'examples' / 'tiny-model.json')
        self.assertEqual((tiny['dim'], tiny['layers'], tiny['spread']), (4, 1, 4))
        self.assertEqual(json.loads((repo / 'examples' / 'grid.json').read_text()),
                         {'dim': [4, 8], 'seed': [11, 22]})

    def test_init_preserves_files(self):
        from tmt.cli import main

        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            main(['init'])
            Path('model.json').write_text('{"dim": 4}')
            before = {p: p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
            for args in (['init'], ['init', '--output', 'model.json']):
                with self.assertRaises(SystemExit) as exit: main(args)
                self.assertEqual(exit.exception.code, 1)
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_config(self):
        from tmt.cli import load_config
        from tmt.main import Model

        defaults = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text('{}')
            self.assertEqual(load_config(path), defaults)
            path.write_text('{"dim": 4}')
            settings = load_config(path)
            self.assertEqual((settings['dim'], settings['layers'], settings['bound']), (4, 16, [40000, 120000]))

        parameters = list(inspect.signature(Model).parameters.values())
        parameters.append(inspect.Parameter('extra', inspect.Parameter.KEYWORD_ONLY, default=17))
        extended = Mock()
        extended.__signature__ = inspect.Signature(parameters)
        with patch('tmt.main.Model', extended):
            self.assertEqual(load_config()['extra'], 17)

    def _tiny_settings(self):
        return {'dim': 4, 'layers': 1, 'spread': 4, 'temp': 0.75,
                'rate': 0.0005, 'bound': [40000, 120000]}

    def test_train(self):
        import mlx.core as mx
        from tmt.cli import main, train
        from tmt.main import Model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'train.bin'
            data.write_bytes(bytes((0, 255, 65, 10)))
            calls = []
            train_step, step = Model.train_step, Model.step

            def record_train(model, current, target, end, ce_only=False):
                calls.append((current, target, end))
                return train_step(model, current, target, end, ce_only)

            from io import StringIO
            stdout = StringIO()
            with (contextlib.redirect_stdout(stdout),
                  patch.object(Model, 'sample', side_effect=AssertionError('sampled')),
                  patch.object(Model, 'train_step', record_train),
                  patch.object(Model, 'step', autospec=True, side_effect=step) as steps):
                first = train(self._tiny_settings(), root / 'one' / 'model.safetensors', str(data), 3, 11)
                second = train(self._tiny_settings(), root / 'two' / 'model.safetensors', str(data), 3, 11)
                resumed = train(None, root / 'one' / 'model.safetensors', str(data), 2, 11, resume=True)

            self.assertEqual(stdout.getvalue().count('run folder:'), 3)
            self.assertEqual(stdout.getvalue().count('saved checkpoint:'), 3)
            self.assertEqual(stdout.getvalue().count('loss:'), 3)
            self.assertIn('seed: 11, config:', stdout.getvalue())
            self.assertNotIn('\r', stdout.getvalue())

            with contextlib.chdir(root), patch('tmt.cli.train') as command:
                Path('model.json').write_text(json.dumps(self._tiny_settings()))
                main(['train', '--data', str(data), '--run', 'named', '--updates', '3'])
                self.assertEqual(command.call_args.args[:2], (self._tiny_settings(), Path('runs/named/model.safetensors')))
                main(['train', '--data', str(data)])
                self.assertEqual(command.call_args.args[1].parent.parent, Path('runs'))
                with self.assertRaises(SystemExit) as exit: main(['train', '--data', str(data), '--run', 'named'])
                self.assertEqual(exit.exception.code, 1)
                for args in (['--run', 'bad', 'model.safetensors'], ['--resume'], ['--resume', '--run', 'bad'],
                             ['model.safetensors', '--resume', '--config', 'model.json']):
                    with self.assertRaises(SystemExit) as exit: main(['train', '--data', str(data), *args])
                    self.assertEqual(exit.exception.code, 2)

            expected = [(0, 255, False), (255, 65, False), (65, 10, True)]
            self.assertEqual(calls, expected + expected + expected[:2])
            self.assertEqual([call.args[1].item() for call in steps.call_args_list if call.kwargs.get('frozen')], [10, 10])
            self.assertEqual([int(model.optimizer.state['step'].item()) for model in (first, second, resumed)], [3, 3, 5])
            self.assertTrue(mx.array_equal(first.decoder.decode.weight, second.decoder.decode.weight).item())

    def test_manifest(self):
        from datetime import datetime, timezone
        from tmt.cli import train
        from tmt.main import Model

        original_dump, records = json.dump, []

        def capture(record, *args, **kwargs):
            records.append(dict(record))
            return original_dump(record, *args, **kwargs)

        train_step = Model.train_step
        for ce_only in (False, True):
            with self.subTest(ce_only=ce_only), tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
                root = Path(directory).resolve()
                Path('data').mkdir()
                paths, calls = {}, []
                for name, content in (('a.bin', b'ab'), ('b.bin', b'cd'), ('c.bin', b'ef')):
                    data = root / 'data' / name
                    data.write_bytes(content)
                    paths[content[0]] = str(data)

                def record_train(model, current, target, end, objective=False):
                    calls.append((paths[current], objective))
                    return train_step(model, current, target, end, objective)

                checkpoint = root / 'run' / 'model.safetensors'
                records.clear()
                with (patch('tmt.cli.json.dump', side_effect=capture) as writes,
                      patch.object(Model, 'train_step', record_train)):
                    train(self._tiny_settings(), checkpoint, 'data/*.bin', 4, 11, ce_only)

                self.assertEqual(writes.call_count, 2)
                self.assertEqual(set(records[0]), self.manifest_fields)
                for record in records:
                    self.assertEqual((record['config'], record['seed'], record['data_glob']), (self._tiny_settings(), 11, 'data/*.bin'))
                    self.assertEqual(set(record['data_files']), set(paths.values()))
                    self.assertEqual(record['data_files'], [path for path, _ in calls[:3]])
                    self.assertEqual((record['updates'], record['objective'], record['resume'], record['initial_optimizer_step']),
                                     (4, 'ce_only' if ce_only else 'tmt', False, 0))
                self.assertEqual(calls, [(path, ce_only) for path in records[0]['data_files'] + records[0]['data_files'][:1]])
                self.assertEqual([(record['status'], record['completed_updates'], record['last_checkpoint']) for record in records],
                                 [('running', 0, None), ('complete', 4, str(checkpoint.resolve()))])
                self.assertEqual(datetime.fromisoformat(records[0]['started_at']).utcoffset(), timezone.utc.utcoffset(None))
                self.assertEqual(json.loads((checkpoint.parent / 'manifest' / 'run.json').read_text()), records[1])
                self.assertTrue(checkpoint.is_file())

    def test_manifest_partial_progress(self):
        from tmt.cli import train
        from tmt.main import Model

        train_step = Model.train_step
        for error_type, status in ((OSError, 'failed'), (KeyboardInterrupt, 'interrupted')):
            for resume in (False, True):
                with self.subTest(status=status, resume=resume), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    data = root / 'train.bin'
                    data.write_bytes(b'abcdef')
                    checkpoint = root / 'run' / 'model.safetensors'
                    original = None
                    if resume:
                        train(self._tiny_settings(), checkpoint, str(data), 2, 11)
                        original = checkpoint.read_bytes()
                    completed = 0

                    def fail(model, current, target, end, ce_only=False):
                        nonlocal completed
                        if completed == 1:
                            raise error_type('injected update failure')
                        result = train_step(model, current, target, end, ce_only)
                        completed += 1
                        return result

                    with patch.object(Model, 'train_step', fail), self.assertRaises(error_type):
                        train(None if resume else self._tiny_settings(), checkpoint, str(data), 3, 17, True, resume)

                    record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
                    self.assertEqual(set(record), self.manifest_fields)
                    self.assertEqual((record['data_glob'], record['data_files'], record['updates'], record['completed_updates']),
                                     (str(data), [str(data.resolve())], 3, 1))
                    self.assertEqual((record['status'], record['seed'], record['objective'], record['resume'], record['initial_optimizer_step']),
                                     (status, 17, 'ce_only', resume, 2 if resume else 0))
                    self.assertEqual(record['last_checkpoint'], str(checkpoint.resolve()) if resume else None)
                    if resume:
                        self.assertEqual(checkpoint.read_bytes(), original)
                    else:
                        self.assertFalse(checkpoint.exists())

    def test_evaluate(self):
        import mlx.core as mx
        from types import SimpleNamespace
        from tmt.benchmark import evaluate

        fields = ('states', 'decaytrace', 'embedtrace'); block = SimpleNamespace(states=mx.array([91.]), decaytrace=mx.array([92.]), embedtrace=mx.array([[93.]]))
        model, uniform = SimpleNamespace(blocks=[block]), [False]
        def reset(): block.states, block.decaytrace, block.embedtrace = mx.zeros((1,)), mx.zeros((1,)), mx.zeros((1, 1))
        def step(current, dummies=None, frozen=False):
            self.assertTrue(frozen)
            byte = int(current.item())
            state = (int(block.states.item()) * 3 + byte) % 17
            block.states = mx.array([float(state)])
            logits = mx.zeros((256,)) if uniform[0] else mx.array([state / 5 if i == byte else 0 for i in range(256)])
            return ((None, [], []), (logits, mx.zeros((1,))))
        model.reset, model.step = reset, step
        saved = [mx.array(getattr(block, field)) for field in fields]
        mx.eval(*saved)
        def unchanged(): self.assertTrue(all(mx.array_equal(getattr(block, field), value).item() for field, value in zip(fields, saved)))
        uniform[0] = True
        for documents in ([], [b''], [b'a', b'b']):
            with self.assertRaisesRegex(ValueError, 'at least one target'):
                evaluate(model, documents)
            unchanged()
        uniform_score = evaluate(model, [b'aa'])
        self.assertAlmostEqual(uniform_score['bpb'], 8.0, delta=1e-6)
        self.assertIsNone(uniform_score['common_bpb'])
        uniform[0], documents, windows = False, [b'aa', b'abcde', b'', b'xy', b'pqrs'], (1, 2)

        def manual(prefix, target):
            state = sum(byte * 3 ** power for byte, power in zip(prefix, reversed(range(len(prefix))))) % 17
            return math.log(255 + math.exp(state / 5)) - (state / 5 if prefix[-1] == target else 0)

        def total(window=None, common=False):
            return math.fsum(manual(document[max(0, offset - window):offset] if window else document[:offset], document[offset])
                             for document in documents for offset in range(1, len(document)) if not common or offset >= 2)
        result = evaluate(model, documents, windows); self.assertEqual((result['input_bytes'], result['targets'], result['common_targets']), (13, 9, 5))
        self.assertAlmostEqual(result['bpb'], total() / (9 * math.log(2)), delta=1e-5)
        self.assertAlmostEqual(result['common_bpb'], total(common=True) / (5 * math.log(2)), delta=1e-5)
        for score, window in zip(result['windows'], windows):
            common = total(window, True) / (5 * math.log(2))
            self.assertAlmostEqual(score['bpb'], total(window) / (9 * math.log(2)), delta=1e-5)
            self.assertAlmostEqual(score['common_bpb'], common, delta=1e-5)
            self.assertAlmostEqual(score['delta_bpb'], common - result['common_bpb'], delta=1e-5)
        unchanged()
        short = evaluate(model, [b'ab'], (3,))
        self.assertEqual((short['common_targets'], short['common_bpb'], short['windows'][0]['delta_bpb']), (0, None, None))
        with patch('tmt.benchmark._nll', side_effect=RuntimeError('injected score failure')), self.assertRaisesRegex(RuntimeError, 'injected'): evaluate(model, documents, windows)
        unchanged()

    def test_benchmark(self):
        import mlx.core as mx
        import mlx.utils as util
        from tmt.benchmark import BYTE_DOCUMENTS, CONTEXT_WINDOWS, mcc, run
        from tmt.cli import new_model, save_checkpoint

        settings = {'dim': 4, 'layers': 1, 'spread': 4, 'temp': 0.75, 'rate': 0.0005, 'bound': [40000, 120000]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = new_model(settings, 11)
            checkpoint = root / 'model.safetensors'
            save_checkpoint(checkpoint, model, settings)
            fields = ('states', 'decaytrace', 'embedtrace')
            before = {name: mx.array(value) for name, value in util.tree_flatten(model.parameters()) if not any(field in name for field in fields)}
            optimizer_before = dict(util.tree_flatten(model.optimizer.state))
            mx.eval(*before.values(), *optimizer_before.values())
            data = root / 'cola.tsv'
            data.write_text('src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t0\tx\tgamma\nsrc\t1\tx\tdelta\n')
            from io import StringIO
            stdout = StringIO()
            states = [mx.array(getattr(model.blocks[0], field)) for field in fields]
            with contextlib.redirect_stdout(stdout):
                byte_only = run(str(checkpoint), model=model)
            self.assertEqual(byte_only['epochs'], [])
            self.assertEqual(byte_only['byte_scores']['targets'], 390)
            self.assertIn('104 targets', stdout.getvalue())
            self.assertTrue(all(mx.array_equal(value, getattr(model.blocks[0], field)).item()
                                for value, field in zip(states, fields)))
            result = run(str(checkpoint), 1, 0.5, str(data), model=model, seed=11)
            self.assertEqual(byte_only['byte_scores'], result['byte_scores'])
            with self.assertRaises(FileNotFoundError): run(str(checkpoint), data=str(root / 'missing.tsv'), model=model)

        scores = result['byte_scores']
        self.assertEqual((sum(map(len, BYTE_DOCUMENTS)), scores['input_bytes'], scores['targets'], scores['common_targets']), (393, 393, 390, 104))
        self.assertEqual(tuple(item['window'] for item in scores['windows']), CONTEXT_WINDOWS)
        held = result['epochs'][0]['held']
        denominator = math.sqrt((held['tp'] + held['fp']) * (held['tp'] + held['fn']) * (held['tn'] + held['fp']) * (held['tn'] + held['fn']))
        self.assertAlmostEqual(held['mcc'], (held['tp'] * held['tn'] - held['fp'] * held['fn']) * 100 / denominator if denominator else 0)
        self.assertEqual(mcc(1, 1, 0, 0), 100.0)
        after = dict(util.tree_flatten(model.parameters()))
        self.assertTrue(all(mx.array_equal(value, after[name]).item() for name, value in before.items()))
        optimizer_after = dict(util.tree_flatten(model.optimizer.state))
        self.assertTrue(all(mx.array_equal(value, optimizer_after[name]).item() for name, value in optimizer_before.items()))

    def _checkpoint(self, path):
        import mlx.core as mx
        from tmt.cli import new_model, save_checkpoint

        settings = {'dim': 4, 'layers': 1, 'spread': 4, 'temp': 0.75, 'rate': 0.0005, 'bound': [40000, 120000]}
        model = new_model(settings, 11)
        save_checkpoint(path, model, settings)
        arrays, metadata = mx.load(path, return_metadata=True)
        mx.eval(arrays)
        return mx, settings, model, arrays, metadata

    @contextlib.contextmanager
    def _saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            mx, settings, model, arrays, metadata = self._checkpoint(path)
            yield path, mx, settings, model, arrays, metadata

    def _assert_rejected(self, path, mx, load_model, arrays, metadata, *parts):
        mx.save_safetensors(path, arrays, metadata=metadata)
        with self.assertRaises(ValueError) as error: load_model(path)
        message = str(error.exception)
        for part in parts: self.assertIn(part, message)

    def test_checkpoint_names(self):
        from tmt.cli import load_model

        with self._saved_checkpoint() as (path, mx, _, _, arrays, metadata):
            key = next(name for name in arrays if name.startswith('m.'))
            missing = dict(arrays)
            missing.pop(key)
            extra = dict(arrays)
            extra['o.extra'] = next(iter(arrays.values()))
            cases = [('missing model tensor', missing, key), ('extra optimizer tensor', extra, 'o.extra')]
            for label, tensors, name in cases:
                with self.subTest(label=label): self._assert_rejected(path, mx, load_model, tensors, metadata, name)

    def test_checkpoint_shapes(self):
        from tmt.cli import load_model

        with self._saved_checkpoint() as (path, mx, _, _, arrays, metadata):
            name = 'm.decoder.decode.weight'
            arrays[name] = mx.zeros((1,), dtype=arrays[name].dtype)
            self._assert_rejected(path, mx, load_model, arrays, metadata, 'decoder.decode.weight', 'shape')

    def test_checkpoint_dtypes(self):
        from tmt.cli import load_model

        with self._saved_checkpoint() as (path, mx, _, _, arrays, metadata):
            name = next(key for key in arrays if key.startswith('o.') and key.endswith('.m'))
            arrays[name] = mx.zeros(arrays[name].shape, dtype=mx.int32)
            self._assert_rejected(path, mx, load_model, arrays, metadata, name[2:], 'dtype')

    def test_generate(self):
        from tmt.cli import generate

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            self._checkpoint(path)
            original = path.read_bytes()
            first_path, second_path = Path(directory) / 'first.bin', Path(directory) / 'second.bin'
            first = generate(path, 'The ', first_path, 11, 16)
            second = generate(path, 'The ', second_path, 11, 16)
            self.assertEqual((len(first), len(second)), (16, 16))
            self.assertEqual(first, second)
            self.assertEqual(first_path.read_bytes(), first)
            self.assertEqual(second_path.read_bytes(), second)
            self.assertEqual(path.read_bytes(), original)

    def test_sweep(self):
        from io import StringIO
        from tmt.cli import load_model, sweep

        from tmt.cli import main
        with patch('tmt.cli.sweep') as command:
            args = ['sweep', '--grid', 'grid.json', '--data', 'train/*', '--development', 'dev/*', '--output', 'runs/grid']
            main(args)
            self.assertEqual(command.call_args.args[4], 'model.json')
            main([*args, '--config', 'custom.json'])
            self.assertEqual(command.call_args.args[4], 'custom.json')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid, config = root / 'grid.json', root / 'model.json'
            grid.write_text(json.dumps({'dim': [4, 8], 'seed': [11, 22]}))
            config.write_text(json.dumps(self._tiny_settings()))
            train, development = root / 'train.bin', root / 'development.bin'
            train.write_bytes(b'\x00\x01\x02')
            development.write_bytes(b'\xf0\xf1\xf2\xf3')
            for ce_only in (False, True):
                with self.subTest(ce_only=ce_only):
                    output, stdout = root / ('runs-ce' if ce_only else 'runs'), StringIO()
                    with contextlib.redirect_stdout(stdout):
                        best = sweep(grid, str(train), str(development), output, config, 2, 11, ce_only, max_bytes=4)
                    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
                    self.assertEqual([(row['settings']['dim'], row['seed']) for row in rows], [(4, 11), (4, 22), (8, 11), (8, 22)])
                    self.assertEqual(best, min(rows, key=lambda row: row['development_bpb']))
                    self.assertIn(json.dumps(best), stdout.getvalue())
                    for index, row in enumerate(rows, 1):
                        checkpoint = Path(row['checkpoint'])
                        self.assertEqual((row['objective'], row['development_targets']), ('ce_only' if ce_only else 'tmt', 3))
                        self.assertEqual(json.loads((checkpoint.parent / 'manifest' / 'model.json').read_text()), row['settings'])
                        manifest = json.loads((checkpoint.parent / 'manifest' / 'run.json').read_text())
                        self.assertEqual((set(manifest), manifest['status'], manifest['last_checkpoint']),
                                         (self.manifest_fields, 'complete', str(checkpoint.resolve())))
                        self.assertEqual(manifest['config'], row['settings'])
                        for field in ('seed', 'updates', 'objective', 'data_glob'):
                            self.assertEqual(manifest[field], row[field])
                        self.assertEqual((manifest['data_files'], manifest['completed_updates'], manifest['resume'], manifest['initial_optimizer_step']),
                                         ([str(train.resolve())], row['updates'], False, 0))
                        model, settings = load_model(checkpoint, row['seed'])
                        self.assertEqual(settings, row['settings'])
                        self.assertEqual(int(model.optimizer.state['step'].item()), 2)
                        self.assertEqual(checkpoint.parent.name, f'run-{index:04d}')

    def test_checkpoint_header(self):
        from tmt.cli import load_model

        with self._saved_checkpoint() as (path, mx, settings, model, arrays, metadata):
            restored, loaded = load_model(path)
            self.assertEqual((loaded, restored.dim, int(restored.optimizer.state['step'].item())),
                             (settings, 4, 0))
            self.assertTrue(mx.array_equal(restored.decoder.decode.weight, model.decoder.decode.weight).item())

            header = json.loads(metadata['tmt'])
            missing = dict(header)
            missing['model'] = dict(header['model'])
            missing['model'].pop('spread')
            bad_version = dict(header)
            bad_version['version'] = True
            cases = [('missing metadata', None, '__metadata__'),
                     ('missing model field', {'tmt': json.dumps(missing)}, 'spread'),
                     ('bad version', {'tmt': json.dumps(bad_version)}, 'version')]
            for label, invalid, field in cases:
                with self.subTest(label=label):
                    self._assert_rejected(path, mx, load_model, arrays, invalid, field)
            with self.subTest(label='truncated header'):
                path.write_bytes((8).to_bytes(8, 'little') + b'{}')
                with self.assertRaisesRegex(ValueError, 'header'): load_model(path)
