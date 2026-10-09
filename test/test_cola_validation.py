import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tmt.benchmark import cola, run
# This is where I realized I fucking hate the way I structured this project. I should have put the cola() function in a separate module, and then imported it into benchmark.py. Instead, I put it in benchmark.py, and now I have to import it from there in order to test it. This is a bad design decision, and I regret it.
# for the record, fuck writing tests. i hate it. i hate it so much

class ColaValidationTests(unittest.TestCase):
    def test_cola_rejects_nonbinary_label_with_tsv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'cola.tsv'
            data.write_text(
                'src\t1\tx\talpha\n'
                'src\t0\tx\tbeta\n'
                'src\t2\tx\tgamma\n'
                'src\t2\tx\tdelta\n',
                encoding='utf-8',
            )

            with self.assertRaisesRegex(ValueError, r'row 3 has invalid label .2.; expected 0 or 1'):
                cola(str(data))

    def test_cola_rejects_noninteger_label_with_tsv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'cola.tsv'
            data.write_text('src\t1\tx\talpha\nsrc\tno\tx\tbeta\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, r'row 2 has invalid label .no.; expected 0 or 1'):
                cola(str(data))

    def test_run_validates_labels_before_model_or_benchmark_work(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'cola.tsv'
            data.write_text(
                'src\t1\tx\talpha\n'
                'src\t0\tx\tbeta\n'
                'src\t2\tx\tgamma\n'
                'src\t2\tx\tdelta\n',
                encoding='utf-8',
            )

            with patch('tmt.cli.load_model') as load_model, patch('tmt.benchmark.evaluate') as evaluate:
                with self.assertRaisesRegex(ValueError, r'row 3 has invalid label .2.; expected 0 or 1'):
                    run('missing.safetensors', data=str(data))

                load_model.assert_not_called()
                evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
