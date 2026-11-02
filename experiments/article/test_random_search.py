"""Shell orchestration tests with a fake MEASE runner; no expensive mining runs."""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("random_search.sh")
DATASETS = ("cancer", "breast-cancer", "carcinoma", "lung", "mgus2", "veteran")


class RandomSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mease shell test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "experiments/article").mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.root / "experiments/article/random_search.sh")
        (self.root / "easd").mkdir()
        (self.root / "datasets/files").mkdir(parents=True)
        for name in ("main.py", "easd/core.py", "pyproject.toml", "uv.lock"):
            (self.root / name).write_text("fixture\n")
        for name in DATASETS:
            (self.root / f"datasets/files/{name}.parquet").write_text("fixture\n")
        bindir = self.root / "bin"
        bindir.mkdir()
        fake_uv = bindir / "uv"
        fake_uv.write_text(f"#!{sys.executable}\n" + '''
import os, pathlib, sys
args = sys.argv[1:]
assert args[:4] == ['run', '--locked', 'python', 'main.py']
def arg(key): return args[args.index(key) + 1]
name = arg('--dataset_name')
with open(os.environ['CALLS'], 'a') as f: f.write(name + '\\n')
if os.environ.get('FAIL_DATASET') == name: sys.exit(7)
out = pathlib.Path(arg('--output_dir')) / name / 'complement'
out.mkdir(parents=True, exist_ok=True)
for suffix in ('Info', 'RulesMetricsResult', 'DetailedRules'):
    if os.environ.get('MISSING_OUTPUT') == suffix: continue
    (out / f'{name}_0_complement_{suffix}.csv').write_text('metric\\n1\\n')
''')
        fake_uv.chmod(0o755)
        self.env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"], "CALLS": str(self.root / "calls")}

    def run_search(self, *args, env=None):
        return subprocess.run(
            ["bash", str(self.root / "experiments/article/random_search.sh"), "--num-configs", "1", *args],
            cwd="/tmp", env=self.env if env is None else env, text=True, capture_output=True,
        )

    @property
    def out(self):
        return self.root / "experiments/random_search_shell"

    def test_deterministic_preview_and_manifest_compatibility(self):
        first = self.run_search("--dry-run", "--num-configs", "4")
        self.assertEqual(first.returncode, 0, first.stderr)
        original = (self.out / "configs.txt").read_bytes()
        second = self.run_search("--dry-run", "--num-configs", "4")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(original, (self.out / "configs.txt").read_bytes())
        self.assertEqual(first.stdout.count("uv run --locked python main.py"), 24)
        configs = [ast.literal_eval(line) for line in original.decode().splitlines()]
        self.assertTrue(all(len(config) == 13 for config in configs))
        self.assertTrue(all(config["min_rule_support"] < config["max_rule_support"] for config in configs))
        self.assertFalse((self.root / "calls").exists())
        other = self.run_search("--dry-run", "--num-configs", "4", "--output-dir", "other")
        self.assertEqual(other.returncode, 0, other.stderr)
        self.assertEqual(original, (self.root / "other/configs.txt").read_bytes())

    def test_resume_and_repair_missing_output(self):
        result = self.run_search()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.root / "calls").read_text().splitlines()), 6)
        result = self.run_search()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Already completed"), 6)
        (self.out / "config_0/lung/complement/lung_0_complement_Info.csv").unlink()
        result = self.run_search()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.root / "calls").read_text().splitlines()), 7)
        self.assertEqual(result.stdout.count("Already completed"), 5)

    def test_failure_stops_and_can_resume(self):
        result = self.run_search(env={**self.env, "FAIL_DATASET": "cancer"})
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse((self.out / "config_0/cancer/complement/.complete").exists())
        self.assertFalse((self.out / ".random_search.lock").exists())
        result = self.run_search()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.root / "calls").read_text().splitlines()), 7)

    def test_missing_output_is_not_complete(self):
        result = self.run_search(env={**self.env, "MISSING_OUTPUT": "Info"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.out / "config_0/cancer/complement/.complete").exists())

    def test_refuse_changed_seed_manifest_and_code(self):
        self.assertEqual(self.run_search("--dry-run").returncode, 0)
        self.assertNotEqual(self.run_search("--dry-run", "--search-seed", "42").returncode, 0)
        manifest = self.out / "configs.txt"
        original = manifest.read_text()
        manifest.write_text("tampered\n")
        self.assertNotEqual(self.run_search("--dry-run").returncode, 0)
        manifest.write_text(original)
        (self.root / "main.py").write_text("changed\n")
        self.assertNotEqual(self.run_search("--dry-run").returncode, 0)

    def test_refuse_legacy_directory_and_invalid_options(self):
        self.out.mkdir(parents=True)
        (self.out / "configs.txt").write_text("legacy\n")
        self.assertNotEqual(self.run_search("--dry-run").returncode, 0)
        self.assertEqual((self.out / "configs.txt").read_text(), "legacy\n")
        for args in [("--search-seed", "0"), ("--num-configs", "0"), ("--run-seed", "-1"), ("--output-dir",)]:
            with self.subTest(args=args):
                self.assertNotEqual(self.run_search(*args).returncode, 0)


if __name__ == "__main__":
    unittest.main()
