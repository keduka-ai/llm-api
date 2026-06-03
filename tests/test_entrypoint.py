"""
Tests for the shell layer: model-defaults.sh, download-models.sh, entrypoint.sh.

The scripts are written to be *sourceable* — sourcing them defines their
functions and sets their defaults without running the download or starting any
process (the side-effecting `main` is guarded by a `BASH_SOURCE == $0` check).
That lets these tests exercise the real script logic (model resolution, server
arg assembly, URL sanitisation) with no GPU, no network, and no llama-server.
"""

import os
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DEFAULTS = os.path.join(REPO_ROOT, "model-defaults.sh")
DOWNLOAD = os.path.join(REPO_ROOT, "download-models.sh")
ENTRYPOINT = os.path.join(REPO_ROOT, "entrypoint.sh")


def run_bash(script, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, env=full_env, cwd=REPO_ROOT,
    )


class ModelDefaultsTests(unittest.TestCase):
    def test_default_alias_and_filename(self):
        r = run_bash(f'source "{MODEL_DEFAULTS}"; echo "$DEFAULT_MODEL_ALIAS|$DEFAULT_MODEL_FILENAME"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "gemma-4-e2b-it|gemma-4-E2B-it-UD-Q6_K_XL.gguf")


class DownloadModelsResolveTests(unittest.TestCase):
    def test_catalog_alias_resolves(self):
        r = run_bash(
            f'source "{DOWNLOAD}"; resolve_model gemma-4-e2b-it; echo "$MODEL_FILE|$MODEL_URL"'
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.strip()
        self.assertTrue(out.startswith("gemma-4-E2B-it-UD-Q6_K_XL.gguf|"), out)
        self.assertIn("https://huggingface.co/", out)

    def test_direct_url_resolves_and_sanitises_filename(self):
        url = "https://example.com/a/b/My.Model-v2.gguf?download=true&t=xyz"
        r = run_bash(f'source "{DOWNLOAD}"; resolve_model "{url}"; echo "$MODEL_FILE"')
        self.assertEqual(r.returncode, 0, r.stderr)
        # Query string stripped; only [A-Za-z0-9._-] kept.
        self.assertEqual(r.stdout.strip(), "My.Model-v2.gguf")

    def test_unknown_alias_exits_nonzero(self):
        r = run_bash(f'source "{DOWNLOAD}"; resolve_model totally-bogus')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Unknown model", r.stderr)

    def test_sourcing_does_not_download(self):
        # Sourcing must not invoke wget / write files.
        r = run_bash(f'source "{DOWNLOAD}"; echo sourced-ok')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("sourced-ok", r.stdout)


class EntrypointResolveFilenameTests(unittest.TestCase):
    def test_explicit_model_file_wins(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_bash(
                f'source "{ENTRYPOINT}"; resolve_model_filename',
                env={"MODEL_FILE": "explicit.gguf", "MODELS_DIR": d},
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "explicit.gguf")

    def test_active_model_marker_used(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, ".active_model"), "w") as f:
                f.write("marked.gguf\n")
            r = run_bash(
                f'source "{ENTRYPOINT}"; resolve_model_filename',
                env={"MODELS_DIR": d, "MODEL_FILE": ""},
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "marked.gguf")

    def test_default_filename_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_bash(
                f'source "{ENTRYPOINT}"; resolve_model_filename',
                env={"MODELS_DIR": d, "MODEL_FILE": ""},
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "gemma-4-E2B-it-UD-Q6_K_XL.gguf")


class EntrypointServerArgsTests(unittest.TestCase):
    def _args(self, env=None):
        r = run_bash(
            f'source "{ENTRYPOINT}"; build_server_args /models/x.gguf; '
            f'printf "%s\\n" "${{SERVER_ARGS[@]}}"',
            env=env,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_core_flags_present(self):
        out = self._args()
        for token in ("--model", "/models/x.gguf", "--alias", "gemma-4-e2b-it",
                      "--host", "0.0.0.0", "--port", "8080", "--ctx-size",
                      "--n-gpu-layers", "--flash-attn", "--jinja", "--metrics"):
            self.assertIn(token, out, f"missing {token}\n{out}")

    def test_draft_model_added_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "draft.gguf"), "w").close()
            out = self._args(env={"MODELS_DIR": d, "DRAFT_MODEL_FILE": "draft.gguf"})
            self.assertIn("--model-draft", out)
            self.assertIn("draft.gguf", out)

    def test_draft_model_absent_by_default(self):
        out = self._args()
        self.assertNotIn("--model-draft", out)

    def test_reasoning_format_opt_in(self):
        out = self._args(env={"REASONING_FORMAT": "deepseek"})
        self.assertIn("--reasoning-format", out)
        self.assertIn("deepseek", out)

    def test_reasoning_format_off_by_default(self):
        out = self._args()
        self.assertNotIn("--reasoning-format", out)


class EntrypointMissingModelTests(unittest.TestCase):
    def test_missing_model_file_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_bash(
                f'bash "{ENTRYPOINT}"',
                env={"MODELS_DIR": d, "MODEL_FILE": "does-not-exist.gguf"},
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("model file not found", r.stderr)


if __name__ == "__main__":
    unittest.main()
