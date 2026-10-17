"""CI regression: Kaggle auth must survive the boundary between workflow steps."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.kaggle_auth import persist_verified_credentials_to_actions, verify_and_configure


class KaggleActionsEnvironmentTests(unittest.TestCase):
    def test_persists_the_verified_legacy_key_after_token_mode_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = root / "github_env"
            env.write_text("EXISTING=untouched\n")
            def fake_auth(args, *, env, **kwargs):
                success = env.get("KAGGLE_USERNAME") == "ffatty" and env.get("KAGGLE_KEY") == "legacy-test-value"
                return SimpleNamespace(returncode=0 if success else 1)
            with patch.dict(os.environ, {"GITHUB_ENV": str(env)}, clear=False), \
                 patch("scripts.kaggle_auth.subprocess.run", side_effect=fake_auth):
                mode = verify_and_configure("legacy-test-value", "ffatty", home=root / "home")
            self.assertEqual(mode, "legacy-key")
            self.assertEqual(env.read_text(),
                "EXISTING=untouched\nKAGGLE_USERNAME=ffatty\nKAGGLE_KEY=legacy-test-value\n")
            self.assertNotIn("KAGGLE_API_TOKEN", env.read_text())

    def test_persists_only_selected_access_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = root / "github_env"
            with patch.dict(os.environ, {"GITHUB_ENV": str(env)}, clear=False), \
                 patch("scripts.kaggle_auth.subprocess.run", return_value=SimpleNamespace(returncode=0)):
                mode = verify_and_configure("access-token-test-value", "ffatty", home=root / "home")
            self.assertEqual(mode, "access-token")
            self.assertEqual(env.read_text(), "KAGGLE_API_TOKEN=access-token-test-value\n")
            self.assertFalse((root / "home" / ".kaggle" / "kaggle.json").exists())

    def test_no_actions_env_file_outside_github_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GITHUB_ENV", None)
                with patch("scripts.kaggle_auth.subprocess.run", return_value=SimpleNamespace(returncode=0)):
                    verify_and_configure("token", "ffatty", home=root)
            self.assertFalse((root / "github_env").exists())

    def test_rejects_newline_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "env")
            with self.assertRaises(ValueError):
                persist_verified_credentials_to_actions({"KAGGLE_API_TOKEN": "x\nINJECTED=yes"}, path)
            with self.assertRaises(ValueError):
                persist_verified_credentials_to_actions({"HARMFUL_KEY": "x"}, path)


if __name__ == "__main__":
    unittest.main()
