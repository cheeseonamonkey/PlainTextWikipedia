"""Offline tests for the reusable publishing workflow and preflight handling."""

import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts.publish_preflight import check
from scripts.structured_fleet import retry, transient_error


ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class ReusableWorkflowTests(unittest.TestCase):
    def test_fleet_queues_same_canonical_single_language_workflow(self):
        fleet = yaml.load((ROOT / "structured-fleet.yml").read_text(), Loader=yaml.BaseLoader)
        single = yaml.load((ROOT / "structured-wikipedia.yml").read_text(),
                           Loader=yaml.BaseLoader)
        controller = fleet["jobs"]
        self.assertEqual(controller["publish"]["uses"],
                         "./.github/workflows/structured-wikipedia.yml")
        self.assertEqual(controller["publish"]["secrets"], "inherit")
        self.assertEqual(controller["publish"]["strategy"]["fail-fast"], "false")
        self.assertEqual(controller["publish"]["strategy"]["max-parallel"],
                         "${{ inputs.parallelism }}")
        self.assertEqual(controller["publish"]["with"]["skip_preflight"], "true")
        self.assertEqual(set(single["on"]), {"workflow_call", "workflow_dispatch"})
        self.assertEqual(single["on"]["workflow_call"]["inputs"]["repo_id"]["required"], "true")
        self.assertEqual(single["on"]["workflow_call"]["inputs"]["skip_preflight"]["default"],
                         "false")
        self.assertEqual(controller["publish"]["needs"], ["plan", "credentials"])
        self.assertEqual(controller["publish"]["if"],
                         "${{ needs.plan.result == 'success' && needs.credentials.result == 'success' }}")
        for name in ("structured-fleet.yml", "structured-wikipedia.yml"):
            self.assertNotIn("upload-artifact", (ROOT / name).read_text())

    def test_all_three_publishers_remain_enabled_by_default(self):
        for name in ("structured-fleet.yml", "structured-wikipedia.yml"):
            workflow = yaml.load((ROOT / name).read_text(), Loader=yaml.BaseLoader)
            form = workflow["on"]["workflow_dispatch"]["inputs"]
            self.assertEqual(form["mirror_kaggle"]["default"], "true")
            self.assertEqual(form["publish_zenodo"]["default"], "true")


class PreflightTests(unittest.TestCase):
    def test_optional_publishers_are_never_touched_when_disabled(self):
        with (patch.dict(os.environ, {"HF_TOKEN": "test"}, clear=True),
              patch("scripts.publish_preflight.HfApi") as hub,
              patch("scripts.publish_preflight.subprocess.run") as command,
              patch("requests.get") as get):
            check(kaggle=False, zenodo=False)
            hub.return_value.whoami.assert_called_once_with()
            command.assert_not_called()
            get.assert_not_called()

    def test_enabled_publishers_are_read_only(self):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: [])
        with (patch.dict(os.environ, {"HF_TOKEN": "test", "KAGGLE_TOKEN": "test",
                                      "ZENODO_TOKEN": "test"}, clear=True),
              patch("scripts.publish_preflight.HfApi"),
              patch("scripts.publish_preflight.subprocess.run") as command,
              patch("requests.get", return_value=response) as get):
            check()
            command.assert_called_once()
            self.assertTrue(command.call_args.kwargs["check"])
            self.assertEqual(get.call_count, 1)

    def test_missing_required_tokens_fail_before_any_upload(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                check()


class SelectiveRetriesTests(unittest.TestCase):
    def test_permanent_urllib_http_failures_never_retry(self):
        def problem(code):
            return urllib.error.HTTPError(
                "https://example.com", code, "error", {}, None)
        self.assertFalse(transient_error(problem(401)))
        self.assertFalse(transient_error(problem(403)))
        self.assertFalse(transient_error(problem(404)))
        self.assertTrue(transient_error(problem(429)))
        self.assertTrue(transient_error(problem(503)))
        calls = []
        with patch("scripts.structured_fleet.time.sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                retry("permanent 404", lambda: (calls.append(1), (_ for _ in ()).throw(
                    problem(404)))[1])
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_transient_errors_retry_with_bounded_backoff(self):
        attempts = []
        def callback():
            attempts.append(1)
            if len(attempts) < 3:
                raise TimeoutError("transient network")
            return "ok"
        with (patch("scripts.structured_fleet.time.sleep") as sleep,
              patch("scripts.structured_fleet.random.random", return_value=0)):
            self.assertEqual(retry("network", callback), "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [8, 16])


if __name__ == "__main__":
    unittest.main()
