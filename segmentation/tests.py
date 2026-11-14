import os
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

from . import hf_service


@override_settings(HF_API_URL="https://example.test")
class ClientReuseTests(SimpleTestCase):
    def setUp(self):
        self.client_patch = patch.object(hf_service, "Client")
        self.factory = self.client_patch.start()
        self.token_patch = patch.dict(os.environ, {"HF_TOKEN": "test-placeholder"})
        self.token_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.addCleanup(self.token_patch.stop)
        self.addCleanup(hf_service.discard_client)

    def test_reuses_client_with_new_session_and_output_directory(self):
        first = hf_service.get_client("first-output", "first")
        second = hf_service.get_client("second-output", "second")
        self.assertIs(first, second)
        self.factory.assert_called_once()
        first.reset_session.assert_called_once()
        self.assertEqual(second.output_dir, "second-output")
        self.assertEqual(second.download_files, "second-output")

    def test_changed_endpoint_replaces_client_and_closes_resources(self):
        first, second = Mock(), Mock()
        self.factory.side_effect = [first, second]
        hf_service.get_client("first-output", "first")
        with override_settings(HF_API_URL="https://other.test"):
            self.assertIs(hf_service.get_client("second-output", "second"), second)
        first.close.assert_called_once()
        first.executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_discarded_client_is_recreated(self):
        hf_service.get_client("first-output", "first")
        hf_service.discard_client()
        self.assertIsNone(hf_service._cached_client)
        hf_service.get_client("second-output", "second")
        self.assertEqual(self.factory.call_count, 2)
