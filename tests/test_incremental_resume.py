import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from evaluate_decision_models import (
    EvalItem, _check_resume_compat, parse_args, run_evaluation, validate_cached_inputs,
)
from si_lab_decision_client import SiLabDecisionError, SiLabOllamaDecisionClient
from jev_only.run_jev_full import parse_args as jev_args, run_full


class IncrementalResumeTests(unittest.TestCase):
    def test_jev_completed_resume_and_summary_need_no_credentials(self):
        for mode in ('--resume', '--summarize-only'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'predictions.jsonl'
                path.write_text(json.dumps({
                    'model': 'typesafe/jev-1.13', 'dataset': 'sentiment', 'id': '1',
                    'status': 'ok', 'text_sha256': hashlib.sha256(b'original').hexdigest(),
                    'gold_label': 'positive',
                }) + '\n')
                args = jev_args([mode, '--datasets', 'sentiment', '--output', directory])
                with patch('jev_only.run_jev_full.load_dataset', return_value=(
                    [EvalItem('sentiment', '1', 'original', 'positive')], {'sha256': 'test'},
                )), patch('jev_only.run_jev_full.get_api_key') as credential, \
                        patch('jev_only.run_jev_full.summarize_predictions') as summarize:
                    self.assertEqual(asyncio.run(run_full(args)), 0)
                    credential.assert_not_called()
                    summarize.assert_called_once()

    def test_added_rows_are_accepted_but_changed_or_removed_cached_rows_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            path.write_text(json.dumps({
                'model': 'm', 'dataset': 'sentiment', 'id': '1', 'status': 'ok',
                'text_sha256': hashlib.sha256(b'original').hexdigest(),
                'gold_label': 'positive',
            }) + '\n', encoding='utf-8')
            original = EvalItem('sentiment', '1', 'original', 'positive')
            added = EvalItem('sentiment', '2', 'new', 'negative')
            validated = validate_cached_inputs(path, {'sentiment': [original, added]}, ['m'])
            old = {'dataset_sha256': {'sentiment': 'old'}}
            new = {'dataset_sha256': {'sentiment': 'new'}}
            with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                _check_resume_compat(old, new)
            _check_resume_compat(old, new, validated_datasets=validated)
            for items in ([added], [EvalItem('sentiment', '1', 'edited', 'positive')],
                          [EvalItem('sentiment', '1', 'original', 'negative')]):
                with self.assertRaisesRegex(RuntimeError, 'Cannot reuse'):
                    validate_cached_inputs(path, {'sentiment': items}, ['m'])

    def test_completed_si_lab_resume_does_not_contact_server(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            path.write_text(json.dumps({'model': 'm', 'dataset': 'sentiment',
                                        'id': '1', 'status': 'ok'}) + '\n')
            args = parse_args(['--resume', '--models', 'm', '--datasets', 'sentiment'])
            with patch('evaluate_decision_models.preflight', new_callable=AsyncMock) as probe:
                result = asyncio.run(run_evaluation(
                    args=args, items_by_dataset={'sentiment': [EvalItem('sentiment', '1', 'a', 'positive')]},
                    specs={}, predictions_path=path, errors_path=path.parent / 'errors.jsonl',
                ))
                probe.assert_not_awaited()
                self.assertEqual(result[:2], (0, 1))


class ModelDiscoveryTests(unittest.TestCase):
    def test_empty_primary_list_falls_back_for_sync_and_async(self):
        def handler(request):
            if request.url.path == '/api/tags':
                return httpx.Response(200, json={'models': [{'name': 'nimble:9b'}]})
            return httpx.Response(200, json={'data': []})
        transport = httpx.MockTransport(handler)
        client = SiLabOllamaDecisionClient('http://localhost:11435/api', 'nimble:9b',
                                         http_transport=transport, async_http_transport=transport)
        try:
            self.assertEqual(client.list_models(), ['nimble:9b'])
            self.assertEqual(asyncio.run(client.alist_models()), ['nimble:9b'])
        finally:
            asyncio.run(client.aclose())

    def test_connection_error_is_not_an_empty_model_list(self):
        def handler(request):
            raise httpx.ConnectError('refused', request=request)
        transport = httpx.MockTransport(handler)
        client = SiLabOllamaDecisionClient('http://localhost:11435', 'nimble:9b',
                                         http_transport=transport, async_http_transport=transport)
        try:
            with self.assertRaisesRegex(SiLabDecisionError, 'server/tunnel.*ConnectError'):
                client.list_models()
            with self.assertRaisesRegex(SiLabDecisionError, 'server/tunnel.*ConnectError'):
                asyncio.run(client.alist_models())
        finally:
            asyncio.run(client.aclose())


if __name__ == '__main__':
    unittest.main()
