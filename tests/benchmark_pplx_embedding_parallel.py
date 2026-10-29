#!/usr/bin/env python3
"""Compare solo vs simultaneous PPLX and EmbeddingGemma 2 API performance.

No third-party Python modules required. Run on the GPU server or through SSH tunnels.
The PPLX route/model are explicit: confirm your service's native API contract.
"""
import argparse
import concurrent.futures
import csv
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

STATE = 'A customer reports HTTP 401 errors after rotating their API credentials.'
CRITERIA = {
    'technical': 'API authentication and integration errors',
    'billing': 'Payments, refunds, and invoices',
    'sales': 'Product purchases and upgrades',
}


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def request_json(url, payload, timeout, key=None):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def call(kind, phase, seq, args):
    start = time.perf_counter()
    result = {'phase': phase, 'service': kind, 'seq': seq, 'success': False,
              'latency_s': None, 'error': '', 'dimension': '', 'decision': ''}
    try:
        if kind == 'pplx':
            payload = {
                'model': args.pplx_model,
                'state': STATE,
                'questions': {'department': {
                    'type': 'choice',
                    'instructions': 'Which department should handle the issue?',
                    'criteria': CRITERIA,
                }},
            }
            data = request_json(args.pplx_url, payload, args.timeout, args.pplx_key)
            answer = data.get('answers', {}).get('department')
            if not isinstance(answer, dict):
                raise ValueError('Response missing answers.department object')
            # The choice result's exact field names can differ by implementation;
            # preserve it for inspection, but require an answer object.
            result['decision'] = json.dumps(answer, ensure_ascii=False)[:180]
            if data.get('error'):
                raise ValueError(f"PPLX reported error: {str(data['error'])[:180]}")
        else:
            payload = {'model': args.embed_model,
                       'input': f'Retrieval-augmented generation using Neo4j GraphRAG. Case {seq % 4}.'}
            data = request_json(args.embed_url, payload, args.timeout, args.embed_key)
            records = data.get('data', [])
            vector = records[0].get('embedding') if records and isinstance(records[0], dict) else None
            if not isinstance(vector, list) or not vector or not all(isinstance(x, (int, float)) for x in vector):
                raise ValueError('Missing/invalid numeric embedding vector')
            result['dimension'] = len(vector)
        result['success'] = True
    except Exception as exc:
        result['error'] = str(exc)[:240]
    finally:
        result['latency_s'] = round(time.perf_counter() - start, 5)
    return result


def phase_run(phase, counts, concurrency, args):
    jobs = [(kind, phase, i, args) for kind, n in counts.items() for i in range(n)]
    rng = random.Random(42)
    rng.shuffle(jobs)
    # Independent per-service quotas allow each service to reach requested concurrency.
    from threading import Semaphore
    sem = {'pplx': Semaphore(concurrency), 'embedding': Semaphore(concurrency)}

    def bounded(job):
        with sem[job[0]]:
            return call(*job)

    print(f'\nRunning {phase}: {counts} at concurrency {concurrency}/service...', flush=True)
    before = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency * len(counts))) as pool:
        futures = [pool.submit(bounded, j) for j in jobs]
        rows = [f.result() for f in concurrent.futures.as_completed(futures)]
    duration = time.perf_counter() - before
    print(f'Finished {phase} in {duration:.2f} s', flush=True)
    return rows, duration


def summarize(rows, duration, phase, kind):
    relevant = [r for r in rows if r['service'] == kind]
    ok = [r['latency_s'] for r in relevant if r['success']]
    failures = [r for r in relevant if not r['success']]
    data = {'phase': phase, 'service': kind, 'requests': len(relevant), 'success': len(ok),
            'failed': len(failures), 'success_pct': round(100 * len(ok) / len(relevant), 1) if relevant else 0,
            'p50_s': round(percentile(ok, 50), 3) if ok else None,
            'p95_s': round(percentile(ok, 95), 3) if ok else None,
            'mean_s': round(statistics.mean(ok), 3) if ok else None,
            'throughput_success_rps': round(len(ok) / duration, 3) if duration else 0,
            'first_error': failures[0]['error'] if failures else ''}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pplx-url', default='http://127.0.0.1:30002/v1/systemone')
    parser.add_argument('--pplx-model', required=True, help='Exact model ID expected by your PPLX direct API')
    parser.add_argument('--pplx-key', help='Optional API key (avoid if command history is shared)')
    parser.add_argument('--embed-url', default='http://127.0.0.1:30003/v1/embeddings')
    parser.add_argument('--embed-model', default='google/embeddinggemma-2')
    parser.add_argument('--embed-key')
    parser.add_argument('--requests', type=int, default=20, help='Requests per service in each phase')
    parser.add_argument('--concurrency', type=int, default=4, help='Concurrent requests per service')
    parser.add_argument('--timeout', type=int, default=90)
    parser.add_argument('--output-prefix', default='parallel_inference_benchmark')
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error('--requests and --concurrency must be positive')

    all_rows = []
    metrics = []
    for phase, counts in (
        ('pplx_alone', {'pplx': args.requests}),
        ('embedding_alone', {'embedding': args.requests}),
        ('both_together', {'pplx': args.requests, 'embedding': args.requests}),
    ):
        rows, duration = phase_run(phase, counts, args.concurrency, args)
        all_rows.extend(rows)
        for kind in counts:
            metric = summarize(rows, duration, phase, kind)
            metrics.append(metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)

    out = Path(args.output_prefix)
    with out.with_suffix('.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with out.with_suffix('.json').open('w', encoding='utf-8') as fp:
        json.dump({'config': {'requests': args.requests, 'concurrency': args.concurrency,
                              'timeout': args.timeout, 'pplx_model': args.pplx_model,
                              'embed_model': args.embed_model},
                   'summary': metrics}, fp, indent=2)
    print(f'\nSaved {out.with_suffix(".csv")} and {out.with_suffix(".json")}')
    if any(m['failed'] for m in metrics):
        print('WARNING: Some requests failed. Review first_error and CSV before interpreting throughput.')


if __name__ == '__main__':
    main()
