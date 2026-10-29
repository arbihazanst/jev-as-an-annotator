#!/usr/bin/env python3
"""Ask the actual Decider to route the supplied 401 example; never guess."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phishbench.decision_api import authenticated_endpoint,post_json,choice_answer
from phishbench.openwebui_decider import request_decision as webui_decision
from phishbench.data import atomic_json

CRITERIA={
    'technical':'API authentication and integration errors',
    'billing':'Payments, refunds, and invoices',
    'sales':'Product purchases and upgrades',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport',choices=['direct','openwebui'],default='direct')
    parser.add_argument('--endpoint-url',default=os.getenv('PPLX_SYSTEMONE_URL'))
    parser.add_argument('--model',default=os.getenv('PPLX_MODEL_ID'))
    parser.add_argument('--api-key-env')
    parser.add_argument('--webui-url',default='https://llms.si-lab.org')
    parser.add_argument('--chat-model',default='openai/gpt-oss-120b')
    parser.add_argument('--tool-id',default='server:pplxdeciderflash')
    parser.add_argument('--response-output',help='Save the synthetic test response for inspecting server tool evidence')
    parser.add_argument('--timeout',type=int,default=180)
    parser.add_argument('--request-attempts',type=int,default=3)
    args=parser.parse_args()
    args.api_key_env=args.api_key_env or ('OPENWEBUI_API_KEY' if args.transport=='openwebui' else 'PPLX_API_KEY')
    if not args.model and args.transport=='direct':parser.error('Set PPLX_MODEL_ID or --model to the exact server model ID')
    def save_response(result):
        if args.response_output:
            # Keys are not sent in the JSON payload; redact any accidental server echo as well.
            serialized=json.dumps(result)
            key=os.getenv(args.api_key_env,'')
            if key:serialized=serialized.replace(json.dumps(key)[1:-1],'[REDACTED]')
            atomic_json(args.response_output,json.loads(serialized))
    try:
        payload={
            'model':args.model,
            'state':'A customer reports HTTP 401 errors after rotating their API credentials.',
            'questions':{'department':{
                'type':'choice','instructions':'Which department should handle the issue?',
                'criteria':CRITERIA,
            }},
        }
        if args.transport=='openwebui':
            result,evidence=webui_decision(args.webui_url,args.chat_model,args.tool_id,payload,
                                          args.api_key_env,args.timeout,args.request_attempts,response_observer=save_response)
        else:
            url,headers=authenticated_endpoint(args.endpoint_url or '',args.api_key_env)
            result=post_json(url,payload,headers,args.timeout,args.request_attempts)
            save_response(result)
        selected,probability,confidence=choice_answer(result,'department',CRITERIA)
    except (RuntimeError,ValueError) as exc:
        print('Decider test failed: '+str(exc)+'; no department inferred.',file=sys.stderr)
        return 1
    print(json.dumps({
        'selected_option':selected,'probability':probability,'confidence':confidence,
        'explanation':'Decider selected the option defined as: '+CRITERIA[selected],
        'source':evidence['decision_tool_source'] if args.transport=='openwebui' else 'Direct structured systemone response',
    },indent=2))
    return 0


if __name__=='__main__':sys.exit(main())
