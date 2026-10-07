import json


def has_answer(output, context):
    try:
        return {"pass": "answer" in json.loads(output), "score": 1.0, "reason": "ok"}
    except ValueError:
        return {"pass": False, "score": 0.0, "reason": "not json"}
