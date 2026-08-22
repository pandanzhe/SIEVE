# Stage-1 SFT Input/Output Example

This note records one real Stage-1 SFT development example and the expected model
output after Stage-1. The example illustrates the SIEVE policy contract: the model
does not directly write low-support observations into executable state when the
downstream action is risky.

## Task

Stage-1 trains the revision policy to map a deployment-visible context to exactly
one structured cognitive action:

- `UPDATE`: commit supported task-relevant evidence into the executable belief state.
- `HOLD`: preserve the current executable value, mark the affected field as pending,
  and request verification when the observation is relevant but not sufficiently
  supported.
- `IGNORE`: discard observations that have no admissible effect on the task state.

The model output must be valid JSON only.

## System Prompt

```text
You are the observation evaluation policy of a multi-step agent.

Given the current executable belief state, task goal, new observation, risk constraints,
evidence ledger, and remaining budget, produce exactly one structured cognitive action.

Decision definitions:
- UPDATE: the observation applies to the current task and has sufficient support to modify the executable belief state.
- HOLD: the observation may affect the current task, but current evidence is insufficient for safe commitment.
- IGNORE: the observation has no admissible effect on the current task state and does not require verification.

Constraints:
1. UPDATE must contain local executable patches.
2. HOLD must preserve the current executable value. It may mark an affected field as pending and request verification.
3. IGNORE must contain no affected fields, patches, or verification request.
4. Do not modify fields outside affected_fields.
5. Return valid JSON only, without explanations.
```

## Example: HOLD With Verification

Source record:

```text
data/sft/clean/dev.jsonl
record_id=sieve-v3-001778
domain=retail
observation_type=tool_error_or_low_trust
reason_code=SOURCE_RELIABILITY
```

### Model Input

The user message is the deployment-visible context JSON:

```json
{
  "belief_state": {
    "max_slots": 8,
    "slots": [
      {
        "entity": "retail-task-44",
        "id": "first_name",
        "observed_at": 1700001852,
        "source": "initial_state",
        "status": "empty",
        "valid_from": 1700000074,
        "value": null
      }
    ]
  },
  "budget": {
    "steps_remaining": 8,
    "tokens_remaining": 4096,
    "tool_remaining": 5,
    "verification_remaining": 2
  },
  "goal": "You want to change the Desk Lamp in order #W9300146 that you've placed for the cheapest Desk Lamp that's available. Any price difference should go to a gift card. You also want to know how much you get back in total.",
  "ledger": {
    "capacity": 4,
    "entries": []
  },
  "observation": {
    "authenticated": false,
    "entity": "retail-task-44",
    "field_id": "first_name",
    "observed_at": 1700001862,
    "source": "forwarded_customer_message",
    "source_authority": "secondary",
    "valid_from": 1700001862,
    "value": "Aarav"
  },
  "risk": {
    "active_subgoal": "You want to change the Desk Lamp in order #W9300146 that you've placed for the cheapest Desk Lamp that's available. Any price difference should go to a gift card. You also want to know how much you get back in total.",
    "dependent_fields": [
      "first_name"
    ],
    "risk": "high",
    "reversible": false
  }
}
```

### Expected Model Output

```json
{"decision":"HOLD","affected_fields":["first_name"],"patches":[{"op":"SET_STATUS","field_id":"first_name","value":"pending_verification"}],"verification":{"tool":"verify_first_name","field_id":"first_name"}}
```

### Why This Is Correct

The new observation says `first_name = Aarav`, but the source is a forwarded
customer message with `authenticated=false` and `source_authority=secondary`.
The task is high risk and irreversible. Committing the value with `UPDATE` would
turn an insufficiently supported observation into executable state. The correct
Stage-1 behavior is therefore `HOLD`: keep the executable value unchanged, mark
the field as pending, and request `verify_first_name`.

## Contrast: UPDATE

When the observation is relevant, authenticated, and comes from a primary record
source, Stage-1 should commit it. A typical expected output is:

```json
{"decision":"UPDATE","affected_fields":["reservation_id"],"patches":[{"op":"SET_VALUE","field_id":"reservation_id","value":"XEHM4B"}],"verification":null}
```

This is the single-step policy learned by SFT. Stage-2 RL then learns the
multi-step consequences of these choices: wrong `UPDATE` can pollute state,
wrong `IGNORE` can lose required evidence, and unnecessary `HOLD` can waste
verification budget.
