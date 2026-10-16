# n8n workflow: engine decision webhook

`engine_decision_webhook.json` is an importable n8n workflow. A system posts a reading to it (`engine`, `cycle`, `pred_rul`, `lo`, `hi`) and gets back `PULL` or `CONTINUE` plus a human-review flag. The rule is the one the simulation found best: pull when the predicted remaining life is 8 cycles or less (0 failures in 100 simulated engines, about 8 cycles of life wasted).

**Status: not run in n8n.** The JavaScript was tested on sample inputs. The node settings were written from the node documentation and may need small fixes after import. The rule comes from simulated engines and is not for real maintenance decisions.

## Use it
1. Import the file in n8n (Workflows, Import from file) and activate it.
2. Test: `curl -X POST <your n8n test URL>/webhook-test/engine-decision -H "Content-Type: application/json" -d "{\"engine\":5,\"cycle\":250,\"pred_rul\":6,\"lo\":0,\"hi\":35}"`
3. Expected reply: action PULL, review auto.

Not built: sending the readings from a live sensor stream, or an AI explanation node (add your own Anthropic credential if you want one).