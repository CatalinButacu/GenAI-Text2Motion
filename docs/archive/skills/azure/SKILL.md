---
name: azure
description: Azure for this thesis - subscription facts, quota reality, least-privilege identities, model registry, blue-green demo endpoints, key rotation, alerting, and auto-deallocating GPU training. Use for any az CLI work, cloud cost/quota question, or deployment design.
---

# Azure

## House style for az commands

The user is learning cloud. Every command must be legible on its own.

- **One command per step.** Never chain with `;` or `&&`. Never pipe a base64 blob.
- **Single scope per command.** One resource, one question.
- Always `--output table` unless JSON is genuinely needed.
- Say what a command will change *before* running it. Reads are free; writes are not.
- Prefer `--query` to trim output over dumping full JSON.

## Subscription facts (verified 2026-09-14)

| Fact | Value |
|---|---|
| Name | `Azure subscription 1 - DIZERTATIE` |
| Offer | `PayAsYouGo_2014-09-01` |
| Spending limit | **Off** - real charges accrue, nothing auto-stops billing |
| Total Regional vCPUs | **10** in every region checked |
| All GPU family quotas | **0** |
| `Standard_NC24ads_A100_v4` | Offered **unrestricted** in 16 regions |
| `Standard_NC*_T4_v3` | `NotAvailableForSubscription` - restricted, do not plan on T4 |

## Quota: what actually works

The A100 SKU is entitled to this subscription; only the quota is zero. The automated path fails:

```
az quota update --resource-name standardNCADSA100v4Family \
  --scope "/subscriptions/<id>/providers/Microsoft.Compute/locations/<region>" \
  --limit-object value=24 --resource-type dedicated
-> (QuotaNotAvailableForResource) Request failed.
```

This was reproduced in `germanywestcentral` and `swedencentral`. **`QuotaNotAvailableForResource`
means the automated grant is refused and a human-reviewed support ticket is required** - it does not
mean the SKU is unavailable. Check entitlement separately with `az vm list-skus`; an empty
`Restriction` column means entitled.

Two limits must BOTH rise before an A100 VM can start:
1. `StandardNCADSA100v4Family` 0 -> 24
2. `Total Regional vCPUs` 10 -> 24 (a 24-vCPU SKU alone exceeds the regional cap)

`Microsoft.Quota` must be registered before the quota API responds:
`az provider register --namespace Microsoft.Quota --wait`

## Identity model - least privilege, one identity per job

Never run training or inference as the user account. One managed identity per function, each with the
narrowest role on the narrowest scope:

- **trainer** - read the data container, write the checkpoint container, write its own run logs.
  No ability to create or delete compute, no read on other containers.
- **inference** - read-only on the model registry and the deployed model blob. Nothing else.
- **deployer** - the only identity that may create/update endpoints. Used by CI, not by a human.

Prefer user-assigned managed identities over service principals: no secret to rotate or leak. Use a
service principal only where a managed identity cannot reach (for example GitHub Actions without
federated credentials - and prefer federated credentials/OIDC there too).

Scope roles at the resource, never the subscription. `--scope` on the resource id, not `/subscriptions/<id>`.

## Traceability

Every resource carries tags so cost and provenance are attributable:
`--tags project=thesis stage=<train|serve> run=<run-id> owner=<identity>`

Any number quoted in the dissertation must trace to a run dir (`app/run_log.py`), and any cloud spend
must trace to a tag. Enable a budget alert before the first GPU hour, not after.

## Demo deployment - free tier only

This is a side project for a demo, not a release. Inference is CPU-only and must stay on free or
near-free SKUs.

- Azure ML **serverless/managed online endpoints bill per hour even when idle** - do not leave one
  running. Prefer a container on a free App Service plan (F1), or scale the endpoint to zero between
  demos.
- The model itself lives in the **Azure ML model registry**, versioned. Deployments reference a
  registry version, never a loose blob path.
- **Blue-green**: an endpoint holds two deployments (`blue`, `green`). Traffic starts 100/0, the new
  version is deployed to the idle colour, smoke-tested at 0% traffic, then shifted. Roll back by
  shifting traffic back - never by redeploying.

## Credentials and rotation

- Endpoint auth uses keys with two slots (primary/secondary). Rotate one slot at a time so the live
  demo never sees an auth gap: regenerate secondary -> switch clients to secondary -> regenerate primary.
- Store keys in Key Vault, never in the repo, never in a config committed to git.
- Prefer token (AAD) auth over keys where the client can obtain a token.

## Training runs must self-terminate

A GPU VM that outlives its job is the main cost risk on a PAYG subscription with no spending limit.

- The training script owns allocation AND deallocation, including on failure - trap the exit and
  deallocate in the handler, not only on the happy path.
- Sync artifacts (checkpoints, run dirs, gates) to Blob **during** the run, not at the end, so a crash
  or a guard kill still leaves usable state. `infra/azure/run_twins.sh` syncs every 10 minutes.
- Auto-shutdown on the VM is a backstop, not the mechanism: `az vm auto-shutdown`.
- `infra/azure/run_twins.sh` already enforces a max-hours budget and a heartbeat stall-kill per stage,
  and marks a stage `.done` only when it genuinely completed.

## Alerting

Wire alerts before the first paid run: build/job failure, budget threshold (50/80/100%), and VM still
running past the expected window. Action group -> email. A failed overnight run that nobody hears
about costs a whole night of credit.

## Gotchas hit in this project

- `az account show` does NOT return `subscriptionPolicies`. Use
  `az rest --method get --url ".../subscriptions/<id>?api-version=2022-12-01"` for the offer type.
- `az vm list-usage` shows only families with an entry; a family absent from the list is still 0.
- The `quota` extension is not installed by default: `az extension add --name quota`.
- PowerShell 5.1 has no `&&`. Use `;` between commands only when failure genuinely does not matter,
  otherwise separate steps.
