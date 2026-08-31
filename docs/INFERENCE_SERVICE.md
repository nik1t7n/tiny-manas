# Tiny Manas inference service

The public demo uses the accepted step-2,900 checkpoint. It does not retrain, switch models, or fall back to a remote language model.

## Boundary

The service exposes two routes:

- `GET /health` reports whether the accepted model is loaded;
- `POST /generate` produces a bounded continuation.

Generation requires the `X-Manas-Token` service credential. The browser never receives this credential. The website calls its own same-origin route, which applies a public rate limit and forwards an authenticated request over the private deployment path.

The request contract is deliberately small:

| Field | Range |
|---|---|
| Prompt | 1–300 characters and fewer than 256 tokenizer tokens |
| New tokens | 16–128 |
| Temperature | 0.5–1.2 |
| Top-k | 1–100 |
| Seed | optional integer from 0 to 2,147,483,647 |

Only one generation runs at a time. A concurrent request receives HTTP 429 rather than creating an unbounded queue. Unknown request fields are rejected. The service does not return token IDs or log prompt bodies.

## Startup integrity

Startup fails unless all of these conditions hold:

- checkpoint, tokenizer, and service credential paths are configured;
- checkpoint SHA-256 equals `cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1`;
- tokenizer SHA-256 equals `5047b4f427bb1af1c06cfb9cefbe83790b56df409b137b887988db6eba4b159f`;
- the configured device is exactly `cpu` or an available Apple `mps` device.

The model and tokenizer are loaded once through FastAPI's lifespan hook. The production container uses one Uvicorn worker so the 102.6 MiB model is not duplicated across processes.

## Required environment

```text
MANAS_CHECKPOINT_PATH=/models/tiny-manas-27m.pt
MANAS_TOKENIZER_PATH=/models/kyrgyz-byte-bpe-v1.json
MANAS_CHECKPOINT_SHA256=cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1
MANAS_API_TOKEN=<stored deployment secret>
MANAS_DEVICE=cpu
```

Model artifacts are mounted read-only at runtime. They are not copied into Git or into the Docker build context.

## Local real smoke

```bash
uv sync --extra serving

MANAS_CHECKPOINT_PATH=artifacts/tiny-manas-27m.pt \
MANAS_TOKENIZER_PATH=data/tokenizer/kyrgyz-byte-bpe-v1.json \
MANAS_API_TOKEN=local-only \
MANAS_DEVICE=cpu \
uv run uvicorn manas_gpt.service:app --host 127.0.0.1 --port 8765
```

The accepted local CPU smoke returned health 200, rejected an unauthenticated generation with 401, and generated 16 real tokens at about 228 tokens/s on the development Mac.

## Production record

The accepted service is deployed from commit
`43ed402cf506f1e263b17ebe99d7761b92474bed` as image
`manas-gpt:43ed402cf506f1e263b17ebe99d7761b92474bed`. Its local OVH image ID is
`sha256:0e429e7c7cd193053effcfe37ae7ca2e6972b49fdbccb3bed3ac7a86b888d962`.

The model runs as the private `manas` component inside the existing
`nik1t7n.com` Coolify service. It has no public domain or host port. The website
proxy is the only caller and authenticates with a deployment-scoped secret. The
checkpoint and tokenizer are mounted read-only from `/opt/manas-gpt/models`.

On `2026-08-31`, the production container became healthy with limits of 1 GiB
RAM and 2 CPUs. A real request through `https://nik1t7n.com/api/manas/generate`
returned 16 generated tokens at 32.51 tokens/s. The public article and reload
returned HTTP 200, and bounded service logs contained no errors.
