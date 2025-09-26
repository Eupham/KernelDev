# Automated Modal Training Launcher

This guide describes how to use the automated launcher script to run the training process on Modal. The script handles all necessary setup, authentication, and execution steps.

## Automated Workflow using `launcher.py`

The `launcher.py` script is the primary entry point for running the training job. It automates the entire process, including:
- Installing local dependencies (`modal`, `proxychains4`).
- Configuring the proxy to connect to the Modal API.
- Trusting the necessary TLS certificates.
- Temporarily modifying `config.yaml` to set the number of training samples to 1000.
- Launching a job on a Modal H100 GPU instance.
- Setting up the remote environment (cloning the repository, installing dependencies).
- Running the training script (`entry.py`) inside the Modal environment.
- Restoring the original `config.yaml` file after the job completes.

### Prerequisites

Before running the launcher, ensure you have set the following environment variables with your Modal credentials:

```bash
export MODAL_TOKEN_ID="<your_token_id>"
export MODAL_TOKEN_SECRET="<your_token_secret>"
```

### How to Run

To start the entire process, simply execute the `launcher.py` script:

```bash
python launcher.py
```

The script will provide real-time output from both the local setup and the remote Modal job.

---
*For reference, the original manual setup instructions are kept below, but they are no longer the recommended workflow.*
---

# Original Manual Modal API Authentication Guide

Use this checklist whenever you need to reach the Modal API from this repository. The environment sits behind an HTTP proxy that performs TLS inspection, so a few extra steps are required beyond the standard SDK setup.

1. **Install the required tooling**  \
   Install the Modal SDK and the proxy helper used for tunnelling gRPC traffic through the corporate proxy:
   ```bash
   pip install modal
   sudo apt-get update
   sudo apt-get install -y proxychains4
   ```

2. **Trust the egress proxy certificate**  \
   Modal's Python client pins to the `certifi` CA bundle. Append the managed MITM certificate to that bundle once per environment:
   ```bash
   python - <<'PY'
   import certifi
   from pathlib import Path

   bundle = Path(certifi.where())
   mitm = Path("/usr/local/share/ca-certificates/envoy-mitmproxy-ca-cert.crt").read_text()
   text = bundle.read_text()
   if "egress-proxy" not in text:
       bundle.write_text(text.rstrip() + "\n" + mitm + "\n")
   PY
   ```

3. **Configure proxychains4 to reach Modal**  \
   Enable the dynamic chain mode and point proxychains4 at the HTTP proxy. The following commands are idempotent and can be re-run safely:
   ```bash
   sudo perl -0pi -e 's/#?dynamic_chain/dynamic_chain/' /etc/proxychains4.conf
   sudo perl -0pi -e 's/^strict_chain/#strict_chain/' /etc/proxychains4.conf
   sudo perl -0pi -e 's#^\s*(socks4|socks5|http)\s+.*#http    172.30.2.211 8080# if $.>0 && /\[ProxyList\]/ .. eof' /etc/proxychains4.conf
   ```
   After running the commands, the first non-comment entry under `[ProxyList]` should read `http    172.30.2.211 8080`.

4. **Load the Modal credentials**  \
   Export the credentials that are injected as environment variables into the current shell session:
   ```bash
   export MODAL_PROFILE="${MODAL_PROFILE:-<profile>}"
   export MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-<token_id>}"
   export MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-<token_secret>}"
   ```
   Replace the fallback placeholders (`<profile>`, `<token_id>`, `<token_secret>`) with the actual values from the provided secret source if they are not already present.

5. **Verify connectivity with the Modal App CLI**  \
   Always run Modal CLI commands through proxychains so the traffic traverses the proxy with the trusted certificate:
   ```bash
   proxychains4 python -m modal app list
   ```
   A successful call lists the accessible apps. If authentication fails, double-check that the three environment variables above are set in the current shell and that the proxychains configuration contains the HTTP proxy entry.

6. **Run future Modal commands through proxychains**  \
   Prefix subsequent invocations with `proxychains4` as well, for example `proxychains4 modal run ...` or `proxychains4 python script.py` when the script uses the Modal SDK.