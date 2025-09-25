# Modal API Authentication Guide

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
   sudo perl -0pi -e 's#^\s*(socks4|socks5|http)\s+.*#http    172.30.3.19 8080# if $.>0 && /\[ProxyList\]/ .. eof' /etc/proxychains4.conf
   ```

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
