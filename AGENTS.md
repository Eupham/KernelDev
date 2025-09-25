# Modal API Authentication Guide

Follow this checklist whenever you need to talk to the Modal API from this repository.

1. **Install the SDK**  \
   Install the official client tools so both the CLI and the Python helpers are available:
   ```bash
   pip install modal
   ```

2. **Load credentials from the environment**  \
   The runtime will inject three environment variables. Export them into your shell session before running any Modal commands:
   ```bash
   export MODAL_PROFILE="${MODAL_PROFILE:-<profile>}"
   export MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-<token_id>}"
   export MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-<token_secret>}"
   ```
   Replace the fallback placeholders (`<profile>`, `<token_id>`, `<token_secret>`) with the actual values from the provided secret source if they are not already present.

3. **Verify connectivity with the Modal App CLI**  \
   Use the `modal app` command group to ensure the SDK can authenticate and enumerate the apps in your account:
   ```bash
   python -m modal app list
   ```
   A successful call lists the accessible apps. If authentication fails, double-check that the three environment variables above are set in the current shell.
