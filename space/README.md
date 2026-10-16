---
title: TrustRUL engine lookup
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# TrustRUL engine lookup

Pick a simulated test engine and see the GRU's range, the calibrated baseline range and the sensor history.

This folder is a ready-to-deploy Hugging Face Space (Gradio). Function checked locally with the real data; the web UI itself has not been launched in a browser.

## Deploy (you do this, with your own Hugging Face account)

1. On huggingface.co create a new Space: SDK Gradio, free CPU hardware, public or private.
2. Clone the Space repository and copy everything from this `space/` folder into it (the files above the README front matter must stay at the top).
3. `git add . && git commit -m "first version" && git push`. The build takes a few minutes.
4. Put the Space link on the project page and in the portfolio card.

Run locally first: `pip install -r requirements.txt` then `python app.py`.
