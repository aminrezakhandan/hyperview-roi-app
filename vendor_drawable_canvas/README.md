# vendor_drawable_canvas

This is a vendored, patched copy of
[`streamlit-drawable-canvas-fix`](https://pypi.org/project/streamlit-drawable-canvas-fix/)
0.9.8, itself a community fork of the original (and unmaintained)
`streamlit-drawable-canvas`.

## Why vendored instead of pip-installed

The compiled frontend (`frontend/build/static/js/main.5369d588.chunk.js`)
builds the background-image URL as:

```js
e.src = n + h
```

where `n` is the *origin* (scheme + host only, no path) parsed from the
`streamlitUrl` query parameter, and `h` is the background-image URL passed
from Python. This unconditionally prepends the page origin to whatever URL
Python provides.

- On Streamlit Community Cloud, the app is served behind a proxy path
  prefix (e.g. `/~/+/`) that isn't part of `origin`, so a root-relative
  MediaFileManager path (`/media/<hash>.png`) resolves to the wrong route
  and the canvas shows no background image.
- `app.py` works around the MediaFileManager/proxy path entirely by
  embedding the background image as a `data:image/png;base64,...` URL
  (see `_background_image_to_data_url` in `app.py`). But the origin-prefix
  line above corrupts a data URL the same way (`https://hostdata:image/...`
  is not a valid URL).

The one-line patch changes that to:

```js
e.src = h
```

so whatever URL Python supplies (in our case, a self-contained data URL)
is used directly, with no origin prefix. That's the only change from the
upstream package.

## Regenerating this vendored copy

If upgrading `streamlit-drawable-canvas-fix` to a newer release, re-apply
the same one-line patch to the new build's JS bundle:

```bash
pip install --target /tmp/sdc streamlit-drawable-canvas-fix==<new-version>
grep -rl 'e.src=n+h' /tmp/sdc/streamlit_drawable_canvas/frontend/build/static/js/*.js
# then replace that exact string with 'e.src=h' in the matched file(s)
```
