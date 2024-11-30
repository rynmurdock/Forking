# Image to Video Prior

Just an ip adapter on LTX. We assume it'd generalize to longer videos.

Don't need to load T5; just image embedding into the adapter, then output that vae-encoded image.
