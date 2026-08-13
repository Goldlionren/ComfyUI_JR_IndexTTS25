# Bundled IndexTTS source

This directory contains the runtime subset of the official IndexTTS source so
ComfyUI users do not need a second Git clone or a manual patch step.

- Upstream repository: `https://github.com/index-tts/index-tts`
- Upstream baseline: `a371df7d0746a0ae7fdf075798b6b04e34a0132e`
- Compatibility commit: `93684de2ee4efa05f64044597dde475db4b7fe6f`
- Sampling/normalization commit: `30fecfa188455a560aeea6f6dc60bc2f7c19bb14`
- Bundled revision: `30fecfa188455a560aeea6f6dc60bc2f7c19bb14`

Included here are the complete `indextts` Python runtime package, the Pinyin
reference vocabulary, the upstream README, disclaimer, and both language
versions of the upstream license. Repository metadata, CI, tests, WebUI,
TensorRT tooling, promotional media, caches, and model weights are excluded.

The compatibility changes cover Python 3.13, PyTorch/TorchAudio 2.11,
Transformers 4.56, SoundFile WAV output, ComfyUI progress callbacks, generation
parameter forwarding, and Arabic normalization behavior.

Any modifications made to the original model in this Derivative Work are not
endorsed, warranted, or guaranteed by the original right-holder of the original
model, and the original right-holder disclaims all liability related to this
Derivative Work.

Redistribution and use remain subject to `LICENSE` and `LICENSE_ZH.txt` in this
directory. The Chinese license text prevails if the two versions conflict.
