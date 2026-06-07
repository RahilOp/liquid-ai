# TTS — Models & Datasets for Synthetic Code-Switch Audio

*Tooling to generate synthetic JP↔EN code-switch audio for fine-tuning. Priority: a TTS whose **license permits commercial use of generated audio** and that can render **JP + EN in one utterance** with natural prosody (no splicing). Target 24 kHz. Verified against model/dataset cards (2026-06-06).*

> **TL;DR.** No permissive open TTS does perfect high-quality natively-code-switched JP↔EN in one utterance today. Closest **commercial-safe native-CS** option = **CosyVoice2-0.5B (Apache-2.0)** via zero-shot cross-lingual cloning. For **bulk monolingual-span volume**, **Kokoro-82M (Apache-2.0, 24 kHz, JP+EN)** and **MeloTTS (MIT)** are clean splice engines. **edge-tts** (our current default) is a **legal liability for commercial synthetic data** (Microsoft service ToS, not the LGPL wrapper) → demote. **XTTS-v2, F5-TTS, fish-speech are non-commercial — do NOT use their outputs as training data.**

---

## 1. Recommended stack

| Goal | Recommendation | Why |
|---|---|---|
| **Primary native-CS engine (JP+EN one utterance)** | **CosyVoice2-0.5B** (`FunAudioLLM/CosyVoice2-0.5B`), **Apache-2.0** | Only permissively-licensed model doing **multilingual/cross-lingual zero-shot** incl. **Japanese**; one reference voice synthesizes mixed-language sentences. Outputs commercial-safe. GPU recommended. |
| **Bulk splice engine (JP quality + speed)** | **Kokoro-82M** (`hexgrad/Kokoro-82M`), **Apache-2.0**, **24 kHz** | Tiny (82M), fast CPU/GPU, native 24 kHz (no resample), 5 JP + ~28 EN voices. One language per `KPipeline` → splice per-language spans. Cleanest license. |
| **Bulk splice alt (EN accents)** | **MeloTTS** (`myshell-ai/MeloTTS-*`), **MIT** | "free for commercial and non-commercial use." JP + 5 EN accents, CPU-friendly. JP model needs English pre-converted to katakana → splice engine. |
| **Speaker diversity** | Kokoro (33 voices) + MeloTTS (6 accents) + Piper (MIT) | rotate voices/speakers for acoustic diversity |
| **Demote** | **edge-tts** | prototyping only — not for shippable/commercial synthetic data (§5) |

**Pipeline:** **CosyVoice2** for genuinely code-switched lines (boundary prosody matters — the realistic meeting case); **Kokoro/MeloTTS splicing** for high-throughput bulk. Both legs Apache/MIT → LFM-Open-compatible.

---

## 2. Models comparison

| Model | License — commercial synth OK? | Languages | JP+EN in **one** utterance? | On-device? | HF ID |
|---|---|---|---|---|---|
| **Kokoro-82M** | **Apache-2.0 → YES** | en, ja, zh, es, fr, hi, it, pt | **No** (one lang/pipeline; splice) | **Yes** — 82M, CPU/GPU, 24 kHz | `hexgrad/Kokoro-82M` |
| **MeloTTS** | **MIT → YES** | en(5 accents), ja, zh, ko, es, fr | **No** for JP+EN (only ZH mixes inline) | **Yes** — CPU-friendly | `myshell-ai/MeloTTS-Japanese`, `-English` |
| **CosyVoice2-0.5B** | **Apache-2.0 → YES** | zh, en, **ja**, ko, de, es, fr, it, ru | **Yes** — cross-lingual zero-shot | Yes — ~0.5B, **GPU rec.** | `FunAudioLLM/CosyVoice2-0.5B` |
| **edge-tts** | Code LGPL, **but MS service ToS ≈ NO** for commercial output w/o Azure billing | many incl. ja, en | **No** (splice) | Cloud call (needs internet) | PyPI `edge-tts` |
| **Style-Bert-VITS2** | **AGPL-3.0** code; model terms vary | JP-strong; JP-Extra = **JP only** | **No** in JP-Extra | Yes | `litagin02/Style-Bert-VITS2` |
| **XTTS-v2 / Coqui** | **CPML → NON-COMMERCIAL. EXCLUDE** | 17 incl. ja, en | Yes but license blocks | Yes | `coqui/XTTS-v2` |
| **Parler-TTS** | **Apache-2.0 → YES** | **EN only** (no JP) | No (no JP) | Yes | `parler-tts/parler-tts-mini-v1` |
| **F5-TTS** | **CC-BY-NC-4.0 + Emilia → NON-COMMERCIAL. EXCLUDE** | en, zh (+community ja) | cross-lingual but blocked | Yes (GPU) | `SWivid/F5-TTS` |
| **fish-speech 1.5 / OpenAudio-S1-mini** | **CC-BY-NC-SA-4.0 → EXCLUDE** | 13 incl. ja, en | yes but blocked | Yes (GPU) | `fishaudio/fish-speech-1.5` |
| **OpenVoice v2** | **MIT → YES** | en, zh (+ tone transfer over MeloTTS) | **No** — voice-cloner, not CS synth | Yes | `myshell-ai/OpenVoiceV2` |
| **Piper** | **MIT → YES** | 30+ — **no official JP** (use `piper-plus` bilingual-ja-en) | mainline no JP | **Yes** — CPU, ONNX, light | `rhasspy/piper-voices` |

---

## 3. Datasets comparison (TTS corpora)

| Dataset | Lang | Hours / Speakers | License | HF ID | Notes |
|---|---|---|---|---|---|
| **JSUT** | JA | ~10h / 1 | Labels CC-BY-SA-4.0; **AUDIO personal/research, commercial via contact. FLAG** | sarulab site | Studio single-speaker; not redistributable. |
| **JVS** | JA | ~30h / **100** | Labels CC-BY-SA-4.0; AUDIO personal/research. **FLAG** | sarulab site | 100-speaker variety. |
| **JVNV** | JA | ~4h / 4 | CC-BY-SA-4.0 family (verify) | `asahi417/jvnv-emotional-speech-corpus` | Emotional + nonverbal. |
| **Common Voice (ja)** | JA | varies | **CC0** ✅✅ | `mozilla-foundation/common_voice_17_0` (`ja`) | Fully commercial-safe; variable quality. |
| **LJSpeech** | EN | ~24h / 1 | **Public domain** ✅✅ | `keithito/lj_speech` | Classic EN baseline (22.05 kHz). |
| **Jenny** (kit example) | EN | ~30h / 1 | **Permissive but no-resale-of-data** — verify | `reach-vb/jenny_tts_dataset` | 48 kHz; the kit's example dataset. |
| **VCTK** | EN | ~44h / **110** | **CC-BY-4.0** ✅ | `CSTR-Edinburgh/vctk` | Multi-accent variety. |
| **LibriTTS-R** | EN | ~585h / 2,456 | **CC-BY-4.0** ✅ | `mythicinfinity/libritts_r` | **24 kHz**, restored, big multi-speaker. |
| **Expresso** | EN | 40h / 4 | **CC-BY-NC-4.0 → FLAG** | `ylacombe/expresso` | Expressive; NC. |
| **Kokoro-Speech-Dataset** | JA | single-speaker | **Public domain** ✅✅ | GH `kaiidams/Kokoro-Speech-Dataset` | JP PD (unrelated to Kokoro-82M model). |

**Bilingual JP-EN single-speaker corpus:** none open + clean. Combine a JP corpus (CV CC0 / JSUT w/ care) + EN (LJSpeech PD / LibriTTS-R CC-BY) per role, or **generate** with CosyVoice2.

---

## 4. Minimal code (top 3 engines)

### CosyVoice2 — native JP↔EN code-switch (Apache-2.0)
```python
# git clone https://github.com/FunAudioLLM/CosyVoice && pip install -r requirements.txt
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
cv = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
prompt = load_wav('ref_speaker.wav', 16000)                       # 3-10s reference
text = "明日のmeetingはconference roomで、aboutは予算のreviewです。"  # JP+EN in one utterance
for i, out in enumerate(cv.inference_cross_lingual(text, prompt, stream=False)):
    torchaudio.save(f'cs_{i}.wav', out['tts_speech'], cv.sample_rate)
```

### Kokoro-82M — fast 24 kHz splice (Apache-2.0)
```python
# pip install kokoro>=0.9.2 "misaki[ja]" soundfile
import numpy as np, soundfile as sf
from kokoro import KPipeline
jp = KPipeline(lang_code='j'); en = KPipeline(lang_code='a')      # one language per pipeline
spans = [(jp,"明日のミーティングは",'jf_alpha'), (en,"conference room",'af_heart'), (jp,"です",'jf_alpha')]
audio = np.concatenate([seg.audio for p,t,v in spans for seg in p(t, voice=v)])
sf.write('cs_spliced.wav', audio, 24000)                         # already 24 kHz
```
- JP voices grade ~C (adequate for ASR training data, not premium TTS).

### MeloTTS — MIT splice, CPU-friendly
```python
# pip install git+https://github.com/myshell-ai/MeloTTS.git && python -m unidic download
from melo.api import TTS
jp = TTS(language='JP', device='cpu'); jp.tts_to_file("会議は10時から", jp.hps.data.spk2id['JP'], 'jp.wav')
en = TTS(language='EN', device='cpu'); en.tts_to_file("in the main hall", en.hps.data.spk2id['EN-US'], 'en.wav')
# concatenate + resample to 24 kHz
```

---

## 5. Licensing posture

**🟢 Commercial-safe outputs:** Kokoro (Apache), MeloTTS (MIT), CosyVoice2 (Apache), Parler (Apache, EN-only), OpenVoice v2 (MIT, cloner), Piper (MIT, no official JP).

**🔴 NOT commercial-safe — exclude outputs:** XTTS-v2/Coqui (CPML NC), F5-TTS (CC-BY-NC + Emilia, "can't be used commercially even after finetuning"), fish-speech/OpenAudio (CC-BY-NC-SA). **edge-tts** — wrapper LGPL but **Microsoft Edge read-aloud service ToS** governs the audio; MS licenses commercial TTS only via a **billed Azure Speech resource**; maintainer treats it as personal-use → **treat output as not commercial-clean.**

**⚠️ Verify per-asset:** Style-Bert-VITS2 / Bert-VITS2 — code AGPL-3.0 (copyleft on code, not synthetic audio); pretrained-model terms vary per uploader.

**Dataset licenses:** CC0 (Common Voice) + PD (LJSpeech, Kokoro-Speech-Dataset) fully open; CC-BY (VCTK, LibriTTS-R) attribution; CC-BY-NC (Expresso) + JSUT/JVS audio (personal/research; commercial via contact) flagged.

---

## 6. How to use in our pipeline

1. **Native CS (one utterance) → CosyVoice2** `inference_cross_lingual` (only commercial-safe native-CS; use where boundary prosody matters).
2. **Splice (per-language spans) → Kokoro (24 kHz) or MeloTTS** — drop-in replacement for our edge-tts splice path, license-clean + local; crossfade-concatenate (5–15 ms fade to hide seams).
3. **Bulk vs JP quality:** Kokoro/MeloTTS for high-count monolingual-span majority; CosyVoice2 for best JP+CS prosody. (Premium JP = Style-Bert-VITS2 JP-Extra, but JP-only → still splice.)
4. **Speaker diversity:** rotate Kokoro 33 voices + MeloTTS 6 accents + Piper + CosyVoice2 reference clips (varied WAVs); randomize speed 0.9–1.1.
5. **Sample rate:** Kokoro + CosyVoice2 emit 24 kHz → no resample (least artifacts); resample MeloTTS/Piper/Jenny(48k)/LJSpeech(22.05k) to 24 kHz; keep single 24 kHz mono pipeline.
6. **Recipe:** LLM emits language-tagged CS transcript → high-CS-density lines → CosyVoice2 native; bulk lines → Kokoro/MeloTTS splice → loudness-normalize (−23 LUFS) → 24 kHz mono → optional light reverb/noise for meeting realism → pair with transcript. All Apache/MIT → outputs LFM-Open-clean.

> **Action for our repo:** add `KokoroBackend` + `CosyVoice2Backend` to `src/csmeeting/tts_backends.py`; keep `EdgeTTSBackend` for offline prototyping only.

---

## 7. Sources & uncertainties

**Sources:** Kokoro https://huggingface.co/hexgrad/Kokoro-82M (VOICES.md) , https://github.com/hexgrad/kokoro · MeloTTS https://github.com/myshell-ai/MeloTTS · CosyVoice2 https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B , https://github.com/FunAudioLLM/CosyVoice , arXiv:2412.10117 · edge-tts ToS https://github.com/rany2/edge-tts/discussions/261 , https://learn.microsoft.com/en-us/answers/questions/2088770/ · Style-Bert-VITS2 https://github.com/litagin02/Style-Bert-VITS2 , arXiv:2505.17320 · XTTS https://huggingface.co/coqui/XTTS-v2 · Parler https://huggingface.co/parler-tts/parler-tts-mini-v1 · F5-TTS https://huggingface.co/SWivid/F5-TTS · fish-speech https://huggingface.co/fishaudio/fish-speech-1.5 · OpenVoice https://huggingface.co/myshell-ai/OpenVoiceV2 · Piper https://huggingface.co/rhasspy/piper-voices , https://github.com/ayutaz/piper-plus · datasets: JSUT/JVS sarulab site · JVNV https://huggingface.co/datasets/asahi417/jvnv-emotional-speech-corpus · LJSpeech https://huggingface.co/datasets/keithito/lj_speech · Jenny https://huggingface.co/datasets/reach-vb/jenny_tts_dataset · VCTK https://huggingface.co/datasets/CSTR-Edinburgh/vctk · LibriTTS-R https://huggingface.co/datasets/mythicinfinity/libritts_r · Expresso https://huggingface.co/datasets/ylacombe/expresso

**Uncertainties:** CosyVoice2 weights — confirm no separate model-agreement beyond Apache-2.0 (high confidence clean); A/B JP↔EN prosody before bulk. Jenny license permissive-but-nonstandard (no resale of data). JSUT/JVS audio not plainly commercial (prefer CV CC0 / PD). fish-speech "MIT for self-host" is secondary-source — verify the exact checkpoint. Style-Bert-VITS2 model terms per-checkpoint. Common Voice access now via Mozilla Data Collective.
