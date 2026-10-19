"""Extract the ASR rail directly from a video's burned-in (hardsub) subtitles via
macOS Vision OCR — the primary transcript source for 短剧 (the on-screen text IS the
script; more accurate than ASR and immune to its homophone/proper-noun mistakes), with
Whisper ASR kept as the fallback when a video has no usable hardsub coverage.

Pipeline (measured ~4s for a 106s episode, 24.5x realtime, no re-encode):
  1. One ffmpeg pass decodes+crops ONLY the subtitle band at a low fps (cheap: a few
     hundred KB of gray pixels, not full frames).
  2. A text-mask (bright-next-to-dark-outline) column profile per frame turns "did the
     CAPTION change" into a cheap numpy diff — this ignores background motion/animation,
     which a raw pixel diff does not (that over-segmented ~8x in testing).
  3. OCR runs ONCE per stable caption run (not per frame) — cost scales with the number
     of distinct lines (~25-80/episode), not frame count.
  4. Coarse (1/fps) start times are handed to the existing Silero-VAD ``refine_segments``
     snap, same as the ASR rail — measured: hardsub timing already matches VAD onset for
     ~96% of cues (creators author it against the real voice); only edge cases need the
     snap. This is why no separate "OCR onset" logic is needed here.
"""

from __future__ import annotations

import json
import subprocess
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .models import AsrSegment
from .text import clean

if TYPE_CHECKING:
    import numpy as np

    from backend.app.agents import AgentConfig

Band = tuple[int, int, int, int]  # (band_y, band_h, width, height) — see hardsub.detect_sub_band
# One gray crop of the (padded) subtitle band per stable caption run in, one text per
# crop out ("" = no caption read). The seam between change detection and the reader.
Recognizer = Callable[[list["np.ndarray"]], list[str]]
_OCR_BATCH = 200  # crops per recognizer call (a multiple of llm_recognizer's batch of 20)


@dataclass(slots=True)
class OcrExtractConfig:
    fps: int = 4  # caption sampling rate; timing is later snapped to VAD, so coarse is fine
    # Vision's "accurate" recognizer doesn't give CJK text a continuous confidence score —
    # measured output only ever lands on {0.3, 0.5, 1.0}. 0.3 is a legitimate read, not
    # noise (verified against source frames: full correct sentences, same error rate as
    # the 0.5/1.0 tiers); a 0.5 floor was silently dropping ~25% of real captions with no
    # fallback. Position + width below already reject the actual junk (icon glyphs,
    # split single characters land under 0.05 width).
    min_conf: float = 0.2
    cx_lo: float = 0.25   # a caption box's centre-x must sit in this fraction of the
    cx_hi: float = 0.75   # band width (horizontally centred — excludes corner UI/logos)
    min_width_frac: float = 0.08   # caption box must span at least this much of the band width
    diff_threshold: float = 0.35   # 1-cosine-similarity of column profiles -> "caption changed"
    empty_frac: float = 0.02       # text-pixel fraction below this -> "no caption" frame
    # Vision reads worse when the crop is flush against the glyphs (clips ascenders/
    # descenders, box-fit gets noisier near the edge — measured: several cues only came
    # back correct with this padding). Only the OCR crop gets it; change detection below
    # still runs on the exact original band so cue-run timing is unaffected.
    ocr_pad_frac: float = 0.18


def ocr_coverage(segments: list[AsrSegment], duration: float) -> float:
    """Fraction of the video's runtime covered by OCR-derived speech segments."""
    if duration <= 0:
        return 0.0
    return sum(seg.duration for seg in segments) / duration


def vision_recognizer(config: OcrExtractConfig | None = None) -> Recognizer | None:
    """macOS Vision reader, or ``None`` when ocrmac is unavailable."""
    try:
        from ocrmac import ocrmac
    except Exception:
        return None
    import cv2
    from PIL import Image

    config = config or OcrExtractConfig()

    def recognize(crops: list["np.ndarray"]) -> list[str]:
        texts: list[str] = []
        for crop in crops:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
            try:
                result = ocrmac.OCR(pil, recognition_level="accurate",
                                    language_preference=["zh-Hans", "en-US"]).recognize()
            except Exception:
                texts.append("")
                continue
            # Vision returns one box PER LINE, so a wrapped two-line caption is two
            # candidates here — keep every box that passes the filters (not just the
            # widest) or the second line silently vanished from the transcript. Vision's
            # bbox is bottom-left origin (larger y = higher on screen), so sort descending
            # to read top line first.
            candidates: list[tuple[float, str]] = []
            for candidate_text, conf, (bx, by, bw, _bh) in result or []:
                cx = bx + bw / 2
                if conf >= config.min_conf and config.cx_lo < cx < config.cx_hi and bw > config.min_width_frac:
                    candidates.append((by, candidate_text.strip()))
            candidates.sort(key=lambda c: -c[0])
            texts.append("".join(part for _, part in candidates if part))
        return texts

    return recognize


OCR_LLM_SYSTEM = (
    "You are an OCR engine for the burned-in subtitles of Chinese short dramas. "
    "Each image is a horizontal strip cut from the bottom of one video frame. "
    "For every image, transcribe the subtitle caption exactly as written (keep the "
    "original Chinese characters and punctuation; do not translate, correct or explain). "
    "A caption wrapped over two lines is one caption: join the lines top to bottom with "
    "no separator. Ignore watermarks, logos, usernames and UI text at the edges. "
    "If an image has no caption, use an empty string. "
    'Reply with ONLY a JSON object mapping each image number to its text, e.g. {"1": "…", "2": ""}.'
)


def _parse_indexed_texts(reply: str, count: int) -> list[str]:
    """``{"1": text, …}`` (fenced or with chatter around it) -> one text per image.

    Out-of-range, non-numeric or non-string entries are ignored; a missing index reads
    as "" (that caption is dropped and the ASR gap-fill covers its speech)."""
    from backend.app.agents import AgentError

    # The first decodable object wins: tolerates code fences and chatter on either
    # side, including chatter that itself contains braces.
    decoder = json.JSONDecoder()
    data = None
    start = reply.find("{")
    while start != -1 and not isinstance(data, dict):
        try:
            data, _end = decoder.raw_decode(reply, start)
        except ValueError:
            data = None
        start = reply.find("{", start + 1)
    if not isinstance(data, dict):
        raise AgentError(f"ocr llm: phản hồi không phải JSON object: {reply[:120]!r}")
    texts = [""] * count
    for key, value in data.items():
        try:
            index = int(str(key).strip())
        except ValueError:
            continue
        if 1 <= index <= count and isinstance(value, str):
            texts[index - 1] = value.strip()
    return texts


def llm_recognizer(agent: "AgentConfig", batch_size: int = 20) -> Recognizer:
    """Reader backed by a vision-capable LLM API: ``batch_size`` PNG crops per request,
    each with the agent runtime's retry/backoff.

    A request that still fails raises ``AgentError`` — the caller treats OCR as unavailable and
    the Whisper rail stands alone, same as a missing Vision."""
    import cv2

    from backend.app.agents import AgentError, complete

    def recognize(crops: list["np.ndarray"]) -> list[str]:
        texts: list[str] = []
        for offset in range(0, len(crops), batch_size):
            batch = crops[offset:offset + batch_size]
            images: list[bytes] = []
            for crop in batch:
                ok, encoded = cv2.imencode(".png", crop)
                if not ok:
                    raise AgentError("ocr llm: không mã hoá được ảnh PNG")
                images.append(encoded.tobytes())
            user = f"{len(batch)} images, numbered 1 to {len(batch)} in the order given."
            texts += _parse_indexed_texts(complete(OCR_LLM_SYSTEM, user, agent, images=images), len(batch))
        return texts

    return recognize


def extract_ocr_segments(
    video: Path,
    band: Band,
    config: OcrExtractConfig | None = None,
    recognizer: Recognizer | None = None,
) -> list[AsrSegment] | None:
    """OCR the burned-in subtitle track into an ASR-shaped rail.

    ``recognizer`` reads the caption crops (default: macOS Vision). Returns ``None``
    when OCR can't run at all — no recognizer (Vision unavailable), or the ffmpeg
    decode failed — so the caller falls back to Whisper; a recognizer's own error
    (e.g. an LLM API failure) propagates for the caller to log, with the same fallback.
    Returns ``[]`` when OCR ran cleanly but found no captions (a same-condition signal
    as ``None`` for the caller's coverage check, kept distinct for logging clarity)."""
    config = config or OcrExtractConfig()
    recognizer = recognizer or vision_recognizer(config)
    if recognizer is None:
        return None
    import cv2
    import numpy as np

    band_y, band_h, width, height = band
    if band_h <= 0 or width <= 0:
        return None

    # Crop wider than the band for OCR's sake (see ocr_pad_frac); change detection
    # below re-slices back to the exact original band so cue-run timing is unaffected.
    pad = max(1, round(config.ocr_pad_frac * band_h))
    crop_y0 = max(0, band_y - pad)
    crop_y1 = min(height, band_y + band_h + pad)
    crop_h = crop_y1 - crop_y0
    band_top = band_y - crop_y0

    # Convert to gray BEFORE cropping: on a 4:2:0 source ffmpeg silently rounds an odd
    # crop height down to even (measured: 175 -> 174 rows), so every frame read here
    # would be one row short, the index drifts 1/crop_h behind the video and the
    # captions land seconds early (19.5s at minute 57 of a 2.6h episode).
    #
    # Frames are streamed one at a time and dropped once seen: the whole band of a long
    # video does not fit in memory (1920x200 gray at 4 fps is ~43 GB for a 7.7h video,
    # and holding it all once took the backend to 145 GB and got it killed).
    x0, x1 = int(width * 0.18), int(width * 0.82)
    empty_floor = config.empty_frac * band_h * (x1 - x0)
    frame_bytes = width * crop_h

    def mask_col(frame: "np.ndarray") -> tuple["np.ndarray", bool]:
        """Normalized text-mask column profile (not raw pixels) and whether the band is empty."""
        bright = (frame >= 195).astype(np.uint8)
        dark = cv2.dilate((frame <= 85).astype(np.uint8), np.ones((5, 5), np.uint8))
        text = (bright & dark)[:, x0:x1]
        col = text.sum(axis=0).astype(np.float32)
        norm = np.linalg.norm(col)
        return (col / norm if norm > 0 else col), float(text.sum()) < empty_floor

    # A run of frames with the same caption ends where the profile changes; its middle
    # frame is the one OCR'd. The middle only moves forward as the run grows, so only
    # the frames from the current middle on are kept. Crops are OCR'd in batches as they
    # come, so a long video's thousands of cues are never all held at once either.
    cue_runs: list[tuple[int, int]] = []
    crops: list["np.ndarray"] = []
    texts: list[str] = []
    run: deque[tuple[int, "np.ndarray"]] = deque()
    cursor = run_empty = n = 0
    prev_col = prev_empty = None

    def recognize_crops() -> None:
        batch = crops.copy()  # the recognizer may keep the list it was handed
        crops.clear()
        read = recognizer(batch)
        if len(read) != len(batch):  # a short reply must not shift captions in time
            raise ValueError(f"ocr-subs: recognizer returned {len(read)} texts for {len(batch)} crops")
        texts.extend(read)

    def close_run(end: int) -> None:
        if run_empty <= 0.5 * (end - cursor):
            cue_runs.append((cursor, end))
            crops.append(run[0][1])
            if len(crops) >= _OCR_BATCH:
                recognize_crops()

    partial = False
    with subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"format=gray,crop={width}:{crop_h}:0:{crop_y0},fps={config.fps}",
         "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ) as proc:
        try:
            while chunk := proc.stdout.read(frame_bytes):
                if len(chunk) < frame_bytes:
                    partial = True
                    break
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(crop_h, width)
                col, is_empty = mask_col(frame[band_top:band_top + band_h])
                if prev_col is not None and (
                    1 - float(np.sum(col * prev_col)) > config.diff_threshold or is_empty != prev_empty
                ):
                    close_run(n)
                    cursor, run_empty = n, 0
                    run.clear()
                run.append((n, frame))
                run_empty += is_empty
                n += 1
                while run[0][0] < (cursor + n) // 2:
                    run.popleft()
                prev_col, prev_empty = col, is_empty
        except BaseException:
            proc.kill()  # don't leave ffmpeg blocked on a full pipe
            raise
    if proc.returncode != 0:
        return None
    if partial:
        # Frames of another size than asked for: slicing them would shift every
        # caption in time, so refuse and let the caller fall back to ASR.
        raise RuntimeError(
            f"ocr-subs: ffmpeg returned a trailing partial frame, not a multiple of the "
            f"{width}x{crop_h} frame — caption timing would drift"
        )
    if n < 2:
        return None
    close_run(n)

    if crops:
        recognize_crops()
    segments: list[AsrSegment] = []
    for (start_i, end_i), raw in zip(cue_runs, texts, strict=True):
        text = clean(raw)
        if text:
            segments.append(AsrSegment(
                id=len(segments) + 1,
                start=round(start_i / config.fps, 3),
                end=round(end_i / config.fps, 3),
                text_zh=text,
            ))
    return segments


def detect_band(video: Path, frames: int = 12, crop_top: float = 0.55, pad_frac: float = 0.01) -> Band | None:
    """Locate the burned-in subtitle band via macOS Vision OCR.

    Samples the bottom strip, keeps bottom/horizontally-centred/wide text boxes (the
    caption, not watermarks or corner UI) and returns their aggregated y-range.
    Returns None when Vision is unavailable or too few sampled frames carry a caption
    (the caller then runs ASR only)."""
    try:
        from ocrmac import ocrmac
    except Exception:
        return None
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        cap.release()
        return None
    crop_y0 = int(max(0.0, min(0.9, crop_top)) * height)
    crop_h = height - crop_y0
    tops: list[float] = []
    bottoms: list[float] = []
    hits = 0
    for index in range(max(6, frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (index + 0.5) / max(6, frames)))
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[crop_y0:, :]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        try:
            result = ocrmac.OCR(pil, recognition_level="accurate", language_preference=["zh-Hans", "en-US"]).recognize()
        except Exception:
            continue
        found = False
        for _text, conf, (bx, by, bw, bh) in result or []:
            cx = bx + bw / 2
            if conf >= 0.5 and 0.25 < cx < 0.75 and bw > 0.12:
                tops.append(crop_y0 + (1 - (by + bh)) * crop_h)
                bottoms.append(crop_y0 + (1 - by) * crop_h)
                found = True
        hits += found
    cap.release()
    if hits < max(6, frames) * 0.25 or not tops:
        return None
    pad = max(1, round(pad_frac * height))
    band_y = max(0, int(np.percentile(tops, 10)) - pad)
    band_bottom = min(height, int(np.percentile(bottoms, 90)) + pad)
    if band_bottom - band_y < height * 0.02:
        return None
    return band_y, band_bottom - band_y, width, height
