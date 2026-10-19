"""ASR client for the pipeline: one CLI, three backends.

``qwen``   — Qwen3-ASR-1.7B + Qwen3-ForcedAligner on Apple Silicon (mlx-qwen3-asr).
             Measured against Whisper turbo on real jobs: 2.4x fewer character errors
             on a drama episode (0.073 vs 0.177 vs the on-screen captions), 51 -> 9
             music hallucinations, no missed-speech holes (Whisper: 5, 11.6s) — at
             ~15x realtime on M4 (Whisper turbo: ~23x).
``local``  — mlx-whisper on Apple Silicon (same large-v3-turbo weights as the worker,
             no network, no quota).
``worker`` — the Cloudflare Workers AI whisper endpoint (portable, but the free tier
             has a daily neuron allocation: measured 4006 "used up your daily free
             allocation of 10,000 neurons", which failed every ASR job that day).
``auto``   — qwen, else local, else worker: the first one installed.
"""

import argparse
import base64
import json
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def duration(video_path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ])
    return float(out.decode().strip())


def extract_audio_chunk(video_path: str, start: float, length: float, out_path: Path) -> None:
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length), "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-f", "mp3", str(out_path),
    ])


def call_worker(worker_url: str, audio_path: Path, offset: float, length: float, language: str, model: str | None) -> dict:
    audio = base64.b64encode(audio_path.read_bytes()).decode()
    body = json.dumps({"audio": audio, "offset": offset, "language": language, "model": model}).encode()
    req = Request(
        worker_url.rstrip("/") + "/transcribe",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "AutoReup-ASR/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=300) as res:
            data = json.loads(res.read().decode())
            data["chunk_start"] = offset
            data["chunk_end"] = offset + length
            return data
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Worker HTTP {exc.code}: {detail}") from exc


LOCAL_DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"


def local_repo(model: str) -> str:
    """Map the pipeline's model names (Cloudflare slugs) onto MLX repos."""
    if model.startswith("mlx-community/"):
        return model
    if "turbo" in model:
        return LOCAL_DEFAULT_REPO
    # the non-turbo ladder rung: '@cf/openai/whisper' (plain whisper) or an explicit
    # large-v3 slug
    if "large-v3" in model or model.rstrip("/").endswith("/whisper"):
        return "mlx-community/whisper-large-v3-mlx"
    return LOCAL_DEFAULT_REPO


def transcribe_local(video_path: str, language: str, model: str) -> list[dict]:
    """Whole-file transcription with mlx-whisper (no chunking needed)."""
    import mlx_whisper

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        subprocess.check_call([
            "ffmpeg", "-v", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", str(audio),
        ])
        result = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=local_repo(model),
            language=None if language in ("", "auto") else language,
        )
    return [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": str(seg.get("text") or "")}
        for seg in result.get("segments", [])
    ]


# 8-bit Qwen3-ASR-1.7B: on a 15-min drama slice the same error rate and rail stats as
# the fp16 "Qwen/Qwen3-ASR-1.7B" (6 cues differ, punctuation/one name), 60s vs 93s.
QWEN_DEFAULT_REPO = "moona3k/mlx-qwen3-asr-1.7b-8bit"


def qwen_repo(model: str) -> str:
    """The pipeline's model setting holds a Whisper slug by default; only an explicit
    Qwen3-ASR repo (or local path naming one) overrides the default checkpoint."""
    return model if "qwen" in model.lower() else QWEN_DEFAULT_REPO


def phrases_from_qwen(words: list[dict], chunks: list[dict], language: str, text: str, max_chars: int) -> list[dict]:
    """Aligner words (one per CJK character) -> timed phrases of at most ``max_chars``.

    Grouping on the words keeps every cue's start/end on a real aligned character,
    instead of letting ``normalize_cues`` split a long phrase proportionally. A chunk
    the aligner returned nothing for (a language it doesn't cover) still yields its
    text, spanning the chunk — the VAD pass tightens it later — so no speech is lost.
    """
    from mlx_qwen3_asr.writers import group_subtitle_segments

    # display width: a CJK character counts 2, so max_chars characters = 2*max_chars
    phrases = group_subtitle_segments(words, language=language, text=text, max_chars=2 * max_chars)
    for chunk in chunks:
        start, end = float(chunk["start"]), float(chunk["end"])
        if str(chunk.get("text") or "").strip() and not any(start <= w["start"] < end for w in words):
            phrases.append({"start": start, "end": end, "text": str(chunk["text"]).strip()})
    return sorted(phrases, key=lambda seg: (seg["start"], seg["end"]))


def transcribe_qwen(video_path: str, language: str, model: str, max_chars: int) -> list[dict]:
    """Whole-file Qwen3-ASR with forced-aligned timestamps (the library chunks at pauses)."""
    from mlx_qwen3_asr import transcribe

    def on_progress(event: dict) -> None:
        # same "ASR <done>/<total>" line as the worker path: the pipeline parses it
        if event.get("event") == "chunk_completed":
            print(f"ASR {event['chunk_index']}/{event['total_chunks']}", flush=True)

    result = transcribe(
        video_path,
        model=qwen_repo(model),
        language=None if language in ("", "auto") else language,
        return_timestamps=True,
        return_chunks=True,
        on_progress=on_progress,
    )
    return phrases_from_qwen(result.segments or [], result.chunks or [], result.language, result.text, max_chars)


def srt_time(second: float) -> str:
    ms = int(round(second * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def split_long_phrase(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i : i + max_chars])
    return chunks


def split_text(text: str, max_chars: int = 18) -> list[dict]:
    text = text.strip().replace(" ", "")
    text = text.replace(",", "，").replace("?", "？").replace("!", "！")
    if not text:
        return []

    phrases: list[str] = []
    current = ""
    breaks = "。！？!?，,、；;：:"
    for char in text:
        current += char
        if char in breaks:
            phrases.extend(split_long_phrase(current.strip(), max_chars))
            current = ""
        elif len(current) >= max_chars:
            phrases.append(current.strip())
            current = ""
    if current.strip():
        phrases.extend(split_long_phrase(current.strip(), max_chars))

    cues: list[str] = []
    current = ""
    for phrase in phrases:
        if not current:
            current = phrase
        elif len(current) + len(phrase) <= max_chars:
            current += phrase
        else:
            cues.append(current)
            current = phrase
    if current:
        cues.append(current)
    parts = []
    for cue in cues:
        display = clean_cue_text(cue)
        if display:
            parts.append({"text": display, "translate_text": cue.strip()})
    return parts


def clean_cue_text(text: str) -> str:
    return text.strip().rstrip("，,、；;：:")


def normalize_cues(segments: list[dict], max_chars: int = 18, max_duration: float = 2.8) -> list[dict]:
    cues: list[dict] = []
    for group_id, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        if not text or end <= start:
            continue
        chunks = split_text(text, max_chars=max_chars)
        expanded: list[dict] = []
        for chunk in chunks:
            if len(chunk["text"]) <= max_chars:
                expanded.append(chunk)
                continue
            for i in range(0, len(chunk["text"]), max_chars):
                piece = chunk["text"][i : i + max_chars]
                expanded.append({"text": piece, "translate_text": piece})
        chunks = expanded or [{"text": clean_cue_text(text), "translate_text": text}]

        total_chars = sum(max(1, len(chunk["text"])) for chunk in chunks)
        cursor = start
        total_duration = end - start
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                cue_end = end
            else:
                weight = max(1, len(chunk["text"])) / total_chars
                cue_end = cursor + total_duration * weight
                cue_end = min(cue_end, cursor + max_duration)
                cue_end = max(cue_end, cursor + 0.8)
            cues.append({
                "group_id": group_id,
                "start": round(cursor, 3),
                "end": round(cue_end, 3),
                "text": chunk["text"],
                "translate_text": chunk["translate_text"],
            })
            cursor = cue_end
    return cues


def trim_chunk_overlaps(results: list[dict], total: float, overlap: float) -> list[dict]:
    kept: list[dict] = []
    margin = overlap / 2
    for result in results:
        chunk_start = float(result.get("chunk_start", 0))
        chunk_end = float(result.get("chunk_end", total))
        keep_start = 0 if chunk_start <= 0 else chunk_start + margin
        keep_end = total if chunk_end >= total else chunk_end - margin
        for seg in result.get("segments") or []:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))
            center = (start + end) / 2
            if keep_start <= center <= keep_end:
                kept.append(seg)
    kept.sort(key=lambda item: (item.get("start", 0), item.get("end", 0)))
    return kept


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def dedupe_overlaps(segments: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if deduped:
            prev = deduped[-1]
            overlap = min(float(prev.get("end", 0)), float(seg.get("end", 0))) - max(float(prev.get("start", 0)), float(seg.get("start", 0)))
            sim = text_similarity(prev.get("text", ""), text)
            repeated_phrase = len(text) >= 4 and text[:4] in prev.get("text", "")
            if overlap > 0.25 and (sim > 0.22 or repeated_phrase):
                if len(text) > len(prev.get("text", "")):
                    prev["text"] = text
                prev["end"] = max(float(prev.get("end", 0)), float(seg.get("end", 0)))
                continue
        deduped.append(seg)
    return deduped


def write_srt(segments: list[dict], output: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.extend([str(i), f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}", text, ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def write_json(segments: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    import importlib.util

    if importlib.util.find_spec("mlx_qwen3_asr"):
        return "qwen"
    return "local" if importlib.util.find_spec("mlx_whisper") else "worker"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--backend", choices=("auto", "qwen", "local", "worker"), default="auto")
    parser.add_argument("--worker-url", default="")
    parser.add_argument("--output", default=None)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--model", default="@cf/openai/whisper-large-v3-turbo")
    parser.add_argument("--chunk-seconds", type=float, default=60)
    parser.add_argument("--overlap", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--max-chars", type=int, default=28)
    args = parser.parse_args()

    total = duration(args.video)
    if args.max_duration:
        total = min(total, args.max_duration)
    output = Path(args.output) if args.output else Path("data/subtitles") / f"{Path(args.video).stem}.asr.zh.srt"

    backend = pick_backend(args.backend)
    if backend == "qwen":
        print(f"ASR backend: qwen ({qwen_repo(args.model)})")
        all_segments = transcribe_qwen(args.video, args.language, args.model, args.max_chars)
    elif backend == "local":
        print(f"ASR backend: local ({local_repo(args.model)})")
        all_segments = transcribe_local(args.video, args.language, args.model)
    else:
        if not args.worker_url:
            raise SystemExit("--worker-url is required for the worker backend")
        print(f"ASR backend: worker ({args.worker_url})")
        starts = [i * (args.chunk_seconds - args.overlap) for i in range(math.ceil(total / (args.chunk_seconds - args.overlap)))]
        starts = [s for s in starts if s < total]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            jobs = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for index, start in enumerate(starts):
                    length = min(args.chunk_seconds, total - start)
                    audio_path = tmp_dir / f"chunk_{index:04}.mp3"
                    extract_audio_chunk(args.video, start, length, audio_path)
                    jobs.append(pool.submit(call_worker, args.worker_url, audio_path, start, length, args.language, args.model))

                results = []
                for idx, job in enumerate(as_completed(jobs), 1):
                    result = job.result()
                    results.append(result)
                    print(f"ASR {idx}/{len(jobs)}")
        all_segments = trim_chunk_overlaps(results, total, args.overlap)
    deduped = dedupe_overlaps(all_segments)
    cues = normalize_cues(deduped, max_chars=args.max_chars)
    write_srt(cues, output)
    json_output = output.with_suffix(".json")
    write_json(cues, json_output)
    print(json.dumps({"output": str(output), "json": str(json_output), "segments": len(cues), "duration": total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
