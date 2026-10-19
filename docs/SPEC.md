# AutoReup Studio — SPEC

> **Source of truth duy nhất.** Cập nhật: 2026-09-14.
> Mọi thay đổi contract (pipeline, artifact, API, UI, quality gates) sửa file này.
> Các doc trong `docs/archive/` chỉ là lịch sử — không dùng để code.

## 1. Sản phẩm

Local personal studio: **video Trung → video Việt hoá**.

- Phụ đề **1 dòng** tiếng Việt trên video.
- Voice-over TTS tiếng Việt (giọng CapCut cố định `BV421_vivn_streaming`), tuỳ chọn bật/tắt.
- Sub và voice-over **cùng một nguồn Việt** (`master_vi`) → nội dung khớp nhau.
- Timing/QA deterministic trước khi đóng gói.
- Đa thể loại, data-driven: glossary/style trích tự động từ source mỗi video, không hardcode.
- Single-user, chạy local. Không multi-tenant, không cloud deploy.

## 2. Bất biến kiến trúc

1. **ASR micro-segment là rail cứng** — không bao giờ gộp làm mất ranh giới thoại; anchor chỉ là khung thời gian.
2. **Một nguồn Việt cho một semantic span**: `SOURCE_ZH + context/glossary + timing → master_vi`; `master_vi → display 1-line` và `master_vi → TTS`. Không dịch subtitle và narration riêng.
3. **Code sở hữu**: segmentation, timing, QA, scheduling, packaging. **Agent (CLI headless) chỉ viết/sửa text bounded** (TSV `ID<TAB>TEXT`), không tự chọn segmentation.
4. **Overflow là lỗi cục bộ trong 1 anchor** (absolute scheduler) — không đẩy lệch anchor sau.
5. **Stage idempotent theo job dir**; job resume được từ manifest.

## 3. Pipeline (in-process, FastAPI)

```text
download → asr ─┬ normalize ASR noise (rule + gated agent pass)
               └ build semantic units (L2)
         → glossary → translate (1 batch/call, rolling context)
                    → edit (1 gated creative pass per batch)
         → qa → voiceover (optional) → package
```

| Stage | Input | Output | Sở hữu | Fail mode |
| --- | --- | --- | --- | --- |
| download | url | video.mp4 + meta | yt-dlp adapter (Douyin: CDP extractor) | network/cookie |
| asr | video | asr_segments → units (sentence-level) | ASR worker + normalize + units builder | thiếu worker URL |
| glossary | source_zh | glossary + style notes | agent 1 pass data-driven | rỗng → fallback |
| translate | units + glossary + rolling context | speech rows (1/dòng lượt) | agent (no char budget) | validation lỗi → fail rõ |
| edit | draft VI + source + glossary | edited VI (fact-gated, max 1 pass) | agent | patch vi phạm → giữ draft |
| qa | speech + rail | qa_report | code thuần | gate không đạt → fail |
| voiceover | speech + anchors | voice.m4a + render_manifest | scheduler + TTS + ffmpeg | overflow → repair loop ≤3 vòng |
| package | video + speech + voice | softsub/dubbed mp4, srt/vtt/json, zip | ffmpeg | — |

Ghi chú: stage `edit` đã **retired** (LLM editor pass rewrite ~nửa anchors, mỗi lần rewrite là một nguồn artifact mới). Chỉ giữ trong `StageName` để manifest job cũ còn parse.

## 4. Data contract (`backend/app/dubbing/`)

Artifact trung tâm là `speech`. `unit` là đơn vị dịch (tên artifact cũ `anchors.tsv` giữ nguyên nhưng nội dung là unit).

```text
asr_segments : (id, start, end, text_zh)                        # micro-beat rail CỨNG
unit         : (id, segment_ids, start, end, source_zh, flags)  # câu/đoạn nghĩa — đơn vị dịch
run          : (id, unit_ids, start, end)                       # đoạn thoại liên tục (gap < 0.8s)
speech       : (sub_id, unit_id, start, end, source_ids, text_vi)  # 1 dòng / lượt thoại
glossary     : (term_zh, term_vi, kind, note + evidence)        # data-driven mỗi video
render_manifest : per-run fill/overflow, per-unit tts_duration
```

- `⟂` = marker tách explicit-turn, truyền giữa translate/repair và splitter.
- Display giữ dấu câu; TTS strip dấu câu riêng. **Không feed display-sliced cue vào TTS.**
- TSV/JSON chỉ là persist artifact — interface giữa các stage là dataclass typed.

## 5. Library (SQLite `data/library.db`)

Quyết định đã khoá: **SQLite** · **xoá series = cascade** · **ghép nối thẳng, không card chuyển tập** · **1 tập → 1 series**.

```sql
series(id, title, cover_url, tags json, created_at, updated_at)
series_source(id, series_id, platform, url, label, last_scanned_at)
episode(id, series_id, platform, platform_video_id, source_url, title,
        thumbnail_url, duration, uploader, discovered_at, dub_job_id, dub_state,
        UNIQUE(series_id, platform, platform_video_id))       -- dedup tập
export(id, series_id, title, description, hashtags json,
       episode_job_ids json ordered, output_path, duration, state, error,
       created_at, updated_at)
```

- `dub_state ∈ {none, translating, dubbed, failed}` (suy từ job state, cache vào episode).
- `export.state ∈ {draft, rendering, done, failed}`.
- JobManifest có `series_id`, `source_video_id`; job xong → cập nhật `episode.dub_state/dub_job_id`.
- Render export: chuẩn hoá từng tập (re-encode spec chung) rồi nối; output `data/exports/<series>/<export_id>.mp4`.

## 6. Jobs

```text
queued → running → succeeded | failed | canceled
```

- Manifest `data/jobs/<job_id>/manifest.json`: `schema_version`, `input`, `state`, `config`, `artifacts[]`, `stages[]` (status, timestamps, metrics, error).
- Artifact kinds: `video`, `source_json`, `subtitle_json/srt/vtt`, `qa_report`, `glossary`, `pipeline_audit`, `tts_manifest`, `voice_track`, `soft_sub_mp4`, `dubbed_mp4`, `hardsub_mp4`, `zip_package`, `thumbnail`.
- SSE: `GET /jobs/{id}/events`.

## 7. Quality gates (mặc định)

- `max_chars 60`, `max_cps 32.0`, `target_cps 22.0`, `min_duration 0.45s`, `max_duration 4.8s`.
- Display: **1 dòng**, reflow width 40 ký tự; cue được phép vượt tối đa `LINE_SLACK = 8` ký tự để tránh ngắt xấu — **QA gate dùng đúng ngưỡng này** (reflow và gate cùng một nguồn).
- Không ký tự Hán trong output Việt.
- Không missing/duplicate source cue coverage; không overlap; gap sau snap ≤ 0.12s.
- Rail OCR (phụ đề cứng, trước ASR): change-detection tìm từng caption ổn định, mỗi caption 1 crop → bộ đọc `tasks.ocr.recognizer`: `vision` (macOS Vision, mặc định) hoặc `llm` (kết nối có vision `openai_compat`/`gemini`, 20 crop/request, trả JSON `{"số": "text"}`, thiếu số = bỏ caption). Band: Vision; với `llm` mà Vision không ra band → heuristic pixel `hardsub.detect_sub_band`. OCR lỗi → chỉ dùng ASR.
- Rail ASR: segment được co về đúng vùng VAD speech chứa nó (bỏ blob nhạc dài), hallucination nhạc bị loại; rail bất biến **sorted, không chồng lấn** sau gap-fill.
- **Nghĩa là đơn vị dịch (F1)**: `translate` là **một workspace agent run** — workspace `data/jobs/<job>/translate/` gồm `SKILL.md` + `input/{units,glossary,notes}`, agent đọc toàn bộ transcript, ghi `out/draft.tsv`, tự soát nghĩa rồi ghi `out/final.tsv`. Pipeline **đọc artifact từ đĩa** (không parse prose), và **gate code** mọi dòng đổi giữa draft→final (số/phủ định/glossary/độ dài) — không bao giờ đặt ngân sách độ dài trong prompt.
- Agent flags — agy: `--add-dir <ws> --dangerously-skip-permissions --sandbox`; claude: cwd = ws,
  `--permission-mode acceptEdits --tools Read,Write,Edit,Glob,Grep` (không Bash), follow-up `--continue`; codex: cwd = ws,
  `-c sandbox_mode="workspace-write"`, follow-up `exec resume --last`. Timeout workspace phải phủ vòng lặp nội bộ của agent (`translate_timeout`).
- **Row = unit = clip**: mỗi unit một clip TTS neo vào cửa sổ của unit; timing không nội suy, không pack theo run, không clamp/hold.
- **Sub = voice, cùng thời điểm (F4)**: khối cue của unit = `[voice_start − lead, max(voice_end, unit_end)]` (clamp theo cue kế) — sub hiện trước voice ~0.12s và phủ trọn cửa sổ caption gốc; text sub dựng lại SAU fit loop nên luôn khớp đúng lời đọc.
- QA gate: mọi unit có dòng Việt đúng cửa sổ rail (`units_missing_coverage`, `rows_misaligned_with_rail`), không Hán, cue không chồng, 1 dòng/cue.

## 8. TTS contract

Engine = kết nối của tác vụ **Giọng đọc** (`tasks.tts`, xem §12): `capcut` tổng hợp native trong process
(session tự đăng nhập bằng email/mật khẩu của kết nối), không có session → dùng `base_url` của chính kết nối
làm máy chủ HTTP dự phòng; `http_tts` = chỉ máy chủ HTTP. Chỉ lỗi khi không engine nào dùng được.
Bật/tắt, giọng, model, `synth_rate` cũng nằm trong tác vụ đó.

Mọi engine (`backend/app/services/tts_engines.py`) cùng giao thức: `synthesize_batch(texts) -> list[bytes]`,
`cache_id`, `voice`. Chọn theo loại kết nối:

| Loại | Gọi | Giọng mặc định | Model |
|---|---|---|---|
| `capcut` | native; không có session → `base_url` (HTTP) | `BV421_vivn_streaming` | — |
| `http_tts` | `POST /v2/synthesize` | như trên | — |
| `edge` | gói `edge-tts` (import lười) | `vi-VN-HoaiMyNeural` | — |
| `vieneu` | gói `vieneu` (VieNeu-TTS v3 Turbo, ONNX/CPU trên máy này, 1 model/process) → WAV 48kHz; câu chỉ có ký hiệu → 0.3s im lặng | `Thiện Minh` | — |
| `openai_compat` | `POST {base_url}/audio/speech` (mp3) | `nova` | bắt buộc |
| `gemini` | `generateContent` modality AUDIO → PCM 24kHz bọc WAV | `Kore` | bắt buộc (vd. model `*-tts`) |
| `elevenlabs` | `POST /v1/text-to-speech/{voice}?output_format=mp3_44100_128` | bắt buộc (voice id) | mặc định `eleven_flash_v2_5` |

Khoá cache clip thô = `cache_id` + giọng + text; `cache_id`: capcut `capcut-native`, http = `base_url`
(cache cũ vẫn dùng được), vieneu `vieneu:v3turbo`, loại khác `<loại>:<model>`. Thiếu thứ engine cần (api_key, model, voice…) → stage
báo lỗi rõ.

`vieneu`: onnxruntime ≥1.24 từ chối `.onnx` có `.data` ngoài thư mục của nó, mà shared blob store của HF hub
để mỗi file một thư mục → engine hardlink file model từ cache HF sang `data/models/vieneu/` (khác ổ → copy),
nên `vieneu` ghim đúng bản. Danh sách giọng tĩnh (`VIENEU_VOICES`) — chỉ synth và nút Kiểm tra mới tải model.

Voice tuning (BV421 'Nguồn nhỏ ngọt ngào', đã A/B chọn):
- text mode **`expressive`**: giữ sắc thái dấu câu (`,` = lấy hơi, `.` = xuống giọng, `…` = ngắt kịch tính, `! ?` = lên giọng) — gom mọi dấu thành một dấu phẩy nghe máy móc;
- `tts_synth_rate 1.0` — không time-stretch post-synth (artifact atempo đọc như robot); độ dài do text quyết;
- trim silence: keep 0.08s, ngưỡng -50dB (cắt sát làm cụt hơi = "khựng");
- mastering: compressor chậm (ratio 1.6, attack 20ms, release 240ms, makeup 1.2), presence 3.2k +1.2, air 10k +1.0, lowpass 15k;
- mix với âm gốc: bed **0.18**, **sidechain duck 6:1** (attack 25ms/release 420ms), **carve 2.5kHz −3dB** — nhạc/SFX nở lại khi ngừng thoại, tiếng gốc chỉ bị hạ lúc có thoại;
- **nâng âm gốc khi không thoại** (`dub_gap_volume`, mặc định **0.7**; chỉnh từng tập trong editor): vùng thoại = Silero VAD trên audio gốc ∪ chỗ voice TTS kêu, nới 0.3s; khe ≥ 1.5s lên mức gap, ramp 0.4s nằm trong khe (envelope wav 1 kHz × bed qua `amultiply`). Không có VAD → không nâng (tránh làm to thoại gốc còn sót);
- timeline mỗi unit neo vào cửa sổ của nó: `start = max(voice trước kết thúc, unit.start - lead)`; voice chỉ được **tăng tốc** (≤ `tempo_ceiling 1.18`), chỉ vừa đủ để khớp, không bao giờ làm chậm; không thêm gap giữa clip (crossfade lo click).
- **Overlay (che + vị trí sub) là dữ liệu của job, ở không gian OUTPUT**: `data/jobs/<job>/overlay.json` (phân số 0..1) gồm các vùng che (blur/darken/solid + cường độ) và vị trí/cỡ sub; render scale/pad về khung canonical TRƯỚC rồi mới che + đặt sub — nên preview trong UI (đúng khung output, có thanh tua) luôn trùng bản burn; không có overlay thì dùng band dò tự động (Vision trước, heuristic pixel là fallback).
- **Ảnh của người dùng (logo/watermark)**: thư viện chung `data/overlay_images/` (upload `POST /api/v1/settings/images`, luôn lưu lại thành PNG RGBA, giữ alpha); vùng overlay `mode: "image"` chỉ lưu TÊN file. Ảnh nằm gọn trong hộp (giữ tỉ lệ, căn giữa), `strength` = độ đậm; được ghép vào lớp phụ đề trong suốt → nằm trên vùng che, dưới phụ đề Việt, suốt cả video; ảnh đã mất khỏi thư viện thì bỏ qua, không làm hỏng render.
- fit loop chấm bằng SỐ ĐO: `voice > free_time (tới hình kế) + 0.4s` → rút gọn; `voice < 0.65 × VAD speech` **và** nguồn đang nói (`vad ≥ 0.8s`) → hoàn thiện; lặp tới hội tụ (≤3 vòng). Không dùng ngân sách ký tự để đoán.



- Giọng cố định `BV421_vivn_streaming`; text mode `continuous`; speed/pitch/volume = 10 (neutral).
- Budget **12 chars/sec** (đo từ voice thật, không đoán); raw median ~14.16 cps.
- Accept nếu `0.92 ≤ tempo_needed ≤ 1.18` và không vi phạm semantic/glossary.
- Overflow → repair `master_vi` các ID lỗi → resynthesize **chỉ các ID đó**; tối đa 3 vòng.
- Package dubbed: giữ audio gốc làm bed, duck dưới voice Việt (sidechain) — **không mute** mặc định.

## 9. Agent runtime (CLI headless hoặc LLM API)

Mọi bước text (glossary, translate, normalize, repair, naming) chạy qua **CLI agent headless** (mặc định, không
API key) hoặc một **kết nối LLM API** (`openai_compat`, `gemini`). Mỗi bước là một *tác vụ* trong settings (§12)
trỏ tới một kết nối + model + effort riêng (`backend/app/settings/resolve.py::chat_agent`); timeout/retries vẫn
lấy từ env. Tác vụ chat trỏ vào kết nối không có năng lực chat (TTS…) → báo lỗi rõ, không chạy ngầm bằng agy.

- **provider `agy`** (mặc định): Antigravity `agy -p <prompt> --output-format json --print-timeout <t> [+ --model, --effort]`;
  login subscription qua HOME — `agent_home` để tách account (mỗi account một HOME riêng).
- **provider `claude_code`**: Claude Code `claude -p <prompt> --output-format json` (đọc `result`), luôn kèm
  `--strict-mcp-config --setting-sources "" --disable-slash-commands` — không nạp CLAUDE.md/MCP/plugin của user
  (đo: ~283k → ~4k token/lượt); one-shot thêm `--tools "" --no-session-persistence --system-prompt …`, chạy trong cwd
  tạm rỗng. `--model` (alias `sonnet`/`opus`/`fable`/`haiku` hoặc id đầy đủ), `--effort`.
- **provider `codex`**: Codex `codex exec --skip-git-repo-check -c sandbox_mode="read-only" --ephemeral
  --output-last-message <file> [--model, -c model_reasoning_effort=…] -- <prompt>`, đọc reply từ file; lỗi lấy dòng
  `ERROR:` cuối của stderr. Cwd tạm rỗng (không nạp AGENTS.md của repo).
- claude/codex/agy dùng login có sẵn của CLI; `home` (HOME override) để tách account. Quota oracle chỉ dành cho agy.
- **provider `command`**: template CLI bất kỳ cho agent khác — ACP (`acpx claude exec --format quiet {prompt}`)…
  `{prompt}` được thay thế, hoặc prompt append cuối argv nếu không có placeholder.
- **provider `openai_compat`**: `POST {base_url}/chat/completions` (OpenAI, OpenRouter, Ollama, vLLM…), header
  `Authorization: Bearer <api_key>` (bỏ khi không có key), `reasoning_effort` = effort khi đặt; ảnh nhúng `data:` URL.
- **provider `gemini`**: `POST {base_url|https://generativelanguage.googleapis.com/v1beta}/models/{model}:generateContent`,
  header `x-goog-api-key`, `systemInstruction` + `contents[0].parts` (text + `inline_data`); bỏ qua part `thought`.
- CLI: prompt gửi đi = `system + "\n\n---\n\n" + user`; API giữ system/user tách riêng. Retry mặc định 3 lần backoff;
  lỗi trả `AgentError` (API: kèm HTTP status + message của provider, không bao giờ chứa api_key — key được strip
  khoảng trắng và che `***` trong mọi thông báo). API 4xx (trừ 408/429) không retry; 429 chờ theo `Retry-After` (≤30s).
- Translate qua API (không có công cụ file) **giả lập workspace**: `SKILL.md` làm system prompt, `input/*` dán inline,
  khối TSV trả về ghi ra `out/draft.tsv` + `out/final.tsv` (raw ở `out/reply-N.txt`) → gate/follow-up (chỉ hỏi ID còn
  thiếu, ≤ `translate_retries`, timeout/2)/bisect y như đường agent; lỗi API (timeout, vượt context…) = session hỏng →
  bisect, không làm hỏng cả stage. Không dùng quota oracle của agy.
- Adapter nằm ở `backend/app/agents/` (`runtime.py`, `agy.py`, `claude_code.py`, `codex.py`, `command.py`, `api_llm.py`); stage modules chỉ gọi
  `complete(system, user, config)`. Settings API: *Kiểm tra* kết nối API = liệt kê model (chứng minh endpoint + key),
  `GET …/models` trả danh sách model cho ô chọn (claude: alias; codex: từ
  `~/.codex/models_cache.json` của CLI; CLI khác: rỗng, nhập tay).

## 10. API v1

```text
# Providers
GET    /providers/sessions                     # trạng thái cookie Douyin/Bilibili
PUT    /providers/{platform}/cookies           # lưu cookies (Cookie-Editor JSON / cookies.txt / header)
DELETE /providers/{platform}/cookies

# Settings (secret luôn trả về dạng mask "••••" + 4 ký tự cuối)
GET    /settings                               # connections + tasks
PUT    /settings                               # lưu toàn bộ; secret rỗng/mask = giữ cũ; tham chiếu sai → 400
POST   /settings/connections/{id}/test         # kiểm tra nhanh (lệnh có tồn tại, server có trả lời; edge/elevenlabs: liệt kê giọng; vieneu: tải model) → {ok, message}
GET    /settings/connections/{id}/models       # {models: []} (CLI agent: model nhập tay)
GET    /settings/tts/voices?connection=        # {voices: [{id, name}]} (edge: giọng tiếng Việt; elevenlabs: giọng của tài khoản); lỗi engine → 502
POST   /settings/tts/preview {connection, voice?, model?, text?}  # nghe thử → bytes audio/mpeg hoặc audio/wav; lỗi → 502 + detail

# Studio (discovery)
GET    /studio/state                           # source videos + queue + assets
POST   /studio/discover  { url }
POST   /studio/download-queue
POST   /studio/add-video
POST   /studio/import-channel · /preview-channel
GET    /studio/thumb

# Jobs
POST   /jobs                                   # tạo job từ URL
GET    /jobs · GET /jobs/{id} · GET /jobs/{id}/events (SSE)
POST   /jobs/{id}/cancel
GET    /jobs/{id}/artifacts/{kind}             # tải artifact
DELETE /jobs/{id}

# Library / Series
GET    /series?q=&sort=&limit=&offset=         # {items, total}
POST   /series · GET/PATCH/DELETE /series/{id} # delete = cascade (bỏ purge)
POST   /series/{id}/sources · DELETE /series/{id}/sources/{sid}
POST   /series/{id}/scan                       # upsert episodes, dedup
GET    /series/{id}/episodes?dub_state=
POST   /series/{id}/translate  { episode_ids[] }
POST   /series/{id}/translate-titles

# Exports (ghép tập)
GET/POST /series/{id}/exports
GET/DELETE /exports/{export_id}
POST   /exports/{export_id}/render
POST   /series/{id}/suggest-export

# Media
GET    /media/...                              # file dưới data/
```

## 11. UI

Màn hình (React + Vite, serve bởi FastAPI):

- **Library** `/` — grid series + import URL/kênh.
- **Series detail** `/series/:id?tab=source|dubbed|export` — 3 tab: Nguồn / Đã dịch / Export; tab nằm trong URL.
- **Settings** (modal) — sắp theo **việc**, field kỹ thuật gập trong **▸ Nâng cao** (mặc định đóng):
  - **Dịch thuật** — chọn 1 dịch vụ cho mọi việc chữ (dịch, thuật ngữ, chuẩn hoá, sửa câu, đặt tên):
    Antigravity (agy) / Claude Code (claude) / Codex CLI (codex) — gói đăng ký, nút Kiểm tra / Gemini API (key + model) / OpenAI-compatible (địa chỉ API +
    key + model) / Lệnh CLI tuỳ chỉnh (chỉ hiện khi đã có). Nâng cao: model + mức nghĩ riêng từng bước con
    (mặc định "giống Dịch"), thư mục HOME, lệnh chạy (agy/claude/codex), mẫu lệnh CLI.
  - **Giọng đọc** — bật lồng tiếng; dịch vụ CapCut / VieNeu (máy này) / Edge TTS / OpenAI-compatible / Gemini / ElevenLabs /
    Máy chủ HTTP, chỉ hiện field của dịch vụ đã chọn (CapCut: email + mật khẩu; API: key, model giọng đọc
    cho OpenAI/Gemini); Giọng (danh sách `GET /settings/tts/voices`) + **▶ Nghe thử** (`POST /settings/tts/preview`,
    cần lưu trước). Nâng cao: tốc độ sau tổng hợp, máy chủ HTTP dự phòng của CapCut, model ElevenLabs.
  - **Phụ đề & lời thoại** — Nguồn lời thoại (`asr.ocr_first`): Tự động (đọc phụ đề trên hình trước,
    Speech-to-Text lấp chỗ thiếu — khuyến nghị) / Chỉ Speech-to-Text; Đọc phụ đề trên hình: macOS Vision /
    AI (Gemini/OpenAI-compatible, dùng key đã nhập + model); Speech-to-Text chạy ở Tự động / Trên máy
    (Qwen3-ASR) / Trên máy (mlx-whisper) / Máy chủ Cloudflare (+ địa chỉ). Tự động = qwen → mlx-whisper →
    worker, cái đầu tiên đã cài. Nâng cao: model Speech-to-Text (backend qwen chỉ nhận repo Qwen3-ASR,
    slug Whisper bị bỏ qua → mặc định `moona3k/mlx-qwen3-asr-1.7b-8bit`).
  - **Tài khoản nền tảng** — cookie/session Douyin, Bilibili.

  UI không hiện id kết nối, không có chọn "loại": mỗi dịch vụ ↔ kết nối **đầu tiên** cùng `type` (tạo khi
  dùng lần đầu, id `c-xxxxxx`); key nhập 1 lần dùng lại ở mọi tab ("đã có ✓"); secret chỉ về dạng mask,
  để trống = giữ. Kết nối thêm tay trong JSON (khác loại / trùng loại) được giữ nguyên khi lưu. Helper map
  thuần ở `frontend/src/settings/model.ts`, test `node --test frontend/tests` (chạy trong pytest).
  Lưu → áp dụng cho job mới.

UX rules (bắt buộc):

1. **Title-first** — hiển thị title video (từ `metrics.download.title`), không phải hostname.
2. **Chỉ file final** trong Export: final SRT/VTT, softsub MP4, dubbed MP4, ZIP; JSON chỉ dưới nhãn "Dữ liệu phụ đề".
3. **Stage timeline đủ trạng thái** pending/running/done; không show log thô, path, tên provider.
4. **4 states rõ ràng** cho library/series: empty / running / failed (có action tiếp theo) / done.
5. **Preview là surface chính**: player + subtitle list sync (auto-scroll, jump tới dòng).
6. Mobile: sidebar thành drawer; `aria-live` cho job status.

## 12. Config (env `AUTOREUP_*`, file `.env`)

```text
Agent:    AGENT_PROVIDER (agy|claude|codex|command), AGENT_BIN, AGENT_HOME, AGENT_COMMAND,
          AGENT_EFFORT, AGENT_TIMEOUT, AGENT_RETRIES, TRANSLATE_MODEL, POLISH_MODEL
Content:  CONTENT_GENRE (hint, optional), CONTENT_NOTES (mặc định per-video)
ASR:      ASR_WORKER_URL, ASR_MODEL, ASR_WORKERS
Download: COOKIES_FILE, BROWSER_PATH, NODE_BIN
TTS:      TTS_ENABLED, TTS_SERVER_URL, TTS_VOICE, TTS_NATIVE, CAPCUT_SESSION_FILE
Limits:   MAX_CHARS, DISPLAY_MAX_CHARS, DISPLAY_MAX_LINES, MAX_CPS, TARGET_CPS,
          MIN_DURATION, MAX_DURATION, SNAP_GAP, MAX_CONCURRENT_JOBS
```

Provider cookies nhập từ UI lưu tại `.autoreup-private/cookies/` (ngoài thư mục media được serve).

**Settings JSON đè env**: `.autoreup-private/settings.json` (chmod 600, ghi atomic, không serve) giữ
`connections` (agy · claude_code · codex · command · openai_compat · gemini · elevenlabs · capcut · edge · http_tts · vieneu; id
`^[a-z0-9][a-z0-9_-]{0,39}$`) và `tasks` (translate, glossary, normalize, repair, naming, ocr, tts, asr).
Không có file → seed từ env, hành vi y hệt chỉ-env: kết nối `agent` (AGENT_*), `capcut` (TTS_NATIVE, kèm
TTS_SERVER_URL làm HTTP dự phòng) hoặc `tts-server`; translate/glossary/normalize/naming = TRANSLATE_MODEL,
repair = POLISH_MODEL. File được đọc mỗi lần (không cache). Job mới ghi `config.settings_snapshot` (đã bỏ
api_key/mật khẩu/email — manifest được serve); resume dùng snapshot, các trường đó lấy lại theo `connection.id`
từ store hiện tại (chỉ khi kết nối còn cùng loại + `base_url`); file hỏng → API báo lỗi rõ, lưu lại từ UI để ghi đè;
job cũ không có snapshot đọc store hiện tại. Env vẫn giữ infra: timeout/retries, workers, đường dẫn,
tham số CapCut ký/batch.

## 13. Dev & test

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app --reload          # mở http://127.0.0.1:8000

.venv/bin/python -m pytest tests/ -q           # phải xanh toàn bộ
cd frontend && npm run dev                     # UI dev
```

## 14. Non-goals (v2)

- Không multi-voice; giữ 1 giọng cố định.
- Không realtime/streaming render.
- Không multi-user.
- Không mã hoá secret at-rest (settings.json chỉ chmod 600, single-user).
- Không subtitle editing workflow trong UI (preview/review ổn trước đã).
