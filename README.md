# AutoReup Studio

> Studio local: **video Trung/Nhật → video Việt hoá** — phụ đề 1 dòng + **voice-over tiếng Việt**,
> cùng một nguồn `master_vi`, đóng gói hardsub (che phụ đề gốc + burn sub Việt + mix bed gốc).
> Mặc định không cần API key: mọi bước chữ nghĩa chạy qua **agent CLI headless** (Antigravity `agy`, Claude Code
> `claude` hay Codex `codex` bằng gói đăng ký đã đăng nhập, hoặc CLI bất kỳ qua template) — hoặc chọn kết nối LLM API (OpenAI-compatible, Gemini) trong Cài đặt.

```
download → asr → glossary → translate → qa → voiceover → package
```

## Cơ chế (8 lớp, xem `docs/SPEC.md` để đầy đủ)

| Lớp | Việc | Ghi chú |
|---|---|---|
| L1 Rail | OCR phụ đề burn-in (macOS Vision, hoặc LLM đọc ảnh) **trước**, ASR chỉ phiên âm phần speech mà caption không phủ | cost model đo được: `none`/`glue`/`full` |
| L2 Unit | gộp caption thành câu/cụm nghĩa (60/90 ký tự) | đơn vị *định thời*, không phải đơn vị lấy context |
| L3 Dịch | **1 workspace agent run**: `SKILL.md` + `input/` → agent đọc toàn transcript, ghi `out/draft.tsv`, tự soát nghĩa → `out/final.tsv`; pipeline đọc file + **gate code** từng dòng đổi | không nhét ngân sách độ dài vào prompt |
| L4 Đo | TTS 1 clip/unit → thời lượng THẬT | cache theo text |
| L5 Fit | repair/expand theo số đo, loop **best-of** (vòng nào xấu hơn thì rollback) | ≤3 vòng |
| L6 Voice | mỗi unit neo cửa sổ của nó; chỉ tăng tốc ≤1.18 vừa đủ; không pack theo run | drift ≤ ~0.2s |
| L7 Sub | khối cue = `[voice_start − lead, max(voice_end, unit_end)]` | sub đi trước voice, phủ trọn caption gốc |
| L8 QA + Package | gate: unit coverage, cửa sổ khớp rail, không Hán, không chồng cue; hardsub + overlay editor | `data/jobs/<id>/overlay.json` |

Chi tiết kiến trúc/quyết định: **[`docs/SPEC.md`](docs/SPEC.md)** (source of truth). `docs/archive/` chỉ là lịch sử.

## Yêu cầu

- **macOS** (bắt buộc cho OCR Vision + tuỳ chọn ASR local MLX; phần còn lại chạy được ở nơi khác)
- Python **3.13**, `ffmpeg`, Node **≥ 22** (tool Douyin CDP + build frontend)
- Một **agent CLI headless** đã đăng nhập (`agy`, `claude` hoặc `codex`) — hoặc bất kỳ CLI nào qua `AUTOREUP_AGENT_COMMAND`

## Cài đặt nhanh

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install "mlx-qwen3-asr[aligner]==0.4.4"  # ASR local Qwen3-ASR (Apple Silicon) — khuyến nghị
pip install mlx-whisper              # ASR local Whisper (tuỳ chọn) — bỏ qua cả hai nếu dùng worker
pip install -r requirements-dev.txt  # pytest

cd frontend && npm install && npm run build && cd ..
cp .env.example .env                 # điền cấu hình của bạn
uvicorn backend.app.main:app --port 8000
```

Mở `http://127.0.0.1:8000`. UI dev: `cd frontend && npm run dev`.

## ASR: ba backend

| Backend | Khi nào | Đo được |
|---|---|---|
| **qwen** (mlx-qwen3-asr: Qwen3-ASR-1.7B 8-bit + ForcedAligner, timestamp từng chữ) | mặc định khi đã cài mlx-qwen3-asr | ~15× realtime trên M4; tập phim 15 phút: lỗi chữ 0.075 vs 0.177 (Whisper), ảo giác nhạc 11 vs 51, lỗ hổng lời 0 vs 5 |
| **local** (mlx-whisper, cùng model large-v3-turbo) | khi chỉ cài mlx-whisper | ~23× realtime trên M4 |
| **worker** (Cloudflare Workers AI) | máy không phải Apple Silicon | ~4.2× realtime, có quota neuron/ngày |

`AUTOREUP_ASR_BACKEND=auto|qwen|local|worker` (`auto` = cái đầu tiên đã cài theo thứ tự qwen → local → worker).
Lần chạy đầu của `qwen` tải ~3.7 GB model (ASR 8-bit + aligner) từ Hugging Face. Gap-fill (nghe lại khoảng trống) vẫn chỉ chạy qua worker. Worker template ở [`worker-ai/`](worker-ai/) (wrangler + Workers AI binding).

## Chỉnh overlay ngay trong UI

Preview từng tập → **Khung che & vị trí sub**:
- kéo-thả **hộp che** (blur / tối / đen đặc) lên band phụ đề gốc hoặc logo kênh; có ô số X/Y/W/H + nudge 0.5%;
- **kéo chính dòng phụ đề** để đặt vị trí, hoặc ghim theo band gốc;
- **Xem thử khung hình** render đúng 1 frame bằng chính config mà bản render dùng;
- **Render lại video** chỉ chạy lại stage package; **Tải video** tải file thành phẩm.

Toạ độ lưu dạng phân số (0..1) nên không phụ thuộc độ phân giải nguồn.

## Cấu hình

Env `AUTOREUP_*` (xem [`.env.example`](.env.example) và `docs/SPEC.md` §12) là cấu hình khởi đầu.
**⚙ Cài đặt** trong UI: tab *Kết nối* (agent CLI, LLM API OpenAI-compatible/Gemini, CapCut, máy chủ TTS, Edge, OpenAI-compatible/Gemini/ElevenLabs TTS…), *Tác vụ* (kết nối + model riêng cho
dịch / thuật ngữ / chuẩn hoá / sửa độ dài / đặt tên, OCR), *Giọng đọc & ASR* (chọn engine/giọng/model, nút *Nghe thử*; Edge TTS cần `edge-tts` trong `requirements.txt`; **VieNeu (máy này)** chạy VieNeu-TTS v3 Turbo ngay trên máy (ONNX/CPU, ~1.2 GB RAM, WAV 48 kHz, 25 giọng Việt), lần đầu tải model ~1.2 GB vào cache HuggingFace rồi hardlink sang `data/models/vieneu/`). Khi đã lưu, `.autoreup-private/settings.json`
đè env (chưa lưu lần nào → chạy y hệt env); job mới dùng ngay, job đang có giữ cấu hình lúc tạo.
Cookie provider nhập từ UI, lưu ở `.autoreup-private/cookies/`; secret (API key, mật khẩu) chỉ ở `settings.json`
(chmod 600) — không bao giờ được serve hay trả về qua API.

## Tests

```bash
. .venv/bin/activate && python -m pytest tests/ -q
```

## Cấu trúc

```
backend/app/         pipeline (FastAPI in-process), dubbing/, agents/, library/, services/
frontend/            SPA React (Vite) — library, preview, overlay editor
worker-ai/           Cloudflare Workers AI ASR worker (template)
scripts/             asr client (local/worker), audit + đo chi phí, audition voice/mix
tools/               scraper Douyin qua headless Chromium (CDP)
docs/                SPEC.md (source of truth) + archive/
```

## Lưu ý & miễn trừ

- Công cụ dành cho **sử dụng cá nhân**: bạn tự chịu trách nhiệm về bản quyền nội dung, điều khoản
  của nền tảng nguồn (Douyin…) và của dịch vụ bạn cắm vào (CapCut TTS, Cloudflare…).
- Không kèm cookie/tài khoản/khóa nào — bạn tự đăng nhập và tự chịu trách nhiệm với tài khoản của mình.
- Dự án không tải hay phát tán nội dung có sẵn; nó xử lý video bạn đã có quyền dùng.

## License

MIT — xem [`LICENSE`](LICENSE).
