import type { Artifact, DubProgress, DubState, ExportState, JobManifest } from "./types";

/** The most recent artifact of a kind on a job (used to find a job's dubbed video). */
export function latestArtifact(job: JobManifest, kind: string): Artifact | undefined {
  return [...job.artifacts].reverse().find((artifact) => artifact.kind === kind);
}

/** Route a remote CDN cover through the backend proxy (browsers can't load Douyin
 *  images cross-origin). Local /media paths and empty values pass through unchanged. */
export function thumbSrc(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  if (url.startsWith("/")) return url;
  return `/api/v1/library/thumb?url=${encodeURIComponent(url)}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return "";
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const mm = m % 60;
  const ss = s % 60;
  return h > 0 ? `${h}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${m}:${String(ss).padStart(2, "0")}`;
}

export const DUB_STATE_VI: Record<DubState, string> = {
  none: "Chưa dịch",
  translating: "Đang dịch",
  review: "Chờ duyệt render",
  dubbed: "Đã dịch",
  failed: "Lỗi",
};

export const EXPORT_STATE_VI: Record<ExportState, string> = {
  draft: "Nháp",
  rendering: "Đang ghép",
  done: "Xong",
  failed: "Lỗi",
};

// Pipeline stage → human label, in run order. Keep in sync with backend StageName.
export const STAGE_VI: Record<string, string> = {
  queued: "Đang chờ",
  download: "Tải video",
  asr: "Nhận dạng lời",
  glossary: "Lập thuật ngữ",
  translate: "Dịch",
  edit: "Biên tập",
  qa: "Kiểm tra",
  voiceover: "Lồng tiếng",
  package: "Ghép & burn phụ đề",
};

/** Turn a raw pipeline error into a short, actionable Vietnamese message. */
export function dubErrorVi(error: string | null | undefined): string {
  const e = error ?? "";
  if (/timed out|timeout/i.test(e) && /douyin_extract|download/i.test(e))
    return "Không tải được video (Douyin chặn hoặc hết thời gian). Thử lại sau ít phút.";
  if (/not recognized as a known model|invalid model selection/i.test(e))
    return "Model slug không hợp lệ — kiểm tra AUTOREUP_TRANSLATE_MODEL / POLISH_MODEL (xem `agy models`).";
  if (/agent call failed|agent binary|agent command|agent exited|no result envelope|no result object|no last message/i.test(e))
    return "Agent CLI lỗi — kiểm tra CLI (agy/claude/codex) đã đăng nhập và lệnh chạy đúng trong Cài đặt.";
  if (/验证码|captcha/i.test(e)) return "Douyin yêu cầu xác minh (CAPTCHA). Cần cookie mới.";
  if (/mã ho|playlet|短剧|DRM|mã hoá/i.test(e)) return "Playlet Douyin bị mã hoá — chỉ tải được trong app, bản web không hỗ trợ.";
  if (/ffmpeg|boxblur|libx264|videotoolbox/i.test(e)) return "Lỗi khi dựng/burn video (ffmpeg).";
  if (/asr/i.test(e)) return "Lỗi nhận dạng lời thoại. Thử dịch lại.";
  return e.length > 160 ? e.slice(0, 160) + "…" : e || "Lỗi không rõ.";
}

/** "bước 5/8 · Lồng tiếng" for a translating episode, or null if no progress data. */
export function dubProgressLabel(ep: {
  dub_stage: string | null;
  dub_stage_index: number | null;
  dub_stage_total: number | null;
  dub_progress?: DubProgress | null;
}): string | null {
  if (!ep.dub_stage) return null;
  const label = STAGE_VI[ep.dub_stage] ?? ep.dub_stage;
  if (ep.dub_stage === "queued") return label;
  const live = ep.dub_progress
    ? [formatProgress(ep.dub_progress), stageElapsed(ep.dub_progress)].filter(Boolean).map((part) => ` · ${part}`).join("")
    : "";
  if (ep.dub_stage_index && ep.dub_stage_total) {
    return `bước ${ep.dub_stage_index}/${ep.dub_stage_total} · ${label}${live}`;
  }
  return label + live;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

/** "Nhận dạng lời (Whisper) · 12/40 đoạn" or "34% · 158 MB/465 MB · 2.1 MB/s · còn 2:26" —
 *  parts that are unknown are left out. */
export function formatProgress(p: DubProgress): string {
  const parts: string[] = [];
  if (p.step) parts.push(p.step);
  if (p.downloaded_bytes != null) {
    if (p.total_bytes) {
      parts.push(`${Math.floor((p.downloaded_bytes / p.total_bytes) * 100)}%`);
      parts.push(`${formatBytes(p.downloaded_bytes)}/${formatBytes(p.total_bytes)}`);
    } else {
      parts.push(formatBytes(p.downloaded_bytes));
    }
    if (p.speed) parts.push(`${(p.speed / 1024 ** 2).toFixed(1)} MB/s`);
    if (p.eta) parts.push(`còn ${formatDuration(p.eta)}`);
  } else if (p.total) {
    parts.push(`${p.done ?? 0}/${p.total}${p.unit ? ` ${p.unit}` : ""}`);
  }
  if (p.sessions) parts.push(`${p.sessions} phiên đang chạy`);
  return parts.join(" · ");
}

/** How long the stage in flight has been running ("12:04"), or "" when unknown. */
export function stageElapsed(p: DubProgress | null | undefined, now: number = Date.now()): string {
  const started = p?.started_at ? Date.parse(p.started_at) : NaN;
  if (Number.isNaN(started)) return "";
  return formatDuration(Math.max(1, (now - started) / 1000));
}
