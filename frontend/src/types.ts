export type JobState = "queued" | "running" | "succeeded" | "failed" | "canceled" | "review";
export type StageStatus = "pending" | "running" | "succeeded" | "failed" | "skipped";

export type Artifact = {
  kind: string;
  path: string;
  mime_type: string;
  size_bytes: number;
  created_at: string;
};

export type Stage = {
  name: string;
  status: StageStatus;
  started_at: string | null;
  ended_at: string | null;
  artifacts: Artifact[];
  metrics: Record<string, unknown>;
  logs: string[];
  error: string | null;
};

export type JobManifest = {
  job_id: string;
  input: { url: string };
  state: JobState;
  config?: { stages?: string[] };
  stages: Stage[];
  artifacts: Artifact[];
  metrics?: Record<string, Record<string, unknown>>;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type JobResponse = { job: JobManifest };

export type SubtitleCue = {
  start?: number;
  end?: number;
  translated?: string;
  text?: string;
  source?: string;
  anchor_id?: number;
};

export type GlossaryEntry = { term_zh: string; term_vi: string; kind: string; note: string };

export type ProviderSession = {
  platform: string;
  label: string;
  configured: boolean;
  updated_at: string | null;
  cookie_count: number;
  cookie_names: string[];
  required_missing: string[];
  recommended_missing: string[];
  note: string;
};

export type SourceVideo = {
  id: string;
  platform: string;
  platform_video_id: string;
  source_url: string;
  title: string;
  thumbnail_url: string | null;
  duration: number | null;
  uploader: string | null;
  download_state: string;
  error?: string | null;
};

/* ---------- library ---------- */

export type DubState = "none" | "translating" | "review" | "dubbed" | "failed";
export type ExportState = "draft" | "rendering" | "done" | "failed";

export type SeriesSource = {
  id: string;
  series_id: string;
  platform: string;
  url: string;
  label: string | null;
  last_scanned_at: string | null;
};

export type Series = {
  id: string;
  title: string;
  cover_url: string | null;
  tags: string[];
  created_at: string;
  updated_at: string;
  episode_count: number;
  dubbed_count: number;
  export_count: number;
  sources: SeriesSource[];
};

export type Episode = {
  id: string;
  series_id: string;
  platform: string;
  platform_video_id: string;
  source_url: string;
  title: string;
  title_vi: string | null;
  thumbnail_url: string | null;
  duration: number | null;
  uploader: string | null;
  discovered_at: string;
  dub_job_id: string | null;
  dub_state: DubState;
  dub_stage: string | null;
  dub_stage_index: number | null;
  dub_stage_total: number | null;
  dub_error: string | null;
  dub_progress: DubProgress | null;
};

/** Live state of the stage in flight. Every key but ``stage`` is optional: a stage
 *  reports what it knows (a sub-step, counts, download bytes, recent log lines). */
export type DubProgress = {
  stage: string;
  started_at?: string | null; // ISO; drives the elapsed clock
  step?: string; // sub-step in flight, e.g. "Nhận dạng lời (Whisper)"
  done?: number;
  total?: number;
  unit?: string; // what done/total count ("đoạn", "dòng", "câu TTS")
  sessions?: number; // translate: agent sessions in flight
  logs?: string[]; // last few log lines of the stage
  downloaded_bytes?: number;
  total_bytes?: number | null;
  speed?: number | null; // bytes/s
  eta?: number | null; // seconds
};

export type Export = {
  id: string;
  series_id: string;
  title: string;
  description: string;
  hashtags: string[];
  episode_job_ids: string[];
  output_path: string | null;
  duration: number | null;
  state: ExportState;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type SeriesPage = { items: Series[]; total: number };

export type ChannelGroup = {
  title: string;
  title_vi: string | null;        // LLM-suggested Vietnamese name for the scan review
  platform: string;
  series_id: string | null;       // matched existing series, or null = new
  existing_title: string | null;
  new_count: number;              // episodes not yet in the matched series
  total: number;
  episodes: Record<string, unknown>[];
};

// ----- provider/model settings (secrets arrive masked: "••••" + last 4) -----
export type ConnectionType = "agy" | "claude_code" | "codex" | "command" | "openai_compat" | "gemini" | "elevenlabs" | "capcut" | "edge" | "http_tts" | "vieneu";

export type Connection = {
  id: string;
  label: string;
  type: ConnectionType;
  base_url: string;
  api_key: string;
  bin: string;
  home: string;
  command: string;
  email: string;
  password: string;
};

export type ChatTask = { connection: string; model: string; effort: string };
export type OcrTask = { recognizer: "vision" | "llm"; connection: string; model: string };
export type TtsTask = { enabled: boolean; connection: string; model: string; voice: string; synth_rate: number };
export type AsrTask = { backend: "auto" | "qwen" | "local" | "worker"; worker_url: string; model: string; ocr_first: boolean };

export type ChatTaskName = "translate" | "glossary" | "normalize" | "repair" | "naming";

export type AppSettings = {
  version: number;
  connections: Connection[];
  tasks: Record<ChatTaskName, ChatTask> & { ocr: OcrTask; tts: TtsTask; asr: AsrTask };
};

export type ConnectionProbe = { ok: boolean; message: string };
export type TtsVoice = { id: string; name: string };
