import type {
  AppSettings,
  ChannelGroup,
  ConnectionProbe,
  Episode,
  Export,
  GlossaryEntry,
  JobManifest,
  JobResponse,
  ProviderSession,
  Series,
  SeriesPage,
  SeriesSource,
  SourceVideo,
  SubtitleCue,
  TtsVoice,
} from "./types";

async function ok(response: Response, message: string) {
  if (!response.ok) throw new Error((await response.text()) || message);
  return response;
}

export async function listJobs(): Promise<JobManifest[]> {
  const response = await ok(await fetch("/api/v1/jobs"), "Không tải được danh sách công việc");
  const payload = (await response.json()) as JobResponse[];
  return payload.map((item) => item.job);
}

export async function fetchJob(jobId: string): Promise<JobManifest> {
  const response = await ok(await fetch(`/api/v1/jobs/${jobId}`), "Không tải được công việc");
  return ((await response.json()) as JobResponse).job;
}

export async function cancelJob(jobId: string): Promise<JobManifest> {
  const response = await ok(await fetch(`/api/v1/jobs/${jobId}/cancel`, { method: "POST" }), "Không dừng được công việc");
  return ((await response.json()) as JobResponse).job;
}

export async function deleteJob(jobId: string): Promise<void> {
  await ok(await fetch(`/api/v1/jobs/${jobId}`, { method: "DELETE" }), "Không xóa được công việc");
}

export async function createJob(url: string): Promise<JobManifest> {
  const response = await ok(
    await fetch("/api/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),
    "Không tạo được công việc",
  );
  return ((await response.json()) as JobResponse).job;
}

export async function fetchArtifactJson<T>(jobId: string, kind: string): Promise<T | null> {
  const response = await fetch(`/api/v1/jobs/${jobId}/artifacts/${kind}`);
  if (!response.ok) return null;
  return (await response.json()) as T;
}

export async function loadCues(jobId: string): Promise<SubtitleCue[]> {
  const payload = await fetchArtifactJson<SubtitleCue[]>(jobId, "subtitle_json");
  return Array.isArray(payload) ? payload : [];
}

export async function loadGlossary(jobId: string): Promise<GlossaryEntry[]> {
  const payload = await fetchArtifactJson<GlossaryEntry[]>(jobId, "glossary");
  return Array.isArray(payload) ? payload : [];
}

export async function loadProviders(): Promise<ProviderSession[]> {
  const response = await fetch("/api/v1/providers/sessions");
  if (!response.ok) return [];
  return ((await response.json()) as { sessions: ProviderSession[] }).sessions;
}

export async function saveCookies(platform: string, cookies: string): Promise<void> {
  await ok(
    await fetch(`/api/v1/providers/${platform}/cookies`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cookies }),
    }),
    "Không lưu được cookie",
  );
}

export async function loadAppSettings(): Promise<AppSettings> {
  const response = await ok(await fetch("/api/v1/settings"), "Không tải được cài đặt");
  return (await response.json()) as AppSettings;
}

export async function saveAppSettings(settings: AppSettings): Promise<AppSettings> {
  const response = await fetch("/api/v1/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : "Không lưu được cài đặt");
  }
  return (await response.json()) as AppSettings;
}

export async function testConnection(id: string): Promise<ConnectionProbe> {
  const response = await ok(
    await fetch(`/api/v1/settings/connections/${encodeURIComponent(id)}/test`, { method: "POST" }),
    "Không kiểm tra được kết nối",
  );
  return (await response.json()) as ConnectionProbe;
}

export async function listConnectionModels(id: string): Promise<string[]> {
  const response = await fetch(`/api/v1/settings/connections/${encodeURIComponent(id)}/models`);
  if (!response.ok) return [];
  return ((await response.json()) as { models: string[] }).models;
}

export async function listTtsVoices(connection: string, type = ""): Promise<TtsVoice[]> {
  const query = new URLSearchParams({ connection, ...(type ? { type } : {}) });
  const response = await ok(
    await fetch(`/api/v1/settings/tts/voices?${query}`),
    "Không tải được danh sách giọng",
  );
  return ((await response.json()) as { voices: TtsVoice[] }).voices;
}

/** Synthesize a short sample with the (saved) connection and the draft voice/model;
 *  a keyless `type` (edge, vieneu) needs no saved connection. */
export async function previewTts(body: { connection: string; voice: string; model: string; type?: string }): Promise<Blob> {
  const response = await fetch("/api/v1/settings/tts/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(typeof payload?.detail === "string" ? payload.detail : "Không nghe thử được giọng");
  }
  return response.blob();
}

export async function discoverVideos(url: string): Promise<SourceVideo[]> {
  const response = await ok(
    await fetch("/api/v1/studio/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, limit: 18 }),
    }),
    "Không quét được nguồn",
  );
  return ((await response.json()) as { videos: SourceVideo[] }).videos;
}

/* ---------- library ---------- */

const LIB = "/api/v1/library";
const jsonHeaders = { "Content-Type": "application/json" };

export async function listSeries(q = ""): Promise<SeriesPage> {
  const response = await ok(await fetch(`${LIB}/series?q=${encodeURIComponent(q)}`), "Không tải được thư viện");
  return (await response.json()) as SeriesPage;
}

export async function previewChannel(url: string): Promise<ChannelGroup[]> {
  const response = await ok(
    await fetch(`${LIB}/preview-channel`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ url }) }),
    "Không quét được kênh",
  );
  return ((await response.json()) as { groups: ChannelGroup[] }).groups;
}

export async function addVideo(url: string, seriesId?: string, title?: string): Promise<{ series_id: string; episode_id: string | null }> {
  const response = await ok(
    await fetch(`${LIB}/add-video`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ url, series_id: seriesId ?? null, title: title ?? null }) }),
    "Không thêm được video",
  );
  return (await response.json()) as { series_id: string; episode_id: string | null };
}

export async function importChannel(
  url: string,
  groups: { title: string; series_id: string | null; episodes: unknown[] }[],
): Promise<string[]> {
  const response = await ok(
    await fetch(`${LIB}/import-channel`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ url, groups }) }),
    "Không tạo được bộ",
  );
  return ((await response.json()) as { series_ids: string[] }).series_ids;
}

export async function createSeries(title: string, sources: { platform: string; url: string }[] = []): Promise<Series> {
  const response = await ok(
    await fetch(`${LIB}/series`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ title, sources }) }),
    "Không tạo được series",
  );
  return (await response.json()) as Series;
}

export async function getSeries(id: string): Promise<Series> {
  const response = await ok(await fetch(`${LIB}/series/${id}`), "Không tải được series");
  return (await response.json()) as Series;
}

export async function deleteSeries(id: string): Promise<void> {
  await ok(await fetch(`${LIB}/series/${id}`, { method: "DELETE" }), "Không xóa được series");
}

export async function addSource(seriesId: string, platform: string, url: string): Promise<SeriesSource> {
  const response = await ok(
    await fetch(`${LIB}/series/${seriesId}/sources`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ platform, url }) }),
    "Không thêm được nguồn",
  );
  return (await response.json()) as SeriesSource;
}

export async function deleteSource(seriesId: string, sourceId: string): Promise<void> {
  await ok(await fetch(`${LIB}/series/${seriesId}/sources/${sourceId}`, { method: "DELETE" }), "Không xóa được nguồn");
}

export async function scanSeries(seriesId: string): Promise<{ added: number; total: number }> {
  const response = await ok(await fetch(`${LIB}/series/${seriesId}/scan`, { method: "POST" }), "Không quét được kênh");
  return (await response.json()) as { added: number; total: number };
}

export async function listEpisodes(seriesId: string): Promise<Episode[]> {
  const response = await ok(await fetch(`${LIB}/series/${seriesId}/episodes`), "Không tải được tập");
  return (await response.json()) as Episode[];
}

export async function translateEpisodes(seriesId: string, episodeIds: string[]): Promise<string[]> {
  const response = await ok(
    await fetch(`${LIB}/series/${seriesId}/translate`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ episode_ids: episodeIds }) }),
    "Không tạo được job dịch",
  );
  return ((await response.json()) as { jobs: string[] }).jobs;
}

export async function translateTitles(seriesId: string): Promise<number> {
  const response = await ok(await fetch(`${LIB}/series/${seriesId}/translate-titles`, { method: "POST" }), "Không dịch được tiêu đề");
  return ((await response.json()) as { translated: number }).translated;
}

export async function deleteEpisodeDub(episodeId: string): Promise<void> {
  await ok(await fetch(`${LIB}/episodes/${episodeId}/dub`, { method: "DELETE" }), "Không xóa được bản dịch");
}

export async function listExports(seriesId: string): Promise<Export[]> {
  const response = await ok(await fetch(`${LIB}/series/${seriesId}/exports`), "Không tải được bản ghép");
  return (await response.json()) as Export[];
}

export async function createExport(
  seriesId: string,
  body: { title: string; description: string; hashtags: string[]; episode_job_ids: string[] },
): Promise<Export> {
  const response = await ok(
    await fetch(`${LIB}/series/${seriesId}/exports`, { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }),
    "Không tạo được bản ghép",
  );
  return (await response.json()) as Export;
}

export async function suggestExport(
  seriesId: string,
  episodeIds: string[],
): Promise<{ title: string; description: string; hashtags: string[] }> {
  const response = await ok(
    await fetch(`${LIB}/series/${seriesId}/suggest-export`, { method: "POST", headers: jsonHeaders, body: JSON.stringify({ episode_ids: episodeIds }) }),
    "Không tạo được gợi ý",
  );
  return (await response.json()) as { title: string; description: string; hashtags: string[] };
}

export async function renderExport(exportId: string): Promise<Export> {
  const response = await ok(await fetch(`${LIB}/exports/${exportId}/render`, { method: "POST" }), "Không render được");
  return (await response.json()) as Export;
}

export async function getExport(exportId: string): Promise<Export> {
  const response = await ok(await fetch(`${LIB}/exports/${exportId}`), "Không tải được bản ghép");
  return (await response.json()) as Export;
}

export async function deleteExport(exportId: string): Promise<void> {
  await ok(await fetch(`${LIB}/exports/${exportId}`, { method: "DELETE" }), "Không xóa được bản ghép");
}


/* ---------- overlay editor (caption position + cover boxes) ---------- */

export type CoverRegion = {
  x: number;
  y: number;
  w: number;
  h: number;
  mode: string;
  strength: number;
  label: string;
  /** mode "image": the library file name (fetchOverlayImages) */
  image?: string;
};

export type OverlayConfig = {
  caption_mode: string;
  caption_y: number | null;
  caption_scale: number;
  cover: CoverRegion[];
  /** caption font file (one of fetchFonts); "" = the app default */
  font?: string;
  /** video bitrate for the render ("2500k", "4M"); "" = the app setting */
  bitrate?: string;
  /** original audio under the dub (0-1); null = the app setting (frame.bed_volume) */
  bed_volume?: number | null;
  /** the dub itself (0-1) */
  voice_volume?: number;
  /** original audio where nobody speaks (0-1); null = the app setting (frame.gap_volume) */
  gap_volume?: number | null;
};

export type BitratePreset = { value: string; label: string; hint: string; vmaf: number };

export type OverlayEncode = {
  source_codec: string;
  source_bitrate: number;
  /** the app's default render bitrate (one of presets, normally) */
  default: string;
  audio_bitrate: number;
  presets: BitratePreset[];
};

export type FontInfo = { path: string; name: string; source: "app" | "user" | "system" };

export async function fetchFonts(): Promise<{ default: string; fonts: FontInfo[] }> {
  const response = await ok(await fetch("/api/v1/settings/fonts"), "Không tải được danh sách font");
  return (await response.json()) as { default: string; fonts: FontInfo[] };
}

export type OverlayImage = { name: string; width: number; height: number };

/** The user's images (logo, watermark…) to stamp on renders; files live under ``base``. */
export async function fetchOverlayImages(): Promise<{ base: string; images: OverlayImage[] }> {
  const response = await ok(await fetch("/api/v1/settings/images"), "Không tải được thư viện ảnh");
  return (await response.json()) as { base: string; images: OverlayImage[] };
}

export async function uploadOverlayImage(file: File): Promise<OverlayImage> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/v1/settings/images", { method: "POST", body });
  if (!response.ok) {
    const detail = await response.json().then((data: { detail?: unknown }) => data.detail, () => null);
    throw new Error(typeof detail === "string" ? detail : "Không tải ảnh lên được");
  }
  return (await response.json()) as OverlayImage;
}

export function fontFileUrl(path: string): string {
  return `/api/v1/settings/fonts/file?path=${encodeURIComponent(path)}`;
}

export type OverlayFrame = {
  out: { width: number; height: number };
  source: { width: number; height: number };
  scale: number;
  pad: { x: number; y: number };
  font_size?: number;
  duration: number;
  encode?: OverlayEncode;
  /** the app's default level for the original audio under the dub */
  bed_volume?: number;
  /** the app's default level for the original audio where nobody speaks */
  gap_volume?: number;
};

export type OverlayState = OverlayConfig & { frame?: OverlayFrame };

export async function fetchOverlay(jobId: string): Promise<OverlayState> {
  const response = await ok(await fetch(`/api/v1/jobs/${jobId}/overlay`), "Không tải được khung che");
  return (await response.json()) as OverlayState;
}

export async function saveOverlay(jobId: string, config: OverlayConfig): Promise<OverlayState> {
  const response = await ok(
    await fetch(`/api/v1/jobs/${jobId}/overlay`, { method: "PUT", headers: jsonHeaders, body: JSON.stringify(config) }),
    "Không lưu được khung che",
  );
  return (await response.json()) as OverlayState;
}

export async function previewOverlayFrame(jobId: string, config: OverlayConfig, t: number, text: string): Promise<string> {
  const response = await ok(
    await fetch(`/api/v1/jobs/${jobId}/overlay/frame?t=${encodeURIComponent(t)}&text=${encodeURIComponent(text)}`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(config),
    }),
    "Không dựng được ảnh xem thử",
  );
  return URL.createObjectURL(await response.blob());
}

export async function repackJob(jobId: string): Promise<JobManifest> {
  const response = await ok(await fetch(`/api/v1/jobs/${jobId}/repack`, { method: "POST" }), "Không render lại được");
  return ((await response.json()) as JobResponse).job;
}

/** Direct download URL for a job artifact (hardsub_mp4, subtitle_srt, …). */
export function artifactUrl(jobId: string, kind: string): string {
  return `/api/v1/jobs/${jobId}/artifacts/${kind}`;
}
