// Settings <-> "by job" view. The settings file stays connections + tasks; the modal
// talks in services ("Gemini API", "CapCut") and drives ONE connection per service
// type — the first of that type. Others (hand-edited JSON) are kept untouched.
// Pure functions only: every helper returns a new AppSettings and never mutates.
import type { AppSettings, ChatTask, ChatTaskName, Connection, ConnectionType } from "../types";

export const CHAT_SUBTASKS: ChatTaskName[] = ["glossary", "normalize", "repair", "naming"];
const CHAT_TASKS: ChatTaskName[] = ["translate", ...CHAT_SUBTASKS];
const VISION_TYPES: ConnectionType[] = ["gemini", "openai_compat"];
const MASK = "••••";

/** Subscription CLI agents and the binary they run when `bin` is blank. */
export const CLI_BINS: Partial<Record<ConnectionType | "", string>> = { agy: "agy", claude_code: "claude", codex: "codex" };

/** Name a new connection gets in settings.json (the UI never shows it). */
export const SERVICE_NAMES: Record<ConnectionType, string> = {
  agy: "Antigravity",
  claude_code: "Claude Code",
  codex: "Codex CLI",
  command: "Lệnh CLI tuỳ chỉnh",
  openai_compat: "OpenAI-compatible",
  gemini: "Gemini API",
  elevenlabs: "ElevenLabs",
  capcut: "CapCut",
  edge: "Edge TTS",
  http_tts: "Máy chủ giọng đọc HTTP",
  vieneu: "VieNeu TTS",
};

export const firstOfType = (settings: AppSettings, type: ConnectionType): Connection | undefined =>
  settings.connections.find((conn) => conn.type === type);

export const connectionOf = (settings: AppSettings, id: string): Connection | undefined =>
  settings.connections.find((conn) => conn.id === id);

/** A stored key comes back masked ("••••abcd"); either way the service has one. */
export const hasKey = (conn: Connection | undefined): boolean => !!conn && conn.api_key.trim() !== "";

export const isMasked = (value: string): boolean => value.startsWith(MASK);

export function newConnectionId(existing: Connection[]): string {
  for (;;) {
    const id = `c-${Math.random().toString(36).slice(2, 8).padEnd(6, "0")}`;
    if (!existing.some((conn) => conn.id === id)) return id;
  }
}

/** The connection the UI uses for a service type, created on first use. */
export function ensureConnection(settings: AppSettings, type: ConnectionType): { settings: AppSettings; id: string } {
  const found = firstOfType(settings, type);
  if (found) return { settings, id: found.id };
  const conn: Connection = {
    id: newConnectionId(settings.connections), label: SERVICE_NAMES[type], type,
    base_url: "", api_key: "", bin: "", home: "", command: "", email: "", password: "",
  };
  return { settings: { ...settings, connections: [...settings.connections, conn] }, id: conn.id };
}

/** Edit one connection's fields (the one a task is bound to). A new API address
 *  drops the stored key (the server would not send it there either). */
export function patchConnection(settings: AppSettings, id: string, change: Partial<Connection>): AppSettings {
  return {
    ...settings,
    connections: settings.connections.map((conn) => {
      if (conn.id !== id) return conn;
      const moved = change.base_url !== undefined && change.base_url !== conn.base_url && isMasked(conn.api_key);
      return { ...conn, ...change, ...(moved ? { api_key: "" } : {}) };
    }),
  };
}

/** Edit a service's fields (key, address, account…) — creates its connection if needed. */
export function patchService(settings: AppSettings, type: ConnectionType, change: Partial<Connection>): AppSettings {
  const { settings: next, id } = ensureConnection(settings, type);
  return patchConnection(next, id, change);
}

function withTasks(settings: AppSettings, change: Partial<AppSettings["tasks"]>): AppSettings {
  return { ...settings, tasks: { ...settings.tasks, ...change } };
}

// ----- Dịch thuật ---------------------------------------------------------------
export const chatService = (settings: AppSettings): ConnectionType | "" =>
  connectionOf(settings, settings.tasks.translate.connection)?.type ?? "";

const sameBinding = (a: ChatTask, b: ChatTask) => a.connection === b.connection && a.model === b.model && a.effort === b.effort;

/** A subtask "giống Dịch" = bound exactly like translate. */
export const followsTranslate = (settings: AppSettings, task: ChatTaskName): boolean =>
  sameBinding(settings.tasks[task], settings.tasks.translate);

/** Point every chat task at the service. Re-choosing the same service changes nothing;
 *  another service starts clean — model names belong to a service, so per-step
 *  overrides go back to "giống Dịch" instead of carrying an id the service can't run. */
export function setChatService(settings: AppSettings, type: ConnectionType): AppSettings {
  if (chatService(settings) === type) return settings;
  const { settings: next, id } = ensureConnection(settings, type);
  const translate: ChatTask = { ...settings.tasks.translate, connection: id, model: "" };
  const tasks: Partial<AppSettings["tasks"]> = { translate };
  for (const task of CHAT_SUBTASKS) tasks[task] = { ...translate };
  return withTasks(next, tasks);
}

/** Change Dịch's model/effort; subtasks following it move along unless pinned. */
export function setTranslate(settings: AppSettings, change: Partial<Omit<ChatTask, "connection">>, pinned?: ChatTaskName[]): AppSettings {
  const translate = { ...settings.tasks.translate, ...change };
  const tasks: Partial<AppSettings["tasks"]> = { translate };
  for (const task of CHAT_SUBTASKS) {
    const follows = pinned ? !pinned.includes(task) : followsTranslate(settings, task);
    if (follows) tasks[task] = { ...translate };
  }
  return withTasks(settings, tasks);
}

/** Override a subtask's model/effort, or `null` = back to "giống Dịch". */
export function setSubtask(settings: AppSettings, task: ChatTaskName, override: { model: string; effort: string } | null): AppSettings {
  const translate = settings.tasks.translate;
  const binding = override ? { ...settings.tasks[task], ...override } : { ...translate };
  return withTasks(settings, { [task]: binding });
}

export const usesChatService = (settings: AppSettings, connId: string): boolean =>
  CHAT_TASKS.some((task) => settings.tasks[task].connection === connId);

// ----- Giọng đọc -------------------------------------------------------------------
/** Voice services with nothing to fill in: voices + preview work before saving. */
export const KEYLESS_TTS: ConnectionType[] = ["edge", "vieneu"];

export const ttsService = (settings: AppSettings): ConnectionType | "" =>
  connectionOf(settings, settings.tasks.tts.connection)?.type ?? "";

/** Another voice service = its default voice/model (voices differ per engine). */
export function setTtsService(settings: AppSettings, type: ConnectionType): AppSettings {
  const { settings: next, id } = ensureConnection(settings, type);
  const tts = settings.tasks.tts;
  const same = ttsService(settings) === type;
  return withTasks(next, { tts: { ...tts, connection: id, voice: same ? tts.voice : "", model: same ? tts.model : "" } });
}

// ----- Phụ đề (OCR) ------------------------------------------------------------------
export const ocrService = (settings: AppSettings): ConnectionType | "" =>
  connectionOf(settings, settings.tasks.ocr.connection)?.type ?? "";

export function setOcrService(settings: AppSettings, type: ConnectionType): AppSettings {
  const { settings: next, id } = ensureConnection(settings, type);
  const ocr = settings.tasks.ocr;
  return withTasks(next, { ocr: { ...ocr, recognizer: "llm", connection: id, model: ocrService(settings) === type ? ocr.model : "" } });
}

/** Reading subtitles by AI reuses an API service already set up: the translation one,
 *  else one that has a key, else Gemini. */
export function setOcrRecognizer(settings: AppSettings, recognizer: "vision" | "llm"): AppSettings {
  const ocr = settings.tasks.ocr;
  if (recognizer === "vision") return withTasks(settings, { ocr: { ...ocr, recognizer } });
  const current = ocrService(settings);
  if (current && VISION_TYPES.includes(current)) return withTasks(settings, { ocr: { ...ocr, recognizer } });
  const chat = chatService(settings);
  const type = chat && VISION_TYPES.includes(chat)
    ? chat
    : VISION_TYPES.find((candidate) => hasKey(firstOfType(settings, candidate))) ?? "gemini";
  return setOcrService(settings, type);
}

// ----- Chưa đủ để chạy ----------------------------------------------------------------
export type SettingsTab = "translation" | "voice" | "subtitles";
export type Problem = { tab: SettingsTab; message: string };

const API_TYPES: ConnectionType[] = ["gemini", "openai_compat", "elevenlabs"];

/** What a job would fail on with these settings (saving is still allowed: the model
 *  list of a new service only loads once it is saved). */
export function problems(settings: AppSettings): Problem[] {
  const found: Problem[] = [];
  const add = (tab: SettingsTab, message: string) => found.push({ tab, message });
  const needsAccess = (tab: SettingsTab, conn: Connection | undefined, label: string) => {
    if (!conn) return;
    if (conn.type === "openai_compat" && !conn.base_url.trim()) add(tab, `${label}: chưa nhập địa chỉ API`);
    else if (conn.type === "openai_compat" && !/^https?:\/\//i.test(conn.base_url.trim()))
      add(tab, `${label}: địa chỉ API phải bắt đầu bằng http:// hoặc https://`);
    if (API_TYPES.includes(conn.type) && !hasKey(conn)) add(tab, `${label}: chưa nhập API key`);
  };
  const { tasks } = settings;

  const chat = connectionOf(settings, tasks.translate.connection);
  needsAccess("translation", chat, "Dịch thuật");
  if (chat?.type === "command" && !chat.command.includes("{prompt}")) add("translation", "Dịch thuật: mẫu lệnh phải chứa {prompt}");
  if (chat && API_TYPES.includes(chat.type)) {
    if (!tasks.translate.model.trim()) add("translation", "Dịch thuật: chưa chọn model");
    else if (CHAT_SUBTASKS.some((task) => !tasks[task].model.trim())) add("translation", "Dịch thuật: có bước dùng model riêng nhưng chưa nhập tên");
  }

  if (tasks.tts.enabled) {
    const tts = connectionOf(settings, tasks.tts.connection);
    if (!tts) add("voice", "Giọng đọc: chưa chọn dịch vụ");
    needsAccess("voice", tts, "Giọng đọc");
    if (tts?.type === "http_tts" && !tts.base_url.trim()) add("voice", "Giọng đọc: chưa nhập địa chỉ máy chủ");
    if ((tts?.type === "gemini" || tts?.type === "openai_compat") && !tasks.tts.model.trim()) add("voice", "Giọng đọc: chưa chọn model giọng đọc");
    if (tts?.type === "elevenlabs" && !tasks.tts.voice.trim()) add("voice", "Giọng đọc: ElevenLabs cần chọn giọng");
  }

  if (tasks.asr.ocr_first && tasks.ocr.recognizer === "llm") {
    const ocr = connectionOf(settings, tasks.ocr.connection);
    needsAccess("subtitles", ocr, "Đọc phụ đề bằng AI");
    if (!tasks.ocr.model.trim()) add("subtitles", "Đọc phụ đề bằng AI: chưa chọn model");
  }
  if (tasks.asr.backend === "worker" && !tasks.asr.worker_url.trim()) add("subtitles", "Speech-to-Text: chưa nhập địa chỉ máy chủ");
  return found;
}
