import { useEffect, useRef, useState } from "react";
import * as api from "../api";
import { useTtsVoices } from "../queries";
import { useToast } from "../toast";
import type { ConnectionType } from "../types";
import { Advanced, CheckButton, Field, isSaved, ModelField, SecretField, Section, TextField, type TabProps } from "./fields";
import { connectionOf, KEYLESS_TTS, patchConnection, setTtsService, ttsService, usesChatService } from "./model";

const SERVICES: { type: ConnectionType; title: string; description: string }[] = [
  { type: "capcut", title: "CapCut", description: "Giọng tiếng Việt tự nhiên của CapCut — đăng nhập bằng tài khoản CapCut." },
  { type: "vieneu", title: "VieNeu (máy này)", description: "Giọng tiếng Việt chạy ngay trên máy này, miễn phí, không cần mạng sau lần tải model đầu (~1.2 GB)." },
  { type: "edge", title: "Edge TTS (miễn phí)", description: "Giọng Microsoft Edge, miễn phí, không cần tài khoản." },
  { type: "openai_compat", title: "OpenAI-compatible", description: "API giọng đọc chuẩn OpenAI (dùng key đã nhập)." },
  { type: "gemini", title: "Gemini", description: "Giọng đọc của Gemini (dùng key Gemini đã nhập)." },
  { type: "elevenlabs", title: "ElevenLabs", description: "Giọng ElevenLabs theo tài khoản của bạn." },
  { type: "http_tts", title: "Máy chủ HTTP", description: "Máy chủ giọng đọc riêng của bạn (giao thức CapCut)." },
];

/** Giọng đọc: on/off, which voice service, its account/key, voice + preview. */
export function VoiceTab({ draft, setDraft, saved }: TabProps) {
  const tts = draft.tasks.tts;
  const service = ttsService(draft);
  const conn = connectionOf(draft, tts.connection);
  const keyless = !!service && KEYLESS_TTS.includes(service);
  const ready = isSaved(saved, conn) || keyless;
  const setTts = (change: Partial<typeof tts>) => setDraft({ ...draft, tasks: { ...draft.tasks, tts: { ...tts, ...change } } });
  const patch = (change: Parameters<typeof patchConnection>[2]) => conn && setDraft(patchConnection(draft, conn.id, change));
  const info = SERVICES.find((item) => item.type === service);
  const needsModel = service === "openai_compat" || service === "gemini";
  const shared = conn && usesChatService(draft, conn.id) ? "dùng chung với Dịch thuật" : undefined;

  return (
    <div className="settings-list">
      <label className="radio-row toggle-row">
        <input type="checkbox" checked={tts.enabled} onChange={(e) => setTts({ enabled: e.target.checked })} />
        Bật lồng tiếng (đọc bản dịch thành giọng tiếng Việt)
      </label>
      <Section title="Dịch vụ giọng đọc">
        <Field label="Dịch vụ" hint={info?.description}>
          <select className="set-input" value={service} onChange={(e) => setDraft(setTtsService(draft, e.target.value as ConnectionType))}>
            {!service ? <option value="">— chọn —</option> : null}
            {SERVICES.map((item) => <option key={item.type} value={item.type}>{item.title}</option>)}
          </select>
        </Field>
        {service === "capcut" ? (
          <>
            <TextField label="Email CapCut" value={conn?.email ?? ""} onChange={(email) => patch({ email })} />
            <SecretField key={conn?.id} label="Mật khẩu CapCut" conn={conn} field="password" onChange={(password) => patch({ password })} />
          </>
        ) : null}
        {service === "http_tts" ? (
          <TextField label="Địa chỉ máy chủ" value={conn?.base_url ?? ""} placeholder="http://127.0.0.1:8000" onChange={(base_url) => patch({ base_url })} />
        ) : null}
        {service === "openai_compat" ? (
          <TextField label="Địa chỉ API" value={conn?.base_url ?? ""} placeholder="https://api.openai.com/v1" hint={shared} onChange={(base_url) => patch({ base_url })} />
        ) : null}
        {service === "openai_compat" || service === "gemini" || service === "elevenlabs" ? (
          <SecretField key={conn?.id} label="API key" conn={conn} field="api_key" hint={shared} onChange={(api_key) => patch({ api_key })} />
        ) : null}
        {needsModel ? (
          <ModelField label="Model giọng đọc" conn={conn} saved={saved} value={tts.model} required
            placeholder={service === "gemini" ? "model *-tts" : "vd. tts-1"} onChange={(model) => setTts({ model })} />
        ) : null}
        {service ? <VoicePicker draft={draft} setDraft={setDraft} saved={saved} ready={ready} keylessType={keyless ? service : ""} /> : null}
        {service ? <CheckButton key={conn?.id} conn={conn} saved={saved} /> : null}
      </Section>
      <Advanced>
        <Field label="Tốc độ sau tổng hợp" hint="1 = giữ nguyên; >1 đọc nhanh hơn">
          <input className="set-input" type="number" min={0.5} max={2} step={0.05} value={tts.synth_rate}
            onChange={(e) => Number.isFinite(e.target.valueAsNumber) && setTts({ synth_rate: e.target.valueAsNumber })} />
        </Field>
        {service === "capcut" ? (
          <TextField label="Máy chủ HTTP dự phòng" value={conn?.base_url ?? ""} placeholder="dùng khi chưa có phiên CapCut (tuỳ chọn)"
            onChange={(base_url) => patch({ base_url })} />
        ) : null}
        {service === "elevenlabs" ? (
          <ModelField label="Model giọng đọc" conn={conn} saved={saved} value={tts.model} placeholder="để trống = mặc định" onChange={(model) => setTts({ model })} />
        ) : null}
      </Advanced>
    </div>
  );
}

/** Voice list with a ▶ per voice: a sample is synthesized once per voice, then replayed. */
function VoicePicker({ draft, setDraft, ready, keylessType }: TabProps & { ready: boolean; keylessType: string }) {
  const tts = draft.tasks.tts;
  const { data: voices = [], isError } = useTtsVoices(ready ? tts.connection : "", keylessType);
  const { toast } = useToast();
  const [sample, setSample] = useState<{ voice: string; phase: "load" | "play" } | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const samples = useRef(new Map<string, Blob>());
  const setVoice = (voice: string) => setDraft({ ...draft, tasks: { ...draft.tasks, tts: { ...tts, voice } } });
  const elevenlabs = ttsService(draft) === "elevenlabs";

  function release() {
    if (!audio.current) return;
    audio.current.pause();
    URL.revokeObjectURL(audio.current.src);
    audio.current = null;
  }
  useEffect(() => release, []);

  async function preview(voice: string) {
    release();
    if (sample?.voice === voice && sample.phase === "play") return setSample(null);
    setSample({ voice, phase: "load" });
    try {
      const key = `${tts.connection}|${keylessType}|${tts.model}|${voice}`;
      const blob = samples.current.get(key) ?? (await api.previewTts({ connection: tts.connection, voice, model: tts.model, type: keylessType }));
      samples.current.set(key, blob);
      const player = new Audio(URL.createObjectURL(blob));
      audio.current = player;
      player.onended = () => {
        if (audio.current !== player) return;
        release();
        setSample(null);
      };
      await player.play();
      setSample({ voice, phase: "play" });
    } catch (error) {
      release();
      setSample(null);
      toast(error instanceof Error ? error.message : "Không nghe thử được giọng", "error");
    }
  }

  const hint = !ready ? "Lưu để tải danh sách giọng và nghe thử" : isError ? "Không tải được danh sách giọng — gõ mã giọng" : elevenlabs && !tts.voice ? "ElevenLabs cần chọn giọng" : undefined;
  const loading = sample?.phase === "load";
  const playButton = (voice: string, name: string) => (
    <button type="button" className="ghost voice-play" onClick={() => preview(voice)} disabled={!ready || loading}
      title={ready ? `Nghe thử ${name}` : "Lưu trước khi nghe thử"} aria-label={`Nghe thử ${name}`}>
      {sample?.voice === voice ? (loading ? "…" : "■") : "▶"}
    </button>
  );

  if (!voices.length) {
    return (
      <Field label="Giọng" hint={hint}>
        <span className="voice-row">
          <input className="set-input" value={tts.voice} placeholder="để trống = giọng mặc định" autoComplete="off" onChange={(e) => setVoice(e.target.value)} />
          {playButton(tts.voice, "giọng đang chọn")}
        </span>
      </Field>
    );
  }
  const rows = [
    ...(elevenlabs ? [] : [{ id: "", name: "Mặc định" }]),
    ...(tts.voice && !voices.some((voice) => voice.id === tts.voice) ? [{ id: tts.voice, name: tts.voice }] : []),
    ...voices,
  ];
  return (
    <div className="set-field">
      <span>Giọng</span>
      <div className="voice-list" role="radiogroup" aria-label="Giọng">
        {rows.map((voice) => (
          <div key={voice.id || "default"} className={`voice-item${voice.id === tts.voice ? " on" : ""}`}>
            <label className="voice-pick">
              <input type="radio" name="tts-voice" checked={voice.id === tts.voice} onChange={() => setVoice(voice.id)} />
              <span>{voice.name}</span>
            </label>
            {playButton(voice.id, voice.name)}
          </div>
        ))}
      </div>
      {hint ? <em className="faint small">{hint}</em> : null}
    </div>
  );
}
