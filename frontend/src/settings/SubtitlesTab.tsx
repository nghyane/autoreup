import type { ConnectionType } from "../types";
import { Advanced, Field, ModelField, SecretField, Section, ServiceChoice, TextField, type TabProps } from "./fields";
import { connectionOf, ocrService, patchConnection, setOcrRecognizer, setOcrService, usesChatService } from "./model";

const OCR_SERVICES: { type: ConnectionType; title: string }[] = [
  { type: "gemini", title: "Gemini API" },
  { type: "openai_compat", title: "OpenAI-compatible" },
];

/** Phụ đề & lời thoại: where the spoken lines come from (on-screen subtitles first,
 *  Speech-to-Text fills the gaps), how subtitles are read, and where STT runs. */
export function SubtitlesTab({ draft, setDraft, saved }: TabProps) {
  const { asr, ocr } = draft.tasks;
  const setAsr = (change: Partial<typeof asr>) => setDraft({ ...draft, tasks: { ...draft.tasks, asr: { ...asr, ...change } } });
  const setOcr = (change: Partial<typeof ocr>) => setDraft({ ...draft, tasks: { ...draft.tasks, ocr: { ...ocr, ...change } } });
  const ocrConn = connectionOf(draft, ocr.connection);
  const service = ocrService(draft);
  const patch = (change: Parameters<typeof patchConnection>[2]) => ocrConn && setDraft(patchConnection(draft, ocrConn.id, change));
  const shared = ocrConn && usesChatService(draft, ocrConn.id) ? "dùng chung với Dịch thuật" : undefined;

  return (
    <div className="settings-list">
      <Section title="Nguồn lời thoại">
        <ServiceChoice name="line-source" checked={asr.ocr_first} onSelect={() => setAsr({ ocr_first: true })}
          title="Tự động (khuyến nghị)"
          description="Đọc phụ đề trên hình trước, Speech-to-Text lấp chỗ hình không có phụ đề — đúng tên riêng, từ đồng âm hơn." />
        <ServiceChoice name="line-source" checked={!asr.ocr_first} onSelect={() => setAsr({ ocr_first: false })}
          title="Chỉ Speech-to-Text"
          description="Bỏ qua phụ đề trên hình, chỉ nghe tiếng trong video." />
      </Section>

      {asr.ocr_first ? (
        <Section title="Đọc phụ đề trên hình (OCR)">
          <ServiceChoice name="ocr" checked={ocr.recognizer === "vision"} onSelect={() => setDraft(setOcrRecognizer(draft, "vision"))}
            title="macOS Vision" description="Miễn phí, chạy ngay trên máy." />
          <ServiceChoice name="ocr" checked={ocr.recognizer === "llm"} onSelect={() => setDraft(setOcrRecognizer(draft, "llm"))}
            title="AI (Gemini / OpenAI-compatible)" description="Mô hình AI đọc ảnh — dùng key đã nhập, tính phí theo dịch vụ.">
            <Field label="Dịch vụ">
              <select className="set-input" value={service} onChange={(e) => setDraft(setOcrService(draft, e.target.value as ConnectionType))}>
                {OCR_SERVICES.map((item) => <option key={item.type} value={item.type}>{item.title}</option>)}
              </select>
            </Field>
            {service === "openai_compat" ? (
              <TextField label="Địa chỉ API" value={ocrConn?.base_url ?? ""} placeholder="https://openrouter.ai/api/v1" hint={shared} onChange={(base_url) => patch({ base_url })} />
            ) : null}
            <SecretField key={ocrConn?.id} label="API key" conn={ocrConn} field="api_key" hint={shared} onChange={(api_key) => patch({ api_key })} />
            <ModelField label="Model đọc ảnh" conn={ocrConn} saved={saved} value={ocr.model} required onChange={(model) => setOcr({ model })} />
          </ServiceChoice>
        </Section>
      ) : null}

      <Section title="Nhận dạng giọng nói (Speech-to-Text)">
        <Field label="Chạy ở">
          <select className="set-input" value={asr.backend} onChange={(e) => setAsr({ backend: e.target.value as typeof asr.backend })}>
            <option value="auto">Tự động</option>
            <option value="qwen">Trên máy (Qwen3-ASR)</option>
            <option value="local">Trên máy (mlx-whisper)</option>
            <option value="worker">Máy chủ Cloudflare</option>
          </select>
        </Field>
        {asr.backend === "worker" ? (
          <TextField label="Địa chỉ máy chủ" value={asr.worker_url} placeholder="https://…workers.dev" onChange={(worker_url) => setAsr({ worker_url })} />
        ) : null}
        <Advanced>
          <TextField label="Model Speech-to-Text" value={asr.model} placeholder="để trống = mặc định" onChange={(model) => setAsr({ model })} />
        </Advanced>
      </Section>
    </div>
  );
}
