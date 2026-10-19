import { useState, type ReactNode } from "react";
import type { ChatTaskName, ConnectionType } from "../types";
import { Advanced, CheckButton, EffortSelect, ModelField, SecretField, Section, ServiceChoice, TextField, type TabProps } from "./fields";
import { CHAT_SUBTASKS, CLI_BINS, chatService, connectionOf, firstOfType, followsTranslate, patchConnection, patchService, setChatService, setSubtask, setTranslate } from "./model";

const SERVICES: { type: ConnectionType; title: string; description: string }[] = [
  { type: "agy", title: "Antigravity (agy)", description: "Dùng gói đăng ký Antigravity trên máy này — không cần API key." },
  { type: "claude_code", title: "Claude Code (claude)", description: "Dùng gói Claude Pro/Max đã đăng nhập trong Claude Code trên máy này — không cần API key." },
  { type: "codex", title: "Codex CLI (codex)", description: "Dùng tài khoản ChatGPT đã đăng nhập trong Codex CLI trên máy này — không cần API key." },
  { type: "gemini", title: "Gemini API", description: "Gọi thẳng API Gemini của Google bằng API key của bạn." },
  { type: "openai_compat", title: "OpenAI-compatible", description: "Mọi dịch vụ chuẩn OpenAI: OpenRouter, DeepSeek, Groq, Ollama…" },
  { type: "command", title: "Lệnh CLI tuỳ chỉnh", description: "Chạy một lệnh CLI của riêng bạn cho mỗi lượt dịch." },
];

const SUBTASK_LABELS: Record<ChatTaskName, string> = {
  translate: "Dịch",
  glossary: "Thuật ngữ",
  normalize: "Chuẩn hoá phụ đề",
  repair: "Sửa độ dài lời đọc",
  naming: "Đặt tên bộ",
};

/** Dịch thuật: one service for every text job (translate, glossary, normalize,
 *  repair, naming); per-step models fold under Nâng cao. */
export function TranslationTab({ draft, setDraft, saved }: TabProps) {
  const service = chatService(draft);
  const conn = connectionOf(draft, draft.tasks.translate.connection);
  const translate = draft.tasks.translate;
  // Subtasks with their own model; the rest follow Dịch. Kept as UI state so an
  // override that happens to equal Dịch is not silently re-linked.
  const [custom, setCustom] = useState<ChatTaskName[]>(() => CHAT_SUBTASKS.filter((task) => !followsTranslate(draft, task)));
  const hasCommand = !!firstOfType(draft, "command");
  // fields of the service in use edit the connection Dịch is bound to
  const patch = (change: Parameters<typeof patchConnection>[2]) => conn && setDraft(patchConnection(draft, conn.id, change));
  const isApi = service === "gemini" || service === "openai_compat";

  function toggleCustom(task: ChatTaskName, on: boolean) {
    setCustom(on ? [...custom, task] : custom.filter((item) => item !== task));
    if (!on) setDraft(setSubtask(draft, task, null));
  }

  return (
    <div className="settings-list">
      <p className="muted-note">Dịch vụ AI làm mọi việc chữ: dịch, gom thuật ngữ, chuẩn hoá phụ đề, sửa câu cho vừa lời đọc, đặt tên bộ.</p>
      {SERVICES.filter((item) => item.type !== "command" || hasCommand || service === "command").map((item) => (
        <ServiceChoice
          key={item.type}
          name="chat-service"
          checked={service === item.type}
          onSelect={() => {
            setDraft(setChatService(draft, item.type));
            setCustom([]); // another service: every step follows Dịch again
          }}
          title={item.title}
          description={item.description}
        >
          {item.type === "openai_compat" ? (
            <TextField label="Địa chỉ API" value={conn?.base_url ?? ""} placeholder="https://openrouter.ai/api/v1" onChange={(base_url) => patch({ base_url })} />
          ) : null}
          {item.type === "gemini" || item.type === "openai_compat" ? (
            <SecretField key={conn?.id} label="API key" conn={conn} field="api_key" onChange={(api_key) => patch({ api_key })} />
          ) : null}
          {item.type === "command" ? (
            <TextField label="Mẫu lệnh" value={conn?.command ?? ""} placeholder="my-llm --prompt {prompt}" onChange={(command) => patch({ command })} />
          ) : null}
          <ModelField
            label="Model"
            conn={conn}
            saved={saved}
            value={translate.model}
            required={isApi}
            placeholder={isApi ? "chọn hoặc gõ tên model" : "để trống = mặc định"}
            onChange={(model) => setDraft(setTranslate(draft, { model }, custom))}
          />
          <CheckButton key={conn?.id} conn={conn} saved={saved} />
        </ServiceChoice>
      ))}
      <Advanced>
        <Section title="Model riêng cho từng bước">
          <div className="step-grid">
            <span className="task-name">Dịch</span>
            <span className="step-note">model ở trên</span>
            <EffortSelect label="Mức nghĩ khi dịch" value={translate.effort} onChange={(effort) => setDraft(setTranslate(draft, { effort }, custom))} />
            {CHAT_SUBTASKS.map((task) => {
              const own = custom.includes(task);
              const binding = draft.tasks[task];
              return (
                <StepRow key={task} label={SUBTASK_LABELS[task]}>
                  <select className="set-input" aria-label={`Model cho ${SUBTASK_LABELS[task]}`} value={own ? "own" : "same"} onChange={(e) => toggleCustom(task, e.target.value === "own")}>
                    <option value="same">giống Dịch</option>
                    <option value="own">model riêng</option>
                  </select>
                  {own ? (
                    <span className="step-own">
                      <input className="set-input" aria-label={`Tên model cho ${SUBTASK_LABELS[task]}`} placeholder="tên model" value={binding.model}
                        onChange={(e) => setDraft(setSubtask(draft, task, { model: e.target.value, effort: binding.effort }))} />
                      <EffortSelect label={`Mức nghĩ cho ${SUBTASK_LABELS[task]}`} value={binding.effort}
                        onChange={(effort) => setDraft(setSubtask(draft, task, { model: binding.model, effort }))} />
                    </span>
                  ) : <span />}
                </StepRow>
              );
            })}
          </div>
        </Section>
        {service in CLI_BINS || service === "command" ? (
          <TextField label="Thư mục HOME" value={conn?.home ?? ""} placeholder="để trống = tài khoản mặc định" hint="tách tài khoản đăng nhập riêng cho app"
            onChange={(home) => patch({ home })} />
        ) : null}
        {service in CLI_BINS ? (
          <TextField label="Lệnh chạy" value={conn?.bin ?? ""} placeholder={CLI_BINS[service]} onChange={(bin) => patch({ bin })} />
        ) : null}
        {service !== "command" ? (
          <TextField label="Mẫu lệnh CLI tuỳ chỉnh" value={firstOfType(draft, "command")?.command ?? ""} placeholder="my-llm --prompt {prompt}"
            hint="nhập để hiện lựa chọn “Lệnh CLI tuỳ chỉnh” ở trên" onChange={(command) => setDraft(patchService(draft, "command", { command }))} />
        ) : null}
      </Advanced>
    </div>
  );
}

function StepRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <span className="task-name">{label}</span>
      {children}
    </>
  );
}
