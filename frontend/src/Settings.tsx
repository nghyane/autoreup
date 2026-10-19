import { useEffect, useState } from "react";
import { useAppSettings, useSaveAppSettings } from "./queries";
import { problems } from "./settings/model";
import { PlatformsTab } from "./settings/PlatformsTab";
import { SubtitlesTab } from "./settings/SubtitlesTab";
import { TranslationTab } from "./settings/TranslationTab";
import { VoiceTab } from "./settings/VoiceTab";
import { useToast } from "./toast";
import type { AppSettings } from "./types";

type Tab = "translation" | "voice" | "subtitles" | "platforms";

const TABS: { id: Tab; label: string }[] = [
  { id: "translation", label: "Dịch thuật" },
  { id: "voice", label: "Giọng đọc" },
  { id: "subtitles", label: "Phụ đề & lời thoại" },
  { id: "platforms", label: "Tài khoản nền tảng" },
];

/** Settings, organised by job (translation, voice, subtitles, platform accounts).
 *  The tabs edit one draft of settings.json; secrets come back masked and posting
 *  the mask back keeps the stored value. Platform cookies save on their own. */
export function Settings({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<Tab>("translation");
  const { data: saved, error: loadError } = useAppSettings();
  const [draft, setDraft] = useState<AppSettings | null>(null);
  // bumped when the draft is replaced wholesale, so tabs drop their local UI state
  const [revision, setRevision] = useState(0);
  const save = useSaveAppSettings();
  const { toast } = useToast();

  useEffect(() => {
    if (saved && !draft) setDraft(structuredClone(saved));
  }, [saved, draft]);

  const dirty = !!saved && !!draft && JSON.stringify(saved) !== JSON.stringify(draft);

  function reset(next: AppSettings) {
    setDraft(structuredClone(next));
    setRevision((value) => value + 1);
  }

  async function submit() {
    if (!draft) return;
    try {
      reset(await save.mutateAsync(draft));
      toast("Đã lưu cài đặt — áp dụng cho công việc mới", "success");
    } catch (error) {
      toast(error instanceof Error ? error.message : "Không lưu được cài đặt", "error");
    }
  }

  const props = draft ? { draft, setDraft, saved } : null;
  const issues = draft && tab !== "platforms" ? problems(draft) : [];
  return (
    // a stray click outside must not throw away unsaved edits (typed keys included)
    <div className="modal-scrim" onClick={() => !dirty && onClose()}>
      <div className="modal settings-modal" onClick={(event) => event.stopPropagation()}>
        <h3>Cài đặt</h3>
        <div className="settings-tabs" role="tablist">
          {TABS.map((item) => (
            <button key={item.id} role="tab" aria-selected={tab === item.id} className={tab === item.id ? "on" : ""} onClick={() => setTab(item.id)}>
              {item.label}
            </button>
          ))}
        </div>
        {issues.length ? (
          <div className="settings-problems" role="status">
            Chưa đủ để chạy:
            <ul>
              {issues.map((issue) => (
                <li key={issue.message}>
                  <button type="button" onClick={() => setTab(issue.tab)}>{issue.message}</button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        <div className="settings-body" key={revision}>
          {tab === "platforms" ? (
            <PlatformsTab />
          ) : !props ? (
            <p className="muted-note">{loadError ? `Không đọc được cài đặt: ${loadError.message}` : "Đang tải…"}</p>
          ) : tab === "translation" ? (
            <TranslationTab {...props} />
          ) : tab === "voice" ? (
            <VoiceTab {...props} />
          ) : (
            <SubtitlesTab {...props} />
          )}
        </div>
        <div className="modal-actions">
          {tab !== "platforms" && dirty && saved ? <button className="ghost" onClick={() => reset(saved)}>Hoàn tác</button> : null}
          <button className="ghost" onClick={onClose}>Đóng</button>
          {tab !== "platforms" ? (
            <button onClick={submit} disabled={!dirty || save.isPending}>{save.isPending ? "Đang lưu…" : "Lưu"}</button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
