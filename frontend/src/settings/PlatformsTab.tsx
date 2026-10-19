import { useState } from "react";
import { useProviders, useSaveCookies } from "../queries";
import { useToast } from "../toast";
import type { ProviderSession } from "../types";

/** Tài khoản nền tảng: Douyin/Bilibili sessions (cookies), saved on their own. */
export function PlatformsTab() {
  const { data: providers = [] } = useProviders();
  const [editing, setEditing] = useState<ProviderSession | null>(null);

  if (editing) return <CookieEditor session={editing} onBack={() => setEditing(null)} />;
  return (
    <div className="settings-list">
      {providers.map((session) => (
        <button key={session.platform} className={`settings-row ${session.configured ? "ok" : "off"}`} onClick={() => setEditing(session)}>
          <span className="prov-dot" aria-hidden />
          <span className="settings-name">{session.label}</span>
          <span className="settings-state">{session.configured ? "phiên ✓" : "cần cookie"}</span>
        </button>
      ))}
      {!providers.length ? <p className="muted-note">Chưa có nền tảng nào.</p> : null}
    </div>
  );
}

function CookieEditor({ session, onBack }: { session: ProviderSession; onBack: () => void }) {
  const [value, setValue] = useState("");
  const save = useSaveCookies();
  const { toast } = useToast();

  async function submit() {
    try {
      await save.mutateAsync({ platform: session.platform, cookies: value });
      toast(`Đã lưu phiên ${session.label}`, "success");
      onBack();
    } catch (error) {
      toast(error instanceof Error ? error.message : "Không lưu được cookie", "error");
    }
  }

  return (
    <>
      <p className="muted-note">{session.note}</p>
      {session.required_missing.length ? <p className="mono small">Cần: {session.required_missing.join(", ")}</p> : null}
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder='Dán cookie JSON từ Cookie-Editor ({"cookies":[…]}), file cookies.txt, hoặc header Cookie…'
        rows={6}
      />
      <div className="modal-actions">
        <button className="ghost" onClick={onBack}>Quay lại</button>
        <button className="primary" onClick={submit} disabled={save.isPending || value.trim().length < 4}>
          {save.isPending ? "Đang lưu…" : "Lưu phiên"}
        </button>
      </div>
    </>
  );
}
