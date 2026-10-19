import { useId, useState, type ReactNode } from "react";
import * as api from "../api";
import { useConnectionModels } from "../queries";
import { useToast } from "../toast";
import type { AppSettings, Connection } from "../types";
import { connectionOf, isMasked } from "./model";

export type TabProps = { draft: AppSettings; setDraft: (next: AppSettings) => void; saved?: AppSettings };

/** Saved exactly as drafted — only then can the server test it, list its models/voices
 *  or synthesize with it (those endpoints read the stored connection). */
export function isSaved(saved: AppSettings | undefined, conn: Connection | undefined): boolean {
  if (!saved || !conn) return false;
  return JSON.stringify(connectionOf(saved, conn.id)) === JSON.stringify(conn);
}

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="set-field">
      <span>{label}</span>
      {children}
      {hint ? <em className="faint small">{hint}</em> : null}
    </label>
  );
}

export function TextField({ label, value, onChange, placeholder, hint }: {
  label: string; value: string; onChange: (value: string) => void; placeholder?: string; hint?: ReactNode;
}) {
  return (
    <Field label={label} hint={hint}>
      <input className="set-input" value={value} placeholder={placeholder} autoComplete="off" onChange={(e) => onChange(e.target.value)} />
    </Field>
  );
}

/** A key/password entered once and reused everywhere: a stored one shows as
 *  "đã có ✓" (the server only ever sends it masked); blank on save keeps it. */
export function SecretField({ label, conn, field, onChange, hint }: {
  label: string; conn: Connection | undefined; field: "api_key" | "password"; onChange: (value: string) => void; hint?: ReactNode;
}) {
  const stored = conn?.[field] ?? "";
  const [editing, setEditing] = useState(!stored);
  // the key was cleared under us (new API address): stay in typing mode from here on
  if (!stored && !editing) setEditing(true);
  if (stored && !editing) {
    // a div, not a <label>: clicking the caption must not press "Đổi"
    return (
      <div className="set-field">
        <span>{label}</span>
        <span className="secret-ok">
          <span className="ok-mark">đã có ✓</span>
          {isMasked(stored) ? <span className="mono small faint">{stored}</span> : null}
          <button type="button" className="ghost small-btn" onClick={() => setEditing(true)}>Đổi</button>
        </span>
        {hint ? <em className="faint small">{hint}</em> : null}
      </div>
    );
  }
  return (
    <Field label={label} hint={hint ?? (isMasked(stored) ? "để trống = giữ nguyên" : undefined)}>
      <input
        className="set-input"
        type="password"
        // Chrome ignores "off" on password inputs and fills a saved login (the email
        // into the text input above, the password here); "new-password" stops that.
        autoComplete="new-password"
        value={isMasked(stored) ? "" : stored}
        onChange={(e) => onChange(e.target.value)}
      />
    </Field>
  );
}

/** Free-text model name with the service's model list as suggestions (once saved). */
export function ModelField({ label, conn, saved, value, onChange, placeholder, required }: {
  label: string; conn: Connection | undefined; saved?: AppSettings; value: string; onChange: (model: string) => void;
  placeholder?: string; required?: boolean;
}) {
  const { data: models = [] } = useConnectionModels(isSaved(saved, conn) && conn ? conn.id : "");
  const listId = useId();
  return (
    <Field label={label} hint={required && !value.trim() ? <span className="warn-text">cần chọn model</span> : undefined}>
      <input
        className="set-input"
        list={models.length ? listId : undefined}
        value={value}
        placeholder={placeholder ?? "tên model"}
        autoComplete="off"
        onChange={(e) => onChange(e.target.value)}
      />
      {models.length ? <datalist id={listId}>{models.map((model) => <option key={model} value={model} />)}</datalist> : null}
    </Field>
  );
}

export function EffortSelect({ value, onChange, label }: { value: string; onChange: (effort: string) => void; label: string }) {
  return (
    <select className="set-input" aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">mức nghĩ mặc định</option>
      <option value="low">nghĩ ít</option>
      <option value="medium">nghĩ vừa</option>
      <option value="high">nghĩ kỹ</option>
    </select>
  );
}

/** One option of a "pick a service" list; its own fields open under it when chosen. */
export function ServiceChoice({ name, checked, onSelect, title, description, children }: {
  name: string; checked: boolean; onSelect: () => void; title: string; description: string; children?: ReactNode;
}) {
  return (
    <div className={`choice ${checked ? "on" : ""}`}>
      <label className="radio-row">
        <input type="radio" name={name} checked={checked} onChange={onSelect} />
        {title}
      </label>
      <span className="add-sub">{description}</span>
      {checked && children ? <div className="choice-body">{children}</div> : null}
    </div>
  );
}

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="conn-card">
      <legend className="add-label">{title}</legend>
      {children}
    </fieldset>
  );
}

/** Technical knobs, folded away by default. */
export function Advanced({ children }: { children: ReactNode }) {
  return (
    <details className="set-advanced">
      <summary>Nâng cao</summary>
      <div className="set-advanced-body">{children}</div>
    </details>
  );
}

/** Ask the server whether a (saved) service answers. */
export function CheckButton({ conn, saved }: { conn: Connection | undefined; saved?: AppSettings }) {
  const { toast } = useToast();
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);
  const ready = isSaved(saved, conn);

  async function check() {
    if (!conn) return;
    setBusy(true);
    try {
      setResult(await api.testConnection(conn.id));
    } catch (error) {
      toast(error instanceof Error ? error.message : "Không kiểm tra được", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="conn-actions">
      {result ? <span className={`small ${result.ok ? "ok-mark" : "warn-text"}`}>{result.message}</span> : null}
      <button type="button" className="ghost" onClick={check} disabled={!ready || busy} title={ready ? "" : "Lưu trước khi kiểm tra"}>
        {busy ? "Đang kiểm tra…" : "Kiểm tra"}
      </button>
    </div>
  );
}
