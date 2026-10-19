import { useState, type FormEvent, type KeyboardEvent } from "react";
import { useAddVideo, useSeriesList } from "../queries";
import { useToast } from "../toast";
import { ChannelImport } from "./ChannelImport";

/** One intake for everything. Paste a link → routed by type: a channel link fans out
 *  into series (ChannelImport); a single video is added as an episode (new or existing
 *  series). Replaces the old scattered "Quét kênh" / "+ Series" buttons. */
export function AddContent({ onOpenSeries }: { onOpenSeries: (seriesId: string) => void }) {
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState("");
  const [stage, setStage] = useState<"input" | "channel" | "single">("input");

  function reset() {
    setStage("input");
    setUrl("");
    setOpen(false);
  }
  const isChannel = (u: string) =>
    /\/user\//.test(u) || /space\.bilibili\.com/.test(u) || /playlet\/detail|object_id=|\/collection\//.test(u);
  // A Bilibili video link often IS a whole show (合集 collection or multi-part video
  // — the "videopod.episodes" player). Let the user pick scope explicitly.
  const isBiliVideo = (u: string) => /bilibili\.com\/video\//.test(u);
  const link = url.trim();
  const askScope = isBiliVideo(link) && !isChannel(link);

  function next(event: FormEvent) {
    event.preventDefault();
    if (!link) return;
    // Enter on an ambiguous Bilibili link takes the primary choice (whole show).
    setStage(askScope || isChannel(link) ? "channel" : "single");
  }
  function onKeyDown(event: KeyboardEvent) {
    if (event.key === "Escape") reset();
  }

  if (!open) {
    return (
      <button className="primary add-btn" onClick={() => setOpen(true)}>+ Thêm nội dung</button>
    );
  }

  return (
    <section className="add-panel panel" aria-labelledby="add-title" onKeyDown={onKeyDown}>
      <div className="add-head">
        <div>
          <h2 id="add-title" className="add-title">Thêm nội dung</h2>
          <p className="add-sub">Dán link kênh, bộ phim hoặc một video lẻ.</p>
        </div>
        <button type="button" className="ghost" onClick={reset}>Đóng</button>
      </div>

      {stage === "input" ? (
        <form className="add-field" onSubmit={next}>
          <label className="add-label" htmlFor="add-url">Link nguồn</label>
          <div className="ci-scan">
            <input
              id="add-url"
              className="lib-input"
              inputMode="url"
              autoComplete="off"
              spellCheck={false}
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="https://www.douyin.com/user/… hoặc link 1 video"
              aria-describedby="add-rules"
              autoFocus
            />
            {askScope ? (
              <>
                <button type="submit" className="primary">Nhập cả bộ</button>
                <button type="button" className="ghost" onClick={() => setStage("single")}>Chỉ video này</button>
              </>
            ) : (
              <button type="submit" className="primary" disabled={!link}>Tiếp</button>
            )}
          </div>
          <ul id="add-rules" className="add-rules">
            <li><code>/user/…</code><span>Cả kênh — tự tách thành nhiều bộ</span></li>
            <li><code>bilibili.com/video/…</code><span>Thuộc 合集 / nhiều phần — chọn nhập cả bộ hoặc 1 tập</span></li>
            <li><code>1 video</code><span>Thêm 1 tập vào bộ mới hoặc bộ có sẵn</span></li>
          </ul>
        </form>
      ) : (
        <div className="add-source">
          <span className="add-label">{stage === "channel" ? "Cả bộ / kênh" : "Video lẻ"}</span>
          <span className="add-source-url mono" title={link}>{link}</span>
          <button type="button" className="link" onClick={() => setStage("input")}>Đổi link</button>
        </div>
      )}

      {stage === "channel" ? <ChannelImport url={link} onDone={reset} /> : null}
      {stage === "single" ? <SingleVideoAdd url={link} onOpenSeries={(id) => { reset(); onOpenSeries(id); }} /> : null}
    </section>
  );
}

function SingleVideoAdd({ url, onOpenSeries }: { url: string; onOpenSeries: (seriesId: string) => void }) {
  const { data } = useSeriesList("");
  const series = data?.items ?? [];
  const addVideo = useAddVideo();
  const { toast } = useToast();
  const [target, setTarget] = useState<"new" | string>("new");
  const [title, setTitle] = useState("");

  async function add(event: FormEvent) {
    event.preventDefault();
    try {
      const result = await addVideo.mutateAsync({
        url,
        seriesId: target === "new" ? undefined : target,
        title: target === "new" ? title.trim() || undefined : undefined,
      });
      toast("Đã thêm video", "success");
      onOpenSeries(result.series_id);
    } catch (error) {
      toast(error instanceof Error ? error.message : "Không thêm được video", "error");
    }
  }

  return (
    <form className="single-add" onSubmit={add}>
      <fieldset>
        <legend className="add-label">Thêm vào</legend>
        <div className={`choice ${target === "new" ? "on" : ""}`}>
          <label className="radio-row">
            <input type="radio" name="add-target" checked={target === "new"} onChange={() => setTarget("new")} />
            Tạo bộ mới
          </label>
          {target === "new" ? (
            <input
              className="lib-input"
              aria-label="Tên bộ mới"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Tên bộ (để trống = lấy tên video)"
            />
          ) : null}
        </div>
        <div className={`choice ${target !== "new" ? "on" : ""} ${series.length ? "" : "disabled"}`}>
          <label className="radio-row">
            <input
              type="radio"
              name="add-target"
              checked={target !== "new"}
              disabled={!series.length}
              onChange={() => setTarget(series[0]?.id ?? "new")}
            />
            Bộ có sẵn {series.length ? "" : <span className="add-sub">— chưa có bộ nào</span>}
          </label>
          {target !== "new" ? (
            <select className="lib-input" aria-label="Chọn bộ" value={target} onChange={(event) => setTarget(event.target.value)}>
              {series.map((s) => (
                <option key={s.id} value={s.id}>{s.title}</option>
              ))}
            </select>
          ) : null}
        </div>
      </fieldset>
      <button type="submit" className="primary" disabled={addVideo.isPending}>
        {addVideo.isPending ? "Đang thêm…" : "Thêm video"}
      </button>
    </form>
  );
}
