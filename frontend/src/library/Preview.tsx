import { useEffect, useRef, useState } from "react";
import * as api from "../api";
import { OverlayEditor } from "./OverlayEditor";
import type { Artifact, Episode, JobManifest, SubtitleCue } from "../types";
import { STAGE_VI, dubErrorVi, formatDuration, formatProgress, latestArtifact, stageElapsed } from "../util";

const STAGE_ORDER = ["download", "asr", "glossary", "translate", "qa", "voiceover", "package"];

type Cue = { start: number; end: number; text: string };

function toCues(rows: SubtitleCue[]): Cue[] {
  return rows
    .filter((row): row is SubtitleCue & { start: number; end: number } =>
      typeof row.start === "number" && typeof row.end === "number")
    .map((row) => ({ start: row.start, end: row.end, text: (row.text ?? row.translated ?? "").trim() }))
    .filter((cue) => cue.text.length > 0);
}

/** Re-render every second while ``active`` (an elapsed clock between polls). */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

/** Pipeline checklist so a translating/failed episode shows how far it got and where:
 *  the stage in flight shows its sub-step, counts, elapsed time and latest log lines. */
function StageProgress({ ep }: { ep: Episode }) {
  // dub_stage_index counts the stage in flight (or the one that failed) as the last one.
  const current = (ep.dub_stage_index ?? 0) - 1;
  const failed = ep.dub_state === "failed";
  const live = failed ? null : ep.dub_progress;
  const now = useNow(Boolean(live?.started_at));
  const detail = live ? formatProgress(live) : "";
  return (
    <div className="pv-stages">
      <ol>
        {STAGE_ORDER.map((name, i) => {
          const state = i < current ? "done" : i === current ? (failed ? "fail" : "now") : "todo";
          return (
            <li key={name} className={`stg ${state}`}>
              <span className="stg-mark" aria-hidden>{state === "done" ? "✓" : state === "fail" ? "✕" : state === "now" ? "●" : "○"}</span>
              {STAGE_VI[name] ?? name}
              {state === "now" && live ? <span className="stg-live mono">{stageElapsed(live, now)}</span> : null}
              {state === "now" && detail ? <span className="stg-detail">{detail}</span> : null}
            </li>
          );
        })}
      </ol>
      {live?.logs?.length ? (
        <pre className="pv-logs mono" aria-live="polite">{live.logs.join("\n")}</pre>
      ) : null}
      {failed && ep.dub_error ? <p className="pv-err" title={ep.dub_error}>⚠ {dubErrorVi(ep.dub_error)}</p> : null}
    </div>
  );
}

type JobStage = { name: string; status: string };

/** Compact live progress for a job that is running again (e.g. a package re-render):
 *  the editor must never look like nothing happened. */
function LiveProgress({ stages }: { stages: JobStage[] }) {
  const running = stages.find((stage) => stage.status === "running") ?? stages[stages.length - 1];
  const done = stages.filter((stage) => stage.status === "succeeded").length;
  return (
    <p className="pv-live mono tiny" role="status" aria-live="polite">
      <span className="pv-spin" aria-hidden /> Đang chạy {done}/{stages.length} · {STAGE_VI[running?.name ?? ""] ?? running?.name ?? "bắt đầu"}
    </p>
  );
}

/** Watch/inspect one episode — the only "studio" surface. A dubbed episode plays its
 *  local video (original/dubbed toggle) next to a synced subtitle list: the active cue
 *  is highlighted and kept in view, clicking a row seeks the player. */
export function Preview({
  episode,
  onClose,
  onRedub,
  onDelete,
  onStop,
  onRender,
}: {
  episode: Episode;
  onClose: () => void;
  onRedub: () => void;
  onDelete: () => void;
  onStop: () => Promise<void>;
  /** A render was started from the editor: refresh the episode list. */
  onRender: () => void;
}) {
  const [job, setJob] = useState<JobManifest | null>(null);
  const [dubbed, setDubbed] = useState(true);
  const [error, setError] = useState("");
  const [cues, setCues] = useState<Cue[]>([]);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [artIndex, setArtIndex] = useState(0);
  const [editor, setEditor] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const activeRowRef = useRef<HTMLLIElement | null>(null);

  useEffect(() => {
    if (!episode.dub_job_id) return;
    setJob(null);
    setCues([]);
    setActiveIndex(-1);
    api.fetchJob(episode.dub_job_id).then(setJob).catch((err) => setError(String(err)));
  }, [episode.dub_job_id]);

  // While a (re)render is running, poll the manifest so the progress strip and the
  // finished video actually update — the repack button used to look like a no-op.
  const running = job ? job.stages.some((stage) => stage.status === "running") || job.state === "queued" : false;
  useEffect(() => {
    if (!episode.dub_job_id || !running) return;
    const timer = setInterval(() => {
      api.fetchJob(episode.dub_job_id!).then(setJob).catch(() => {});
    }, 2500);
    return () => clearInterval(timer);
  }, [episode.dub_job_id, running]);

  useEffect(() => {
    setArtIndex(0);
  }, [episode.dub_job_id, job, dubbed]);

  // Load the final subtitle track once the manifest is known (served from /media).
  useEffect(() => {
    const artifact = job ? latestArtifact(job, "subtitle_json") : undefined;
    if (!artifact) return;
    let cancelled = false;
    fetch(`/media/${artifact.path}`)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(String(response.status)))))
      .then((rows: SubtitleCue[]) => {
        if (!cancelled) setCues(toCues(rows));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [job]);

  useEffect(() => {
    activeRowRef.current?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  const dubArt = job ? latestArtifact(job, "hardsub_mp4") ?? latestArtifact(job, "dubbed_mp4") : undefined;
  const srcArt = job ? latestArtifact(job, "video") : undefined;
  const voiceArt = job ? latestArtifact(job, "voice_track") : undefined;
  // Prefer the dubbed render, but fall back to the original if that file is gone
  // (a stale manifest artifact is dropped server-side; a broken file may still 404).
  const candidates = (dubbed ? [dubArt, srcArt] : [srcArt]).filter((artifact): artifact is Artifact => Boolean(artifact));
  const art = candidates[Math.min(artIndex, Math.max(0, candidates.length - 1))];
  // A re-render writes the same path with new bytes: version the URL by size so the
  // player actually reloads it instead of replaying the browser cache.
  const versioned = art ? `${art.path}?v=${art.size_bytes ?? art.created_at ?? ""}` : "";
  const isDubbed = episode.dub_state === "dubbed";
  const isRunning = episode.dub_state === "translating";
  // Translated and voiced; the render waits for the user to review the overlay.
  const isReview = episode.dub_state === "review";
  // A failed or stopped dub resumes in place: finished steps and translated parts stay.
  const canResume = !isDubbed && !isReview && !isRunning && Boolean(episode.dub_job_id);
  const [stopping, setStopping] = useState(false);
  useEffect(() => setStopping(false), [episode.dub_state]);
  // Dubbed but the final video was never rendered (or the render failed): say so and
  // offer the render instead of silently playing the original without voice or subs.
  const missingRender = isDubbed && !!job && !dubArt;
  const fellBackToSource = dubbed && !!dubArt && artIndex > 0;

  async function rerender() {
    if (!job) return;
    try {
      setJob(await api.repackJob(job.job_id));
    } catch (err) {
      setError(String(err));
    }
  }
  const showSubs = isDubbed && dubbed && !!dubArt && cues.length > 0 && !fellBackToSource;

  function handleTimeUpdate(event: React.SyntheticEvent<HTMLVideoElement>) {
    const time = event.currentTarget.currentTime;
    let index = -1;
    for (let i = 0; i < cues.length; i += 1) {
      const cue = cues[i]!;
      if (cue.start > time) break;
      if (time < cue.end) {
        index = i;
        break;
      }
    }
    if (index !== activeIndex) setActiveIndex(index);
  }

  function seekTo(cue: Cue) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = cue.start;
    void video.play();
  }

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="preview-modal" onClick={(event) => event.stopPropagation()}>
        <div className="pv-head">
          <div className="pv-titles">
            <p className="pv-title">{episode.title_vi || episode.title}</p>
            {episode.title_vi ? <p className="pv-title-zh">{episode.title}</p> : null}
          </div>
          <button className="link" onClick={onClose}>Đóng</button>
        </div>

        {running && job ? <LiveProgress stages={job.stages as JobStage[]} /> : null}
        <div className={`pv-body ${showSubs ? "with-subs" : ""}`}>
          <div className="pv-screen">
            {missingRender ? (
              <div className="pv-empty">
                {running ? (
                  <p>Đang ghép video lồng tiếng (giọng đọc + phụ đề)… Tập dài có thể mất vài phút.</p>
                ) : (
                  <>
                    <p>Đã dịch và đọc giọng xong, nhưng chưa có video thành phẩm — bước ghép video chưa chạy hoặc bị lỗi.</p>
                    {error ? <p className="pv-err" title={error}>⚠ {dubErrorVi(error)}</p> : null}
                    <button className="primary" onClick={rerender}>Ghép video</button>
                  </>
                )}
              </div>
            ) : isReview ? (
              <div className="pv-empty">
                {running ? (
                  <p>Đang render video (giọng đọc + phụ đề)… Tập dài có thể mất vài phút.</p>
                ) : (
                  <>
                    <p>Đã dịch và lồng tiếng xong — video chưa được render.</p>
                    <p className="tiny">Duyệt trước khi render: che chữ gốc/logo, đặt vị trí phụ đề Việt, chọn font, cỡ chữ và chất lượng video.</p>
                    <button className="primary" onClick={() => setEditor(true)} disabled={!srcArt}>Duyệt &amp; render</button>
                  </>
                )}
              </div>
            ) : isDubbed && art ? (
              <video
                key={art.path}
                ref={videoRef}
                controls
                autoPlay
                playsInline
                src={`/media/${versioned}`}
                onTimeUpdate={handleTimeUpdate}
                onError={() => setArtIndex((index) => (index + 1 < candidates.length ? index + 1 : index))}
              />
            ) : (
              <div className="pv-empty">
                {episode.dub_state === "translating" || episode.dub_state === "failed" ? (
                  <StageProgress ep={episode} />
                ) : (
                  <>
                    {episode.thumbnail_url ? <img className="pv-cover" src={`/api/v1/library/thumb?url=${encodeURIComponent(episode.thumbnail_url)}`} alt="" /> : <span className="reel" aria-hidden />}
                    <p>{error ? "Không tải được video." : "Tập này chưa lồng tiếng. Xem bản gốc rồi bấm Dịch."}</p>
                  </>
                )}
              </div>
            )}
          </div>

          {showSubs ? (
            <aside className="pv-subs" aria-label="Danh sách phụ đề">
              <p className="pv-subs-head mono">{cues.length} dòng phụ đề</p>
              <ol>
                {cues.map((cue, index) => (
                  <li
                    key={index}
                    ref={index === activeIndex ? activeRowRef : undefined}
                    className={index === activeIndex ? "on" : ""}
                    aria-current={index === activeIndex ? "true" : undefined}
                  >
                    <button onClick={() => seekTo(cue)} title="Nhảy tới dòng này">
                      <span className="mono tiny">{formatDuration(cue.start)}</span>
                      <span className="pv-sub-text">{cue.text}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </aside>
          ) : null}
        </div>

        <div className="pv-actions">
          {isDubbed && dubArt ? (
            <div className="seg-toggle">
              <button className={dubbed ? "" : "on"} onClick={() => setDubbed(false)}>Gốc</button>
              <button className={dubbed ? "on" : ""} onClick={() => setDubbed(true)}>Lồng tiếng</button>
            </div>
          ) : (
            <a className="ghost" href={episode.source_url} target="_blank" rel="noreferrer">Xem gốc trên nền tảng ↗</a>
          )}
          {fellBackToSource ? <span className="mono tiny amber">Thiếu file lồng tiếng — đang phát bản gốc</span> : null}
          <div className="pv-right">
            {isDubbed && dubArt ? (
              <a className="ghost" href={api.artifactUrl(episode.dub_job_id ?? "", "hardsub_mp4")} download>
                Tải video
              </a>
            ) : null}
            {(isDubbed || isReview) && job ? (
              <button
                className={isReview ? "primary" : "ghost"}
                onClick={() => setEditor(true)}
                disabled={!srcArt || running}
                title="Che chữ gốc/logo, đặt phụ đề Việt, chọn font và chất lượng rồi render video"
              >
                {isReview ? "Duyệt & render" : "Chỉnh & render lại"}
              </button>
            ) : null}
            {isRunning ? (
              <button
                className="ghost danger"
                disabled={stopping}
                title="Dừng ngay; các bước và phần dịch đã xong được giữ lại"
                onClick={() => { setStopping(true); onStop().catch(() => setStopping(false)); }}
              >
                {stopping ? "Đang dừng…" : "Dừng"}
              </button>
            ) : (
              <button className={isReview ? "ghost" : "primary"} onClick={onRedub}>
                {isDubbed || isReview ? "Làm lại" : canResume ? "Dịch tiếp" : "Dịch tập này"}
              </button>
            )}
            {isDubbed || isReview ? <button className="ghost danger" onClick={onDelete}>Xóa bản dịch</button> : null}
          </div>
        </div>
      </div>
      {editor && job && srcArt ? (
        <OverlayEditor
          jobId={job.job_id}
          videoPath={`/media/${srcArt.path}`}
          // a re-dub rewrites the same path: version it so the editor never replays a stale voice
          voicePath={voiceArt ? `/media/${voiceArt.path}?v=${voiceArt.size_bytes ?? voiceArt.created_at ?? ""}` : ""}
          cues={cues}
          firstRender={isReview}
          onClose={() => setEditor(false)}
          onRepack={() => {
            api.fetchJob(job.job_id).then(setJob).catch(() => {});
            onRender();
          }}
        />
      ) : null}
    </div>
  );
}
