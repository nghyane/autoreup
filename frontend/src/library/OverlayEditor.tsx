import { useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api";
import type { CoverRegion, FontInfo, OverlayConfig, OverlayEncode, OverlayFrame, OverlayImage } from "../api";
import { useToast } from "../toast";

const MODES: { id: string; label: string }[] = [
  { id: "blur", label: "Làm mờ" },
  { id: "darken", label: "Tối đi" },
  { id: "solid", label: "Đen đặc" },
];

type Handle = "move" | "se" | "caption" | null;

type Sound = "dub" | "source" | "off";

/** Re-seek the voice when it drifts this far (s) from the picture. */
const MAX_VOICE_DRIFT = 0.3;

/** WYSIWYG review of the render: drag boxes over the paused frame to hide the original
 *  subtitle band / a channel logo, place the Vietnamese caption, pick font and quality,
 *  then start the render. The pipeline waits here before an episode's first render.
 *  Everything is stored as fractions of the frame, so it survives any resolution. */
export function OverlayEditor({
  jobId,
  videoPath,
  voicePath,
  cues,
  firstRender,
  onClose,
  onRepack,
}: {
  jobId: string;
  videoPath: string;
  /** The finished voiceover track ("" when the episode has no voice). */
  voicePath: string;
  cues: { start: number; end: number; text: string }[];
  /** The episode has never been rendered (the dub waits for this review). */
  firstRender: boolean;
  onClose: () => void;
  onRepack: () => void;
}) {
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const voiceRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const [config, setConfig] = useState<OverlayConfig | null>(null);
  const [frame, setFrame] = useState<OverlayFrame | null>(null);
  const [selected, setSelected] = useState(-1);
  const [t, setT] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [sound, setSound] = useState<Sound>(voicePath ? "dub" : "off");
  const [canvasWidth, setCanvasWidth] = useState(0);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState("");
  const drag = useRef<{ handle: Handle; startX: number; startY: number; box: CoverRegion } | null>(null);
  const pointer = useRef({ x: 0, y: 0 });
  const raf = useRef<number | null>(null);
  const [fonts, setFonts] = useState<FontInfo[]>([]);
  const [defaultFont, setDefaultFont] = useState("");
  const [captionFamily, setCaptionFamily] = useState("");
  const [images, setImages] = useState<OverlayImage[]>([]);
  const [imageBase, setImageBase] = useState("/media/overlay_images");
  const [uploading, setUploading] = useState(false);
  const imageInput = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    api
      .fetchFonts()
      .then((list) => {
        setFonts(list.fonts);
        setDefaultFont(list.default);
      })
      .catch(() => {});
    api
      .fetchOverlayImages()
      .then((library) => {
        setImages(library.images);
        setImageBase(library.base);
      })
      .catch(() => {});
  }, []);

  // Draw the on-canvas caption in the font the render will burn.
  const fontPath = config?.font || defaultFont;
  useEffect(() => {
    if (!fontPath) return;
    let cancelled = false;
    const family = `ov-${fontPath.replace(/[^a-z0-9]/gi, "")}`;
    const face = new FontFace(family, `url(${api.fontFileUrl(fontPath)})`);
    face
      .load()
      .then((loaded) => {
        if (cancelled) return;
        document.fonts.add(loaded);
        setCaptionFamily(family);
      })
      .catch(() => {
        if (!cancelled) setCaptionFamily("");
      });
    return () => {
      cancelled = true;
    };
  }, [fontPath]);

  useEffect(() => {
    api
      .fetchOverlay(jobId)
      .then((state) => {
        const { frame: info, ...cfg } = state;
        setFrame(info ?? null);
        setConfig(cfg);
        setSelected(cfg.cover.length ? 0 : -1);
      })
      .catch(() => setConfig({ caption_mode: "band", caption_y: null, caption_scale: 1, cover: [] }));
  }, [jobId]);

  // The video is only a moving picture inside the OUTPUT frame: the canvas carries the
  // output aspect ratio, the video letterboxes inside it, and every box/caption
  // fraction is measured against the canvas — the same space the render uses.
  const outAspect = frame ? frame.out.width / frame.out.height : 16 / 9;
  useEffect(() => {
    const element = canvasRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => setCanvasWidth(entries[0]?.contentRect.width ?? 0));
    observer.observe(element);
    return () => observer.disconnect();
  }, [frame]);
  // caption font in on-screen pixels = rendered font x (canvas width / output width)
  const captionFontPx =
    frame && canvasWidth > 0 ? ((frame.font_size ?? 58) * (config?.caption_scale ?? 1) * canvasWidth) / frame.out.width : 16;

  // The render's mix levels; the "dub" preview plays the original at the bed level
  // (the render additionally ducks it while the voice speaks).
  const bedVolume = config?.bed_volume ?? frame?.bed_volume ?? 0.18;
  const voiceVolume = config?.voice_volume ?? 1;
  const gapVolume = config?.gap_volume ?? frame?.gap_volume ?? 0.7;
  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.muted = sound === "off";
      video.volume = sound === "dub" ? bedVolume : 1;
    }
    if (voiceRef.current) voiceRef.current.volume = voiceVolume;
  }, [sound, bedVolume, voiceVolume]);

  // The video carries the original audio; the dub is a separate track kept in step
  // with the picture (started from its position, paused with it).
  useEffect(() => {
    const video = videoRef.current;
    const voice = voiceRef.current;
    if (!voice) return;
    if (sound === "dub" && playing && video) {
      voice.currentTime = video.currentTime;
      voice.play().catch(() => {});
    } else {
      voice.pause();
    }
  }, [sound, playing]);

  function syncVoice(time: number) {
    const voice = voiceRef.current;
    if (voice && sound === "dub" && !voice.paused && Math.abs(voice.currentTime - time) > MAX_VOICE_DRIFT) {
      voice.currentTime = time;
    }
  }

  const captionText = useMemo(() => {
    const cue = cues.find((item) => item.start <= t && t < item.end);
    return cue?.text ?? "";
  }, [cues, t]);

  function patch(next: Partial<OverlayConfig>) {
    setConfig((current) => (current ? { ...current, ...next } : current));
    setPreview("");
  }

  /** Mix levels change only the sound: keep the rendered preview frame. */
  function patchVolume(next: Partial<Pick<OverlayConfig, "bed_volume" | "voice_volume" | "gap_volume">>) {
    setConfig((current) => (current ? { ...current, ...next } : current));
  }

  function patchRegion(index: number, next: Partial<CoverRegion>) {
    setConfig((current) => {
      if (!current) return current;
      const cover = current.cover.map((region, i) => (i === index ? { ...region, ...next } : region));
      return { ...current, cover };
    });
    setPreview("");
  }

  function addRegion(label: string, mode: string) {
    setConfig((current) => {
      if (!current) return current;
      const region: CoverRegion = { x: 0.05, y: 0.05, w: 0.25, h: 0.1, mode, strength: 1, label };
      setSelected(current.cover.length);
      return { ...current, cover: [...current.cover, region] };
    });
    setPreview("");
  }

  /** Stamp a library image: a box with the image's own shape, in the top-right corner. */
  function addImage(image: OverlayImage) {
    const aspect = frame ? frame.out.width / frame.out.height : 16 / 9;
    let w = 0.18;
    let h = (w * aspect * image.height) / image.width;
    if (h > 0.4) {
      w *= 0.4 / h;
      h = 0.4;
    }
    setConfig((current) => {
      if (!current) return current;
      const region: CoverRegion = { x: round4(0.97 - w), y: 0.04, w: round4(w), h: round4(h), mode: "image", strength: 1, label: "image", image: image.name };
      setSelected(current.cover.length);
      return { ...current, cover: [...current.cover, region] };
    });
    setPreview("");
  }

  /** Upload into the shared library, then stamp it (or swap it into the selected image box). */
  async function uploadImage(file: File) {
    setUploading(true);
    try {
      const image = await api.uploadOverlayImage(file);
      setImages((list) => [image, ...list.filter((item) => item.name !== image.name)]);
      if (active?.mode === "image") patchRegion(selected, { image: image.name });
      else addImage(image);
    } catch (error) {
      toast(error instanceof Error ? error.message : "Không tải ảnh lên được", "error");
    } finally {
      setUploading(false);
    }
  }

  function removeRegion(index: number) {
    setConfig((current) => {
      if (!current) return current;
      const cover = current.cover.filter((_region, i) => i !== index);
      setSelected((sel) => (sel === index ? -1 : sel > index ? sel - 1 : sel));
      return { ...current, cover };
    });
    setPreview("");
  }

  function onPointerDown(event: React.PointerEvent, index: number, handle: Handle) {
    if (!config) return;
    event.stopPropagation();
    event.preventDefault();
    setSelected(index);
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    drag.current = { handle, startX: event.clientX, startY: event.clientY, box: { ...config.cover[index]! } };
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
  }

  /** One state commit per animation frame: pointermove fires far faster than the
   *  browser paints, and a React render per move made dragging feel sticky. */
  function onPointerMove(event: React.PointerEvent) {
    pointer.current = { x: event.clientX, y: event.clientY };
    if (raf.current !== null) return;
    raf.current = requestAnimationFrame(() => {
      raf.current = null;
      applyDrag();
    });
  }

  function applyDrag() {
    const state = drag.current;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!state || !rect) return;
    const dx = (pointer.current.x - state.startX) / rect.width;
    const dy = (pointer.current.y - state.startY) / rect.height;
    if (state.handle === "move") {
      const box = state.box;
      const x = round4(Math.min(Math.max(box.x + dx, 0), 1 - box.w));
      const y = round4(Math.min(Math.max(box.y + dy, 0), 1 - box.h));
      patchRegion(selected, { x, y });
    } else if (state.handle === "caption") {
      patch({ caption_mode: "manual", caption_y: round4(Math.min(Math.max(state.box.y + dy, 0.05), 0.97)) });
    } else {
      const box = state.box;
      const w = round4(Math.min(Math.max(box.w + dx, 0.02), 1 - box.x));
      const h = round4(Math.min(Math.max(box.h + dy, 0.02), 1 - box.y));
      patchRegion(selected, { w, h });
    }
  }

  function endDrag() {
    drag.current = null;
    if (raf.current !== null) {
      cancelAnimationFrame(raf.current);
      raf.current = null;
    }
  }

  /** The caption preview sits where the render will burn it: the band region's centre
   *  in band mode, the manual fraction otherwise. */
  const captionFraction = (() => {
    if (!config) return 0.88;
    if (config.caption_mode === "manual") return config.caption_y ?? 0.88;
    const band = config.cover.find((region) => region.label === "band");
    return band ? band.y + band.h / 2 : 0.88;
  })();

  function startCaptionDrag(event: React.PointerEvent) {
    if (!config) return;
    event.stopPropagation();
    event.preventDefault();
    drag.current = { handle: "caption", startX: event.clientX, startY: event.clientY, box: { x: 0, y: captionFraction, w: 1, h: 0.06, mode: "blur", strength: 1, label: "caption" } };
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
  }

  const active = selected >= 0 && config ? config.cover[selected] : undefined;

  async function previewFrame() {
    if (!config) return;
    setBusy("preview");
    try {
      const url = await api.previewOverlayFrame(jobId, config, t, captionText);
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return url;
      });
    } catch (error) {
      toast(String(error), "error");
    } finally {
      setBusy("");
    }
  }

  async function save() {
    if (!config) return;
    setBusy("save");
    try {
      const saved = await api.saveOverlay(jobId, config);
      setConfig(saved);
      toast("Đã lưu khung che", "success");
    } catch (error) {
      toast(String(error), "error");
    } finally {
      setBusy("");
    }
  }

  async function repack() {
    if (!config) return;
    setBusy("repack");
    try {
      await save();
      await api.repackJob(jobId);
      toast(firstRender ? "Đang render video…" : "Đang render lại video…", "success");
      onRepack();
      onClose();
    } catch (error) {
      toast(String(error), "error");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="ov-scrim" onClick={onClose}>
      <div className="ov-panel" onClick={(event) => event.stopPropagation()}>
        <div className="ov-head">
          <p>{firstRender ? "Duyệt & render video" : "Chỉnh & render lại video"}</p>
          <button className="link" onClick={onClose}>
            Đóng
          </button>
        </div>
        <p className="ov-hint tiny">
          Kéo hộp mờ đè lên chữ Trung có sẵn trong video (hoặc logo) để che đi, rồi kéo dòng phụ đề Việt tới chỗ muốn đặt và chọn font, cỡ chữ.
          Bấm “Xem thử khung hình” để xem đúng như bản xuất, chọn chất lượng video, xong bấm “{firstRender ? "Render video" : "Render lại video"}” để áp dụng cho cả tập.
        </p>

        <div className="ov-body">
          <div
            className="ov-canvas"
            ref={canvasRef}
            style={{ aspectRatio: `${outAspect}` }}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerLeave={endDrag}
          >
            <video
              ref={videoRef}
              src={videoPath}
              playsInline
              preload="metadata"
              onLoadedMetadata={() => {
                const video = videoRef.current;
                if (video) video.currentTime = t;
              }}
              onTimeUpdate={(event) => {
                if (!playing) return;
                const time = event.currentTarget.currentTime;
                setT(time);
                syncVoice(time);
              }}
              onEnded={() => setPlaying(false)}
            />
            {voicePath ? <audio ref={voiceRef} src={voicePath} preload="auto" /> : null}
            {frame && Math.abs(frame.source.width / frame.source.height - outAspect) > 0.01 ? (
              <div
                className="ov-source-area"
                style={
                  frame.source.width / frame.source.height > outAspect
                    ? { left: 0, width: "100%", top: `${(frame.pad.y / frame.out.height) * 100}%`, height: `${(frame.source.height * frame.scale / frame.out.height) * 100}%` }
                    : { top: 0, height: "100%", left: `${(frame.pad.x / frame.out.width) * 100}%`, width: `${(frame.source.width * frame.scale / frame.out.width) * 100}%` }
                }
                title="Vùng hình gốc (ngoài là viền đen)"
              />
            ) : null}
            {config?.cover.map((region, index) => (
              <div key={index} className="ov-shape">
                {/* A thin band (2-4% of the frame) is nearly impossible to grab, so every
                    region gets an extended transparent rail around it: same drag, easy hit. */}
                <div
                  className="ov-rail"
                  style={{ left: `${region.x * 100}%`, top: `${region.y * 100}%`, width: `${region.w * 100}%`, height: `${region.h * 100}%` }}
                  onPointerDown={(event) => onPointerDown(event, index, "move")}
                />
                <div
                  className={`ov-box ${index === selected ? "on" : ""} ${region.mode}`}
                  style={{ left: `${region.x * 100}%`, top: `${region.y * 100}%`, width: `${region.w * 100}%`, height: `${region.h * 100}%` }}
                  onPointerDown={(event) => onPointerDown(event, index, "move")}
                  title={region.label || region.mode}
                >
                  {region.mode === "image" && region.image ? (
                    <img className="ov-image" src={`${imageBase}/${region.image}`} alt="" style={{ opacity: region.strength }} draggable={false} />
                  ) : null}
                  <span className="ov-tag">{regionName(region)}</span>
                </div>
                {index === selected ? (
                  <>
                    <span className="ov-handle" style={{ left: `calc(${(region.x + region.w) * 100}% - 6px)`, top: `calc(${(region.y + region.h) * 100}% - 6px)` }} onPointerDown={(event) => onPointerDown(event, index, "se")} />
                    <span className="ov-handle edge" style={{ left: `calc(${(region.x + region.w) * 100}% - 5px)`, top: `${(region.y + region.h / 2) * 100}%` }} onPointerDown={(event) => onPointerDown(event, index, "se")} />
                  </>
                ) : null}
              </div>
            ))}
            {/* the caption follows the frame: drag it (or the band box) and it moves */}
            <div
              className="ov-caption"
              style={{
                top: `${(captionFraction * 100).toFixed(2)}%`,
                fontSize: `${captionFontPx.toFixed(1)}px`,
                fontFamily: captionFamily ? `"${captionFamily}", sans-serif` : undefined,
              }}
              onPointerDown={startCaptionDrag}
              title="Kéo để đặt vị trí phụ đề Việt"
            >
              <span className="ov-caption-text">{captionText || "(không có phụ đề ở giây này)"}</span>
            </div>
          </div>

          <aside className="ov-side">
            <div className="ov-group">
              <p className="mono tiny">
                Khung xem {frame ? `${frame.out.width}×${frame.out.height}` : ""} · nguồn{" "}
                {frame ? `${frame.source.width}×${frame.source.height}` : "?"}
              </p>
              <div className="ov-row">
                <button
                  className="ghost"
                  onClick={() => {
                    const video = videoRef.current;
                    if (!video) return;
                    if (playing) {
                      video.pause();
                      setPlaying(false);
                    } else {
                      setPreview("");
                      void video.play();
                      setPlaying(true);
                    }
                  }}
                >
                  {playing ? "⏸ Dừng" : "▶ Chạy"}
                </button>
                <input
                  className="ov-scrub"
                  type="range"
                  min={0}
                  max={frame?.duration || 1}
                  step={0.1}
                  value={t}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    setT(next);
                    const video = videoRef.current;
                    if (video) video.currentTime = next;
                    setPreview("");
                  }}
                />
                <span className="mono tiny">{formatTime(t)}</span>
              </div>
              <div className="seg-toggle" role="group" aria-label="Âm thanh khi chạy">
                {voicePath ? (
                  <button
                    className={sound === "dub" ? "on" : ""}
                    onClick={() => setSound("dub")}
                    title="Giọng lồng tiếng trên nền âm gốc, theo mức âm lượng bên dưới"
                  >
                    Lồng tiếng
                  </button>
                ) : null}
                <button className={sound === "source" ? "on" : ""} onClick={() => setSound("source")}>
                  Gốc
                </button>
                <button className={sound === "off" ? "on" : ""} onClick={() => setSound("off")}>
                  Tắt tiếng
                </button>
              </div>
            </div>

            {voicePath ? (
              <div className="ov-group">
                <p className="mono tiny">Âm lượng bản render</p>
                <label className="ov-field" title="Âm thanh gốc (nhạc, tiếng động) giữ làm nền; khi có giọng, bản render còn tự hạ nền xuống thêm">
                  <span>Âm gốc {Math.round(bedVolume * 100)}%</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={bedVolume}
                    onChange={(event) => patchVolume({ bed_volume: Number(event.target.value) })}
                  />
                </label>
                <label
                  className="ov-field"
                  title="Những đoạn không ai nói (cả thoại gốc lẫn lồng tiếng) tự nâng âm gốc lên mức này. Không cao hơn âm gốc = tắt. Chỉ nghe được trong bản render"
                >
                  <span>
                    Âm gốc khi không thoại {gapVolume > bedVolume ? `${Math.round(gapVolume * 100)}%` : "tắt"}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={gapVolume}
                    onChange={(event) => patchVolume({ gap_volume: Number(event.target.value) })}
                  />
                </label>
                <label className="ov-field">
                  <span>Lồng tiếng {Math.round(voiceVolume * 100)}%</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={voiceVolume}
                    onChange={(event) => patchVolume({ voice_volume: Number(event.target.value) })}
                  />
                </label>
              </div>
            ) : null}

            <div className="ov-group">
              <p className="mono tiny">Vùng che &amp; ảnh chèn ({config?.cover.length ?? 0})</p>
              {config?.cover.map((region, index) => (
                <button
                  key={index}
                  className={`ov-region ${index === selected ? "on" : ""}`}
                  onClick={() => setSelected(index)}
                  onDoubleClick={() => removeRegion(index)}
                  title="Nháy đúp để xoá"
                >
                  <span className={`dot ${region.mode}`} />
                  {regionName(region)}
                  <span className="mono tiny">
                    {Math.round(region.x * 100)}%,{Math.round(region.y * 100)}% · {Math.round(region.w * 100)}×{Math.round(region.h * 100)}
                  </span>
                </button>
              ))}
              <div className="ov-row">
                <button className="ghost" onClick={() => addRegion("logo", "blur")}>
                  + Logo
                </button>
                <button className="ghost" onClick={() => addRegion("text", "blur")}>
                  + Vùng chữ
                </button>
                <button className="ghost" onClick={() => addRegion("band", "blur")}>
                  + Phụ đề gốc
                </button>
                <button
                  className="ghost"
                  disabled={uploading}
                  onClick={() => imageInput.current?.click()}
                  title="Logo, watermark… (PNG trong suốt là đẹp nhất). Ảnh được lưu lại để dùng cho mọi tập"
                >
                  {uploading ? "Đang tải ảnh…" : "+ Ảnh của bạn"}
                </button>
                <input
                  ref={imageInput}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/gif"
                  hidden
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (file) void uploadImage(file);
                  }}
                />
              </div>
              {images.length ? (
                <div className="ov-library" role="list" aria-label="Ảnh đã tải lên">
                  {images.map((image) => (
                    <button
                      key={image.name}
                      role="listitem"
                      className={active?.mode === "image" && active.image === image.name ? "on" : ""}
                      onClick={() => (active?.mode === "image" ? patchRegion(selected, { image: image.name }) : addImage(image))}
                      title={active?.mode === "image" ? "Đổi ảnh của hộp đang chọn" : "Chèn ảnh này"}
                    >
                      <img src={`${imageBase}/${image.name}`} alt={image.name} loading="lazy" />
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            {active ? (
              <div className="ov-group">
                <p className="mono tiny">Hộp đang chọn</p>
                {active.mode === "image" ? (
                  <p className="mono tiny">Ảnh nằm gọn trong hộp, giữ đúng tỉ lệ. Bấm ảnh ở thư viện phía trên để đổi.</p>
                ) : (
                  <label className="ov-field">
                    <span>Cách che</span>
                    <select value={active.mode} onChange={(event) => patchRegion(selected, { mode: event.target.value })}>
                      {MODES.map((mode) => (
                        <option key={mode.id} value={mode.id}>
                          {mode.label}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <label className="ov-field">
                  <span>
                    {active.mode === "image" ? "Độ đậm" : "Cường độ"} {active.strength.toFixed(2)}
                  </span>
                  <input
                    type="range"
                    min={active.mode === "image" ? 0.1 : 0.2}
                    max={1}
                    step={0.05}
                    value={active.strength}
                    onChange={(event) => patchRegion(selected, { strength: Number(event.target.value) })}
                  />
                </label>
                <label className="ov-field">
                  <span>Nhãn</span>
                  <input value={active.label} onChange={(event) => patchRegion(selected, { label: event.target.value })} />
                </label>
                <div className="ov-nums">
                  {(["x", "y", "w", "h"] as const).map((key) => (
                    <label key={key} className="ov-field">
                      <span>{key.toUpperCase()} %</span>
                      <input
                        type="number"
                        min={0}
                        max={100}
                        step={0.5}
                        value={(active[key] * 100).toFixed(1)}
                        onChange={(event) => patchRegion(selected, { [key]: Math.max(0, Math.min(1, Number(event.target.value) / 100)) })}
                      />
                    </label>
                  ))}
                </div>
                <div className="ov-row">
                  {([["↑", 0, -0.005], ["↓", 0, 0.005], ["←", -0.005, 0], ["→", 0.005, 0]] as const).map(([label, dx, dy]) => (
                    <button
                      key={label}
                      className="ghost"
                      title="Dịch 0.5% khung hình"
                      onClick={() => patchRegion(selected, { x: active.x + dx, y: active.y + dy })}
                    >
                      {label}
                    </button>
                  ))}
                  <button className="ghost" onClick={() => patchRegion(selected, { h: Math.min(1 - active.y, active.h + 0.005) })} title="Cao thêm 0.5%">
                    Cao +
                  </button>
                  <button className="ghost" onClick={() => patchRegion(selected, { h: Math.max(0.02, active.h - 0.005) })} title="Thấp bớt 0.5%">
                    Cao −
                  </button>
                </div>
                <button className="ghost danger" onClick={() => removeRegion(selected)}>
                  Xoá hộp này
                </button>
              </div>
            ) : null}

            <div className="ov-group">
              <p className="mono tiny">Phụ đề Việt</p>
              <div className="seg-toggle">
                <button
                  className={config?.caption_mode === "band" ? "on" : ""}
                  onClick={() => patch({ caption_mode: "band" })}
                  title="Đặt phụ đề Việt đúng chỗ phụ đề gốc đã bị che"
                >
                  Đè lên phụ đề gốc
                </button>
                <button className={config?.caption_mode === "manual" ? "on" : ""} onClick={() => patch({ caption_mode: "manual" })}>
                  Tự đặt
                </button>
              </div>
              <label className="ov-field">
                <span>Vị trí dọc {Math.round((config?.caption_y ?? 0.88) * 100)}%</span>
                <input
                  type="range"
                  min={0.3}
                  max={0.98}
                  step={0.01}
                  value={config?.caption_y ?? 0.88}
                  onChange={(event) => patch({ caption_y: Number(event.target.value), caption_mode: "manual" })}
                />
              </label>
              <label className="ov-field">
                <span>Font chữ</span>
                <select value={config?.font ?? ""} onChange={(event) => patch({ font: event.target.value })}>
                  <option value="">Mặc định ({fontName(fonts, defaultFont)})</option>
                  {(["app", "user", "system"] as const).map((source) => {
                    const group = fonts.filter((font) => font.source === source);
                    return group.length ? (
                      <optgroup key={source} label={FONT_SOURCES[source]}>
                        {group.map((font) => (
                          <option key={font.path} value={font.path}>
                            {font.name}
                          </option>
                        ))}
                      </optgroup>
                    ) : null;
                  })}
                </select>
              </label>
              <label className="ov-field">
                <span>Cỡ chữ ×{(config?.caption_scale ?? 1).toFixed(2)}</span>
                <input
                  type="range"
                  min={0.6}
                  max={1.8}
                  step={0.05}
                  value={config?.caption_scale ?? 1}
                  onChange={(event) => patch({ caption_scale: Number(event.target.value) })}
                />
              </label>
            </div>
          </aside>
        </div>

        {preview ? (
          <div className="ov-preview">
            <img src={preview} alt="Xem thử" />
            <span className="mono tiny">Xem thử tại {t.toFixed(1)}s — đúng như bản render</span>
          </div>
        ) : null}

        <div className="ov-actions">
          <button className="ghost" onClick={previewFrame} disabled={!!busy}>
            {busy === "preview" ? "Đang dựng…" : "Xem thử khung hình"}
          </button>
          <div className="pv-right">
            {frame?.encode ? (
              <label className="ov-quality">
                <span className="mono tiny">Chất lượng video</span>
                <select
                  value={config?.bitrate || frame.encode.default}
                  onChange={(event) => patch({ bitrate: event.target.value })}
                  title={sourceNote(frame.encode)}
                >
                  {bitrateChoices(frame.encode, config?.bitrate ?? "").map((choice) => (
                    <option key={choice.value} value={choice.value} title={choice.hint}>
                      {choice.label} · {formatKbps(choice.rate)} · ~{formatSize(choice.rate, frame.encode!, frame.duration)}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <button className="ghost" onClick={save} disabled={!!busy}>
              {busy === "save" ? "Đang lưu…" : "Lưu"}
            </button>
            <button className="primary" onClick={repack} disabled={!!busy}>
              {busy === "repack" ? "Đang gửi…" : firstRender ? "Render video" : "Render lại video"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

type BitrateChoice = { value: string; label: string; hint: string; rate: number };

/** The measured quality tiers (VMAF vs the source); the app default is marked, and a
 *  rate saved outside the tiers stays selectable instead of silently changing. */
function bitrateChoices(encode: OverlayEncode, current: string): BitrateChoice[] {
  const choices = encode.presets.map((preset) => ({
    value: preset.value,
    label: preset.value === encode.default ? `${preset.label} (mặc định)` : preset.label,
    hint: preset.hint,
    rate: parseRate(preset.value),
  }));
  if (!choices.some((choice) => choice.value === encode.default)) {
    choices.unshift({ value: encode.default, label: "Mặc định", hint: "", rate: parseRate(encode.default) });
  }
  if (current && !choices.some((choice) => choice.value === current)) {
    choices.push({ value: current, label: "Đã chọn trước đó", hint: "", rate: parseRate(current) });
  }
  return choices;
}

function sourceNote(encode: OverlayEncode): string {
  if (!encode.source_bitrate) return "";
  return `Video gốc: ${encode.source_codec.toUpperCase()} ${formatKbps(encode.source_bitrate)}`;
}

function parseRate(value: string): number {
  const unit = value.slice(-1).toLowerCase();
  const scale = unit === "k" ? 1_000 : unit === "m" ? 1_000_000 : 1;
  return Number(scale > 1 ? value.slice(0, -1) : value) * scale;
}

function formatKbps(rate: number): string {
  return rate >= 1_000_000 ? `${(rate / 1_000_000).toFixed(1).replace(".", ",")} Mbit/s` : `${Math.round(rate / 1000)} kbit/s`;
}

/** File size of the whole episode at ``rate`` (video + the AAC track). */
function formatSize(rate: number, encode: OverlayEncode, duration: number): string {
  const bytes = ((rate + encode.audio_bitrate) * duration) / 8;
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1).replace(".", ",")} GB` : `${Math.round(bytes / 1e6)} MB`;
}

const FONT_SOURCES = { app: "Font của app", user: "Font bạn đã cài", system: "Font hệ thống" } as const;

function fontName(fonts: FontInfo[], path: string): string {
  return fonts.find((font) => font.path === path)?.name ?? (path.split("/").pop() ?? "").replace(/\.(ttf|otf)$/i, "");
}

const REGION_NAMES: Record<string, string> = { band: "phụ đề gốc", logo: "logo", text: "vùng chữ", image: "ảnh" };

/** The stored label stays a stable id ("band"); the editor shows it in Vietnamese. */
function regionName(region: CoverRegion): string {
  return REGION_NAMES[region.label] ?? (region.label || region.mode);
}

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}
