// Pure settings <-> "by job" view helpers. Run: node --test frontend/tests
// (Node strips the TypeScript types itself — no test framework dependency.)
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  CHAT_SUBTASKS,
  chatService,
  firstOfType,
  followsTranslate,
  hasKey,
  patchConnection,
  patchService,
  problems,
  setChatService,
  setOcrRecognizer,
  setOcrService,
  setSubtask,
  setTranslate,
  setTtsService,
  ttsService,
} from "../src/settings/model.ts";

const blank = { base_url: "", api_key: "", bin: "", home: "", command: "", email: "", password: "" };
const conn = (id, type, extra = {}) => ({ ...blank, id, label: id, type, ...extra });
const chat = (connection, model = "m1", effort = "") => ({ connection, model, effort });

function seeded() {
  return {
    version: 1,
    connections: [conn("agent", "agy", { bin: "agy" }), conn("capcut", "capcut", { email: "a@b.c", password: "••••" })],
    tasks: {
      translate: chat("agent"), glossary: chat("agent"), normalize: chat("agent"),
      repair: chat("agent", "polish"), naming: chat("agent"),
      ocr: { recognizer: "vision", connection: "", model: "" },
      tts: { enabled: true, connection: "capcut", model: "", voice: "BV421_vivn_streaming", synth_rate: 1 },
      asr: { backend: "auto", worker_url: "", model: "", ocr_first: true },
    },
  };
}

test("chat service is the translate connection's type", () => {
  assert.equal(chatService(seeded()), "agy");
});

test("choosing a translation service points every chat task at one connection", () => {
  const next = setChatService(seeded(), "gemini");
  const gemini = firstOfType(next, "gemini");
  assert.ok(gemini, "gemini connection created");
  assert.match(gemini.id, /^c-[a-z0-9]{6}$/);
  for (const task of ["translate", ...CHAT_SUBTASKS]) assert.equal(next.tasks[task].connection, gemini.id, task);
  assert.equal(chatService(next), "gemini");
  // the agy model name means nothing to Gemini
  assert.equal(next.tasks.translate.model, "");
});

test("same service: overrides survive; another service: model names don't carry over", () => {
  const before = seeded();
  assert.equal(followsTranslate(before, "glossary"), true);
  assert.equal(followsTranslate(before, "repair"), false);
  const same = setChatService(before, "agy");
  assert.equal(same.tasks.repair.model, "polish");
  assert.equal(same.tasks.translate.model, "m1");
  // an agy model id ("polish") means nothing to an API service: every step follows Dịch
  const next = setChatService(before, "openai_compat");
  for (const task of CHAT_SUBTASKS) assert.deepEqual(next.tasks[task], next.tasks.translate, task);
});

test("setTranslate carries following subtasks along, leaves pinned ones", () => {
  const next = setTranslate(seeded(), { model: "m2", effort: "high" });
  assert.deepEqual(next.tasks.glossary, next.tasks.translate);
  assert.equal(next.tasks.translate.model, "m2");
  assert.equal(next.tasks.repair.model, "polish");
  // an explicitly pinned subtask stays put even if it currently matches
  const pinned = setTranslate(seeded(), { model: "m3" }, ["naming"]);
  assert.equal(pinned.tasks.naming.model, "m1");
  assert.equal(pinned.tasks.glossary.model, "m3");
});

test("setSubtask overrides or resets to 'giống Dịch'", () => {
  const custom = setSubtask(seeded(), "glossary", { model: "g", effort: "low" });
  assert.equal(custom.tasks.glossary.model, "g");
  assert.equal(custom.tasks.glossary.connection, "agent");
  const reset = setSubtask(custom, "glossary", null);
  assert.deepEqual(reset.tasks.glossary, reset.tasks.translate);
});

test("a key typed once lands on the single connection of that type and is reused", () => {
  const withKey = patchService(seeded(), "gemini", { api_key: "AIza-secret" });
  const gemini = firstOfType(withKey, "gemini");
  assert.equal(gemini.api_key, "AIza-secret");
  assert.equal(hasKey(gemini), true);
  const chatNext = setChatService(withKey, "gemini");
  const ttsNext = setTtsService(chatNext, "gemini");
  const ocrNext = setOcrService(ttsNext, "gemini");
  assert.equal(ocrNext.connections.filter((c) => c.type === "gemini").length, 1);
  assert.equal(ocrNext.tasks.tts.connection, gemini.id);
  assert.equal(ocrNext.tasks.ocr.connection, gemini.id);
  assert.equal(ocrNext.tasks.translate.connection, gemini.id);
});

test("hasKey accepts a masked stored key and rejects a blank one", () => {
  assert.equal(hasKey(conn("x", "gemini", { api_key: "••••abcd" })), true);
  assert.equal(hasKey(conn("x", "gemini")), false);
  assert.equal(hasKey(undefined), false);
});

test("unknown and duplicate connections survive every edit", () => {
  const start = seeded();
  start.connections.push(conn("my-proxy", "openai_compat", { base_url: "http://a" }), conn("my-proxy-2", "openai_compat", { base_url: "http://b" }));
  let next = setChatService(start, "openai_compat");
  next = patchService(next, "openai_compat", { api_key: "k" });
  next = setTtsService(next, "edge");
  next = setOcrRecognizer(next, "llm");
  for (const id of ["agent", "capcut", "my-proxy", "my-proxy-2"]) assert.ok(next.connections.some((c) => c.id === id), id);
  // the first of the type is the one the UI drives
  assert.equal(next.tasks.translate.connection, "my-proxy");
  assert.equal(firstOfType(next, "openai_compat").api_key, "k");
  assert.equal(next.connections.find((c) => c.id === "my-proxy-2").api_key, "");
});

test("helpers never mutate their input", () => {
  const start = seeded();
  const frozen = JSON.stringify(start);
  setChatService(start, "gemini");
  patchService(start, "gemini", { api_key: "k" });
  setTtsService(start, "edge");
  setOcrRecognizer(start, "llm");
  setTranslate(start, { model: "z" });
  assert.equal(JSON.stringify(start), frozen);
});

test("choosing a voice service resets voice/model to the engine default", () => {
  const next = setTtsService(seeded(), "edge");
  assert.equal(ttsService(next), "edge");
  assert.equal(next.tasks.tts.voice, "");
  assert.equal(next.tasks.tts.model, "");
  assert.equal(next.tasks.tts.enabled, true);
  // re-choosing the same service keeps what the user picked
  assert.equal(setTtsService(seeded(), "capcut").tasks.tts.voice, "BV421_vivn_streaming");
});

test("OCR by AI reuses the translation API service, else an API with a key, else Gemini", () => {
  const viaChat = setOcrRecognizer(setChatService(seeded(), "openai_compat"), "llm");
  assert.equal(viaChat.tasks.ocr.recognizer, "llm");
  assert.equal(firstOfType(viaChat, "openai_compat").id, viaChat.tasks.ocr.connection);

  const keyed = patchService(seeded(), "openai_compat", { api_key: "k" });
  const viaKey = setOcrRecognizer(keyed, "llm");
  assert.equal(viaKey.tasks.ocr.connection, firstOfType(keyed, "openai_compat").id);

  const fallback = setOcrRecognizer(seeded(), "llm");
  assert.equal(firstOfType(fallback, "gemini").id, fallback.tasks.ocr.connection);

  const back = setOcrRecognizer(fallback, "vision");
  assert.equal(back.tasks.ocr.recognizer, "vision");
});

test("changing an API address drops the stored (masked) key — it must be typed again", () => {
  const start = seeded();
  start.connections.push(conn("oa", "openai_compat", { base_url: "https://a/v1", api_key: "••••abcd" }));
  const moved = patchConnection(start, "oa", { base_url: "https://b/v1" });
  assert.equal(moved.connections.find((c) => c.id === "oa").api_key, "");
  const same = patchConnection(start, "oa", { base_url: "https://a/v1" });
  assert.equal(same.connections.find((c) => c.id === "oa").api_key, "••••abcd");
  const typed = patchConnection({ ...start, connections: start.connections.map((c) => (c.id === "oa" ? { ...c, api_key: "new" } : c)) }, "oa", { base_url: "https://b/v1" });
  assert.equal(typed.connections.find((c) => c.id === "oa").api_key, "new");
});

test("problems lists what would make a job fail, by tab", () => {
  assert.deepEqual(problems(seeded()), []);
  let s = setChatService(seeded(), "gemini");
  assert.deepEqual(problems(s).map((p) => p.tab), ["translation", "translation"]); // no key, no model
  s = patchService(s, "gemini", { api_key: "k" });
  s = setTranslate(s, { model: "gemini-x" });
  assert.deepEqual(problems(s), []);
  s = setTtsService(s, "gemini");
  assert.deepEqual(problems(s).map((p) => p.tab), ["voice"]); // TTS model
  s = setTtsService(s, "elevenlabs");
  assert.deepEqual(problems(s).map((p) => p.tab), ["voice", "voice"]); // key + voice
  s = setTtsService(s, "edge");
  s = setOcrRecognizer(s, "llm");
  assert.deepEqual(problems(s).map((p) => p.tab), ["subtitles"]); // OCR model
  // OCR is not used when lines come from Speech-to-Text only
  s = { ...s, tasks: { ...s.tasks, asr: { ...s.tasks.asr, ocr_first: false } } };
  assert.deepEqual(problems(s), []);
  // a disabled voice is not a problem
  const off = setTtsService({ ...seeded(), tasks: { ...seeded().tasks, tts: { ...seeded().tasks.tts, enabled: false } } }, "gemini");
  assert.deepEqual(problems(off), []);
});

test("an OpenAI-compatible address that is not a URL (a browser-autofilled email) is a problem", () => {
  let s = setChatService(seeded(), "openai_compat");
  s = patchService(s, "openai_compat", { base_url: "someone@gmail.com", api_key: "k" });
  s = setTranslate(s, { model: "m" });
  assert.deepEqual(problems(s).map((p) => p.message), ["Dịch thuật: địa chỉ API phải bắt đầu bằng http:// hoặc https://"]);
  s = patchService(s, "openai_compat", { base_url: "https://openrouter.ai/api/v1", api_key: "k" });
  assert.deepEqual(problems(s), []);
});
