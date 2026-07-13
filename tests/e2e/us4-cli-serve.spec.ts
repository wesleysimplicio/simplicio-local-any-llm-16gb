import {expect, test, type TestInfo} from "@playwright/test";
import {spawn, type ChildProcess} from "node:child_process";
import {existsSync} from "node:fs";
import {createServer} from "node:net";
import path from "node:path";
import {setTimeout as sleep} from "node:timers/promises";

const repoRoot = path.resolve(__dirname, "..", "..");
const nativeBin = path.join(repoRoot, "build", "apps", "us4-cli");
const serveScript = path.join(repoRoot, "scripts", "openai_serve.py");

type FetchResult = {
  status: number;
  body: string;
  json: unknown;
};

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const address = srv.address();
      if (typeof address === "object" && address !== null) {
        const port = address.port;
        srv.close(() => resolve(port));
        return;
      }
      srv.close();
      reject(new Error("could not allocate port"));
    });
  });
}

async function waitReady(url: string, deadlineMs = 20000): Promise<void> {
  const start = Date.now();
  let lastErr: unknown = null;
  while (Date.now() - start < deadlineMs) {
    try {
      const resp = await fetch(url);
      if (resp.ok) return;
    } catch (err) {
      lastErr = err;
    }
    await sleep(200);
  }
  throw new Error(
      `serve never became ready at ${url}: ${String(lastErr ?? "timeout")}`);
}

async function getJson(url: string): Promise<FetchResult> {
  const resp = await fetch(url);
  const body = await resp.text();
  let json: unknown = null;
  try {
    json = JSON.parse(body);
  } catch {
    json = null;
  }
  return {status : resp.status, body, json};
}

async function postJson(url: string, payload: unknown): Promise<FetchResult> {
  const resp = await fetch(url, {
    method : "POST",
    headers : {"Content-Type" : "application/json"},
    body : JSON.stringify(payload),
  });
  const body = await resp.text();
  let json: unknown = null;
  try {
    json = JSON.parse(body);
  } catch {
    json = null;
  }
  return {status : resp.status, body, json};
}

async function postStream(url: string, payload: unknown): Promise<string> {
  const resp = await fetch(url, {
    method : "POST",
    headers : {"Content-Type" : "application/json"},
    body : JSON.stringify(payload),
  });
  if (!resp.ok) {
    throw new Error(`stream request failed with ${resp.status}`);
  }
  return await resp.text();
}

async function options(url: string): Promise<Response> {
  return await fetch(url, {method : "OPTIONS"});
}

async function attach(testInfo: TestInfo, label: string,
                      payload: unknown): Promise<void> {
  await testInfo.attach(`serve-${label}`, {
    body : typeof payload === "string" ? payload
                                       : JSON.stringify(payload, null, 2),
    contentType : "text/plain",
  });
}

const haveBinary = existsSync(nativeBin) && existsSync(serveScript);

test.describe("us4-cli serve OpenAI-compat smoke", () => {
  test.skip(!haveBinary,
            "native us4-cli or scripts/openai_serve.py not built; " +
                "build with: cmake --build build --target us4-cli");

  let serveProc: ChildProcess|null = null;
  let basePort = 0;

  test.beforeAll(async () => {
    basePort = await freePort();
    serveProc = spawn(nativeBin,
                      [
                        "serve",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        String(basePort),
                        "--no-chat",
                        "--no-embed",
                      ],
                      {
                        cwd : repoRoot,
                        env : {...process.env, NO_COLOR : "1"},
                        stdio : [ "ignore", "pipe", "pipe" ],
                      });
    await waitReady(`http://127.0.0.1:${basePort}/health`);
  });

  test.afterAll(async () => {
    if (serveProc && !serveProc.killed) {
      serveProc.kill("SIGINT");
      await new Promise<void>((resolve) => {
        if (!serveProc) return resolve();
        serveProc.once("exit", () => resolve());
        setTimeout(() => {
          if (serveProc && !serveProc.killed) {
            serveProc.kill("SIGKILL");
          }
          resolve();
        }, 3000);
      });
    }
  });

  test("health reports both backends disabled", async ({}, testInfo) => {
    const result = await getJson(`http://127.0.0.1:${basePort}/health`);
    await attach(testInfo, "health", result.body);
    expect(result.status).toBe(200);
    expect(result.json).toMatchObject(
        {status : "ok", chat : false, embed : false});
  });

  test("models list is empty when both backends disabled",
       async ({}, testInfo) => {
         const result =
             await getJson(`http://127.0.0.1:${basePort}/v1/models`);
         await attach(testInfo, "models", result.body);
         expect(result.status).toBe(200);
         expect(result.json).toMatchObject({object : "list", data : []});
       });

  test("embeddings POST returns 503 when --no-embed", async ({}, testInfo) => {
    const result = await postJson(
        `http://127.0.0.1:${basePort}/v1/embeddings`, {input : "hello"});
    await attach(testInfo, "embeddings-disabled", result.body);
    expect(result.status).toBe(503);
    expect(result.json).toMatchObject({
      error : {
        type : "service_unavailable",
      },
    });
  });

  test("chat completions POST returns 503 when --no-chat",
       async ({}, testInfo) => {
         const result = await postJson(
             `http://127.0.0.1:${basePort}/v1/chat/completions`, {
               model : "any",
               messages : [ {role : "user", content : "hi"} ],
             });
         await attach(testInfo, "chat-disabled", result.body);
         expect(result.status).toBe(503);
         expect(result.json).toMatchObject({
           error : {
             type : "service_unavailable",
           },
         });
       });

  test("unknown route returns 404 not_found", async ({}, testInfo) => {
    const result = await getJson(`http://127.0.0.1:${basePort}/bogus`);
    await attach(testInfo, "not-found", result.body);
    expect(result.status).toBe(404);
    expect(result.json).toMatchObject({
      error : {
        type : "not_found",
      },
    });
  });

  test("proxy mode answers browser preflight for web chat", async () => {
    const response =
        await options(`http://127.0.0.1:${basePort}/v1/chat/completions`);
    expect(response.status).toBe(204);
    expect(response.headers.get("access-control-allow-origin")).toBe("*");
  });
});

// Issue #81.10: `serve --native` must answer OpenAI-compatible requests
// directly from this runtime's own Generate() pipeline -- no external
// mlx_lm.server/Ollama process, unlike the proxy mode exercised above.
test.describe("us4-cli serve --native (real runtime, no external process)",
             () => {
  test.skip(!haveBinary, "native us4-cli not built");

  let nativeProc: ChildProcess|null = null;
  let nativePort = 0;

  test.beforeAll(async () => {
    nativePort = await freePort();
    nativeProc = spawn(nativeBin,
                       [
                         "serve",
                         "--native",
                         "--host",
                         "127.0.0.1",
                         "--port",
                         String(nativePort),
                       ],
                       {
                         cwd : repoRoot,
                         env : {...process.env, NO_COLOR : "1"},
                         stdio : [ "ignore", "pipe", "pipe" ],
                       });
    await waitReady(`http://127.0.0.1:${nativePort}/v1/models`);
  });

  test.afterAll(async () => {
    if (nativeProc && !nativeProc.killed) {
      nativeProc.kill("SIGINT");
      await new Promise<void>((resolve) => {
        if (!nativeProc) return resolve();
        nativeProc.once("exit", () => resolve());
        setTimeout(() => {
          if (nativeProc && !nativeProc.killed) {
            nativeProc.kill("SIGKILL");
          }
          resolve();
        }, 3000);
      });
    }
  });

  test("models list is served from the adapter registry, not a proxy",
       async ({}, testInfo) => {
         const result =
             await getJson(`http://127.0.0.1:${nativePort}/v1/models`);
         await attach(testInfo, "native-models", result.body);
         expect(result.status).toBe(200);
         expect(result.json).toMatchObject({object : "list"});
         const data = (result.json as {data: Array<{id: string}>}).data;
         expect(data.some((entry) => entry.id === "qwen-0.5b")).toBe(true);
       });

  test("chat completions with a real model_path use real weights end to end",
       async ({}, testInfo) => {
         const tensorPath = path.join(
             repoRoot,
             "tests",
             "fixtures",
             "models",
             "toy-dense-real",
             "toy-dense-real.safetensors",
         );
         const result = await postJson(
             `http://127.0.0.1:${nativePort}/v1/chat/completions`, {
               model : "qwen-0.5b",
               model_path : tensorPath,
               messages : [ {role : "user", content : "alpha"} ],
               max_tokens : 1,
             });
         await attach(testInfo, "native-chat-real-weights", result.body);
         expect(result.status).toBe(200);
         // Same external-oracle prediction as the #85 evidence: embedding
         // "alpha" is one-hot over these real weights and argmaxes to
         // "delta" -- this is the native runtime computing it live, not a
         // canned response.
         expect(result.json).toMatchObject({
           used_real_weights : true,
           choices : [ {message : {role : "assistant", content : "delta"}} ],
         });
       });

  test("chat completions support SSE framing in native mode",
       async ({}, testInfo) => {
         const tensorPath = path.join(
             repoRoot,
             "tests",
             "fixtures",
             "models",
             "toy-dense-real",
             "toy-dense-real.safetensors",
         );
         const body = await postStream(
             `http://127.0.0.1:${nativePort}/v1/chat/completions`, {
               model : "qwen-0.5b",
               model_path : tensorPath,
               messages : [ {role : "user", content : "alpha"} ],
               max_completion_tokens : 1,
               stream : true,
             });
         await attach(testInfo, "native-chat-sse", body);
         expect(body).toContain("data: ");
         expect(body).toContain("\"object\":\"chat.completion.chunk\"");
         expect(body).toContain("\"content\":\"delta\"");
         expect(body).toContain("data: [DONE]");
       });

  test("unknown model returns an explicit error, never a fabricated answer",
       async ({}, testInfo) => {
         const result = await postJson(
             `http://127.0.0.1:${nativePort}/v1/chat/completions`, {
               model : "does-not-exist",
               messages : [ {role : "user", content : "hi"} ],
             });
         await attach(testInfo, "native-chat-unknown-model", result.body);
         expect(result.status).toBe(404);
         expect(result.json).toMatchObject({
           error : {message : expect.stringContaining("unknown model")},
         });
       });

  test("native mode answers browser preflight for streaming web chat",
       async () => {
         const response =
             await options(`http://127.0.0.1:${nativePort}/v1/chat/completions`);
         expect(response.status).toBe(204);
         expect(response.headers.get("access-control-allow-origin")).toBe("*");
       });
});
