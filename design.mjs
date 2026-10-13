// A brief in, a finished design out: the OpenAI Agents SDK runs the model through IMG.LY's gateway, the engine runs in one local executor.
import { mkdir, writeFile } from 'node:fs/promises';

import {
  googleFonts,
  prepareCoDesign,
  remoteExecutors
} from '@imgly/codesign-sdk';
import {
  Agent,
  MaxTurnsExceededError,
  OpenAIProvider,
  Runner,
  tool
} from '@openai/agents';

const brief = process.argv[2];
const missing = [
  ...(brief ? [] : ['a brief (the first argument)']),
  ...['IMGLY_CESDK_LICENSE', 'EXECUTOR_TOKEN'].filter((n) => !process.env[n])
];
if (missing.length > 0) {
  console.error(
    `Missing ${missing.join(', ')}.\nUsage: npm start -- "<brief>", with the variables of .env.template set in .env`
  );
  process.exit(1);
}

// IMG.LY's AI gateway speaks OpenAI's Chat Completions; your CoDesign license key is its API key.
const imglyGateway = new OpenAIProvider({
  baseURL: 'https://gateway.img.ly/v1',
  apiKey: process.env.IMGLY_CESDK_LICENSE,
  useResponses: false // the gateway has Chat Completions, not OpenAI's Responses API
});

const codesign = prepareCoDesign({
  // Your id for this job. Every design of a conversation opens on one executor
  // that serves no other conversation. Every run here is yours, so all share one.
  conversation: 'quickstart',
  executors: remoteExecutors({
    // The executor for a conversation. Here there is one; a product starts a
    // new one for each conversation.
    acquire: () => ({
      url: process.env.EXECUTOR_URL || 'http://localhost:8080'
    }),
    token: process.env.EXECUTOR_TOKEN
  }),
  // The CE.SDK plugins the engine applies: everything the agent can search and
  // place. Google Fonts is the smallest useful list; your own plugins go here.
  plugins: [googleFonts()],
  design: `quickstart-${Date.now()}`, // any id; with no `designs` store, the design lives in memory
  gateway: false // no image generation; the model still runs through the gateway
});

const textOf = (parts) =>
  parts
    .map((p) => (p.type === 'text' ? p.text : ''))
    .join('\n')
    .trim();

// The SDK's tools in the Agents SDK's shape.
const tools = codesign.tools.map((t) =>
  tool({
    name: t.name,
    description: t.description ?? '',
    // Plain JSON Schema, sent as it is: strict mode would make every property required.
    parameters:
      /** @type {{ type: 'object', properties: {}, required: [], additionalProperties: true }} */ (
        t.inputSchema
      ),
    strict: false,
    execute: async (args, _context, details) => {
      const result = await t.call(args ?? {}, { signal: details?.signal });
      console.log(`${result.isError ? '✗' : '✓'} ${t.name}`);
      if (result.isError) return textOf(result.content); // the refusal, in words the model can act on
      return result.content.map((p) =>
        p.type === 'text'
          ? { type: /** @type {const} */ ('text'), text: p.text }
          : {
              type: /** @type {const} */ ('image'),
              image: { data: p.data, mediaType: p.mimeType }
            }
      );
    },
    // A call the SDK rejects (the executor does not answer) ends the run with
    // its error; by default the model would get a generic message and go on.
    errorFunction: null
  })
);

// Chat Completions has no images in a tool result: move each render into a
// user message after it, so the model sees its work. Left out, nothing fails;
// the model designs blind.
/** @type {import('@openai/agents').CallModelInputFilter} */
const imagesAfterToolResults = ({ modelData }) => {
  /** @type {import('@openai/agents').AgentInputItem[]} */
  const input = [];
  /** @type {{ type: 'input_image', image: string }[]} */
  let images = [];
  const flush = () => {
    if (images.length === 0) return;
    input.push({
      role: 'user',
      content: [
        { type: 'input_text', text: 'Attached image(s) from tool result:' },
        ...images
      ]
    });
    images = [];
  };
  for (const item of modelData.input) {
    if (item.type !== 'function_call_result' || !Array.isArray(item.output)) {
      if (item.type !== 'function_call_result') flush();
      input.push(item);
      continue;
    }
    const kept = item.output.filter((part) => part.type !== 'input_image');
    for (const part of item.output) {
      if (part.type === 'input_image' && typeof part.image === 'string')
        images.push({ type: 'input_image', image: part.image });
    }
    input.push({
      ...item,
      output:
        kept.length > 0
          ? kept
          : [
              {
                type: /** @type {const} */ ('input_text'),
                text: '(see attached image)'
              }
            ]
    });
  }
  flush();
  return { ...modelData, input };
};

const agent = new Agent({
  name: 'Designer',
  instructions: `You design marketing visuals. When you are done, say in one sentence what you made.\n\n${codesign.instructions}`,
  model: process.env.MODEL || '@openrouter/anthropic/claude-opus-5',
  tools
});

const runner = new Runner({
  modelProvider: imglyGateway,
  callModelInputFilter: imagesAfterToolResults,
  tracingDisabled: true // the Agents SDK would otherwise send a trace of every run to OpenAI
});

const started = Date.now();
const seconds = () => Math.round((Date.now() - started) / 1000);
try {
  const run = await runner.run(agent, brief, { maxTurns: 60 }); // the default is 10
  console.log(`\n${run.finalOutput}`);
  const { inputTokens, outputTokens } = run.runContext.usage;
  console.log(
    `${run.rawResponses.length} steps, ${seconds()} s, ${inputTokens} input / ${outputTokens} output tokens`
  );
  await mkdir('out', { recursive: true });
  await writeFile('out/design.png', await codesign.export('png'));
  await writeFile('out/design.imgly', await codesign.export('imgly'));
  console.log('Wrote out/design.png and out/design.imgly');
} catch (err) {
  // The Agents SDK's errors carry the whole run's state; its message is enough.
  const what =
    err instanceof MaxTurnsExceededError
      ? 'The model stopped early'
      : 'The run failed';
  console.error(
    `${what} after ${seconds()} s: ${err instanceof Error ? err.message : err}; nothing exported.`
  );
  process.exitCode = 1;
} finally {
  await codesign.release(); // ends the design's session now; the engine closes about 30 s later
}
