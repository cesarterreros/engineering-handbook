# AI Agents & Modern LLM Patterns — Technical Interview Study Guide

*A datasource for retrieval, routing, tool-use, and evaluation — with production workflows and TypeScript / .NET code after every section.*

**Prepared:** August 2026 · **Scope:** Current standards and best practices for building LLM-based AI agents.

---

## How to use this document

This guide is written as a **study datasource** (e.g., for Gemini NotebookLM). Each major topic includes **Concept → How it works → Best practices → Code → Interview angles**. Code follows the explanation of each section, in both **TypeScript** (Vercel AI SDK v5+/v7, MCP SDK) and **.NET** (`Microsoft.Extensions.AI`, **Microsoft Agent Framework** — the 2025/2026 successor to Semantic Kernel + AutoGen).

> **Framing note.** SDK class names evolve quickly; the *patterns* are stable. Learn the pattern first and verify the exact symbol against current docs.

### Code conventions (so snippets stay short)

- **TypeScript** uses the **Vercel AI SDK**: `generateText`, `generateObject`, `embed`/`embedMany`, `tool`, `stepCountIs`, with `openai(...)` as the model.
- `openai.getChatCompletions(DEPLOYMENT, messages, opts)` denotes an **Azure OpenAI-style** enterprise client (used where it mirrors common infra).
- `llm(prompt)` is a thin helper = one `generateText` call returning `.text`.
- `retrieve`, `vectorSearch`, `keywordSearch`, `MemoryStore`, `cache` are **your infrastructure adapters**.
- **.NET** uses `Microsoft.Extensions.AI` (`IChatClient`, `IEmbeddingGenerator`) and **Microsoft Agent Framework** (`Microsoft.Agents.AI`).
- A shared type used throughout:

```ts
type Chunk = {
  id: string;
  title: string;
  content: string;
  source: string;       // URL or document id, for citations
  parentId?: string;    // parent document (sentence-window / parent-doc retrieval)
  score?: number;
};
```

**.NET / C#** — the same shared type as a record:

```csharp
public record Chunk(
    string Id,
    string Title,
    string Content,
    string Source,          // URL or document id, for citations
    string? ParentId = null, // parent document (sentence-window / parent-doc retrieval)
    double? Score = null);
```

---

# Part 1 — Foundations

## 1.1 What is an LLM agent?

An **agent** is a system in which a large language model **dynamically directs its own control flow and tool usage** to accomplish a goal, using feedback from the environment across multiple steps. Contrast this with a **workflow**, where LLM calls and tools are orchestrated through **fixed, developer-defined code paths**.

Anthropic's widely-cited framing ("Building Effective Agents"):

- **Workflow** = predefined, deterministic orchestration of LLM calls. Predictable, testable, cheaper.
- **Agent** = the model decides what to do next, when to call tools, and when it is done. Flexible, but higher cost/latency and harder to constrain.

Microsoft's Agent Framework uses the same distinction:

| Use an **agent** when… | Use a **workflow** when… |
|---|---|
| The task is open-ended or conversational | The process has well-defined steps |
| You need autonomous tool use and planning | You need explicit control over execution order |
| A single LLM call (with tools) suffices | Multiple agents/functions must coordinate deterministically |

> **The single most important best practice:** *If you can write a plain function to handle the task, do that instead of using an agent.* Add agentic complexity only when it **demonstrably** improves outcomes.

```ts
// WORKFLOW — the developer fixes the path. Deterministic and easy to test.
async function summarizeThenTranslate(doc: string) {
  const summary = await llm(`Summarize:\n${doc}`);
  return llm(`Translate the summary to Spanish:\n${summary}`);
}

// AGENT — the model decides what to do next, in a loop, using tools.
async function research(goal: string) {
  return runAgent({ goal, tools: [webSearch, readPage, writeNote], maxSteps: 12 });
}
```

**.NET / C#** — a fixed workflow is plain code; an agent is a `Microsoft.Agents.AI` `AIAgent`:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

// WORKFLOW — the developer fixes the path. Deterministic and easy to test.
async Task<string> SummarizeThenTranslateAsync(IChatClient chat, string doc)
{
    var summary = (await chat.GetResponseAsync($"Summarize:\n{doc}")).Text;
    return (await chat.GetResponseAsync($"Translate the summary to Spanish:\n{summary}")).Text;
}

// AGENT — the model decides what to do next, in a loop, using tools.
AIAgent research = chatClient.CreateAIAgent(
    instructions: "Research the goal autonomously. Stop when it is met.",
    tools: [
        AIFunctionFactory.Create(WebSearch),
        AIFunctionFactory.Create(ReadPage),
        AIFunctionFactory.Create(WriteNote)
    ]);

var result = await research.RunAsync("Summarize the latest agent research.");
```

## 1.2 The Augmented LLM (the base building block)

Every agent pattern is built on an **augmented LLM** — a model with three add-ons:

1. **Retrieval** — fetch relevant external knowledge (RAG).
2. **Tools** — call functions/APIs to read or change the world.
3. **Memory** — persist information across turns and sessions.

```
              +----------------------------+
   query ---> |        Augmented LLM        | ---> response
              |                             |
              |  +----------+  +---------+  |
              |  |Retrieval |  |  Tools  |  |
              |  +----------+  +---------+  |
              |        +----------+         |
              |        |  Memory  |         |
              |        +----------+         |
              +----------------------------+
```

```ts
type AugmentedLLM = {
  retrieve: (q: string) => Promise<Chunk[]>;
  tools: Record<string, Tool>;
  memory: MemoryStore;
};

async function ask(a: AugmentedLLM, userId: string, query: string) {
  const [context, memories] = await Promise.all([
    a.retrieve(query),            // retrieval
    a.memory.recall(userId, query), // long-term memory
  ]);
  return generateText({
    model: openai("gpt-5.4"),
    system: `Relevant long-term memory:\n${memories.join("\n")}`,
    tools: a.tools,               // tools
    prompt: `Context:\n${context.map((c) => c.content).join("\n")}\n\nQ: ${query}`,
  });
}
```

**.NET / C#** — retrieval + long-term memory injected into an `AIAgent` that owns the tools:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

async Task<string> AskAsync(
    IChatClient chatClient, MemoryStore memory,
    Func<string, Task<IReadOnlyList<Chunk>>> retrieve,
    IList<AITool> tools, string userId, string query)
{
    var contextTask   = retrieve(query);                 // retrieval
    var memoriesTask  = memory.RecallAsync(userId, query); // long-term memory
    await Task.WhenAll(contextTask, memoriesTask);

    var memoryText = string.Join("\n", memoriesTask.Result);
    var agent = chatClient.CreateAIAgent(
        instructions: $"Relevant long-term memory:\n{memoryText}",
        tools: tools);                                    // tools

    var context = string.Join("\n", contextTask.Result.Select(c => c.Content));
    return (await agent.RunAsync($"Context:\n{context}\n\nQ: {query}")).Text;
}
```

## 1.3 The agent loop (ReAct)

Most autonomous agents implement a **reason–act loop** (ReAct = *Reasoning + Acting*):

```
loop:
  THOUGHT   -> model reasons about the goal and current state
  ACTION    -> model emits a tool call (name + arguments)
  OBSERVE   -> runtime executes the tool, returns the result
  (repeat until the model emits a final answer or a stop condition triggers)
```

Key concerns: a **stop condition** (max steps, budget, or a `finish` signal), **state/context management** between steps, and **error recovery** when a tool fails.

```ts
// A minimal, explicit ReAct loop — what agent frameworks do under the hood.
async function reactLoop(goal: string, tools: Record<string, Tool>, maxSteps = 8) {
  const messages: Msg[] = [{ role: "user", content: goal }];

  for (let step = 0; step < maxSteps; step++) {
    const res = await openai.getChatCompletions(CHAT_DEPLOYMENT, messages, {
      tools: toToolSchemas(tools),
      temperature: 0,
    });
    const msg = res.choices[0].message;
    messages.push(msg);

    const calls = msg.toolCalls ?? [];
    if (calls.length === 0) return msg.content; // final answer -> done

    // OBSERVE: execute each requested tool and feed results back.
    for (const call of calls) {
      const tool = tools[call.function.name];
      const result = await tool.execute(JSON.parse(call.function.arguments));
      messages.push({ role: "tool", toolCallId: call.id, content: JSON.stringify(result) });
    }
  }
  throw new Error("Step budget exhausted without a final answer"); // guardrail
}
```

**.NET / C#** — Agent Framework runs the ReAct loop for you; `AIAgent.RunAsync` reasons,
calls tools, and returns the final answer (loop control via `ChatOptions`):

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

// The runtime performs THOUGHT -> ACTION -> OBSERVE automatically.
AIAgent agent = chatClient.CreateAIAgent(
    instructions: "Achieve the goal, using tools as needed.",
    tools: [AIFunctionFactory.Create(WebSearch), AIFunctionFactory.Create(ReadPage)]);

// Threads persist state between steps/turns.
AgentThread thread = agent.GetNewThread();
var answer = await agent.RunAsync(goal, thread);

// To bound the tool loop explicitly, cap iterations via ChatClientAgentRunOptions:
var bounded = await agent.RunAsync(goal, thread, new ChatClientAgentRunOptions(
    new ChatOptions { MaxOutputTokens = 2000 }));
```

## 1.4 When NOT to build an agent

- Single, deterministic transformation → **just call the model once** (or write code).
- Steps known in advance → **use a workflow**, not an agent.
- No tolerance for unpredictable cost/latency → prefer constrained workflows with gates.
- No evaluation harness → you cannot safely operate *any* agent; build evals first.

```ts
// The best "agent" is often no agent. If the logic is deterministic, write code.
function shippingCost(weightKg: number, zone: "dom" | "intl"): number {
  const base = zone === "intl" ? 20 : 5;
  return base + weightKg * (zone === "intl" ? 4 : 1.5); // no LLM needed
}
```

**.NET / C#** — the best "agent" is often no agent; deterministic logic is just code:

```csharp
static decimal ShippingCost(decimal weightKg, string zone) // "dom" | "intl"
{
    var @base = zone == "intl" ? 20m : 5m;
    return @base + weightKg * (zone == "intl" ? 4m : 1.5m); // no LLM needed
}
```

---

# Part 2 — Retrieval (RAG)

## 2.1 What & why

**Retrieval-Augmented Generation (RAG)** injects relevant external context into the prompt at inference time so the model can answer using **fresh, private, or domain-specific** knowledge it was not trained on. It reduces hallucination, enables citations, and avoids the cost/staleness of fine-tuning for knowledge that changes.

**RAG vs fine-tuning vs long-context:**
- **RAG** → dynamic/changing knowledge, citations, access control, large corpora.
- **Fine-tuning** → teach *behavior/format/style*, not facts.
- **Long-context / "just paste it in"** → small, static corpora; simplest, but costs tokens per call and degrades on very large contexts ("lost in the middle").

```ts
// Decision helper: does this request even need retrieval?
async function needsRetrieval(query: string): Promise<boolean> {
  const { object } = await generateObject({
    model: openai("gpt-5.4-mini"),
    schema: z.object({ needsKnowledge: z.boolean() }),
    prompt: `Does answering this require looking up private/factual knowledge?\n${query}`,
  });
  return object.needsKnowledge; // route chit-chat around the RAG pipeline
}
```

**.NET / C#** \u2014 typed decision via `Microsoft.Extensions.AI` structured output:

```csharp
using Microsoft.Extensions.AI;

record RetrievalDecision(bool NeedsKnowledge);

async Task<bool> NeedsRetrievalAsync(IChatClient chat, string query)
{
    var decision = (await chat.GetResponseAsync<RetrievalDecision>(
        $"Does answering this require looking up private/factual knowledge?\n{query}")).Result;
    return decision.NeedsKnowledge; // route chit-chat around the RAG pipeline
}
```

## 2.2 The RAG pipeline

```
INGESTION (offline)                         QUERY (online)
------------------                          --------------
documents                                   user query
   | load & parse                              | (optional) query transform
   v                                           v
chunking                                    embed query
   |                                           |
   v                                           v
embed chunks --> vector store <---- retrieve top-k (ANN) + keyword (BM25)
   |                                           |
   v                                           v
metadata index                             fuse (RRF) -> rerank (cross-encoder)
                                               |
                                               v
                                          build prompt (context + query) --> LLM --> answer + citations
```

The end-to-end **grounded generation** step — multi-tenant, cited, reproducible (the shape most production RAG endpoints take):

```ts
function buildMessages(query: string, chunks: Chunk[]) {
  const sources = chunks
    .map((c, i) => `[${i + 1}] (title: ${c.title}) ${c.content}`)
    .join("\n\n");

  const system = [
    "You are a product knowledge assistant. Answer ONLY using the sources provided.",
    "If the sources do not contain the answer, say you don't have enough information.",
    "Cite sources inline using their [number]. Never invent attributes, prices, or claims.",
  ].join(" ");

  return [
    { role: "system" as const, content: system },
    { role: "user"   as const, content: `Sources:\n${sources}\n\nQuestion: ${query}` },
  ];
}

async function answer(query: string, tenantId: string) {
  const chunks = await retrieve(query, tenantId); // tenant-isolated retrieval

  if (chunks.length === 0) {
    return { text: "I don't have enough information to answer that.", citations: [] };
  }

  const res = await openai.getChatCompletions(CHAT_DEPLOYMENT, buildMessages(query, chunks), {
    temperature: 0,     // grounded + reproducible
    maxTokens: 500,
  });

  return {
    text: res.choices[0].message?.content ?? "",
    citations: chunks.map((c, i) => ({
      id: i + 1, title: c.title, source: c.source, parentId: c.parentId,
    })),
  };
}
```

**.NET / C#** — the same grounded generation with `Microsoft.Extensions.AI`:

```csharp
using Microsoft.Extensions.AI;

async Task<(string Text, IReadOnlyList<Citation> Citations)> AnswerAsync(
    IChatClient chat, string query, string tenantId)
{
    var chunks = await RetrieveAsync(query, tenantId); // tenant-isolated
    if (chunks.Count == 0)
        return ("I don't have enough information to answer that.", []);

    var sources = string.Join("\n\n",
        chunks.Select((c, i) => $"[{i + 1}] (title: {c.Title}) {c.Content}"));

    var messages = new List<ChatMessage>
    {
        new(ChatRole.System,
            "Answer ONLY using the sources. Cite inline as [n]. " +
            "If the sources lack the answer, say so. Never invent facts."),
        new(ChatRole.User, $"Sources:\n{sources}\n\nQuestion: {query}")
    };

    var res = await chat.GetResponseAsync(messages,
        new ChatOptions { Temperature = 0f, MaxOutputTokens = 500 });

    var citations = chunks.Select((c, i) =>
        new Citation(i + 1, c.Title, c.Source, c.ParentId)).ToList();
    return (res.Text, citations);
}
```

## 2.3 Chunking

The model retrieves **chunks**, not documents. Chunking quality dominates RAG quality.

- **Fixed-size + overlap** — e.g., 500–1,000 tokens with 10–20% overlap. Simple baseline.
- **Recursive/structural** — split on document structure (headings, paragraphs, code blocks). Preferred default.
- **Semantic chunking** — split where embedding similarity between adjacent sentences drops.
- **Sentence-window / parent-document** — embed small units for precise matching; return a larger surrounding window (via `parentId`) to the LLM.

**Best practices:** keep chunks semantically coherent; store rich **metadata** (source, title, section, timestamp, ACL); tune size to the embedding model and content type.

```ts
// Recursive/structural chunking: split on the coarsest boundary that keeps
// chunks under the target size, carrying overlap for context continuity.
function chunkText(text: string, targetChars = 3000, overlap = 300): string[] {
  const separators = ["\n## ", "\n### ", "\n\n", "\n", ". "]; // coarse -> fine
  const out: string[] = [];

  function split(segment: string, depth: number) {
    if (segment.length <= targetChars || depth >= separators.length) {
      if (segment.trim()) out.push(segment.trim());
      return;
    }
    let buf = "";
    for (const part of segment.split(separators[depth])) {
      if ((buf + part).length > targetChars && buf) {
        out.push(buf.trim());
        buf = buf.slice(-overlap); // overlap keeps neighboring context
      }
      buf += (buf ? separators[depth] : "") + part;
    }
    if (buf.trim()) split(buf, depth + 1);
  }

  split(text, 0);
  return out;
}
```

**.NET / C#** — recursive/structural chunking with overlap:

```csharp
List<string> ChunkText(string text, int targetChars = 3000, int overlap = 300)
{
    string[] separators = ["\n## ", "\n### ", "\n\n", "\n", ". "]; // coarse -> fine
    var outList = new List<string>();

    void Split(string segment, int depth)
    {
        if (segment.Length <= targetChars || depth >= separators.Length)
        {
            if (segment.Trim().Length > 0) outList.Add(segment.Trim());
            return;
        }
        var buf = "";
        foreach (var part in segment.Split(separators[depth]))
        {
            if ((buf + part).Length > targetChars && buf.Length > 0)
            {
                outList.Add(buf.Trim());
                buf = buf[Math.Max(0, buf.Length - overlap)..]; // overlap keeps context
            }
            buf += (buf.Length > 0 ? separators[depth] : "") + part;
        }
        if (buf.Trim().Length > 0) Split(buf, depth + 1);
    }

    Split(text, 0);
    return outList;
}
```

## 2.4 Embeddings & similarity

An **embedding** maps text to a dense vector such that semantically similar text is close in vector space. Retrieval ranks chunks by **cosine similarity** (or dot product) to the query embedding.

- Use the **same model** for indexing and querying.
- Examples: OpenAI `text-embedding-3-large/small`, Cohere `embed-v3`, open models like `bge`/`e5`.
- **Asymmetric search:** some models offer query/document prefixes — use them.

```ts
import { embed, embedMany } from "ai";
const embedModel = openai.textEmbeddingModel("text-embedding-3-small");

// Ingestion: embed many chunks in one batched call.
async function indexChunks(chunks: Chunk[], store: VectorStore, tenantId: string) {
  const { embeddings } = await embedMany({ model: embedModel, values: chunks.map((c) => c.content) });
  await store.upsert(chunks.map((c, i) => ({ ...c, tenantId, vector: embeddings[i] })));
}

// Cosine similarity (when you need it explicitly).
function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
```

**.NET / C#** — batched embedding via `IEmbeddingGenerator` + cosine similarity:

```csharp
using Microsoft.Extensions.AI;

// Ingestion: embed many chunks in one batched call.
async Task IndexChunksAsync(
    IEmbeddingGenerator<string, Embedding<float>> embedder,
    IReadOnlyList<Chunk> chunks, VectorStore store, string tenantId)
{
    var embeddings = await embedder.GenerateAsync(chunks.Select(c => c.Content));
    await store.UpsertAsync(chunks.Zip(embeddings,
        (c, e) => new StoredChunk(c, tenantId, e.Vector)));
}

// Cosine similarity (when you need it explicitly).
static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
{
    float dot = 0, na = 0, nb = 0;
    for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (MathF.Sqrt(na) * MathF.Sqrt(nb));
}
```

## 2.5 Vector databases & ANN indexes

At scale you use **Approximate Nearest Neighbor (ANN)** indexes instead of brute force:

- **HNSW** — graph-based; excellent recall/latency; high memory. The common default.
- **IVF / IVF-PQ** — inverted file + product quantization; compresses vectors, lower memory, tunable recall.

**Stores:** Postgres + `pgvector`, Qdrant, Weaviate, Milvus, Pinecone, Redis, Elasticsearch/OpenSearch, Azure AI Search. Choose by ops model, filtering, and hybrid support.

```ts
// ANN search MUST filter by tenant to prevent cross-tenant leakage (see LLM08).
async function vectorSearch(query: string, tenantId: string, topK = 20): Promise<Chunk[]> {
  const { embedding } = await embed({ model: embedModel, value: query });
  return store.search(embedding, {
    topK,
    filter: { tenantId },        // metadata filter enforced at the index
    // ef/nprobe tune the recall/latency trade-off for HNSW/IVF respectively
  });
}
```

**.NET / C#** — ANN search with a mandatory tenant filter:

```csharp
using Microsoft.Extensions.AI;

// ANN search MUST filter by tenant to prevent cross-tenant leakage (see LLM08).
async Task<IReadOnlyList<Chunk>> VectorSearchAsync(
    IEmbeddingGenerator<string, Embedding<float>> embedder,
    VectorStore store, string query, string tenantId, int topK = 20)
{
    var embedding = (await embedder.GenerateAsync([query]))[0].Vector;
    return await store.SearchAsync(embedding, new SearchOptions
    {
        TopK = topK,
        Filter = new { TenantId = tenantId } // metadata filter enforced at the index
    });
}
```

## 2.6 Hybrid search & fusion

Dense retrieval misses **exact matches** (IDs, error codes). **Sparse/keyword** (BM25) misses paraphrases. **Hybrid search** runs both and fuses results with **Reciprocal Rank Fusion (RRF)**:

```
score(d) = Σ_over_retrievers  1 / (k + rank_i(d))     # k ~ 60
```

RRF combines ranked lists using rank position only, so scores need no normalization.

```ts
function reciprocalRankFusion(lists: Chunk[][], k = 60): Chunk[] {
  const scored = new Map<string, { chunk: Chunk; score: number }>();
  for (const list of lists) {
    list.forEach((chunk, rank) => {
      const add = 1 / (k + rank + 1);
      const prev = scored.get(chunk.id);
      if (prev) prev.score += add;
      else scored.set(chunk.id, { chunk, score: add });
    });
  }
  return [...scored.values()].sort((a, b) => b.score - a.score).map((s) => s.chunk);
}

async function hybridSearch(query: string, tenantId: string, topK = 20): Promise<Chunk[]> {
  const [dense, sparse] = await Promise.all([
    vectorSearch(query, tenantId, topK),  // ANN over embeddings
    keywordSearch(query, tenantId, topK), // BM25 / full-text
  ]);
  return reciprocalRankFusion([dense, sparse]).slice(0, topK);
}
```

**.NET / C#** — Reciprocal Rank Fusion over dense + sparse lists:

```csharp
List<Chunk> ReciprocalRankFusion(IEnumerable<IReadOnlyList<Chunk>> lists, int k = 60)
{
    var scored = new Dictionary<string, (Chunk Chunk, double Score)>();
    foreach (var list in lists)
        for (var rank = 0; rank < list.Count; rank++)
        {
            var chunk = list[rank];
            var add = 1.0 / (k + rank + 1);
            scored[chunk.Id] = scored.TryGetValue(chunk.Id, out var prev)
                ? (prev.Chunk, prev.Score + add)
                : (chunk, add);
        }
    return scored.Values.OrderByDescending(s => s.Score).Select(s => s.Chunk).ToList();
}

async Task<List<Chunk>> HybridSearchAsync(string query, string tenantId, int topK = 20)
{
    var denseTask  = VectorSearchAsync(query, tenantId, topK);  // ANN over embeddings
    var sparseTask = KeywordSearchAsync(query, tenantId, topK); // BM25 / full-text
    await Task.WhenAll(denseTask, sparseTask);
    return ReciprocalRankFusion([denseTask.Result, sparseTask.Result]).Take(topK).ToList();
}
```

## 2.7 Reranking

Initial retrieval favors recall (candidates cheaply). A **reranker** — a **cross-encoder** that jointly encodes (query, chunk) — reorders the top ~50–100 for precision; keep the top ~5. Cross-encoders are far more accurate than bi-encoders but too slow for the whole corpus — hence two stages.

```ts
// Two-stage retrieval: cheap recall (hybrid) -> precise rerank -> top N.
async function retrieve(query: string, tenantId: string, finalK = 5): Promise<Chunk[]> {
  const candidates = await hybridSearch(query, tenantId, 50);
  if (candidates.length === 0) return [];

  const reranked = await cohere.rerank({
    model: "rerank-v3.5",
    query,
    documents: candidates.map((c) => c.content),
    topN: finalK,
  });

  // Map reranked positions back to full chunk objects (preserve metadata/citations).
  return reranked.results.map((r) => ({ ...candidates[r.index], score: r.relevanceScore }));
}
```

**.NET / C#** — two-stage retrieval: hybrid recall then cross-encoder rerank:

```csharp
async Task<List<Chunk>> RetrieveAsync(
    ICohereReranker cohere, string query, string tenantId, int finalK = 5)
{
    var candidates = await HybridSearchAsync(query, tenantId, 50);
    if (candidates.Count == 0) return [];

    var reranked = await cohere.RerankAsync(new RerankRequest
    {
        Model = "rerank-v3.5",
        Query = query,
        Documents = candidates.Select(c => c.Content).ToList(),
        TopN = finalK
    });

    // Map reranked positions back to full chunk objects (preserve metadata/citations).
    return reranked.Results
        .Select(r => candidates[r.Index] with { Score = r.RelevanceScore })
        .ToList();
}
```

## 2.8 Query transformation

- **Multi-query** — generate paraphrases, retrieve for each, union/fuse.
- **HyDE** — draft a hypothetical answer, embed *that*, retrieve against it.
- **Decomposition** — split a complex question into sub-questions.

```ts
// Multi-query expansion: paraphrase, retrieve per variant, fuse the lists.
async function multiQueryRetrieve(query: string, tenantId: string): Promise<Chunk[]> {
  const { object } = await generateObject({
    model: openai("gpt-5.4-mini"),
    schema: z.object({ queries: z.array(z.string()).length(3) }),
    prompt: `Rewrite this question 3 different ways to maximize retrieval recall:\n${query}`,
  });

  const lists = await Promise.all(
    [query, ...object.queries].map((q) => hybridSearch(q, tenantId, 20)),
  );
  return reciprocalRankFusion(lists).slice(0, 20);
}
```

**.NET / C#** — multi-query expansion with typed output, then fuse:

```csharp
using Microsoft.Extensions.AI;

record QueryExpansion(string[] Queries);

async Task<List<Chunk>> MultiQueryRetrieveAsync(IChatClient chat, string query, string tenantId)
{
    var expansion = (await chat.GetResponseAsync<QueryExpansion>(
        $"Rewrite this question 3 different ways to maximize retrieval recall:\n{query}")).Result;

    var variants = new[] { query }.Concat(expansion.Queries);
    var lists = await Task.WhenAll(variants.Select(q => HybridSearchAsync(q, tenantId, 20)));
    return ReciprocalRankFusion(lists).Take(20).ToList();
}
```

## 2.9 Advanced patterns

- **Agentic RAG** — the model treats retrieval as a **tool** it can call iteratively (search → read → refine → search). Where most production systems are heading.
- **Contextual Retrieval** (Anthropic) — prepend an LLM-generated context blurb to each chunk before embedding; sharply reduces failed retrievals.
- **GraphRAG** (Microsoft) — build a knowledge graph; retrieve via entities/communities for global "summarize the corpus" questions.

```ts
// Agentic RAG: expose search as a tool so the model can iterate on its own.
const searchKB = tool({
  description: "Search the tenant knowledge base. Call repeatedly to refine the query.",
  inputSchema: z.object({ query: z.string() }),
  execute: async ({ query }) => retrieve(query, currentTenantId, 5),
});

const { text, steps } = await generateText({
  model: openai("gpt-5.4"),
  tools: { searchKB },
  stopWhen: stepCountIs(4),        // bound the retrieve/refine loop
  system: "Search as many times as needed, then answer with citations.",
  prompt: userQuestion,
});
```

**.NET / C#** — Agentic RAG: expose search as a tool so the agent iterates on its own:

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

[Description("Search the tenant knowledge base. Call repeatedly to refine the query.")]
async Task<IReadOnlyList<Chunk>> SearchKb([Description("Search query")] string query)
    => await RetrieveAsync(query, currentTenantId, 5);

AIAgent agent = chatClient.CreateAIAgent(
    instructions: "Search as many times as needed, then answer with citations.",
    tools: [AIFunctionFactory.Create(SearchKb)]);

var answer = await agent.RunAsync(userQuestion);
```

## 2.10 Evaluating RAG

Evaluate **retrieval** and **generation** separately (RAGAS-style):

- **Context precision / recall** — did retrieval return the right chunks (and rank them high)?
- **Faithfulness / groundedness** — is the answer supported by the context?
- **Answer relevancy** — does the answer address the question?

Failure modes to name: bad chunking, embedding/query mismatch, no hybrid search for exact terms, no reranking, stale index, retrieved-but-ignored context, context overflow.

```ts
// Faithfulness check: is every claim in the answer supported by the sources?
async function faithfulness(answerText: string, sources: Chunk[]) {
  const { object } = await generateObject({
    model: openai("gpt-5.4"),        // judge model
    schema: z.object({ supported: z.boolean(), unsupportedClaims: z.array(z.string()) }),
    prompt:
      `Sources:\n${sources.map((s) => s.content).join("\n")}\n\n` +
      `Answer:\n${answerText}\n\nList any claims NOT supported by the sources.`,
  });
  return object; // score = supported ? 1 : 0, plus the offending claims
}
```

**.NET / C#** — faithfulness check via typed judge output:

```csharp
using Microsoft.Extensions.AI;

record Faithfulness(bool Supported, string[] UnsupportedClaims);

async Task<Faithfulness> CheckFaithfulnessAsync(
    IChatClient judge, string answerText, IReadOnlyList<Chunk> sources)
{
    var joined = string.Join("\n", sources.Select(s => s.Content));
    return (await judge.GetResponseAsync<Faithfulness>(
        $"Sources:\n{joined}\n\nAnswer:\n{answerText}\n\n" +
        "List any claims NOT supported by the sources.")).Result;
}
```

---

# Part 3 — Routing

## 3.1 What & why

**Routing** classifies a request and directs it to the best handler — a specialized prompt, a specific model, a tool, or a sub-agent. It lets you optimize **cost, latency, and quality** simultaneously and keep prompts focused.

## 3.2 Types of routing

- **Intent / semantic routing** — classify (billing vs technical vs sales) → specialized flow.
- **Model routing** — easy queries → small/cheap/fast model; hard → large model. A major cost lever.
- **Tool / capability routing** — which tool or data source is relevant.
- **Retrieve-or-not routing** — is RAG even needed?
- **Escalation routing** — low-confidence/high-risk → human.

## 3.3 Implementation approaches & best practices

1. **LLM classifier** (flexible; force a valid label via structured output).
2. **Embedding similarity** (fast/cheap; nearest prototype route).
3. **Rules / regex / metadata** (deterministic, cheapest).

Layer them: rules → embeddings → LLM fallback. **Constrain output to an enum**, handle the **low-confidence** case, keep the router **cheap and fast**, and **log/evaluate** routing decisions.

```ts
import { generateObject, generateText } from "ai";
import { z } from "zod";

async function route(query: string) {
  const { object } = await generateObject({
    model: openai("gpt-5.4-mini"), // cheap, fast router
    schema: z.object({
      category: z.enum(["billing", "technical", "sales", "other"]),
      complexity: z.enum(["simple", "hard"]),
      confidence: z.number().min(0).max(1),
    }),
    prompt: `Classify this request:\n${query}`,
  });

  if (object.confidence < 0.5) return escalateToHuman(query); // low-confidence fallback

  const model = object.complexity === "hard" ? openai("gpt-5.4") : openai("gpt-5.4-mini");
  const system = {
    billing: "You are a billing specialist.",
    technical: "You are a senior technical support engineer.",
    sales: "You are a sales assistant.",
    other: "You are a helpful general assistant.",
  }[object.category];

  return generateText({ model, system, prompt: query }); // model + prompt routing
}
```

**.NET / C#** — typed classification via `Microsoft.Extensions.AI`:

```csharp
using Microsoft.Extensions.AI;

public enum Category { Billing, Technical, Sales, Other }
public record Routing(Category Category, bool IsHard, double Confidence);

async Task<string> HandleAsync(IChatClient chat, string query)
{
    var routing = (await chat.GetResponseAsync<Routing>(
        $"Classify into category, difficulty, and confidence:\n{query}")).Result;

    if (routing.Confidence < 0.5) return await EscalateAsync(query);

    var system = routing.Category switch
    {
        Category.Billing   => "You are a billing specialist.",
        Category.Technical => "You are a senior support engineer.",
        Category.Sales     => "You are a sales assistant.",
        _                  => "You are a helpful assistant."
    };

    var answer = await chat.GetResponseAsync(
        [ new(ChatRole.System, system), new(ChatRole.User, query) ]);
    return answer.Text;
}
```

---

# Part 4 — Tool use / function calling

## 4.1 How function calling works

```
1. Send the model the user message + tool schemas (name, description, JSON-schema params).
2. The model responds with text OR one/more tool calls (name + JSON arguments).
3. Your runtime executes the tool(s) and appends the results to the conversation.
4. Call the model again with the results; it calls more tools or produces a final answer.
```

The **model never executes anything** — it only *requests* calls. Your code executes and stays in control (critical for security).

## 4.2 Designing tools — the Agent-Computer Interface (ACI)

- Write tool descriptions like **"a great docstring for a junior developer."**
- Keep schemas **simple and unambiguous**; prefer formats the model has seen a lot.
- **"Poka-yoke"** — design so misuse is hard (e.g., destructive actions need explicit confirmation).
- Return **structured, informative results and errors** so the model can self-correct.
- **Fewer, well-scoped tools** beat many overlapping ones.

```ts
import { generateText, tool, stepCountIs } from "ai";
import { z } from "zod";

const getWeather = tool({
  description: "Get current weather for a city. Use when the user asks about weather.",
  inputSchema: z.object({ city: z.string().describe('City name, e.g. "Madrid"') }),
  execute: async ({ city }) => {
    try {
      return await weatherApi.current(city); // structured result
    } catch (e) {
      return { error: `Could not fetch weather for ${city}. Ask the user to confirm the city.` };
    }
  },
});

const result = await generateText({
  model: openai("gpt-5.4"),
  tools: { getWeather },
  stopWhen: stepCountIs(5),      // loop control / guardrail
  prompt: "What should I wear in Madrid today?",
});
```

> **v7 note:** the same loop is encapsulated by `new ToolLoopAgent({ model, tools }).generate({ prompt })`, the current recommended abstraction; loop control uses `stopWhen` + `prepareStep`.

**.NET / C#** — automatic function invocation:

```csharp
using Microsoft.Extensions.AI;
using System.ComponentModel;

[Description("Get the current weather for a city.")]
static string GetWeather([Description("City name, e.g. Madrid")] string city)
    => $"{city}: 21C, sunny";

IChatClient chat = baseClient
    .AsBuilder()
    .UseFunctionInvocation() // runs the tool loop automatically
    .Build();

var options = new ChatOptions { Tools = [ AIFunctionFactory.Create(GetWeather) ] };
var response = await chat.GetResponseAsync("What should I wear in Madrid today?", options);
```

## 4.3 Parallel calls, structured outputs, error handling

- Models can emit **multiple tool calls per turn** — execute independent ones concurrently.
- **Structured outputs** (JSON-schema-constrained) guarantee parseable results — use instead of "please return JSON".
- **Validate arguments**, set **timeouts**, **retry** transient failures, cap the loop with **max-step/budget**, make tools **idempotent**.

```ts
// Validate + guard a side-effecting tool; return a clear error the model can act on.
const refund = tool({
  description: "Issue a refund. Requires an explicit confirm=true to prevent accidental refunds.",
  inputSchema: z.object({
    orderId: z.string(),
    amount: z.number().positive(),
    confirm: z.literal(true).describe("Must be true; ask the user to confirm first."),
  }),
  execute: async ({ orderId, amount }) => {
    const order = await orders.get(orderId);
    if (!order) return { error: "Order not found." };
    if (amount > order.total) return { error: "Refund exceeds order total." };
    return payments.refund(orderId, amount); // idempotent by orderId
  },
});
```

**.NET / C#** — a guarded, side-effecting tool that returns clear errors the model can act on:

```csharp
using System.ComponentModel;
using Microsoft.Extensions.AI;

[Description("Issue a refund. Requires confirm=true to prevent accidental refunds.")]
async Task<object> Refund(
    [Description("Order id")] string orderId,
    [Description("Positive amount to refund")] decimal amount,
    [Description("Must be true; ask the user to confirm first.")] bool confirm)
{
    if (!confirm) return new { error = "Set confirm=true after the user confirms." };
    var order = await Orders.GetAsync(orderId);
    if (order is null)       return new { error = "Order not found." };
    if (amount > order.Total) return new { error = "Refund exceeds order total." };
    return await Payments.RefundAsync(orderId, amount); // idempotent by orderId
}

var options = new ChatOptions { Tools = [AIFunctionFactory.Create(Refund)] };
```

## 4.4 Model Context Protocol (MCP)

**MCP** is an open standard (Anthropic, late 2024; broad adoption 2025–2026) for connecting AI apps to external tools/data — *"a USB-C port for AI apps."* Build once, reuse across clients (Claude, ChatGPT, VS Code, Cursor, …).

- **Host** (the AI app) → **Client** (1:1 connection) → **Server** (exposes capabilities).
- **Primitives:** **tools** (actions), **resources** (readable data/context), **prompts** (reusable templates).
- **Transports:** local **stdio**, remote **streamable HTTP**.

MCP decouples *tool/data providers* from *agent applications*: a provider ships one server; every MCP-capable client can use it.

```ts
// TypeScript — consume an MCP server's tools with the AI SDK.
import { experimental_createMCPClient as createMCPClient, generateText, stepCountIs } from "ai";

const mcp = await createMCPClient({
  transport: { type: "stdio", command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"] },
});
const mcpTools = await mcp.tools();

const { text } = await generateText({
  model: openai("gpt-5.4"),
  tools: mcpTools,               // MCP tools used exactly like native tools
  stopWhen: stepCountIs(5),
  prompt: "Summarize the newest markdown file in ./data.",
});
await mcp.close();
```

```csharp
// .NET — expose an MCP server's tools to a Microsoft Agent Framework agent.
using Microsoft.Agents.AI;
using ModelContextProtocol.Client;

await using var mcp = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Command = "npx",
    Arguments = ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
}));
var tools = await mcp.ListToolsAsync();

AIAgent agent = chatClient.CreateAIAgent(
    instructions: "Use the file tools when asked about files.",
    tools: [.. tools]);

Console.WriteLine(await agent.RunAsync("List the markdown files in the data folder."));
```

## 4.5 Guardrails around tools

- **Least privilege** — scope credentials per tool; never hand the model raw secrets.
- **Human-in-the-loop / approval** for high-risk actions (payments, deletes, external sends).
- **Sandboxing** for code/file access; **allow/deny lists**; rate limits.

```ts
// Human-in-the-loop gate: pause the agent for approval before risky actions.
async function guardedExecute(call: ToolCall, ctx: Ctx) {
  const HIGH_RISK = new Set(["refund", "deleteAccount", "sendEmail"]);
  if (HIGH_RISK.has(call.name)) {
    const approval = await ctx.requestHumanApproval(call); // blocks until approved/denied
    if (!approval.approved) return { error: "Action denied by human reviewer." };
  }
  return tools[call.name].execute(call.args);
}
```

**.NET / C#** — wrap tools with a human-in-the-loop approval gate. Agent Framework also
supports `FunctionApprovalRequest`/`FunctionApprovalResponse` for built-in approvals:

```csharp
using Microsoft.Extensions.AI;

static readonly HashSet<string> HighRisk = ["refund", "deleteAccount", "sendEmail"];

// Human-in-the-loop gate: pause before risky actions.
AIFunction Guarded(AIFunction inner, IApprovalContext ctx) =>
    AIFunctionFactory.Create(async (AIFunctionArguments args, CancellationToken ct) =>
    {
        if (HighRisk.Contains(inner.Name))
        {
            var approval = await ctx.RequestHumanApprovalAsync(inner.Name, args); // blocks
            if (!approval.Approved) return "Action denied by human reviewer.";
        }
        return await inner.InvokeAsync(args, ct);
    }, inner.Name, inner.Description);
```

---

# Part 5 — Agent architectures & workflow patterns

Anthropic's five composable **workflow** patterns plus the autonomous **agent** — the vocabulary interviewers use.

## 5.1 Prompt chaining
Decompose into a fixed sequence; each call feeds the next, with optional programmatic **gates**. Trades latency for accuracy.

```ts
async function writeArticle(topic: string) {
  const outline = await llm(`Outline a short article about: ${topic}`);

  // GATE: cheap programmatic check before spending more tokens.
  if (outline.split("\n").filter(Boolean).length < 3) {
    throw new Error("Outline too thin; aborting chain.");
  }

  const draft = await llm(`Write the article from this outline:\n${outline}`);
  return llm(`Tighten prose and fix grammar, keep meaning:\n${draft}`);
}
```

**.NET / C#** — prompt chaining with a programmatic gate:

```csharp
using Microsoft.Extensions.AI;

async Task<string> WriteArticleAsync(IChatClient chat, string topic)
{
    var outline = (await chat.GetResponseAsync($"Outline a short article about: {topic}")).Text;

    // GATE: cheap programmatic check before spending more tokens.
    if (outline.Split('\n').Count(l => l.Length > 0) < 3)
        throw new InvalidOperationException("Outline too thin; aborting chain.");

    var draft = (await chat.GetResponseAsync($"Write the article from this outline:\n{outline}")).Text;
    return (await chat.GetResponseAsync($"Tighten prose and fix grammar, keep meaning:\n{draft}")).Text;
}
```

## 5.2 Routing
Classify input, dispatch to a specialized follow-up (see Part 3).

## 5.3 Parallelization
- **Sectioning** — split independent subtasks, run in parallel, merge (speed).
- **Voting** — run the same task N times, aggregate (confidence).

```ts
// SECTIONING: independent reviews in parallel, then merge.
async function reviewPR(diff: string) {
  const [security, style, tests] = await Promise.all([
    llm(`Review for SECURITY issues only:\n${diff}`),
    llm(`Review for STYLE issues only:\n${diff}`),
    llm(`Review for MISSING TESTS only:\n${diff}`),
  ]);
  return { security, style, tests };
}

// VOTING: run N times, take the majority verdict (reduces variance on hard calls).
async function isToxic(text: string, n = 5): Promise<boolean> {
  const votes = await Promise.all(
    Array.from({ length: n }, () =>
      generateObject({
        model: openai("gpt-5.4-mini"),
        schema: z.object({ toxic: z.boolean() }),
        prompt: `Is this text toxic?\n${text}`,
      }).then((r) => r.object.toxic)),
  );
  return votes.filter(Boolean).length > n / 2;
}
```

**.NET / C#** — sectioning (parallel) and voting (majority):

```csharp
using Microsoft.Extensions.AI;

// SECTIONING: independent reviews in parallel, then merge.
async Task<(string Security, string Style, string Tests)> ReviewPrAsync(IChatClient chat, string diff)
{
    var security = chat.GetResponseAsync($"Review for SECURITY issues only:\n{diff}");
    var style    = chat.GetResponseAsync($"Review for STYLE issues only:\n{diff}");
    var tests    = chat.GetResponseAsync($"Review for MISSING TESTS only:\n{diff}");
    await Task.WhenAll(security, style, tests);
    return (security.Result.Text, style.Result.Text, tests.Result.Text);
}

record ToxicVerdict(bool Toxic);

// VOTING: run N times, take the majority verdict.
async Task<bool> IsToxicAsync(IChatClient chat, string text, int n = 5)
{
    var votes = await Task.WhenAll(Enumerable.Range(0, n).Select(async _ =>
        (await chat.GetResponseAsync<ToxicVerdict>($"Is this text toxic?\n{text}")).Result.Toxic));
    return votes.Count(v => v) > n / 2;
}
```

## 5.4 Orchestrator–workers
A central **orchestrator** dynamically decomposes a task, delegates to **workers**, and synthesizes results. Subtasks are **not known in advance**.

```ts
async function orchestrate(task: string) {
  // 1. Orchestrator decomposes dynamically.
  const { object } = await generateObject({
    model: openai("gpt-5.4"),
    schema: z.object({ subtasks: z.array(z.object({ id: z.string(), instruction: z.string() })) }),
    prompt: `Break this task into independent subtasks:\n${task}`,
  });

  // 2. Workers run in parallel.
  const results = await Promise.all(
    object.subtasks.map((s) =>
      generateText({ model: openai("gpt-5.4"), prompt: s.instruction })
        .then((r) => ({ id: s.id, output: r.text }))),
  );

  // 3. Orchestrator synthesizes.
  const { text } = await generateText({
    model: openai("gpt-5.4"),
    prompt: `Task: ${task}\n\nWorker outputs:\n${JSON.stringify(results)}\n\nSynthesize the final answer.`,
  });
  return text;
}
```

**.NET / C#** — orchestrator dynamically decomposes, workers run in parallel, orchestrator synthesizes.
Agent Framework also ships typed graph **workflows** for this pattern:

```csharp
using System.Text.Json;
using Microsoft.Extensions.AI;

record Subtask(string Id, string Instruction);
record Plan(Subtask[] Subtasks);

async Task<string> OrchestrateAsync(IChatClient chat, string task)
{
    // 1. Orchestrator decomposes dynamically.
    var plan = (await chat.GetResponseAsync<Plan>(
        $"Break this task into independent subtasks:\n{task}")).Result;

    // 2. Workers run in parallel.
    var results = await Task.WhenAll(plan.Subtasks.Select(async s =>
        new { s.Id, Output = (await chat.GetResponseAsync(s.Instruction)).Text }));

    // 3. Orchestrator synthesizes.
    return (await chat.GetResponseAsync(
        $"Task: {task}\n\nWorker outputs:\n{JsonSerializer.Serialize(results)}\n\n" +
        "Synthesize the final answer.")).Text;
}
```

## 5.5 Evaluator–optimizer
A **generator** produces a candidate; an **evaluator** critiques against explicit criteria; loop until it passes.

```
generate --> evaluate --(pass?)--> done
   ^                          |
   +--------- feedback <-------+ (fail)
```

```ts
async function generateWithFeedback(task: string, maxRounds = 3) {
  let candidate = await llm(`Complete the task:\n${task}`);

  for (let round = 0; round < maxRounds; round++) {
    const { object: verdict } = await generateObject({
      model: openai("gpt-5.4"),
      schema: z.object({ pass: z.boolean(), feedback: z.string() }),
      prompt: `Task: ${task}\n\nCandidate:\n${candidate}\n\nDoes it fully meet the criteria?`,
    });
    if (verdict.pass) return candidate;
    candidate = await llm(
      `Task:\n${task}\n\nPrevious attempt:\n${candidate}\n\nApply this feedback:\n${verdict.feedback}`);
  }
  return candidate; // best effort after the budget
}
```

**.NET / C#** — evaluator–optimizer loop with a typed verdict:

```csharp
using Microsoft.Extensions.AI;

record Verdict(bool Pass, string Feedback);

async Task<string> GenerateWithFeedbackAsync(IChatClient chat, string task, int maxRounds = 3)
{
    var candidate = (await chat.GetResponseAsync($"Complete the task:\n{task}")).Text;

    for (var round = 0; round < maxRounds; round++)
    {
        var verdict = (await chat.GetResponseAsync<Verdict>(
            $"Task: {task}\n\nCandidate:\n{candidate}\n\nDoes it fully meet the criteria?")).Result;
        if (verdict.Pass) return candidate;
        candidate = (await chat.GetResponseAsync(
            $"Task:\n{task}\n\nPrevious attempt:\n{candidate}\n\nApply this feedback:\n{verdict.Feedback}")).Text;
    }
    return candidate; // best effort after the budget
}
```

## 5.6 Autonomous agent
The model plans and acts in an open-ended loop using **environmental feedback**, deciding its own steps until done or stopped. Use for open-ended tasks in **trusted, sandboxed** environments with **guardrails** and a **stop condition**.

```ts
import { generateText, stepCountIs } from "ai";

const result = await generateText({
  model: openai("gpt-5.4"),
  tools: { webSearch, runCode, writeFile, openPR },
  stopWhen: stepCountIs(15),        // hard guardrail on the loop
  system: "Work autonomously. Plan, act, verify each step. Stop when the goal is met.",
  prompt: "Investigate the failing CI build and open a fix PR.",
});
```

**.NET / C#** — an autonomous `AIAgent` that plans and acts until the goal is met:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

AIAgent agent = chatClient.CreateAIAgent(
    instructions: "Work autonomously. Plan, act, verify each step. Stop when the goal is met.",
    tools: [
        AIFunctionFactory.Create(WebSearch),
        AIFunctionFactory.Create(RunCode),
        AIFunctionFactory.Create(WriteFile),
        AIFunctionFactory.Create(OpenPr)
    ]);

var result = await agent.RunAsync("Investigate the failing CI build and open a fix PR.");
```

## 5.7 Planning & reasoning strategies

- **ReAct** — interleave reasoning and tool actions (the default).
- **Plan-and-Execute** — plan up front, then execute (fewer calls, better for long tasks; less adaptive).
- **Reflection / Reflexion** — the agent critiques its own trajectory and retries.
- **Tree/graph search** — explore branches, pick the best (expensive).

```ts
// Plan-and-Execute: generate a plan once, then execute steps in order.
async function planAndExecute(goal: string) {
  const { object: plan } = await generateObject({
    model: openai("gpt-5.4"),
    schema: z.object({ steps: z.array(z.string()) }),
    prompt: `Produce a concise ordered plan to achieve:\n${goal}`,
  });

  const log: string[] = [];
  for (const step of plan.steps) {
    const { text } = await generateText({
      model: openai("gpt-5.4"),
      tools: { webSearch, runCode },
      stopWhen: stepCountIs(4),
      prompt: `Goal: ${goal}\nProgress so far:\n${log.join("\n")}\n\nDo this step: ${step}`,
    });
    log.push(`${step} -> ${text}`);
  }
  return log;
}
```

**.NET / C#** — Plan-and-Execute: plan once, then execute steps with a tool-using agent:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

record ExecutionPlan(string[] Steps);

async Task<List<string>> PlanAndExecuteAsync(IChatClient chatClient, string goal)
{
    var plan = (await chatClient.GetResponseAsync<ExecutionPlan>(
        $"Produce a concise ordered plan to achieve:\n{goal}")).Result;

    AIAgent worker = chatClient.CreateAIAgent(
        instructions: "Execute the given step using tools.",
        tools: [AIFunctionFactory.Create(WebSearch), AIFunctionFactory.Create(RunCode)]);

    var log = new List<string>();
    foreach (var step in plan.Steps)
    {
        var progress = string.Join("\n", log);
        var text = (await worker.RunAsync(
            $"Goal: {goal}\nProgress so far:\n{progress}\n\nDo this step: {step}")).Text;
        log.Add($"{step} -> {text}");
    }
    return log;
}
```

## 5.8 Memory

- **Short-term (working)** — the current context window.
- **Long-term** — persisted across sessions: **episodic** (events, via RAG), **semantic** (durable facts/preferences), **procedural** (skills).
- **Implementation** — vector/key-value store; summarize at session end; retrieve relevant memories at the next session start.

```ts
// Write durable facts at session end; recall relevant ones next time (semantic memory).
async function persistMemory(store: MemoryStore, userId: string, transcript: string) {
  const { object } = await generateObject({
    model: openai("gpt-5.4-mini"),
    schema: z.object({ facts: z.array(z.string()) }),
    prompt: `Extract durable user facts/preferences worth remembering long-term:\n${transcript}`,
  });
  for (const fact of object.facts) {
    const { embedding } = await embed({ model: embedModel, value: fact });
    await store.upsert({ userId, text: fact, vector: embedding });
  }
}

async function recallMemory(store: MemoryStore, userId: string, query: string) {
  const { embedding } = await embed({ model: embedModel, value: query });
  return store.search(embedding, { userId, topK: 5 }); // inject into the next system prompt
}
```

**.NET / C#** — persist durable facts at session end, recall them next session:

```csharp
using Microsoft.Extensions.AI;

record Facts(string[] Items);

async Task PersistMemoryAsync(
    IChatClient chat, IEmbeddingGenerator<string, Embedding<float>> embedder,
    MemoryStore store, string userId, string transcript)
{
    var facts = (await chat.GetResponseAsync<Facts>(
        $"Extract durable user facts/preferences worth remembering long-term:\n{transcript}")).Result;

    foreach (var fact in facts.Items)
    {
        var embedding = (await embedder.GenerateAsync([fact]))[0].Vector;
        await store.UpsertAsync(new { UserId = userId, Text = fact, Vector = embedding });
    }
}

async Task<IReadOnlyList<MemoryHit>> RecallMemoryAsync(
    IEmbeddingGenerator<string, Embedding<float>> embedder,
    MemoryStore store, string userId, string query)
{
    var embedding = (await embedder.GenerateAsync([query]))[0].Vector;
    return await store.SearchAsync(embedding, userId, topK: 5); // inject into next system prompt
}
```

## 5.9 Context engineering

As context grows, quality and cost degrade ("lost in the middle"). Techniques: **compaction/summarization**, **selective retrieval**, **structured scratchpads/todo tracking**, and **sub-agent isolation**.

```ts
// Compaction: summarize old turns once the window gets large, keep recent turns verbatim.
async function compact(messages: Msg[], keepRecent = 6): Promise<Msg[]> {
  if (messages.length <= keepRecent + 2) return messages;
  const old = messages.slice(0, -keepRecent);
  const recent = messages.slice(-keepRecent);
  const summary = await llm(`Summarize this conversation, preserving decisions and open tasks:\n${JSON.stringify(old)}`);
  return [{ role: "system", content: `Conversation summary so far:\n${summary}` }, ...recent];
}
```

**.NET / C#** — compaction: summarize old turns, keep recent turns verbatim:

```csharp
using Microsoft.Extensions.AI;

async Task<List<ChatMessage>> CompactAsync(IChatClient chat, List<ChatMessage> messages, int keepRecent = 6)
{
    if (messages.Count <= keepRecent + 2) return messages;

    var old    = messages[..^keepRecent];
    var recent = messages[^keepRecent..];
    var oldText = string.Join("\n", old.Select(m => $"{m.Role}: {m.Text}"));
    var summary = (await chat.GetResponseAsync(
        $"Summarize this conversation, preserving decisions and open tasks:\n{oldText}")).Text;

    return [new(ChatRole.System, $"Conversation summary so far:\n{summary}"), .. recent];
}
```

## 5.10 Multi-agent systems

Topologies: **supervisor/orchestrator** (most controllable), **handoff**, **network/group chat**. Multi-agent helps with separable roles, parallelizable subtasks, and context/tool isolation; it hurts via latency, cost, and error compounding. **Prefer a single well-designed agent until you have a concrete reason to split.**

```ts
// Supervisor routes to specialist sub-agents and owns the final answer.
const specialists = {
  billing:   (q: string) => generateText({ model: openai("gpt-5.4"), system: "Billing expert.", prompt: q }),
  technical: (q: string) => generateText({ model: openai("gpt-5.4"), system: "Senior SRE.",     prompt: q }),
};

async function supervisor(query: string) {
  const { object } = await generateObject({
    model: openai("gpt-5.4-mini"),
    schema: z.object({ agent: z.enum(["billing", "technical"]) }),
    prompt: `Which specialist should handle this?\n${query}`,
  });
  const { text } = await specialists[object.agent](query);
  return { handledBy: object.agent, text };
}
```

**.NET / C#** — multiple agents with a supervisor (Microsoft Agent Framework):

```csharp
using Microsoft.Agents.AI;

AIAgent billing   = chatClient.CreateAIAgent("You are a billing expert.");
AIAgent technical = chatClient.CreateAIAgent("You are a senior SRE.");

// Simple supervisor: classify, then delegate. (Agent Framework also provides
// built-in graph workflows with type-safe routing, handoff, and checkpointing.)
async Task<string> SupervisorAsync(string query)
{
    var route = (await chatClient.GetResponseAsync<string>(
        $"Reply with exactly 'billing' or 'technical' for:\n{query}")).Result.Trim();
    var agent = route == "billing" ? billing : technical;
    return (await agent.RunAsync(query)).Text;
}
```

---

# Part 6 — Evaluation & observability

> *"You cannot improve — or safely ship — what you cannot measure."*

## 6.1 Why evals
LLM systems are **non-deterministic** and regress silently when you change a prompt, model, or retrieval setting. Evals give a repeatable score to compare versions and catch regressions in CI.

## 6.2 Types of evaluation
- **Deterministic / assertion-based** — exact match, regex, JSON-schema validity, "did it call tool X", latency/cost budgets. Cheap and reliable.
- **LLM-as-judge** — reference-based, reference-free/rubric, or **pairwise**.
- **Human evaluation** — gold standard; used to calibrate the judge.

## 6.3 What to measure

| Capability | Metrics |
|---|---|
| **Retrieval** | context precision, context recall, MRR/NDCG |
| **Generation** | faithfulness/groundedness, answer relevancy, correctness |
| **Tool use** | correct tool selected, valid arguments, task completion |
| **Routing** | classification accuracy, confusion matrix |
| **Agent trajectory** | goal reached, # steps, cost, error recovery |
| **Safety** | refusal on unsafe input, injection resistance |
| **Ops** | latency (p50/p95), cost/req, error rate |

## 6.4 Component vs end-to-end
- **Component (unit) evals** — retrieval, routing, a single tool/prompt in isolation.
- **End-to-end / trajectory evals** — the whole run, including the **sequence of steps** — essential for agents.

## 6.5 Datasets, online eval, CI
Build a **golden set** from real traffic + edge cases; augment with **synthetic data** (validate a sample). Add **guardrails** (real-time input/output checks), **online metrics** (thumbs, completion, escalation), **A/B + canary**, and **regression testing in CI** (fail the build on regression).

```ts
// Minimal offline harness: deterministic assertions + LLM-as-judge over a dataset.
type Case = { input: string; mustContain?: string; expectTool?: string };
const dataset: Case[] = [
  { input: "What is 2+2?", mustContain: "4" },
  { input: "Refund order 123", expectTool: "refund" },
];

async function judge(question: string, answerText: string) {
  const { object } = await generateObject({
    model: openai("gpt-5.4"),          // judge model (ideally different from the SUT)
    schema: z.object({ correct: z.boolean(), reason: z.string() }),
    prompt: `Q: ${question}\nA: ${answerText}\nIs the answer correct and relevant?`,
  });
  return object;
}

for (const c of dataset) {
  const { text, steps } = await runSystemUnderTest(c.input);
  const assertionPass = !c.mustContain || text.includes(c.mustContain);
  const toolPass = !c.expectTool || steps.some((s) => s.toolCalls?.some((t) => t.toolName === c.expectTool));
  const verdict = await judge(c.input, text);
  console.log({ input: c.input, assertionPass, toolPass, judged: verdict.correct });
}
```

**.NET / C#** — `Microsoft.Extensions.AI.Evaluation`:

```csharp
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;

IEvaluator evaluator = new RelevanceTruthAndCompletenessEvaluator();

var messages = new List<ChatMessage> { new(ChatRole.User, "Capital of France?") };
var modelResponse = await chat.GetResponseAsync(messages);

var config = new ChatConfiguration(chat); // the judge model
EvaluationResult result = await evaluator.EvaluateAsync(messages, modelResponse, config);

foreach (var metric in result.Metrics.Values)
    Console.WriteLine($"{metric.Name}: {metric}"); // scores + diagnostics
```

## 6.6 Observability & tracing
Instrument every LLM call, tool call, retrieval, and sub-agent as a **span** (inputs, outputs, tokens, latency, cost). The emerging standard is **OpenTelemetry GenAI semantic conventions**. Tooling: Langfuse, LangSmith, Braintrust, Arize Phoenix, W&B Weave; Azure AI Evaluation SDK for .NET.

```ts
import { trace } from "@opentelemetry/api";

async function tracedAnswer(query: string, tenantId: string) {
  return trace.getTracer("rag").startActiveSpan("rag.answer", async (span) => {
    span.setAttributes({ "gen_ai.operation.name": "rag", tenantId });
    try {
      const out = await answer(query, tenantId);
      span.setAttribute("gen_ai.usage.citations", out.citations.length);
      return out;
    } finally {
      span.end();
    }
  });
}
```

**.NET / C#** — OpenTelemetry span around a RAG call (GenAI semantic conventions):

```csharp
using System.Diagnostics;

static readonly ActivitySource Tracer = new("rag");

async Task<AnswerResult> TracedAnswerAsync(string query, string tenantId)
{
    using var span = Tracer.StartActivity("rag.answer");
    span?.SetTag("gen_ai.operation.name", "rag");
    span?.SetTag("tenantId", tenantId);

    var outResult = await AnswerAsync(query, tenantId);
    span?.SetTag("gen_ai.usage.citations", outResult.Citations.Count);
    return outResult;
}
```

> The Agent Framework and `Microsoft.Extensions.AI` clients emit these GenAI spans
> automatically when you add `.UseOpenTelemetry()` to the client/agent builder.

## 6.7 LLM-as-judge pitfalls (and fixes)
- **Position bias** → randomize order, average both orders.
- **Verbosity/length bias** → control for length; use rubrics.
- **Self-preference** → use a different judge model.
- **Low agreement** → calibrate against human labels (report Cohen's κ).
- Prefer **pairwise + rubric** over absolute 1–10 scores.

```ts
// Pairwise judging with position-bias mitigation: evaluate both orders, require agreement.
async function pairwise(question: string, a: string, b: string) {
  const ask = (x: string, y: string) =>
    generateObject({
      model: openai("gpt-5.4"),
      schema: z.object({ winner: z.enum(["X", "Y"]) }),
      prompt: `Q: ${question}\nX:\n${x}\n\nY:\n${y}\n\nWhich answer is better?`,
    }).then((r) => r.object.winner);

  const [r1, r2] = await Promise.all([ask(a, b), ask(b, a)]);
  if (r1 === "X" && r2 === "Y") return "A"; // A won in both orders
  if (r1 === "Y" && r2 === "X") return "B";
  return "tie";                              // disagreement => no reliable winner
}
```

**.NET / C#** — pairwise judging with position-bias mitigation:

```csharp
using Microsoft.Extensions.AI;

record Winner(string Value); // "X" or "Y"

async Task<string> PairwiseAsync(IChatClient judge, string question, string a, string b)
{
    async Task<string> Ask(string x, string y) =>
        (await judge.GetResponseAsync<Winner>(
            $"Q: {question}\nX:\n{x}\n\nY:\n{y}\n\nWhich answer is better?")).Result.Value;

    var r1Task = Ask(a, b);
    var r2Task = Ask(b, a);
    await Task.WhenAll(r1Task, r2Task);

    if (r1Task.Result == "X" && r2Task.Result == "Y") return "A"; // A won in both orders
    if (r1Task.Result == "Y" && r2Task.Result == "X") return "B";
    return "tie";                                                 // disagreement => no winner
}
```

---

# Part 7 — Production concerns

## 7.1 Latency
Stream tokens, parallelize independent work, route easy work to smaller models, and cache.

```ts
import { streamText } from "ai";
// Stream tokens to the client for perceived speed.
const stream = await streamText({ model: openai("gpt-5.4"), prompt });
for await (const delta of stream.textStream) process.stdout.write(delta);
```

**.NET / C#** — stream tokens with `GetStreamingResponseAsync`:

```csharp
using Microsoft.Extensions.AI;

// Stream tokens to the client for perceived speed.
await foreach (var update in chat.GetStreamingResponseAsync(prompt))
    Console.Write(update.Text);
```

## 7.2 Caching
- **Prompt caching** — providers cache stable prompt prefixes (system prompt, tool defs, context). Put the stable part first.
- **Semantic caching** — cache answers keyed by query-embedding similarity.
- **Exact-match caching** — for identical requests.

```ts
// Semantic cache: serve a stored answer when a near-duplicate query arrives.
async function cachedAnswer(query: string, tenantId: string, threshold = 0.95) {
  const { embedding } = await embed({ model: embedModel, value: query });

  const [hit] = await cache.search(embedding, { tenantId, topK: 1 });
  if (hit && hit.score >= threshold) return { ...hit.payload, cached: true };

  const fresh = await answer(query, tenantId);
  await cache.upsert({ tenantId, vector: embedding, payload: fresh });
  return { ...fresh, cached: false };
}
```

**.NET / C#** — semantic cache keyed by query-embedding similarity:

```csharp
using Microsoft.Extensions.AI;

async Task<CachedAnswer> CachedAnswerAsync(
    IEmbeddingGenerator<string, Embedding<float>> embedder,
    SemanticCache cache, string query, string tenantId, double threshold = 0.95)
{
    var embedding = (await embedder.GenerateAsync([query]))[0].Vector;

    var hits = await cache.SearchAsync(embedding, tenantId, topK: 1);
    if (hits.FirstOrDefault() is { } hit && hit.Score >= threshold)
        return hit.Payload with { Cached = true };

    var fresh = await AnswerAsync(query, tenantId);
    await cache.UpsertAsync(tenantId, embedding, fresh);
    return fresh with { Cached = false };
}
```

## 7.3 Cost control
Model routing, prompt/semantic caching, shorter contexts, capped steps, batching. Track **cost per request** as a first-class metric.

## 7.4 Reliability
Timeouts on every external call, retries with exponential backoff + jitter, **fallback models/providers**, circuit breakers, idempotency keys, and a **max-step/token budget** on agent loops.

```ts
async function withResilience<T>(
  primary: () => Promise<T>,
  fallback: () => Promise<T>,
  { retries = 2, baseMs = 200, timeoutMs = 20_000 } = {},
): Promise<T> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await withTimeout(primary(), timeoutMs);
    } catch {
      if (attempt === retries) break;
      await sleep(baseMs * 2 ** attempt + Math.random() * 100); // backoff + jitter
    }
  }
  return fallback(); // e.g. a smaller/cheaper model or a canned safe response
}
```

**.NET / C#** — timeouts, backoff-with-jitter retries, and a fallback (pairs well with `Microsoft.Extensions.Resilience` / Polly):

```csharp
async Task<T> WithResilienceAsync<T>(
    Func<CancellationToken, Task<T>> primary, Func<Task<T>> fallback,
    int retries = 2, int baseMs = 200, int timeoutMs = 20_000)
{
    for (var attempt = 0; attempt <= retries; attempt++)
    {
        try
        {
            using var cts = new CancellationTokenSource(timeoutMs);
            return await primary(cts.Token);
        }
        catch
        {
            if (attempt == retries) break;
            var jitter = Random.Shared.Next(0, 100);
            await Task.Delay(baseMs * (1 << attempt) + jitter); // backoff + jitter
        }
    }
    return await fallback(); // e.g. a smaller/cheaper model or a canned safe response
}
```

## 7.5 Security — OWASP LLM Top 10 (2025) highlights
- **LLM01 Prompt Injection** — untrusted text (docs, web, tool output) hijacks instructions. *Mitigate:* separate system vs data, don't grant tools authority off untrusted input, output filtering, human approval.
- **LLM02 Sensitive Information Disclosure** — least privilege, output scanning, no secrets in prompts.
- **LLM05 Improper Output Handling** — validate/escape/parameterize downstream (SQL/HTML/shell).
- **LLM06 Excessive Agency** — minimal tools, scoped permissions, confirmation gates.
- **LLM08 Vector & Embedding Weaknesses** — access control on the index, validate ingested content.
- **LLM10 Unbounded Consumption** — rate limits, budgets, step caps.

**Indirect prompt injection** via retrieved/tool content is the #1 practical agent risk — treat all external content as **data, never instructions**.

```ts
const INJECTION = [/ignore (all|previous) instructions/i, /system prompt/i, /reveal your|exfiltrate/i];

// Input guardrail: flag injection attempts in the query AND in retrieved content.
function inputGuardrail(userText: string, retrieved: Chunk[]) {
  const suspect = [userText, ...retrieved.map((c) => c.content)]
    .some((t) => INJECTION.some((re) => re.test(t)));
  return { suspect, note: suspect ? "Enforce data-only handling of external content." : null };
}

// Output guardrail: block ungrounded or PII-leaking answers before returning.
async function outputGuardrail(answerText: string, sources: Chunk[]) {
  const { object } = await generateObject({
    model: openai("gpt-5.4-mini"),
    schema: z.object({ grounded: z.boolean(), leaksPII: z.boolean() }),
    prompt: `Sources:\n${sources.map((s) => s.content).join("\n")}\n\nAnswer:\n${answerText}\n\n` +
            `Is the answer fully grounded in the sources? Does it leak PII?`,
  });
  if (!object.grounded || object.leaksPII) throw new Error("Output blocked by guardrail");
  return answerText;
}
```

**.NET / C#** — input/output guardrails for prompt injection and ungrounded/PII output:

```csharp
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;

static readonly Regex[] Injection =
[
    new("ignore (all|previous) instructions", RegexOptions.IgnoreCase),
    new("system prompt", RegexOptions.IgnoreCase),
    new("reveal your|exfiltrate", RegexOptions.IgnoreCase)
];

// Input guardrail: flag injection attempts in the query AND in retrieved content.
(bool Suspect, string? Note) InputGuardrail(string userText, IReadOnlyList<Chunk> retrieved)
{
    var suspect = new[] { userText }.Concat(retrieved.Select(c => c.Content))
        .Any(t => Injection.Any(re => re.IsMatch(t)));
    return (suspect, suspect ? "Enforce data-only handling of external content." : null);
}

record OutputCheck(bool Grounded, bool LeaksPII);

// Output guardrail: block ungrounded or PII-leaking answers before returning.
async Task<string> OutputGuardrailAsync(IChatClient chat, string answerText, IReadOnlyList<Chunk> sources)
{
    var joined = string.Join("\n", sources.Select(s => s.Content));
    var check = (await chat.GetResponseAsync<OutputCheck>(
        $"Sources:\n{joined}\n\nAnswer:\n{answerText}\n\n" +
        "Is the answer fully grounded in the sources? Does it leak PII?")).Result;
    if (!check.Grounded || check.LeaksPII)
        throw new InvalidOperationException("Output blocked by guardrail");
    return answerText;
}
```

## 7.6 Determinism, structured output, versioning
Low temperature for classification/extraction; **structured outputs** for anything parsed by code (validate + repair/retry on failure). Version prompts, tools, models, and eval datasets together; treat prompt changes like code (PR, review, eval-in-CI, canary, rollback).

```ts
// Validate + one repair attempt for structured output.
async function extractInvoice(text: string) {
  const schema = z.object({ total: z.number(), currency: z.string().length(3), dueDate: z.string() });
  try {
    return (await generateObject({ model: openai("gpt-5.4-mini"), schema, prompt: text })).object;
  } catch {
    // Repair: feed the schema error back once before failing hard.
    return (await generateObject({
      model: openai("gpt-5.4"),
      schema,
      prompt: `Return STRICTLY valid data for this invoice:\n${text}`,
    })).object;
  }
}
```

**.NET / C#** — structured output with a single repair attempt on failure:

```csharp
using Microsoft.Extensions.AI;

record Invoice(decimal Total, string Currency, string DueDate);

async Task<Invoice> ExtractInvoiceAsync(IChatClient chat, string text)
{
    try
    {
        return (await chat.GetResponseAsync<Invoice>(text)).Result;
    }
    catch
    {
        // Repair: restate the requirement once before failing hard.
        return (await chat.GetResponseAsync<Invoice>(
            $"Return STRICTLY valid data for this invoice:\n{text}")).Result;
    }
}
```

---

# Part 8 — Building an agent end-to-end (reference workflow)

1. **Define the job & success criteria.** Write 10–20 example inputs with expected outcomes *first* — your eval set.
2. **Start simple.** A single augmented-LLM call before any multi-step agent. Measure it.
3. **Add retrieval** if it needs private/fresh knowledge (Part 2). Evaluate retrieval separately.
4. **Design tools carefully** (Part 4.2). Prefer MCP for reusable integrations. Keep the set minimal.
5. **Choose the pattern** (Part 5): workflow if steps are predictable; agent loop only if open-ended.
6. **Add memory & context engineering** as needed (Part 5.8–5.9).
7. **Instrument everything** — tracing/spans from day one (Part 6.6).
8. **Build the eval harness** — assertions + judge + trajectory checks; wire into CI (Part 6).
9. **Add guardrails & security** — input/output checks, least privilege, human-in-the-loop, step/budget caps (Part 7).
10. **Ship behind canary/A-B, monitor online metrics, iterate.** Every change re-runs evals.

## 8.1 Recommended stacks

**TypeScript / Node** — Vercel AI SDK (`generateText`/`ToolLoopAgent`, structured output, streaming), `@modelcontextprotocol/sdk`, LangGraph.js/Mastra for graph orchestration, pgvector/Qdrant/Pinecone, promptfoo/Braintrust/Langfuse for evals.

**.NET / C#** — `Microsoft.Extensions.AI` (`IChatClient`/`IEmbeddingGenerator` + function invocation), **Microsoft Agent Framework** (`Microsoft.Agents.AI`: agents, threads, graph workflows, MCP, middleware, telemetry — successor to Semantic Kernel + AutoGen), `Microsoft.Extensions.VectorData`, `Microsoft.Extensions.AI.Evaluation` + Azure AI Evaluation SDK.

## 8.2 End-to-end example — a support agent (TypeScript)

Combines RAG-as-tool → tool loop → guardrails → bounded stop condition:

```ts
import { generateText, tool, stepCountIs } from "ai";
import { z } from "zod";

const searchDocs = tool({
  description: "Search the product knowledge base. Use for how-to and troubleshooting questions.",
  inputSchema: z.object({ query: z.string() }),
  execute: async ({ query }) => retrieve(query, currentTenantId, 5), // hybrid + rerank
});

const createTicket = tool({
  description: "Escalate to a human by creating a support ticket. Use only when docs cannot resolve it.",
  inputSchema: z.object({ summary: z.string(), severity: z.enum(["low", "high"]) }),
  execute: async (args) => ({ ticketId: await tickets.create(args) }),
});

export async function supportAgent(userMessage: string, tenantId: string) {
  currentTenantId = tenantId;

  const { text, steps } = await generateText({
    model: openai("gpt-5.4"),
    system:
      "You are a support agent. Prefer answering from the knowledge base with citations. " +
      "Escalate with createTicket only if the docs do not contain the answer. Be concise.",
    tools: { searchDocs, createTicket },
    stopWhen: stepCountIs(6),          // guardrail: bounded loop
    prompt: userMessage,
  });

  return { answer: text, toolCalls: steps.flatMap((s) => s.toolCalls) };
}
```

## 8.3 End-to-end example — an agent with tools (.NET, Agent Framework)

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using System.ComponentModel;

[Description("Search the product knowledge base for how-to and troubleshooting answers.")]
static async Task<string> SearchDocs([Description("User's question")] string query)
    => await Rag.RetrieveTopKAsync(query, 5); // hybrid + rerank retrieval

[Description("Escalate to a human by creating a support ticket when docs cannot resolve it.")]
static async Task<string> CreateTicket(string summary, string severity)
    => $"Created ticket {await Tickets.CreateAsync(summary, severity)}";

AIAgent agent = chatClient.CreateAIAgent(
    instructions:
        "You are a support agent. Answer from the knowledge base with citations. " +
        "Escalate with CreateTicket only if the docs lack the answer. Be concise.",
    tools:
    [
        AIFunctionFactory.Create(SearchDocs),
        AIFunctionFactory.Create(CreateTicket)
    ]);

AgentThread thread = agent.GetNewThread(); // carries session state across turns
var reply = await agent.RunAsync("How do I reset my API key?", thread);
Console.WriteLine(reply);
```

---

# Part 9 — Interview preparation

## 9.1 Rapid-fire Q&A

**Agent vs workflow?** A workflow orchestrates LLM calls through fixed code paths; an agent lets the model decide its own control flow and tool use dynamically. Start with workflows; use agents only for open-ended tasks.

**When would you NOT use RAG?** Tiny/static corpus (paste it in), behavior change (fine-tune), or chit-chat with no knowledge need (route around it).

**Why hybrid search?** Dense captures semantics but misses exact tokens; BM25 captures exact matches but misses paraphrases. Fuse with RRF.

**Why two-stage retrieval?** Cross-encoders are precise but slow → cheap ANN recall (top-100) then expensive rerank (top-5).

**How does function calling work?** The model returns a structured request to call a named tool with JSON args; your code executes and feeds the result back; repeat. The model never executes code.

**What is MCP and why does it matter?** An open protocol standardizing how apps expose tools/resources/prompts to LLM clients — build once, use across Claude/ChatGPT/IDEs. Decouples integration providers from agent apps.

**Biggest agent security risk?** Prompt injection — especially *indirect* via retrieved docs or tool output. Treat external content as untrusted data, enforce least privilege, gate risky actions.

**How do you evaluate an agent?** Separate component evals from end-to-end trajectory evals. Deterministic assertions where possible, LLM-as-judge (pairwise + rubric) for quality, a versioned golden dataset, all in CI. Instrument with traces.

**LLM-as-judge risks?** Position/length/self-preference bias, low human agreement. Mitigate with randomized order, rubrics, a different judge model, pairwise comparisons, calibration.

**Control cost/latency?** Model routing, prompt & semantic caching, parallel tool calls, streaming, less context, capped steps/budget.

**Stop an agent from looping forever?** Hard stop condition: max steps, token/cost budget, or a `finish` signal — plus per-tool timeouts and retries.

**Orchestrator-workers vs parallelization?** Parallelization splits *known* subtasks up front; orchestrator-workers *dynamically* decomposes when subtasks aren't known.

## 9.2 System-design prompt: "Design a customer-support agent"

1. **Requirements & metrics** (resolution rate, escalation rate, latency, cost, safety).
2. **Data & retrieval** — ingest docs/tickets; chunk + embed; hybrid + rerank; citations.
3. **Tools** — KB search, order lookup, refund (human approval), ticket creation. Least privilege.
4. **Control flow** — route by intent; answer from RAG; escalate on low confidence; bounded loop.
5. **Guardrails** — PII/injection filters, output validation, confirmation for money-moving actions.
6. **Evaluation** — golden set, faithfulness + resolution judges, trajectory checks, CI.
7. **Observability** — traces/spans, online metrics, feedback loop.
8. **Rollout** — canary/A-B, monitor, iterate.

## 9.3 Glossary

- **ANN** — approximate nearest neighbor search over vectors.
- **BM25** — sparse keyword ranking function.
- **Cross-encoder** — model that scores (query, doc) jointly; used for reranking.
- **HNSW / IVF-PQ** — ANN index structures.
- **RAG** — retrieval-augmented generation.
- **RRF** — reciprocal rank fusion.
- **ReAct** — reason+act agent loop.
- **MCP** — Model Context Protocol.
- **ACI** — agent-computer interface (tool design).
- **LLM-as-judge** — using an LLM to score outputs.
- **Groundedness/Faithfulness** — answer supported by provided context.
- **Trajectory eval** — evaluating the sequence of agent steps.
- **Context engineering** — managing what goes into the context window.
- **Guardrail** — real-time safety/validation check.
- **Prompt injection** — malicious instructions embedded in inputs/content.

## 9.4 Talking-point checklist

- [ ] Agent vs workflow, and when to use each.
- [ ] Draw the RAG pipeline and name each stage's failure mode.
- [ ] Justify hybrid search + reranking.
- [ ] Explain the function-calling loop and who executes tools.
- [ ] Describe MCP's architecture and value.
- [ ] Name the five workflow patterns + the autonomous agent.
- [ ] Design an eval strategy (component + trajectory, assertions + judge, CI).
- [ ] List top agent security risks and mitigations (esp. prompt injection).
- [ ] Talk about latency/cost levers (routing, caching, parallelism).
- [ ] Sketch an end-to-end production agent.

## 9.5 Further reading (primary sources)

- Anthropic — *Building Effective Agents*; *Contextual Retrieval*; prompt caching docs.
- Model Context Protocol — `modelcontextprotocol.io` (spec, architecture, SDKs).
- Microsoft Learn — *Microsoft Agent Framework*; *Microsoft.Extensions.AI*; .NET RAG & evaluation tutorials.
- Vercel — *AI SDK* docs (agents, tools, loop control).
- OWASP — *Top 10 for LLM Applications (2025)*.
- OpenTelemetry — *GenAI semantic conventions*.
- RAGAS / DeepEval / promptfoo — evaluation frameworks.

---

*End of study guide. Verify exact SDK symbol names and model IDs against current documentation before use — the patterns are stable; the APIs move fast.*
