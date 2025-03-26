Below is a **comprehensive overview** of how this API integrates with your Supabase‑backed application, tying together **the agent workflow** (Decision/RAG/Tool Shed/Finish) and **your data schema** (Workspaces, Spaces, Files, etc.). The goal is to illustrate how user queries flow through the system, how data is retrieved from Supabase, and how the AI agent orchestrates everything to deliver answers.

---

## 1. **High‑Level App Context**

Your app is a **workspace** for deep work, where users:
- Create **Workspaces** (like folders).  
- Inside each Workspace, they have **Spaces** (like projects or subfolders).  
- In each Space, users store their **files**, **notes**, and **links**, along with any code or research items.  
- Each Space has an **AI Agent** that can read/manipulate files, run external searches, summarize notes, and more.

All of this data (Workspaces, Spaces, Files, etc.) is stored in **Supabase**. Some highlights from the schema:

1. **workspaces** – Top-level groupings/folders.  
2. **spaces** – Sub-groupings within a workspace, containing the actual content.  
3. **space_files** – Each file (document, note, code file, etc.) belongs to a Space.  
4. **file_metadata** – Metadata/analysis of these files (e.g., embeddings, text content).  
5. **space_research** – Additional links, references, or research items in a Space.  
6. **users** – Each user has a record, including toggled_files (JSON) that track which files they’ve enabled or disabled for the AI to use.  
7. **workspace_spaces** – A join table linking Spaces to Workspaces.  

In essence, the app centers on letting users collect and organize files/notes within Spaces, then leverages the AI Agent to do specialized tasks on those items.

---

## 2. **API Inputs & Outputs**

### **Input to the API**

1. **Space ID**  
   - Identifies which Space (and thus which user data) the query pertains to.  
   - This also implies knowledge of the associated Workspace (via `workspace_spaces`).

2. **Query**  
   - The user’s question or command, e.g., “Summarize note X” or “Refactor this code.”

3. **View** (could be a note, code snippet, or current file)  
   - Often the piece of content the user is actively editing/reading.  
   - This can be used as additional context for the AI.

4. **Stream (Boolean)**  
   - Indicates whether the response from the language model should be streamed token-by-token or returned all at once.

### **Output from the API**

- **Streamer**  
  - A streaming response object (or single message) from the final LLM.  
  - Contains the AI’s complete, context-enriched answer or action output.

### **Automatically Managed**

- **Toggled Files** (from `users`.`toggled_files`)  
  - Tracks which files a user has explicitly toggled on (or off) for the AI to access.  
  - When the AI needs to read from a user’s set of files (e.g., for RAG), it should only consider those that are toggled on.

- **Toggled Research** (from `users`.`toggled_research`)  
  - Tracks which research a user has explicitly toggled on (or off) for the AI to access.  
  - When the AI needs to read from a user’s set of research (e.g., for RAG), it should only consider those that are toggled on.

---

## 3. **Agent Workflow**

Below is the step-by-step description of the API’s internal workflow, referencing the **Decision Node** (Gemini-2-Flash), **RAG**, **Tool Shed**, and **Finish** steps.

### **1. Decision Node (“Gemini‑2‑Flash”)**

- **Input:** `Space ID`, user `Query`, `View`, plus any context the system already has.  
- **Process:**
  1. The user’s request enters the **Decision Node**.  
  2. The Decision Node calls **Gemini‑2‑Flash** to decide:
     - **Do we need RAG** (Retrieval-Augmented Generation)?  
       - E.g., “We want to fetch relevant notes, code, or references from `space_files` or `space_research` to incorporate context.”  
     - **Do we need a Tool** (e.g., a specialized function like code execution, a PDF parser, or any custom logic in the “Tool Shed”)?  
       - E.g., “We need to run a sentiment analysis tool or a code refactoring function.”

- **Output:** One of three possibilities:  
  1. **RAG needed** → Route to RAG Node.  
  2. **Tool needed** → Route to Tool Shed Node.  
  3. **Neither** → Proceed to Finish Node.

- **Note:** This **decision** may happen multiple times if the AI’s plan calls for repeated retrieval or tool usage.

### **2. RAG Node**

- **Input:** The user Query + any previously gathered context.  
- **Process:**
  1. The API constructs a RAG query. This can include:
     - The user’s `Query`.  
     - The user’s toggled files from Supabase (`users.toggled_files`).  
     - The user’s toggled research from Supabase (`users.toggled_research`).  
     - The relevant `space_files`, `file_metadata`, or `space_research` records.  
  2. Calls the RAG system to retrieve text or embeddings from your data (e.g., using Pinecone IDs stored in `file_metadata.pinecone_id`, or searching text indexes like `description gin_trgm_ops`).
  3. Returns the retrieved data (summaries, chunks of text, or references) back to the Decision Node.

- **Output:** Additional textual/contextual data needed to refine the answer.

### **3. Tool Shed Node**

- **Input:** The user Query + context, along with a list/manifest of available tools.  
- **Process:**
  1. The system shows **Gemini‑2‑Flash** the tool list (function names, parameters, descriptions).  
  2. Gemini decides which tool to invoke, if any, and with which parameters.  
  3. The chosen tool is executed (e.g., code runner, translator, some custom domain-specific function).
  4. The tool result (output) is returned to the Decision Node.

- **Output:** The tool’s result (e.g., code output, analysis, transformations, etc.).

### **4. Finish Node**

- **Input:** The original Query + **all** gathered context (from RAG, Tools, the user’s `View`, toggled files, toggled research, etc.).  
- **Process:**
  1. Pass everything to a Base LLM’s `generate_answer()` function.  
  2. This final LLM call weaves together all the data to produce the best possible answer or action.  
  3. The result is streamed (or returned as one block) to the end-user.

- **Output:** The final LLM answer, typically via a streaming interface.

---

## 4. **Putting It All Together**

1. **User calls the API** with `(space_id, query, view, stream)`.  
2. **Decision Node**:
   - Invokes Gemini-2-Flash to see if it needs RAG or a tool.  
   - Example: “I want to refine code in `file_path=foo.py`” → the Decision Node sees it needs the code from that file → calls RAG to fetch its contents from `space_files` or `file_metadata`.  
3. **RAG Node** (if needed):  
   - Gathers text from the relevant files (only those toggled on in `users.toggled_files`).
   - Gathers text from the relevant research (only those toggled on in `users.toggled_research`).  
   - Returns the text to the Decision Node.  
4. **Decision Node** might again ask Gemini if a specialized tool is needed (e.g., a “Python code refactor” tool). If so:
   - Calls the **Tool Shed Node**, which obtains the tool’s result.  
5. **Decision Node** eventually determines it has all the info it needs (no more RAG or tools).  
6. **Finish Node**:
   - Sends everything to `generate_answer()` from the Base LLM.  
   - Streams back the final response to the user.

Throughout this process, **logging/streaming** can capture intermediate LLM responses (e.g., how Gemini-2-Flash decided to invoke a tool), enabling traceability and debugging.

---

## 5. **Data Flow & Supabase Integration**

- **Selecting which files or notes** to retrieve:
  - The system checks the user’s toggled files (`users.toggled_files`) to know which file IDs are “on.”  
  - Queries `space_files` (and potentially `file_metadata`) for only those toggled file IDs.  
  - The system checks the user’s toggled research (`users.toggled_research`) to know which file IDs are “on.”  
  - Queries `space_research'.  
  - If relevant, uses indexes like `gin (description gin_trgm_ops)` or `metadata` to match the user’s text query.  
- **Retrieving or searching research links**:
  - The system may also query `space_research` for relevant links or embedded data.  
- **Tying it to the correct user**:
  - The API is called under a user’s context, so `user_id` in `spaces`, `space_files`, `workspace_spaces`, etc., ensures no cross-user data mixing.  
- **Workspaces**:
  - Tied to a user and can nest other Workspaces (via `parent_workspace_id`).  
  - Each Workspace can hold multiple Spaces (via `workspace_spaces`).  
- **Spaces**:
  - Each Space belongs to a single user, organizes the actual working items (`space_files`, `space_research`).  
  - Each also has its own AI Agent instance.  

This schema ensures data integrity while letting the AI agent fetch the right content for the user’s query.

---

## 6. **Conclusion**

Your API orchestrates an **intelligent workflow** where:

1. **Decision Node** (Gemini-2-Flash) routes the request, possibly multiple times, to either:
   - **RAG** (to gather data from Supabase & Pinecone)  
   - **Tool Shed** (to run specialized tools/functions)  
2. When enough context has been collected, the **Finish Node** finalizes the response using a Base LLM.  
3. The **Supabase** schema underpins the entire system, storing all the user’s files, notes, metadata, toggled states, and organizational hierarchy (Workspaces/Spaces).

The result is an app that **“supercharges your thinking”**: it centralizes your content, organizes it in a structured database, and leverages an AI Agent that is aware of the user’s environment (toggled files, toggled research, research items, code) to provide powerful, context-rich answers.