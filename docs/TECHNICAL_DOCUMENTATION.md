# D&D Adventure - Complete Technical Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Backend Files](#3-backend-files)
   - [app.py - Flask API Server](#31-apppy---flask-api-server)
   - [lstm_agent.py - LSTM Neural Network](#32-lstm_agentpy---lstm-neural-network)
   - [game_utils.py - Game Utilities](#33-game_utilspy---game-utilities)
4. [RAG System Package](#4-rag-system-package)
   - [rag/__init__.py - Package Init](#41-raginitpy---package-initialization)
   - [rag/config.py - Configuration](#42-ragconfigpy---configuration)
   - [rag/llm_handler.py - LLM Integration](#43-ragllm_handlerpy---llm-integration)
   - [rag/rag_chain.py - RAG Pipeline](#44-ragrag_chainpy---rag-pipeline)
   - [rag/rag_system.py - Wrapper Interface](#45-ragrag_systempy---wrapper-interface)
5. [Frontend Files](#5-frontend-files)
   - [index.html - Game Interface](#51-indexhtml---game-interface)
   - [app.js - Game Logic](#52-appjs---game-logic)
   - [game_config.js - Game Data](#53-game_configjs---game-data)
   - [maze_renderer.js - SVG Map](#54-maze_rendererjs---svg-map)
   - [style.css - Styling](#55-stylecss---styling)
6. [Data Flow](#6-data-flow)
7. [API Reference](#7-api-reference)
8. [Neural Network Details](#8-neural-network-details)
9. [RAG System Details](#9-rag-system-details)
10. [Generated Files](#10-generated-files)

---

## 1. Project Overview

This is an interactive **Dungeons & Dragons themed adventure game** featuring two AI systems:

1. **RAG-Based Dungeon Master (Chatbot)**: Uses Retrieval-Augmented Generation with Google Gemini to provide contextual game hints in a fantasy medieval style.

2. **LSTM Action Agent**: A Long Short-Term Memory neural network that learns to predict optimal game actions by observing player behavior.

### Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | Vanilla HTML5, CSS3, JavaScript (ES6+) |
| Backend | Python 3.x, Flask |
| AI/ML | PyTorch (LSTM), SentenceTransformers, FAISS |
| LLM | Google Gemini 2.5 Flash Lite API |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Browser)                         │
├─────────────────────────────────────────────────────────────────────┤
│  index.html ─────► app.js ─────────► game_config.js                 │
│       │              │                     │                         │
│       │              ├── Game State        ├── Rooms (10)            │
│       │              ├── Actions Handler   ├── Items (11)            │
│       │              ├── Training Logic    ├── Puzzles (5)           │
│       │              └── API Calls         └── Actions (30+)         │
│       │                                                              │
│       └─────► maze_renderer.js (SVG Map)                            │
│       └─────► style.css (Dark Gothic Theme)                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP (localhost:5000)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           BACKEND (Flask)                            │
├─────────────────────────────────────────────────────────────────────┤
│  app.py ─────────────────────────────────────────────────────────── │
│     │                                                                │
│     ├── POST /chatbot ──────► rag/rag_system.py                     │
│     │                              │                                 │
│     │                              └──► rag/rag_chain.py            │
│     │                                      │                         │
│     │                                      ├── FAISS Vector Search   │
│     │                                      └── Gemini LLM            │
│     │                                                                │
│     ├── POST /agent/act ────► lstm_agent.py                         │
│     ├── POST /agent/train ──► lstm_agent.py                         │
│     ├── POST /agent/batch_train ─► lstm_agent.py                    │
│     └── POST /agent/reset ──► lstm_agent.py                         │
│                                    │                                 │
│                                    └──► game_utils.py               │
│                                           │                          │
│                                           └── vectorize_state()     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Backend Files

### 3.1 `app.py` - Flask API Server

**Location**: `final project/app.py`  
**Lines**: ~351  
**Purpose**: Main application entry point providing REST API endpoints.

#### Environment Setup (Lines 14-20)

```python
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TORCH"] = "1"
sys.modules['tensorflow'] = None
```

These lines **prevent TensorFlow from loading** (which conflicts with PyTorch) and suppress unnecessary warnings.

#### Flask App Initialization (Lines 32-37)

```python
app = Flask(__name__)
CORS(app)
```

- Creates a Flask application instance
- Enables Cross-Origin Resource Sharing (CORS) to allow frontend-backend communication

#### Component Initialization (Lines 48-59)

```python
model, optimizer, criterion, state_buffer = create_agent(num_actions=len(ACTION_MAP))

rag_system = None
def get_rag():
    global rag_system
    if rag_system is None:
        rag_system = get_rag_system()
    return rag_system
```

- **LSTM Agent**: Created immediately on startup using `create_agent()`
- **RAG System**: Lazy-loaded on first chatbot request (saves startup time)

#### Endpoint: `/chatbot` (Lines 66-97)

```python
@app.route('/chatbot', methods=['POST'])
def chatbot():
```

**Purpose**: AI Dungeon Master chatbot using RAG

**Request Body**:
```json
{
    "query": "What should I do?",
    "currentRoom": "hall",
    "step": 1,
    "inventory": ["torch"],
    "puzzlesSolved": [],
    "doorLocked": true
}
```

**Response**:
```json
{
    "message": "Take the torch and explore!",
    "intent": "get_item"
}
```

**Flow**:
1. Extract game state from request
2. Call `rag.generate_response(query, game_state)`
3. Return message and detected intent

#### Endpoint: `/agent/act` (Lines 104-153)

```python
@app.route('/agent/act', methods=['POST'])
def agent_act():
```

**Purpose**: Get LSTM agent's action prediction

**Request Body**:
```json
{
    "state": {
        "step": 5,
        "inventory": ["torch"],
        "currentRoom": "library",
        "doorLocked": true
    },
    "intent": "get_item",
    "mask": [1,1,1,1,1,1,1,1,1]
}
```

**Response**:
```json
{
    "action_id": "take_item",
    "action_index": 6,
    "sequence_length": 10
}
```

**Flow**:
1. Convert game state to 25-feature vector via `vectorize_state()`
2. Add to state history buffer
3. Get sequence from buffer
4. Forward pass through LSTM
5. Apply action mask (if provided)
6. Return argmax action

#### Endpoint: `/agent/train` (Lines 156-221)

```python
@app.route('/agent/train', methods=['POST'])
def agent_train():
```

**Purpose**: Single-step online training

**Request Body**:
```json
{
    "state": {...},
    "intent": "get_item",
    "correct_action_id": 6
}
```

**Response**:
```json
{
    "status": "trained",
    "loss": 0.523,
    "sequence_length": 10
}
```

**Flow**:
1. Vectorize state and add to buffer
2. Get state sequence
3. Forward pass → compute CrossEntropyLoss
4. Backpropagate with gradient clipping (max_norm=1.0)
5. Optimizer step

#### Endpoint: `/agent/batch_train` (Lines 224-323)

```python
@app.route('/agent/batch_train', methods=['POST'])
def agent_batch_train():
```

**Purpose**: Train on entire game history (more effective than single-step)

**Request Body**:
```json
{
    "history": [
        {"state": {...}, "intent": "inspect", "actionIndex": 0},
        {"state": {...}, "intent": "get_item", "actionIndex": 6},
        ...
    ],
    "epochs": 3
}
```

**Response**:
```json
{
    "status": "success",
    "final_loss": 0.45,
    "samples_trained": 600,
    "epochs": 3
}
```

**Flow**:
1. Convert all history moves to state vectors
2. For each epoch:
   - Clear state buffer
   - Reset LSTM hidden state
   - Iterate through training data sequentially
   - Accumulate loss
3. Save model to disk after training

#### Endpoint: `/agent/reset` (Lines 326-339)

```python
@app.route('/agent/reset', methods=['POST'])
def agent_reset():
```

**Purpose**: Reset agent state for new game

**Response**:
```json
{
    "status": "reset",
    "message": "LSTM state buffer cleared"
}
```

---

### 3.2 `lstm_agent.py` - LSTM Neural Network

**Location**: `final project/lstm_agent.py`  
**Lines**: 242  
**Purpose**: Defines the LSTM neural network architecture and state management.

#### Class: `LSTMActionAgent` (Lines 20-127)

**Architecture**:
```
Input (25 features)
    │
    ▼
Input Embedding (Linear: 25 → 64)
    │
    ▼ ReLU
LSTM (64 hidden, 2 layers, 30% dropout)
    │
    ▼ (take last timestep output)
Layer Normalization (64)
    │
    ▼
Output MLP:
    ├── Linear (64 → 32)
    ├── ReLU
    ├── Dropout (30%)
    └── Linear (32 → 9 actions)
    │
    ▼
Action Logits (9)
```

**Key Methods**:

1. **`__init__()`** (Lines 42-72)
   - `input_size`: 25 (room + intent + game state features)
   - `hidden_size`: 64
   - `num_layers`: 2
   - `num_actions`: 9
   - `dropout`: 0.3

2. **`forward()`** (Lines 80-123)
   ```python
   def forward(self, x, mask=None, hidden=None):
   ```
   - Handles both single timestep `(batch, features)` and sequence `(batch, seq_len, features)`
   - Applies action masking via `logits + ((1.0 - mask) * -1e9)` (invalid actions get -∞)
   - Returns `(logits, hidden_state)`

3. **`init_hidden()`** (Lines 74-78)
   - Creates zero-initialized hidden states `(h0, c0)`

4. **`reset_hidden()`** (Lines 125-127)
   - Called at start of new game

#### Class: `StateHistoryBuffer` (Lines 130-169)

**Purpose**: Maintains rolling window of recent game states for LSTM input.

```python
buffer = StateHistoryBuffer(max_length=10)
buffer.add(state_vector)  # Add state (12,) tensor
sequence = buffer.get_sequence()  # Returns (1, seq_len, 12) tensor
```

**Key Methods**:

1. **`add(state_vector)`** (Lines 142-149)
   - Ensures state is 1D tensor
   - Appends to history
   - Removes oldest if exceeds max_length

2. **`get_sequence()`** (Lines 151-162)
   - Stacks history tensors
   - Returns shape `(1, seq_len, features)` for LSTM

#### Model Persistence (Lines 172-192)

```python
def save_model(model, optimizer, path=MODEL_SAVE_PATH):
def load_model(model, optimizer, path=MODEL_SAVE_PATH):
```

- Saves/loads model weights and optimizer state
- Default path: `./model_weights.pth`

#### Factory Function: `create_agent()` (Lines 195-241)

```python
model, optimizer, criterion, state_buffer = create_agent(num_actions=9, input_size=25)
```

**Creates**:
1. `LSTMActionAgent` model
2. `Adam` optimizer (lr=0.0005, weight_decay=1e-5)
3. `CrossEntropyLoss` criterion
4. `StateHistoryBuffer` (max_length=10)

**Auto-loads** saved weights if architecture matches.

---

### 3.3 `game_utils.py` - Game Utilities

**Location**: `final project/game_utils.py`  
**Lines**: 152  
**Purpose**: Game constants and state vectorization for the neural network.

#### Constants (Lines 20-63)

**`ACTION_MAP`** - 9 action categories the LSTM can predict:
| Index | Action | Description |
|-------|--------|-------------|
| 0 | look | Look around |
| 1 | read_books | Read books/scrolls |
| 2 | take_key | Take key items |
| 3 | use_keys | Use keys/unlock |
| 4 | open_exit | Open exit/escape |
| 5 | navigate | Move between rooms |
| 6 | take_item | Take general items |
| 7 | solve_puzzle | Solve puzzles |
| 8 | interact | General interaction |

**`INTENT_MAP`** - 9 intents recognized by chatbot:
```python
["inspect", "navigate", "get_item", "use_item", "unlock", "read", "solve_puzzle", "interact", "escape"]
```

**`ROOM_IDS`** - 10 room identifiers:
```python
['hall', 'library', 'study', 'cellar', 'gallery', 'chapel', 'attic', 'crypt', 'dungeon', 'vault']
```

**`STATE_SIZE`** = 25 features

#### Function: `vectorize_state()` (Lines 70-132)

```python
def vectorize_state(state_data, intent_str) -> torch.FloatTensor:
```

**Purpose**: Convert game state dict to 25-feature tensor for LSTM input.

**Feature Breakdown** (25 total):

| Features | Count | Description |
|----------|-------|-------------|
| Room encoding | 10 | One-hot vector for current room |
| Intent encoding | 9 | One-hot vector for intent |
| step_progress | 1 | `min(step / 50, 1.0)` |
| inventory_count | 1 | `min(len(inventory) / 10, 1.0)` |
| has_torch | 1 | Binary: has light source |
| has_key | 1 | Binary: has any key |
| has_dagger | 1 | Binary: has weapon |
| door_locked | 1 | Binary: portal status |

**Example**:
```python
state_data = {
    'currentRoom': 'library',
    'step': 10,
    'inventory': ['torch', 'dagger'],
    'doorLocked': True
}
vector = vectorize_state(state_data, 'get_item')
# Returns tensor of shape (1, 25)
```

---

## 4. RAG System Package

The RAG system is organized as a Python package in `final project/rag/`.

### 4.1 `rag/__init__.py` - Package Initialization

**Lines**: 17  
**Purpose**: Exports public interface.

```python
from rag.config import config, RAGConfig
from rag.rag_system import get_rag_system, RAGSystem

__all__ = ['config', 'RAGConfig', 'get_rag_system', 'RAGSystem']
```

---

### 4.2 `rag/config.py` - Configuration

**Lines**: 55  
**Purpose**: Centralized configuration using dataclass.

#### `RAGConfig` Dataclass (Lines 14-52)

| Setting | Default | Description |
|---------|---------|-------------|
| `DOCS_DIR` | `"docs/"` | Documentation directory |
| `CHUNK_SIZE` | 400 | Max characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | SentenceTransformer model |
| `VECTOR_STORE_PATH` | `"./vector_db"` | FAISS index location |
| `TOP_K` | 3 | Number of chunks to retrieve |
| `SCORE_THRESHOLD` | 0.3 | Minimum similarity score |
| `LLM_TYPE` | `"gemini"` | LLM provider |
| `GEMINI_MODEL` | `"gemini-2.5-flash-lite"` | Gemini model name |
| `GEMINI_API_KEY` | `os.getenv("GEMINI_API_KEY")` | API key from environment |
| `TEMPERATURE` | 0.7 | LLM creativity |

---

### 4.3 `rag/llm_handler.py` - LLM Integration

**Lines**: 140  
**Purpose**: Handles LLM initialization and inference.

#### Class: `LLMHandler` (Lines 16-139)

**Supports**:
- **Primary**: Google Gemini API
- **Fallback**: Ollama (local LLM)

**Key Methods**:

1. **`_init_gemini()`** (Lines 39-52)
   ```python
   genai.configure(api_key=self.config.GEMINI_API_KEY)
   self.llm = genai.GenerativeModel(self.config.GEMINI_MODEL)
   ```

2. **`generate(prompt)`** (Lines 72-99)
   - Sends prompt to Gemini/Ollama
   - Returns text response
   - Falls back on error

3. **`generate_json(prompt)`** (Lines 101-116)
   - Generates JSON response
   - Parses markdown code blocks
   - Returns Python dict

---

### 4.4 `rag/rag_chain.py` - RAG Pipeline

**Lines**: 422  
**Purpose**: Complete RAG pipeline combining retrieval and generation.

#### Class: `GameRAGChain` (Lines 20-421)

**Components**:
- `_embedder`: SentenceTransformer model
- `_index`: FAISS vector index
- `_chunks`: List of document chunks
- `_llm`: LLMHandler instance

**Initialization Flow** (`_setup()`, Lines 73-86):
1. Load embedding model
2. Load or build FAISS index
3. Initialize LLM handler

#### Document Processing

**`_load_and_chunk_docs()`** (Lines 156-170)
- Finds all `.md` files in docs directory
- Applies smart chunking

**`_smart_chunk()`** (Lines 172-215)
- Splits by markdown headers first (`#`, `##`, `###`)
- Then splits large sections by paragraphs
- Preserves semantic boundaries

**`_build_index()`** (Lines 119-154)
1. Load and chunk documents
2. Create embeddings with SentenceTransformer
3. Build FAISS IndexFlatIP (inner product for cosine similarity)
4. Normalize embeddings with L2
5. Save index to disk

#### Retrieval

**`retrieve(query, top_k)`** (Lines 222-254)
```python
results = rag.retrieve("Where is the key?", top_k=3)
# Returns: [{'content': '...', 'source': 'file.md', 'score': 0.85}, ...]
```

**Flow**:
1. Embed query using same model
2. Normalize with L2
3. FAISS search for top_k
4. Filter by score threshold
5. Return chunks with scores

#### Response Generation

**`generate_response(query, game_state)`** (Lines 256-317)

**Prompt Template**:
```
You are the Dungeon Master for a D&D adventure game.
Speak in a fantasy medieval style, referencing dice rolls and ability checks.
Keep responses concise (1-2 sentences).

GAME STATE:
- Current Location: {room_name}
- Rooms Visited: {step}
- Inventory: {inventory}
- Puzzles Solved: {puzzles_solved}
- Portal Status: Sealed/Activated

RELEVANT GAME KNOWLEDGE:
{retrieved_context}

ADVENTURER ASKS: {query}

Choose ONE intent from: [inspect, navigate, get_item, ...]

Reply as JSON only: {"message": "...", "intent": "..."}
```

#### Fallback Logic

**`ITEM_INFO`** Dictionary (Lines 319-337)
- Comprehensive knowledge about all game items
- Used when RAG retrieval fails

**`_fallback_response()`** (Lines 339-390)
**Priority Order**:
1. Item-specific queries (check ITEM_INFO)
2. Game rules/help
3. Game introduction
4. Keys information
5. "What to do next" guidance
6. Room-specific hints
7. Generic "explore" response

---

### 4.5 `rag/rag_system.py` - Wrapper Interface

**Lines**: 88  
**Purpose**: Simple interface for Flask app.

#### Class: `RAGSystem` (Lines 22-73)

```python
rag = RAGSystem()
message, intent = rag.generate_response("Help me!", game_state)
```

**Methods**:
- `generate_response()`: Main interface
- `retrieve_context()`: Get raw chunks
- `rebuild_index()`: Force rebuild
- `get_stats()`: Debug info

#### Singleton Pattern (Lines 80-87)

```python
def get_rag_system() -> RAGSystem:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance
```

---

## 5. Frontend Files

### 5.1 `index.html` - Game Interface

**Lines**: 328  
**Purpose**: HTML structure for the game UI.

#### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HEADER (game-header)                         │
│                    🐉 D&D Adventure ⚔️                               │
│                      Quest for Freedom                               │
├─────────────┬───────────────────────────────────┬───────────────────┤
│ LEFT PANEL  │        CENTER PANEL               │    RIGHT PANEL    │
│             │                                   │                   │
│ ┌─────────┐ │ ┌───────────────────────────────┐ │ ┌───────────────┐ │
│ │ Quest   │ │ │      Room Display             │ │ │   Inventory   │ │
│ │ Map     │ │ │  Room Name + Description      │ │ │   (12 slots)  │ │
│ │ (SVG)   │ │ └───────────────────────────────┘ │ └───────────────┘ │
│ └─────────┘ │                                   │                   │
│             │ ┌───────────────────────────────┐ │ ┌───────────────┐ │
│ ┌─────────┐ │ │      Actions Panel            │ │ │ AI Assistant  │ │
│ │Progress │ │ │  [Look] [Take] [Use] ...      │ │ │ Chat Input    │ │
│ │ Stats   │ │ └───────────────────────────────┘ │ │ Response Box  │ │
│ │ Rooms   │ │                                   │ │ Agent Button  │ │
│ │ Items   │ │ ┌───────────────────────────────┐ │ └───────────────┘ │
│ │ Puzzles │ │ │      Navigation               │ │                   │
│ └─────────┘ │ │         ▲ North               │ │ ┌───────────────┐ │
│             │ │    ◄ W [Hall] E ►             │ │ │ Training      │ │
│             │ │         ▼ South               │ │ │ Panel         │ │
│             │ └───────────────────────────────┘ │ │ Stats/Chart   │ │
│             │                                   │ │ Controls      │ │
│             │ ┌───────────────────────────────┐ │ └───────────────┘ │
│             │ │      Game Log                 │ │                   │
│             │ │  [Log entries scroll here]    │ │                   │
│             │ └───────────────────────────────┘ │                   │
├─────────────┴───────────────────────────────────┴───────────────────┤
│                          FOOTER                                      │
│              [🔄 Restart]    Time: 00:00  |  Moves: 0               │
└─────────────────────────────────────────────────────────────────────┘
```

#### Key Elements

| Element ID | Purpose |
|------------|---------|
| `maze-svg` | SVG container for map |
| `room-name` | Current room title |
| `room-description` | Room description text |
| `actions-container` | Dynamic action buttons |
| `nav-north/south/east/west` | Navigation buttons |
| `inventory-container` | 12 inventory slots |
| `chat-input` | Chatbot input field |
| `ai-response` | Chatbot response display |
| `agent-act` | "Let Agent Act" button |
| `training-mode` | Training toggle checkbox |
| `games-completed` | Games counter |
| `training-count` | Samples counter |
| `avg-loss` | Average loss display |
| `accuracy` | Accuracy display |
| `batch-train-btn` | Batch training button |
| `loss-chart` | Canvas for loss graph |

---

### 5.2 `app.js` - Game Logic

**Lines**: ~1500  
**Purpose**: Main frontend application logic.

#### Game State Object (Lines 10-40)

```javascript
const gameState = {
    currentRoom: 'hall',
    previousRoom: null,
    visitedRooms: ['hall'],
    inventory: [],
    puzzlesSolved: [],
    unlockedDoors: [],
    takenItems: [],
    moveCount: 0,
    startTime: Date.now(),
    gameOver: false,
    lastIntent: null,
    lastAgentPrediction: null,
    gameHistory: [],
    gamesCompleted: 0,
    training: {
        samples: [],
        totalCount: 0,
        correctPredictions: 0,
        totalPredictions: 0,
        lossHistory: [],
        actionDistribution: [0,0,0,0,0,0,0,0,0]
    }
};
```

#### Key Functions

**Navigation** - `navigate(direction)`:
- Checks if path exists
- Handles locked doors (checks requirements)
- Consumes keys when unlocking
- Updates `previousRoom` for anti-loop navigation
- Calls `trainNavigationAction()` for LSTM learning

**Actions** - `performAction(actionId)`:
- Executes game actions
- Updates inventory
- Solves puzzles
- Logs to game log
- Trains agent if training mode enabled

**Chatbot** - `askChatbot()`:
- Sends query to `/chatbot` endpoint
- Displays response
- Sets `lastIntent` for agent

**Agent** - `askAgent()`:
- Calls `/agent/act` endpoint
- Maps action category to specific action via `mapAgentAction()`
- Highlights suggested button
- Executes navigation automatically

**Smart Navigation** - `selectSmartNavigation()`:
- Scores each available direction
- Penalizes previous room (-200)
- Prefers unvisited rooms (+100)
- Prefers rooms with items (+50)
- Avoids recently visited rooms

**Training** - `trainAgent(actionId)`:
- Sends action to `/agent/train`
- Updates loss history
- Updates accuracy counters

**Batch Training** - `batchTrainOnHistory(epochs)`:
- Sends full `gameHistory` to `/agent/batch_train`
- Updates UI with progress
- Shows final loss

**Victory** - `handleVictory()`:
- Displays victory modal
- Increments `gamesCompleted`
- Triggers auto-train if enabled

---

### 5.3 `game_config.js` - Game Data

**Lines**: 377  
**Purpose**: Complete game content definition.

#### Rooms (10 rooms)

```javascript
rooms: {
    hall: {
        id: 'hall',
        name: 'Tavern Entrance',
        description: '...',
        connections: {
            north: 'library',
            east: { room: 'chapel', requires: 'torch', locked: true },
            south: 'dungeon',
            west: 'cellar'
        },
        items: ['torch'],
        actions: ['look', 'take_torch'],
        isStart: true
    },
    // ... 9 more rooms
}
```

**Room Properties**:
| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Display name |
| `description` | string | Full description |
| `connections` | object | N/S/E/W directions |
| `items` | array | Items in room |
| `actions` | array | Available actions |
| `puzzle` | string | Puzzle ID (optional) |
| `isStart` | boolean | Starting room |
| `isEnd` | boolean | Final room |

**Locked Connections**:
```javascript
east: { room: 'chapel', requires: 'torch', locked: true }
// or multiple requirements:
east: { room: 'vault', requires: ['key1', 'key2', 'key3'], locked: true }
```

#### Items (11 items)

```javascript
items: {
    torch: {
        id: 'torch',
        name: 'Everburning Torch',
        description: 'A magical torch...',
        icon: '🔥',
        usable: true,
        useAction: 'light_area'
    },
    // ... more items
}
```

#### Puzzles (5 puzzles)

```javascript
puzzles: {
    candle_sequence: {
        id: 'candle_sequence',
        name: 'Arcane Ritual Circle',
        description: '...',
        hint: 'Light from smallest to greatest',
        solution: [2, 0, 3, 1],
        solved: false,
        reward: 'Dispels the ward'
    },
    // ... more puzzles
}
```

#### Actions (30+ actions)

```javascript
actions: {
    look: { id: 'look', name: 'Look Around', icon: '👁️', intent: 'inspect' },
    take_torch: { id: 'take_torch', name: 'Take Torch', icon: '🔥', intent: 'get_item', item: 'torch' },
    // ... more actions
}
```

---

### 5.4 `maze_renderer.js` - SVG Map

**Lines**: 334  
**Purpose**: Interactive SVG dungeon map visualization.

#### Class: `MazeRenderer`

**Constructor** (Lines 6-59):
- `roomSize`: 60px
- `roomGap`: 40px
- Room positions (x, y grid)
- Room icons (emoji)
- Connections between rooms

**`drawRooms()`** (Lines 144-207):
- Creates SVG `<g>` groups for each room
- Draws rectangle, icon, and name
- Attaches click handlers

**`update(currentRoom, visitedRooms, unlockedDoors)`** (Lines 240-295):
- Updates room colors:
  - Current: Gold border, glow filter
  - Visited: Purple gradient
  - Unknown: Gray gradient
- Updates connection line colors

**`highlightPath(fromRoom, toRoom)`** (Lines 297-318):
- Animates a gold dot moving between rooms

---

### 5.5 `style.css` - Styling

**Purpose**: Dark gothic fantasy theme.

**Key Features**:
- CSS Custom Properties (variables)
- Dark color scheme (#1a1a2e, #2a2a4a)
- Gold accents (#ffd700)
- Cinzel & Crimson Text fonts
- CSS Grid layout
- Animations & transitions
- Responsive design

---

## 6. Data Flow

### 6.1 User Asks Chatbot

```
User types "Where is the key?"
         │
         ▼
    app.js: askChatbot()
         │
         ▼
    POST /chatbot
    Body: {query, currentRoom, inventory, ...}
         │
         ▼
    app.py: chatbot()
         │
         ▼
    rag_system.generate_response()
         │
         ├── retrieve() → FAISS search
         │       │
         │       └── Returns top 3 chunks
         │
         └── LLM generate → Gemini API
                 │
                 └── Returns JSON {message, intent}
         │
         ▼
    Frontend displays response
    Sets gameState.lastIntent
```

### 6.2 User Performs Action (Training Mode)

```
User clicks "Take Torch"
         │
         ▼
    app.js: performAction('take_torch')
         │
         ├── Update game state
         ├── Update UI
         │
         └── trainAgent('take_torch')
                 │
                 ▼
            Map action to index (6)
                 │
                 ▼
            POST /agent/train
            Body: {state, intent, correct_action_id: 6}
                 │
                 ▼
            app.py: agent_train()
                 │
                 ▼
            vectorize_state() → 25 features
                 │
                 ▼
            state_buffer.add()
                 │
                 ▼
            LSTM forward pass
                 │
                 ▼
            CrossEntropyLoss + Backprop
                 │
                 ▼
            Return {loss: 0.5}
                 │
                 ▼
            Frontend updates loss chart
```

### 6.3 Agent Predicts Action

```
User clicks "Let Agent Act"
         │
         ▼
    app.js: askAgent()
         │
         ▼
    Build available actions list
    (including go_north, go_south, etc.)
         │
         ▼
    POST /agent/act
    Body: {state, intent, mask}
         │
         ▼
    app.py: agent_act()
         │
         ▼
    vectorize_state() → 25 features
         │
         ▼
    state_buffer.get_sequence()
         │
         ▼
    LSTM forward pass (eval mode)
         │
         ▼
    argmax(logits) → action_index
         │
         ▼
    Return {action_id: "navigate", action_index: 5}
         │
         ▼
    mapAgentAction("navigate", availableActions)
         │
         ├── Is it navigation?
         │       │
         │       └── selectSmartNavigation()
         │               │
         │               └── Score directions, pick best
         │
         └── Return "go_north"
         │
         ▼
    Execute navigate('north')
```

---

## 7. API Reference

### `POST /chatbot`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | User's question |
| currentRoom | string | No | Current room ID |
| step | int | No | Move count |
| inventory | array | No | Item IDs in inventory |
| puzzlesSolved | array | No | Solved puzzle IDs |
| doorLocked | boolean | No | Portal status |

**Response**: `{message: string, intent: string}`

### `POST /agent/act`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| state | object | Yes | Game state dict |
| intent | string | Yes | Current intent |
| mask | array | No | 9-element action mask |

**Response**: `{action_id: string, action_index: int, sequence_length: int}`

### `POST /agent/train`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| state | object | Yes | Game state dict |
| intent | string | Yes | Current intent |
| correct_action_id | int | Yes | Correct action index (0-8) |

**Response**: `{status: string, loss: float, sequence_length: int}`

### `POST /agent/batch_train`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| history | array | Yes | Array of moves |
| epochs | int | No | Training epochs (default 3) |

**Response**: `{status: string, final_loss: float, samples_trained: int, epochs: int}`

### `POST /agent/reset`

**Response**: `{status: string, message: string}`

---

## 8. Neural Network Details

### Architecture Summary

```
LSTMActionAgent(
  (input_embedding): Linear(in_features=25, out_features=64)
  (lstm): LSTM(64, 64, num_layers=2, batch_first=True, dropout=0.3)
  (layer_norm): LayerNorm((64,))
  (output_layers): Sequential(
    (0): Linear(in_features=64, out_features=32)
    (1): ReLU()
    (2): Dropout(p=0.3)
    (3): Linear(in_features=32, out_features=9)
  )
)
```

### Parameter Count

| Layer | Parameters |
|-------|------------|
| input_embedding | 25 × 64 + 64 = 1,664 |
| LSTM | 4 × (64+64+1) × 64 × 2 = 66,048 |
| layer_norm | 64 + 64 = 128 |
| output_layers | 64×32 + 32 + 32×9 + 9 = 2,377 |
| **Total** | **~14,057** |

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning Rate | 0.0005 | Stable learning |
| Weight Decay | 1e-5 | Regularization |
| Dropout | 0.3 | Prevent overfitting |
| Hidden Size | 64 | Balance capacity/speed |
| Num Layers | 2 | Capture patterns |
| Sequence Length | 10 | Recent context |
| Gradient Clip | 1.0 | Prevent exploding gradients |

---

## 9. RAG System Details

### Embedding Model

**Model**: `all-MiniLM-L6-v2`  
**Dimensions**: 384  
**Type**: Sentence Transformer  

### Vector Store

**Type**: FAISS IndexFlatIP  
**Similarity**: Inner Product (cosine after L2 normalization)  
**Persistence**: `vector_db/index.faiss`, `vector_db/chunks.json`

### Chunking Strategy

1. Split by markdown headers (`#`, `##`, `###`)
2. If chunk > 400 chars, split by paragraphs
3. Minimum chunk size: 20 chars

### Retrieval

- Top K: 3 chunks
- Score Threshold: 0.3 (cosine similarity)

### LLM Configuration

- Model: Gemini 2.5 Flash Lite
- Temperature: 0.7
- Max Response: 200 tokens
- Output Format: JSON

---

## 10. Generated Files

Files created at runtime (should be in `.gitignore`):

| File | Purpose | Can Delete? |
|------|---------|-------------|
| `model_weights.pth` | Trained LSTM model | Yes (resets training) |
| `vector_db/index.faiss` | FAISS vector index | Yes (rebuilds on startup) |
| `vector_db/chunks.json` | Document chunks | Yes (rebuilds on startup) |

---

## Appendix: File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 351 | Flask API server |
| `lstm_agent.py` | 242 | LSTM neural network |
| `game_utils.py` | 152 | Game constants & vectorization |
| `rag/__init__.py` | 17 | Package init |
| `rag/config.py` | 55 | Configuration |
| `rag/llm_handler.py` | 140 | LLM integration |
| `rag/rag_chain.py` | 422 | RAG pipeline |
| `rag/rag_system.py` | 88 | Wrapper interface |
| `index.html` | 328 | Game UI |
| `app.js` | ~1500 | Game logic |
| `game_config.js` | 377 | Game data |
| `maze_renderer.js` | 334 | SVG map |
| `style.css` | ~800 | Styling |
| `requirements.txt` | 9 | Python dependencies |

---

*Documentation generated for NLP Final Project - D&D Adventure Game*
