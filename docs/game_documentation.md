# D&D Adventure - Complete Game Documentation

## Game Overview

You are an adventurer trapped in a magical dungeon! Your quest is to explore 10 interconnected chambers, collect magical items, solve puzzles, and activate the escape portal to complete your quest and earn legendary status!

## Setting

A fantasy dungeon inspired by Dungeons & Dragons. From the Dragon's Rest Tavern to the Portal Chamber, each room holds secrets, treasures, and challenges worthy of a true hero.

## Quest Objectives

1. Explore all 10 chambers of the dungeon
2. Collect 10 magical items scattered throughout
3. Solve 5 puzzles to unlock passages
4. Perform the Portal Activation Ritual
5. Step through the portal to freedom!

---

## Technical Architecture

### AI Systems

This game features two AI systems working together:

#### 1. RAG-Based Dungeon Master (Chatbot)
- **Technology**: Retrieval-Augmented Generation (RAG)
- **Embedding Model**: SentenceTransformer (all-MiniLM-L6-v2)
- **Vector Store**: FAISS with persistent indexing (saved to `vector_db/`)
- **LLM**: Google Gemini 2.5 Flash Lite
- **Features**:
  - Smart document chunking (section-aware)
  - Semantic similarity search
  - Context-aware responses based on game state
  - Comprehensive item knowledge database
  - Fallback logic for offline operation

#### 2. LSTM Action Agent
- **Architecture**: Long Short-Term Memory (LSTM) Neural Network
- **Input**: 25 features:
  - Room encoding (10 one-hot)
  - Intent encoding (9 one-hot)
  - Progress, inventory count, has_torch, has_key, has_dagger, door_locked (6 features)
- **Hidden Layers**: 64 units, 2 LSTM layers with 30% dropout
- **Output**: 9 action categories
- **Training**: 
  - Online learning (every action)
  - Batch training on game history
  - Model persistence to `model_weights.pth`
- **Features**:
  - State history buffer (remembers last 10 states)
  - Gradient clipping (max_norm=1.0)
  - Action masking for valid moves
  - Can navigate between rooms autonomously
  - Lower learning rate (0.0005) for stable training

### Project Structure

```
final project/
├── app.py              # Flask API server (main entry point)
├── lstm_agent.py       # LSTM neural network model + persistence
├── game_utils.py       # Game constants & 25-feature state vectorization
├── rag/                # RAG system package
│   ├── __init__.py
│   ├── config.py       # RAG configuration (API keys, models)
│   ├── llm_handler.py  # Gemini/Ollama integration
│   ├── rag_chain.py    # Main RAG pipeline with item knowledge
│   └── rag_system.py   # Simple wrapper interface
├── docs/
│   ├── game_documentation.md  # This file (RAG knowledge base)
│   └── TRAINING_GUIDE.md      # Detailed training instructions
├── index.html          # Game interface
├── app.js              # Frontend logic + agent integration
├── game_config.js      # Rooms, items, puzzles, actions
├── maze_renderer.js    # SVG map visualization
├── style.css           # Dark gothic styling
├── model_weights.pth   # Saved LSTM weights (generated)
├── vector_db/          # FAISS index cache (generated)
└── requirements.txt
```

---

## The 10 Chambers

### 1. Tavern Entrance (Starting Room) - "hall"
- Description: The Dragon's Rest Tavern where adventures begin. Quest board on the wall.
- Connections: North to Wizard's Study, East to Temple (requires Torch), South to Goblin Prison, West to Dwarven Forge
- Items: Everburning Torch
- Actions: Look around, take torch

### 2. Wizard's Study - "library"
- Description: Arcane tomes and spell scrolls. Crystal ball glows on reading desk. Runes pulse with magic.
- Connections: South to Tavern, North to Dragon's Hoard, East to Hall of Heroes (requires Candle), West to Treasure Chamber (requires Thieves' Tools)
- Items: Ritual Candle, Scroll of Teleportation
- Puzzle: Arcane Ritual Circle - light candles from smallest to greatest
- Actions: Read spellbooks, light candles, take items

### 3. Treasure Chamber - "study"
- Description: Gold coins and gems everywhere. Dragon slayer portrait guards a locked chest.
- Connections: East to Wizard's Study, South to Dwarven Forge
- Items: Thieves' Tools (after puzzle)
- Puzzle: Dragon Slayer's Riddle - count dragon heads (3-7-4-1)
- Actions: Examine portrait, open safe, take key

### 4. Dwarven Forge - "cellar"
- Description: Ancient forges and dwarven runes. Legendary weapons on display.
- Connections: North to Treasure Chamber, East to Tavern, South to Tomb of the Lich
- Items: None
- Actions: Look around

### 5. Hall of Heroes - "gallery"
- Description: Statues of legendary adventurers. Magical scrying mirror.
- Connections: West to Wizard's Study, South to Temple (requires Torch)
- Items: Vorpal Dagger +1
- Puzzle: Scrying Mirror Challenge - present weapon at midnight position
- Actions: Examine mirror, study portraits, take dagger

### 6. Temple of Pelor - "chapel"
- Description: Sacred temple to the sun god. Golden light through stained glass.
- Connections: North to Hall of Heroes, West to Tavern (requires Holy Symbol), South to Portal Chamber
- Items: Blessed Potion, Holy Symbol
- Actions: Pray for blessings, examine altar, take items

### 7. Dragon's Hoard - "attic"
- Description: Mountains of treasure! Dragon scales and magical artifacts.
- Connections: South to Wizard's Study
- Items: Dungeon Map (reveals all rooms)
- Actions: Look around, take map

### 8. Tomb of the Lich - "crypt"
- Description: Ancient sarcophagi with necrotic energy. Inscriptions tell of dark magic.
- Connections: North to Dwarven Forge, East to Goblin Prison
- Items: Lich's Phylactery Key
- Secret: Inscriptions reveal lever sequence: LEFT, RIGHT, MIDDLE
- Actions: Read inscriptions, open coffin, take skeleton key

### 9. Goblin Prison - "dungeon"
- Description: Cages and chains from goblin occupation. Lever mechanism controls doors.
- Connections: North to Tavern, West to Tomb, East to Portal Chamber (requires all 3 keys)
- Items: Dragon Scale Key
- Puzzle: Goblin Lock Mechanism - use all three keys and correct lever sequence
- Actions: Examine mechanism, pull lever, use keys, take key

### 10. Portal Chamber (Final Room) - "vault"
- Description: Mystical chamber where reality bends. Arcane circles glow. Shimmering portal awaits.
- Connections: West to Goblin Prison, North to Temple, Exit (requires ritual)
- Items: Tome of Portal Magic
- Puzzle: Portal Activation Ritual
- Actions: Take tome, read tome, perform ritual, open exit

---

## All 10 Magical Items

### Light Sources
1. **Everburning Torch** 🔥 - Magical continual flame. Required to access Temple and Hall of Heroes.
2. **Ritual Candle** 🕯️ - Essential for spellcasting. Required to access Hall of Heroes from Study.

### Keys (Consumed When Used)
3. **Thieves' Tools** 🗝️ - +2 to lockpicking checks. Unlocks Treasure Chamber. Consumed after use.
4. **Dragon Scale Key** 🔑 - Forged from dragon scales. One of three keys for Portal Chamber.
5. **Lich's Phylactery Key** 💀 - Carved from lich bone. One of three keys for Portal Chamber.

### Weapons & Holy Items
6. **Vorpal Dagger +1** 🗡️ - Extra damage to undead. Required for mirror puzzle.
7. **Blessed Potion** 💧 - Holy water blessed by Pelor. Required for final ritual. Consumed when used.
8. **Holy Symbol** ✝️ - Advantage on saves vs evil. Unlocks passage between Tavern and Temple.

### Quest Items
9. **Dungeon Map** 🗺️ - Reveals entire dungeon layout when taken.
10. **Scroll of Teleportation** 📜 - Contains teleportation ritual. Required for final ritual. Consumed when used.
11. **Tome of Portal Magic** 📕 - Final ritual spellbook. Required for final ritual. Consumed when used.

---

## The 5 Puzzles

### 1. Arcane Ritual Circle (Wizard's Study)
- Light the candles from smallest flame to greatest
- Reward: Dispels ward, reveals Thieves' Tools in safe
- Hint: "Light from smallest flame to greatest power"

### 2. Dragon Slayer's Riddle (Treasure Chamber)
- Count dragon heads in the portrait sections
- Combination: 3-7-4-1
- Reward: Opens safe containing Thieves' Tools

### 3. Scrying Mirror Challenge (Hall of Heroes)
- Present your weapon at midnight position
- Requirement: Must have Vorpal Dagger
- Reward: Reveals passage to Temple

### 4. Goblin Lock Mechanism (Goblin Prison)
- Lever sequence: LEFT, RIGHT, MIDDLE (found in Tomb inscriptions)
- Requirement: All three magical keys (Thieves' Tools, Dragon Scale Key, Lich's Key)
- Reward: Opens passage to Portal Chamber
- Note: All three keys are consumed when the mechanism is activated

### 5. Portal Activation Ritual (Portal Chamber)
- Requirements: Blessed Potion, Scroll of Teleportation, Tome of Portal Magic
- Steps: Pour potion on circle, recite scroll, invoke tome
- Reward: Portal opens - QUEST COMPLETE!
- Note: All three ritual items are consumed when performing the ritual

---

## Locked Passages & Requirements

| From | To | Requires |
|------|-----|----------|
| Tavern | Temple of Pelor | Everburning Torch |
| Wizard's Study | Hall of Heroes | Ritual Candle |
| Hall of Heroes | Temple of Pelor | Everburning Torch |
| Temple of Pelor | Tavern | Holy Symbol |
| Wizard's Study | Treasure Chamber | Thieves' Tools (consumed) |
| Goblin Prison | Portal Chamber | All 3 keys (consumed) |
| Portal Chamber | EXIT | Complete final ritual |

---

## Item Consumption Rules

Some items are **consumed** (removed from inventory) when used:
- **Keys**: Thieves' Tools consumed when unlocking Treasure Chamber
- **Three Portal Keys**: All consumed when opening Portal Chamber
- **Ritual Items**: Holy Water, Scroll, and Tome consumed when performing the final ritual

---

## Tips for Adventurers

1. **Roll for Perception** - Always look around in new rooms
2. **Get the torch first** - Required to access Temple and Hall of Heroes
3. **Read everything** - Spellbooks and inscriptions hold puzzle solutions
4. **The map reveals all** - Find it in the Dragon's Hoard
5. **Collect all three keys** - You need Thieves' Tools, Dragon Scale Key, and Lich's Key for the Portal Chamber
6. **The dagger is magical** - Required for the mirror puzzle in Hall of Heroes
7. **Holy items are essential** - The Symbol allows travel between Tavern and Temple
8. **Prepare the ritual** - Gather Holy Water, Scroll, and Tome before attempting the final ritual
9. **Items disappear after use** - Keys and ritual items are consumed, so use them wisely

---

## Game Controls

| Key | Action |
|-----|--------|
| Arrow Keys / WASD | Navigate between chambers |
| L | Look around (Roll for Perception!) |
| E | Export game log to clipboard |
| Click action buttons | Perform specific actions |
| Enter in chat | Consult the Dungeon Master |

---

## LSTM Agent Action Categories

The agent predicts one of 9 action types:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | look | Look around the room |
| 1 | read_books | Read books, scrolls, inscriptions |
| 2 | take_key | Take key items |
| 3 | use_keys | Use keys, pull levers |
| 4 | open_exit | Open exit, escape |
| 5 | navigate | Move between rooms (N/S/E/W) |
| 6 | take_item | Take general items |
| 7 | solve_puzzle | Solve puzzles |
| 8 | interact | General interaction |

---

## Training Metrics Explained

### Loss (Lower = Better)
| Value | Status |
|-------|--------|
| > 2.0 | Untrained |
| 1.5-2.0 | Learning |
| 1.0-1.5 | Good |
| < 1.0 | Excellent |

### Model Health Indicators
- 🔴 **Untrained**: < 20 samples
- 🟡 **Learning**: Loss > 1.5 or < 3 games
- 🟢 **Good**: Loss 1.0-1.5
- 🟢 **Excellent**: Loss < 1.0

---

## Optimal Path (Speedrun)

1. Tavern → Take Torch
2. Go East to Temple → Take Holy Water, Cross
3. Go North to Hall of Heroes → Take Dagger → Solve Mirror Puzzle
4. Go West to Wizard's Study → Take Candle, Scroll → Light Candles (solve puzzle)
5. Go West to Treasure Chamber → Examine Portrait → Open Safe (3-7-4-1) → Take Thieves' Tools
6. Go North to Dragon's Hoard → Take Map
7. Return South, South to Dwarven Forge
8. Go South to Tomb of the Lich → Read Inscriptions → Take Skeleton Key
9. Go East to Goblin Prison → Take Dragon Scale Key → Pull Lever (with all keys)
10. Go East to Portal Chamber → Take Tome → Read Tome → Perform Ritual → Open Exit
11. 🎉 VICTORY! 🎉

---

## API Endpoints

### Chatbot (RAG)
```
POST /chatbot
Body: {query, step, inventory, currentRoom, puzzlesSolved, doorLocked}
Response: {message, intent}
```

### Agent Actions
```
POST /agent/act       - Get action prediction (includes navigation)
POST /agent/train     - Train on single action
POST /agent/batch_train - Train on game history (saves model)
POST /agent/reset     - Reset agent state buffer
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| GEMINI_API_KEY | Google Gemini API key for RAG chatbot |

---

## Files Generated at Runtime

| File | Purpose |
|------|---------|
| `model_weights.pth` | Trained LSTM weights (persists between sessions) |
| `vector_db/index.faiss` | FAISS vector index |
| `vector_db/chunks.json` | Document chunks for retrieval |
