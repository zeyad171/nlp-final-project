# D&D Adventure 🐉⚔️

An interactive Dungeons & Dragons themed adventure game featuring a RAG-based AI Dungeon Master and a trainable DNN agent.

## Project Structure

```
final project/
├── index.html          # Main game interface
├── style.css           # Dark gothic styling
├── app.js              # Frontend game logic
├── game_config.js      # Game data (rooms, items, puzzles)
├── maze_renderer.js    # SVG map visualization
├── DeepNN_Agent.py     # Backend API server
├── docs/
│   └── game_documentation.md  # RAG knowledge base
└── README.md           # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors torch google-generativeai sentence-transformers faiss-cpu
```

### 2. Start the Backend

```bash
python DeepNN_Agent.py
```

### 3. Open the Game

Open `index.html` in your browser (double-click or use Live Server).

## Features

### 🎮 Game
- 10 D&D themed chambers to explore (Tavern, Wizard's Study, Dragon's Hoard, etc.)
- 12 magical items to collect (Vorpal Dagger, Holy Symbol, Thieves' Tools, etc.)
- 5 puzzles to solve (Arcane Rituals, Dragon Riddles, Scrying Mirrors)
- Fantasy medieval atmosphere with animated UI

### 🤖 AI Dungeon Master
- Uses Google Gemini for natural language responses in DM style
- RAG system retrieves relevant quest hints
- Speaks in fantasy medieval style with dice roll references

### 🧠 DNN Agent
- Simple neural network learns from player actions
- Training visualization with loss chart
- Can play the game autonomously after training

## Controls

| Key | Action |
|-----|--------|
| W / ↑ | Move North |
| S / ↓ | Move South |
| A / ← | Move West |
| D / → | Move East |
| L | Look around |
| Enter | Send chat message |

## File Overview

### Frontend Files

| File | Purpose |
|------|---------|
| `index.html` | Page layout and structure |
| `style.css` | CSS variables, dark theme, animations |
| `app.js` | Game state, rendering, API calls |
| `game_config.js` | All game data in one place |
| `maze_renderer.js` | SVG map with room highlighting |

### Backend Files

| File | Purpose |
|------|---------|
| `DeepNN_Agent.py` | Flask server with 3 endpoints |
| `docs/game_documentation.md` | Knowledge base for RAG |

## API Endpoints

### POST `/chatbot`
Ask the AI for help.

```json
// Request
{"query": "Where is the key?", "step": 1, "inventory": [], "doorLocked": true}

// Response
{"message": "The key is hidden in the library desk.", "intent": "inspect"}
```

### POST `/agent/act`
Get agent's action prediction.

```json
// Request
{"state": {"step": 1, "inventory": [], "doorLocked": true}, "intent": "inspect"}

// Response
{"action_id": "look", "action_index": 0}
```

### POST `/agent/train`
Train the agent on correct actions.

```json
// Request
{"state": {...}, "intent": "inspect", "correct_action_id": 0}

// Response
{"status": "trained", "loss": 0.5}
```

## How Training Works

1. Enable **Training Mode** toggle
2. Ask the chatbot a question (sets the intent)
3. Click the **correct** action button
4. The agent learns to associate that state+intent with the action
5. After ~20 samples, the agent becomes "well trained"

## Tech Stack

- **Frontend**: Vanilla HTML/CSS/JS
- **Backend**: Python Flask
- **AI**: Google Gemini 2.5 Flash
- **ML**: PyTorch (simple neural network)
- **Search**: FAISS + Sentence Transformers

## Troubleshooting

### "The spirits are silent..."
- Check if backend is running on port 5000
- Check browser console for network errors

### Agent not learning
- Make sure Training Mode is enabled
- Ask chatbot first to set intent

### CORS errors
- Backend must be running before opening game
- Try clearing browser cache

---

Built for NLP Final Project 🎓

*May your dice rolls be ever in your favor!* 🎲
