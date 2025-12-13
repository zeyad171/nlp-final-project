# LSTM Agent Training Guide

## Overview

The D&D Adventure Game features an **LSTM (Long Short-Term Memory) neural network** that learns to predict player actions based on game state sequences. This guide explains how to train the model effectively.

---

## How The Training Works

### Architecture
```
Input (25 features) → Embedding (64) → LSTM (2 layers, 64 hidden) → MLP → Output (9 actions)
```

### State Features (25 total)
| Feature Group | Count | Description |
|---------------|-------|-------------|
| Room encoding | 10 | One-hot vector for current room |
| Intent encoding | 9 | One-hot vector for player intent |
| Step progress | 1 | Normalized game progress (0-1) |
| Inventory count | 1 | Normalized item count (0-1) |
| Has torch | 1 | Binary: has light source |
| Has key | 1 | Binary: has any key item |
| Has dagger | 1 | Binary: has weapon |
| Door locked | 1 | Binary: portal status |

### Action Classes (9 total)
| Index | Action | Description |
|-------|--------|-------------|
| 0 | look | Look around the room |
| 1 | read_books | Read books/scrolls |
| 2 | take_key | Take key items |
| 3 | use_keys | Use keys to unlock |
| 4 | open_exit | Open exit/escape |
| 5 | navigate | Move between rooms |
| 6 | take_item | Take general items |
| 7 | solve_puzzle | Solve puzzles |
| 8 | interact | General interaction |

---

## Training Methods

### Method 1: Online Learning (Automatic)
Every action you take trains the model in real-time.

**Pros:** Continuous learning  
**Cons:** Sequential data, no shuffling, prone to overfitting

### Method 2: Batch Training (Recommended) ⭐
Click **"Train on History"** button after completing games.

**Pros:** 
- Processes full game sequences
- More stable learning
- Model is saved to disk

**Cons:** Requires completing games first

---

## Step-by-Step Training Instructions

### Phase 1: Data Collection
1. **Start a new game** (click Play or Restart)
2. **Play through the entire game** to victory
3. **Make diverse actions** - don't repeat the same path every time
4. **Complete 3-5 full games** before batch training

### Phase 2: Batch Training
1. After completing games, click **"🧠 Train on History"**
2. Wait for training to complete (watch the console/logs)
3. Check the training metrics:
   - **Loss should decrease** (target: < 1.5)
   - **Samples trained** should match your moves

### Phase 3: Verification
1. **Start a new game**
2. Click **"Ask Agent"** to see predictions
3. Compare agent predictions to your intended actions
4. **Accuracy should improve** over time

### Phase 4: Iteration
1. Play more games with varied strategies
2. Batch train again
3. Repeat until accuracy > 60%

---

## Training Metrics Explained

### Loss
| Value | Status | Action |
|-------|--------|--------|
| > 2.0 | Poor | Train more, check for issues |
| 1.5 - 2.0 | Learning | Continue training |
| 1.0 - 1.5 | Good | Model is learning well |
| < 1.0 | Excellent | Model has converged |

### Accuracy
| Value | Status | Meaning |
|-------|--------|---------|
| 0-20% | Random | Model hasn't learned |
| 20-40% | Learning | Patterns emerging |
| 40-60% | Good | Reasonable predictions |
| 60-80% | Very Good | Strong predictions |
| 80%+ | Excellent | Near-optimal |

### Loss Trend (Critical!)
- ✅ **Decreasing loss** = Model is learning
- ❌ **Increasing loss** = Problem! (see troubleshooting)
- ➡️ **Flat loss** = Model has converged or is stuck

---

## Best Practices

### DO ✅
1. **Play complete games** - Partial games have incomplete sequences
2. **Vary your strategy** - Take different paths each playthrough
3. **Train in batches** - After 3-5 games, not after each action
4. **Check loss trend** - Should decrease across epochs
5. **Restart the server** after code changes

### DON'T ❌
1. **Don't spam Train on History** - Training 10 times in a row causes overfitting
2. **Don't always take the same path** - Model won't generalize
3. **Don't train with < 50 moves** - Not enough data
4. **Don't ignore increasing loss** - Sign of a problem

---

## Troubleshooting

### Problem: Loss is INCREASING
**Cause:** Learning rate too high, or overfitting to repeated data

**Solution:**
1. Delete `model_weights.pth` to reset
2. Play 3-5 diverse games
3. Train once, not multiple times
4. If persists, lower learning rate in `lstm_agent.py`

### Problem: Model always predicts same action
**Cause:** Mode collapse - model learned one action dominates

**Solution:**
1. Delete `model_weights.pth`
2. Play games with more action variety
3. Ensure all action types are represented in training data

### Problem: 0% Accuracy
**Cause:** Model predictions not matching user actions (possible action mapping issue)

**Solution:**
1. Check console for errors
2. Verify `ACTION_MAP` in `game_utils.py` matches frontend
3. Restart both backend and frontend

### Problem: Training fails
**Cause:** Tensor shape mismatch or missing data

**Solution:**
1. Check console for exact error
2. Restart Python backend
3. Clear browser cache and refresh

---

## Model Persistence

The trained model saves to `model_weights.pth` after batch training.

### Save Location
```
final project/
├── model_weights.pth  ← Saved model weights
├── app.py
└── ...
```

### When Model Loads
- Automatically on server startup
- Checks if architecture matches (25 input features)
- Falls back to fresh model if mismatch

### Reset Model
To start fresh, delete the weights file:
```powershell
Remove-Item model_weights.pth
```

---

## Expected Training Timeline

| Games Played | Expected Loss | Expected Accuracy |
|--------------|---------------|-------------------|
| 1-2 | 2.0+ | 10-20% |
| 3-5 | 1.5-2.0 | 20-40% |
| 5-10 | 1.2-1.5 | 40-60% |
| 10-20 | 1.0-1.2 | 60-80% |
| 20+ | < 1.0 | 80%+ |

---

## Technical Details

### Hyperparameters
| Parameter | Value | Location |
|-----------|-------|----------|
| Learning Rate | 0.0005 | `lstm_agent.py` |
| Weight Decay | 1e-5 | `lstm_agent.py` |
| Dropout | 0.3 | `lstm_agent.py` |
| Hidden Size | 64 | `lstm_agent.py` |
| LSTM Layers | 2 | `lstm_agent.py` |
| Gradient Clip | 1.0 | `app.py` |
| Batch Epochs | 3 | `app.js` |

### Training Flow
```
User Action → Frontend records move → /agent/train API
                                           ↓
                                    Add state to buffer
                                           ↓
                                    Get state sequence
                                           ↓
                                    Forward pass (LSTM)
                                           ↓
                                    Compute CrossEntropyLoss
                                           ↓
                                    Backprop + Gradient Clip
                                           ↓
                                    Update weights
```

---

## Quick Reference

### Minimum Training
1. Play 3 complete games
2. Click "Train on History" once
3. Verify loss < 2.0

### Optimal Training
1. Play 10+ diverse games
2. Train after every 3-5 games
3. Target loss < 1.5, accuracy > 50%

### Commands
```powershell
# Start backend
python app.py

# Reset model
Remove-Item model_weights.pth

# Delete vector index (forces RAG rebuild)
Remove-Item -Recurse vector_db/
```

---

## Summary

1. **Play 3-5 complete games** with varied strategies
2. **Click "Train on History"** to batch train
3. **Watch the loss** - it should decrease
4. **Check accuracy** - should improve over time
5. **Repeat** until satisfied with predictions

The key to good training is **diverse training data** and **patience**. The LSTM needs to see many different game states and actions to learn meaningful patterns.
