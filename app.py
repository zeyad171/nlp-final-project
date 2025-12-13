"""
==============================================
D&D ADVENTURE - FLASK API SERVER
==============================================
Main application entry point providing:
1. RAG-based chatbot (Dungeon Master AI)
2. LSTM agent for action prediction
3. Training endpoints for the neural network

Author: NLP Final Project
==============================================
"""

# IMPORTANT: Set these BEFORE any imports to prevent TensorFlow loading
import os
import sys
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TORCH"] = "1"
sys.modules['tensorflow'] = None

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our modules
from lstm_agent import create_agent, save_model
from rag.rag_system import get_rag_system
from game_utils import ACTION_MAP, vectorize_state, STATE_SIZE
import random


# ==============================================
# FLASK APP SETUP
# ==============================================

app = Flask(__name__)
CORS(app)


# ==============================================
# INITIALIZE COMPONENTS
# ==============================================

print("\n" + "=" * 50)
print("  Initializing D&D Adventure Backend")
print("=" * 50)

# Initialize LSTM Agent with correct input size
model, optimizer, criterion, state_buffer = create_agent(
    num_actions=len(ACTION_MAP),
    input_size=STATE_SIZE  # 25 features
)

# Initialize RAG System (lazy loading on first request)
rag_system = None

def get_rag():
    """Lazy load the RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = get_rag_system()
    return rag_system


# ==============================================
# CHATBOT ENDPOINT
# ==============================================

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    RAG-based chatbot endpoint (Dungeon Master AI).
    
    Expects JSON: {query, step, inventory, doorLocked, currentRoom, puzzlesSolved}
    Returns JSON: {message, intent}
    """
    data = request.json
    query = data.get('query', 'What should I do next?')
    
    # Build game state dict
    game_state = {
        'currentRoom': data.get('currentRoom', 'hall'),
        'step': data.get('step', 1),
        'inventory': data.get('inventory', []),
        'puzzlesSolved': data.get('puzzlesSolved', []),
        'doorLocked': data.get('doorLocked', True)
    }
    
    try:
        rag = get_rag()
        message, intent = rag.generate_response(query, game_state)
        return jsonify({"message": message, "intent": intent})
    except Exception as e:
        print(f"Chatbot error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "message": "The spirits are unclear... Try looking around for clues!",
            "intent": "inspect"
        })


# ==============================================
# AGENT ENDPOINTS
# ==============================================

@app.route('/agent/act', methods=['POST'])
def agent_act():
    """
    LSTM Agent action prediction endpoint.
    
    Uses the state history buffer to provide sequential context.
    
    Expects JSON: {state, intent, mask}
    Returns JSON: {action_id, action_index, sequence_length}
    """
    try:
        data = request.json
        
        # Prepare current state vector
        state_vec = vectorize_state(data['state'], data['intent'])
        
        # Add to state history buffer
        state_buffer.add(state_vec.squeeze(0))
        
        # Get action mask
        mask = torch.FloatTensor(data.get('mask', [1] * len(ACTION_MAP))).unsqueeze(0)
        
        # Get the sequence of states from buffer
        state_sequence = state_buffer.get_sequence()
        
        # Handle empty buffer - use current state only
        if state_sequence is None:
            state_sequence = state_vec
        
        # Predict action using LSTM with full history
        model.eval()
        with torch.no_grad():
            logits, _ = model(state_sequence, mask=mask)
            action_idx = torch.argmax(logits).item()
        
        return jsonify({
            "action_id": ACTION_MAP[action_idx],
            "action_index": action_idx,
            "sequence_length": len(state_buffer)
        })
        
    except Exception as e:
        print(f"Agent act error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "action_id": ACTION_MAP[0],
            "action_index": 0,
            "error": str(e)
        })


@app.route('/agent/train', methods=['POST'])
def agent_train():
    """
    LSTM Agent single-step training endpoint.
    
    Trains on the current state sequence from the history buffer.
    
    Expects JSON: {state, intent, correct_action_id}
    Returns JSON: {status, loss, sequence_length}
    """
    try:
        data = request.json
        
        # Prepare current state vector and add to buffer
        state_vec = vectorize_state(data['state'], data['intent'])
        state_buffer.add(state_vec.squeeze(0))
        
        # Get target action (ensure it's within valid range)
        action_id = data.get('correct_action_id', 0)
        if action_id >= len(ACTION_MAP):
            action_id = 0
        target = torch.LongTensor([action_id])
        
        # Get the sequence of states from buffer
        state_sequence = state_buffer.get_sequence()
        
        # Handle empty buffer case
        if state_sequence is None:
            return jsonify({
                "status": "skipped",
                "loss": 0.0,
                "message": "Buffer empty, state added",
                "sequence_length": len(state_buffer)
            })
        
        # Training step with LSTM
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with sequence
        logits, _ = model(state_sequence)
        
        # Compute loss and backpropagate
        loss = criterion(logits, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return jsonify({
            "status": "trained",
            "loss": loss.item(),
            "sequence_length": len(state_buffer)
        })
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e),
            "loss": 0.0
        })


@app.route('/agent/batch_train', methods=['POST'])
def agent_batch_train():
    """
    Batch training endpoint - trains on entire game history.
    
    More effective for LSTM as it processes full sequences.
    
    Expects JSON: {history: [{state, intent, actionIndex}], epochs: int}
    Returns JSON: {status, final_loss, samples_trained}
    """
    try:
        data = request.json
        history = data.get('history', [])
        epochs = data.get('epochs', 3)
        
        if len(history) == 0:
            return jsonify({
                "status": "error",
                "error": "No history provided",
                "final_loss": 0.0,
                "samples_trained": 0
            })
        
        print(f"Batch training on {len(history)} moves for {epochs} epochs...")
        
        # Convert history to training sequences
        training_data = []
        for move in history:
            state_data = move.get('state', {})
            intent = move.get('intent', 'inspect')
            action_idx = move.get('actionIndex', 0)
            
            state_vec = vectorize_state(state_data, intent)
            training_data.append({
                'state': state_vec.squeeze(0),
                'action': action_idx
            })
        
        total_loss = 0.0
        samples_trained = 0
        
        model.train()
        
        # Create sliding window sequences for better LSTM training
        sequence_length = 5  # Use last 5 states to predict action
        sequences = []
        for i in range(len(training_data)):
            # Get states from max(0, i-seq_len+1) to i+1
            start_idx = max(0, i - sequence_length + 1)
            seq_states = [training_data[j]['state'] for j in range(start_idx, i + 1)]
            target_action = training_data[i]['action']
            sequences.append({
                'states': torch.stack(seq_states, dim=0),  # (seq_len, features)
                'action': target_action
            })
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle sequences each epoch (critical for learning!)
            random.shuffle(sequences)
            model.reset_hidden()
            
            for seq in sequences:
                state_sequence = seq['states'].unsqueeze(0)  # (1, seq_len, features)
                target = torch.LongTensor([seq['action']])
                
                optimizer.zero_grad()
                logits, _ = model(state_sequence)
                loss = criterion(logits, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                samples_trained += 1
            
            avg_epoch_loss = epoch_loss / max(1, len(sequences))
            total_loss += avg_epoch_loss
            print(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_epoch_loss:.4f}")
        
        final_loss = total_loss / max(1, epochs)
        print(f"Batch training complete! Final avg loss: {final_loss:.4f}, trained on {samples_trained} sequences")
        
        # Save model after batch training
        save_model(model, optimizer)
        
        return jsonify({
            "status": "success",
            "final_loss": final_loss,
            "samples_trained": samples_trained,
            "epochs": epochs
        })
        
    except Exception as e:
        print(f"Batch training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e),
            "final_loss": 0.0,
            "samples_trained": 0
        })


@app.route('/agent/reset', methods=['POST'])
def agent_reset():
    """
    Reset the LSTM agent's state buffer.
    
    Call this when starting a new game.
    """
    state_buffer.clear()
    model.reset_hidden()
    
    return jsonify({
        "status": "reset",
        "message": "LSTM state buffer cleared"
    })


# ==============================================
# MAIN ENTRY POINT
# ==============================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  D&D Adventure - Backend Server")
    print("  Running on http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(port=5000, debug=True)
