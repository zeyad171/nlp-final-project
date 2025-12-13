"""
==============================================
GAME UTILITIES
==============================================
Game configuration constants and helper functions
for state vectorization.

Author: NLP Final Project
==============================================
"""

import torch


# ==============================================
# ACTION CONFIGURATION
# ==============================================

# Actions the LSTM agent can predict
ACTION_MAP = [
    "look",           # 0 - Look around
    "read_books",     # 1 - Read books/scrolls
    "take_key",       # 2 - Take key items
    "use_keys",       # 3 - Use keys/unlock
    "open_exit",      # 4 - Open exit/escape
    "navigate",       # 5 - Move between rooms
    "take_item",      # 6 - Take general items
    "solve_puzzle",   # 7 - Solve puzzles
    "interact"        # 8 - General interaction
]

# Intents recognized by the chatbot
INTENT_MAP = [
    "inspect",        # 0 - Look around, examine
    "navigate",       # 1 - Move between rooms
    "get_item",       # 2 - Pick up items
    "use_item",       # 3 - Use an item
    "unlock",         # 4 - Unlock doors
    "read",           # 5 - Read books, scrolls
    "solve_puzzle",   # 6 - Solve a puzzle
    "interact",       # 7 - Generic interaction
    "escape"          # 8 - Final escape
]

# Room name mapping for display
ROOM_NAMES = {
    'hall': "Dragon's Rest Tavern",
    'library': "Wizard's Study", 
    'study': "Treasure Chamber",
    'cellar': "Dwarven Forge",
    'gallery': "Hall of Heroes",
    'chapel': "Temple of Pelor",
    'attic': "Dragon's Hoard",
    'crypt': "Tomb of the Lich",
    'dungeon': "Goblin Prison",
    'vault': "Portal Chamber"
}


# ==============================================
# STATE VECTORIZATION
# ==============================================

def vectorize_state(state_data, intent_str):
    """
    Convert game state to neural network input tensor.
    
    The state vector has 12 features:
        - step_progress (0-1): Normalized progress through the game
        - has_key (0 or 1): Whether player has any key
        - door_locked (0 or 1): Whether final door is locked
        - intent_vector (9 values): One-hot encoded intent
    
    Args:
        state_data: Dict with 'step', 'inventory', 'doorLocked'
        intent_str: String intent from INTENT_MAP
    
    Returns:
        torch.FloatTensor of shape (1, 12)
    """
    # Extract state features
    step = state_data.get('step', 1)
    inventory = state_data.get('inventory', [])
    door_locked = state_data.get('doorLocked', True)
    
    # Normalize features
    step_norm = min(step / 10.0, 1.0)
    has_key = 1.0 if any('key' in str(item).lower() for item in inventory) else 0.0
    door_locked_val = 1.0 if door_locked else 0.0
    
    # One-hot encode intent
    intent_vec = [0.0] * len(INTENT_MAP)
    if intent_str in INTENT_MAP:
        intent_vec[INTENT_MAP.index(intent_str)] = 1.0
    else:
        intent_vec[0] = 1.0  # Default to 'inspect'
    
    # Combine features: [step_norm, has_key, door_locked, intent(9)]
    features = [step_norm, has_key, door_locked_val] + intent_vec
    
    return torch.FloatTensor(features).unsqueeze(0)


def get_action_name(action_idx):
    """Get the action name for a given index."""
    if 0 <= action_idx < len(ACTION_MAP):
        return ACTION_MAP[action_idx]
    return ACTION_MAP[0]


def get_intent_index(intent_str):
    """Get the index for an intent string."""
    if intent_str in INTENT_MAP:
        return INTENT_MAP.index(intent_str)
    return 0


def get_room_display_name(room_id):
    """Get the display name for a room ID."""
    return ROOM_NAMES.get(room_id, room_id)
