"""
==============================================
LSTM ACTION AGENT
==============================================
LSTM-based neural network for sequential action prediction.
Learns to choose actions based on game state history.

Author: NLP Final Project
==============================================
"""

import torch
import torch.nn as nn


class LSTMActionAgent(nn.Module):
    """
    LSTM-based neural network that learns to choose actions from sequential game states.
    
    The LSTM processes a sequence of past game states to predict the next action,
    allowing the agent to learn temporal patterns and context from game history.
    
    Input per timestep (12 features):
        - step_progress: How far the player is (0-1)
        - has_key: Does player have any key? (0 or 1)
        - door_locked: Is final door locked? (0 or 1)
        - intent_vector: One-hot encoded intent (9 values)
    
    Architecture:
        - Input embedding layer: 12 -> 64
        - LSTM: 64 hidden units, 2 layers with dropout
        - Output layers: 64 -> 32 -> 9 (action logits)
    
    Output:
        - Logits for each of the 9 possible actions
    """
    
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_actions=9, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # LSTM layer - processes sequences of game states
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layers (MLP head)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_actions)
        )
        
        # Initialize hidden state storage
        self.hidden = None
        
    def init_hidden(self, batch_size=1):
        """Initialize LSTM hidden states to zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)
    
    def forward(self, x, mask=None, hidden=None):
        """
        Forward pass with optional action masking.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) or (batch, features)
            mask: Optional action mask tensor
            hidden: Optional tuple of (h0, c0) hidden states
        
        Returns:
            logits: Action logits of shape (batch, num_actions)
            hidden: Updated hidden states for next forward pass
        """
        # Handle single timestep input (add sequence dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Embed input features
        embedded = self.input_embedding(x)  # (batch, seq_len, hidden_size)
        embedded = torch.relu(embedded)
        
        # Process through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Take the output from the last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply layer normalization
        normalized = self.layer_norm(last_output)
        
        # Generate action logits
        logits = self.output_layers(normalized)
        
        # Apply action mask if provided
        if mask is not None:
            logits = logits + ((1.0 - mask) * -1e9)
        
        return logits, hidden
    
    def reset_hidden(self):
        """Reset hidden states (call at start of new game)."""
        self.hidden = None


class StateHistoryBuffer:
    """
    Maintains a rolling buffer of recent game states for LSTM input.
    
    Expected tensor shapes:
        - Each state: (12,) - 12 features
        - Sequence output: (1, seq_len, 12) - batch=1, variable seq_len, 12 features
    """
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.history = []
    
    def add(self, state_vector):
        """Add a state to the history buffer."""
        # Ensure state is 1D tensor of shape (12,)
        if state_vector.dim() > 1:
            state_vector = state_vector.squeeze()
        self.history.append(state_vector.clone())
        if len(self.history) > self.max_length:
            self.history.pop(0)
    
    def get_sequence(self):
        """
        Get the current state sequence as a tensor.
        
        Returns:
            Tensor of shape (1, seq_len, 12) or None if buffer is empty
        """
        if not self.history:
            return None
        # Stack along dim=0 to get (seq_len, features), then add batch dimension
        stacked = torch.stack(self.history, dim=0)  # (seq_len, 12)
        return stacked.unsqueeze(0)  # (1, seq_len, 12)
    
    def clear(self):
        """Clear the history buffer (call at game reset)."""
        self.history = []
    
    def __len__(self):
        return len(self.history)


def create_agent(num_actions=9):
    """
    Factory function to create and initialize the LSTM agent.
    
    Returns:
        tuple: (model, optimizer, criterion, state_buffer)
    """
    model = LSTMActionAgent(
        input_size=12,
        hidden_size=64,
        num_layers=2,
        num_actions=num_actions,
        dropout=0.2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    state_buffer = StateHistoryBuffer(max_length=10)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"LSTM Agent initialized: {param_count:,} parameters")
    
    return model, optimizer, criterion, state_buffer
