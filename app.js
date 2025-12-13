// ============================================
// DARK GOTHIC MAZE - MAIN APPLICATION
// ============================================

const API_URL = "http://127.0.0.1:5000";

// ============================================
// GAME STATE
// ============================================
const gameState = {
    currentRoom: 'hall',
    visitedRooms: ['hall'],
    inventory: [],
    puzzlesSolved: [],
    unlockedDoors: [],
    takenItems: [],  // Track items that have been picked up (never reappear)
    moveCount: 0,
    startTime: Date.now(),
    gameOver: false,
    lastIntent: null,
    lastAgentPrediction: null,  // Store agent's predicted action type (0-8)
    
    // Game history for batch training
    gameHistory: [],  // Array of {state, intent, actionId} for replay training
    gamesCompleted: 0,  // Track how many games finished
    
    // Training metrics
    training: {
        samples: [],
        totalCount: 0,
        correctPredictions: 0,
        totalPredictions: 0,
        lossHistory: [],
        epochsCompleted: 0,  // Full game replays
        actionDistribution: [0, 0, 0, 0, 0, 0, 0, 0, 0]  // Track action type frequency
    }
};

// Make gameState globally accessible for maze renderer
window.gameState = gameState;

// ============================================
// DOM ELEMENTS
// ============================================
const elements = {
    // Room display
    roomName: document.getElementById('room-name'),
    roomIcon: document.getElementById('room-icon'),
    roomDescription: document.getElementById('room-description'),
    
    // Navigation
    navNorth: document.getElementById('nav-north'),
    navSouth: document.getElementById('nav-south'),
    navEast: document.getElementById('nav-east'),
    navWest: document.getElementById('nav-west'),
    currentRoomIndicator: document.getElementById('current-room-indicator'),
    
    // Actions
    actionsContainer: document.getElementById('actions-container'),
    
    // Inventory
    inventoryContainer: document.getElementById('inventory-container'),
    
    // Stats
    roomsVisited: document.getElementById('rooms-visited'),
    itemsCollected: document.getElementById('items-collected'),
    puzzlesSolved: document.getElementById('puzzles-solved'),
    
    // Game log
    gameLog: document.getElementById('game-log'),
    
    // AI Controls
    chatInput: document.getElementById('chat-input'),
    askChatbot: document.getElementById('ask-chatbot'),
    agentAct: document.getElementById('agent-act'),
    agentStatus: document.getElementById('agent-status'),
    trainingMode: document.getElementById('training-mode'),
    aiResponse: document.getElementById('ai-response'),
    
    // Training visualization
    trainingCount: document.getElementById('training-count'),
    avgLoss: document.getElementById('avg-loss'),
    accuracy: document.getElementById('accuracy'),
    gamesCompleted: document.getElementById('games-completed'),
    learningProgress: document.getElementById('learning-progress'),
    learningStatus: document.getElementById('learning-status'),
    trainingHistory: document.getElementById('training-history'),
    lossChart: document.getElementById('loss-chart'),
    modelHealth: document.getElementById('model-health'),
    batchTrainBtn: document.getElementById('batch-train-btn'),
    autoTrainBtn: document.getElementById('auto-train-btn'),
    autoTrainStatus: document.getElementById('auto-train-status'),
    historyCount: document.getElementById('history-count'),
    actionDist: document.getElementById('action-dist'),
    
    // Footer
    gameTime: document.getElementById('game-time'),
    moveCountDisplay: document.getElementById('move-count'),
    restartBtn: document.getElementById('restart-btn'),
    
    // Modals
    restartModal: document.getElementById('restart-modal'),
    confirmRestart: document.getElementById('confirm-restart'),
    cancelRestart: document.getElementById('cancel-restart'),
    victoryModal: document.getElementById('victory-modal'),
    playAgain: document.getElementById('play-again'),
    
    // Maze
    mazeSvg: document.getElementById('maze-svg')
};

// ============================================
// MAZE RENDERER INITIALIZATION
// ============================================
let mazeRenderer;
document.addEventListener('DOMContentLoaded', () => {
    mazeRenderer = new MazeRenderer(elements.mazeSvg, GAME_CONFIG);
    render();
    startGameTimer();
    initLossChart();
});

// ============================================
// GAME TIMER
// ============================================
let gameTimer;
function startGameTimer() {
    gameTimer = setInterval(() => {
        if (!gameState.gameOver) {
            const elapsed = Math.floor((Date.now() - gameState.startTime) / 1000);
            const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const secs = (elapsed % 60).toString().padStart(2, '0');
            elements.gameTime.textContent = `Time: ${mins}:${secs}`;
        }
    }, 1000);
}

// ============================================
// LOSS CHART (Simple Canvas Chart)
// ============================================
let lossChartCtx;
function initLossChart() {
    lossChartCtx = elements.lossChart.getContext('2d');
    drawLossChart();
}

function drawLossChart() {
    const ctx = lossChartCtx;
    const canvas = elements.lossChart;
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = '#2a2a3a';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const y = (height / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    
    // Draw loss history
    const losses = gameState.training.lossHistory;
    if (losses.length > 1) {
        const maxLoss = Math.max(...losses, 2);
        const step = width / Math.max(losses.length - 1, 1);
        
        ctx.strokeStyle = '#8b0000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        losses.forEach((loss, i) => {
            const x = i * step;
            const y = height - (loss / maxLoss) * height * 0.9;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        
        ctx.stroke();
        
        // Add glow effect
        ctx.strokeStyle = 'rgba(139, 0, 0, 0.5)';
        ctx.lineWidth = 4;
        ctx.stroke();
    }
    
    // Draw "No data" text if empty
    if (losses.length === 0) {
        ctx.fillStyle = '#5a5a6a';
        ctx.font = '12px Crimson Text';
        ctx.textAlign = 'center';
        ctx.fillText('Training data will appear here', width / 2, height / 2);
    }
}

// ============================================
// RENDER FUNCTION
// ============================================
function render() {
    const room = GAME_CONFIG.rooms[gameState.currentRoom];
    
    // Update room display
    elements.roomName.textContent = room.name;
    elements.roomIcon.textContent = getIcon(room.id);
    elements.roomDescription.textContent = room.description;
    elements.currentRoomIndicator.textContent = room.name.split(' ')[0];
    
    // Update navigation buttons
    updateNavigation(room);
    
    // Update actions
    updateActions(room);
    
    // Update inventory
    updateInventory();
    
    // Update stats
    elements.roomsVisited.textContent = gameState.visitedRooms.length;
    elements.itemsCollected.textContent = gameState.inventory.length;
    elements.puzzlesSolved.textContent = gameState.puzzlesSolved.length;
    elements.moveCountDisplay.textContent = `Moves: ${gameState.moveCount}`;
    
    // Update maze
    if (mazeRenderer) {
        mazeRenderer.update(gameState.currentRoom, gameState.visitedRooms, gameState.unlockedDoors);
    }
    
    // Update training visualization
    updateTrainingDisplay();
}

function getIcon(roomId) {
    const icons = {
        hall: '🍺',      // Tavern
        library: '🧙',    // Wizard's Study
        study: '💰',      // Treasure Chamber
        cellar: '⚒️',     // Dwarven Forge
        gallery: '⚔️',    // Hall of Heroes
        chapel: '☀️',     // Temple of Pelor
        attic: '🐉',      // Dragon's Hoard
        crypt: '💀',      // Tomb of the Lich
        dungeon: '👺',    // Goblin Prison
        vault: '🌀'       // Portal Chamber
    };
    return icons[roomId] || '❓';
}

// ============================================
// NAVIGATION
// ============================================
function updateNavigation(room) {
    const directions = ['north', 'south', 'east', 'west'];
    
    directions.forEach(dir => {
        const btn = elements[`nav${dir.charAt(0).toUpperCase() + dir.slice(1)}`];
        const connection = room.connections[dir];
        
        if (connection) {
            btn.disabled = false;
            
            // Check if locked
            if (typeof connection === 'object' && connection.locked) {
                const hasKey = checkRequirements(connection.requires);
                if (!hasKey && !gameState.unlockedDoors.includes(`${room.id}-${connection.room}`)) {
                    btn.classList.add('locked');
                    btn.title = `Locked - Requires: ${formatRequirements(connection.requires)}`;
                } else {
                    btn.classList.remove('locked');
                }
            } else {
                btn.classList.remove('locked');
            }
        } else {
            btn.disabled = true;
            btn.classList.remove('locked');
        }
    });
}

function checkRequirements(requires) {
    if (Array.isArray(requires)) {
        return requires.every(item => gameState.inventory.includes(item));
    }
    return gameState.inventory.includes(requires);
}

function formatRequirements(requires) {
    if (Array.isArray(requires)) {
        return requires.map(r => GAME_CONFIG.items[r]?.name || r).join(', ');
    }
    return GAME_CONFIG.items[requires]?.name || requires;
}

function navigate(direction) {
    if (gameState.gameOver) return;
    
    const room = GAME_CONFIG.rooms[gameState.currentRoom];
    const connection = room.connections[direction];
    
    if (!connection) {
        addLog("There's no path in that direction.", 'danger');
        return;
    }

    let targetRoom = connection;
    
    // Handle locked doors
    if (typeof connection === 'object') {
        if (connection.locked && !gameState.unlockedDoors.includes(`${room.id}-${connection.room}`)) {
            if (checkRequirements(connection.requires)) {
                // Unlock the door
                gameState.unlockedDoors.push(`${room.id}-${connection.room}`);
                addLog(`You used ${formatRequirements(connection.requires)} to unlock the passage!`, 'success');
                
                // Consume the key(s) used - remove from inventory
                const requiredItems = Array.isArray(connection.requires) ? connection.requires : [connection.requires];
                requiredItems.forEach(item => {
                    // Only consume key items (items with 'key' in the name or id)
                    if (item.toLowerCase().includes('key')) {
                        const index = gameState.inventory.indexOf(item);
                        if (index > -1) {
                            gameState.inventory.splice(index, 1);
                            const itemConfig = GAME_CONFIG.items[item];
                            addLog(`${itemConfig?.icon || '🔑'} ${itemConfig?.name || item} was consumed.`, 'warning');
                        }
                    }
                });
            } else {
                addLog(`The passage is locked. You need: ${formatRequirements(connection.requires)}`, 'danger');
                return;
            }
        }
        targetRoom = connection.room;
    }
    
    // Move to new room
    const oldRoom = gameState.currentRoom;
    gameState.currentRoom = targetRoom;
    gameState.moveCount++;
    
    // Mark as visited
    if (!gameState.visitedRooms.includes(targetRoom)) {
        gameState.visitedRooms.push(targetRoom);
        addLog(`You enter the ${GAME_CONFIG.rooms[targetRoom].name}...`, 'important');
    } else {
        addLog(`You return to the ${GAME_CONFIG.rooms[targetRoom].name}.`);
    }
    
    // Animate movement on map
    if (mazeRenderer) {
        mazeRenderer.highlightPath(oldRoom, targetRoom);
    }
    
    // Check for victory
    if (targetRoom === 'vault' && gameState.puzzlesSolved.includes('final_ritual')) {
        handleVictory();
    }
    
    render();
}

// Navigation button handlers
elements.navNorth.addEventListener('click', () => navigate('north'));
elements.navSouth.addEventListener('click', () => navigate('south'));
elements.navEast.addEventListener('click', () => navigate('east'));
elements.navWest.addEventListener('click', () => navigate('west'));

// Map room click handler
window.handleMapRoomClick = function(roomId) {
    if (gameState.visitedRooms.includes(roomId) && roomId !== gameState.currentRoom) {
        addLog(`You can't teleport! Navigate using the compass.`, 'danger');
    }
};

// ============================================
// ACTIONS
// ============================================
function updateActions(room) {
    elements.actionsContainer.innerHTML = '';
    
    const availableActions = getAvailableActions(room);
    
    availableActions.forEach(actionId => {
        const action = GAME_CONFIG.actions[actionId];
        if (!action) return;
        
        const btn = document.createElement('button');
        btn.className = 'action-btn';
        btn.innerHTML = `<span class="action-icon">${action.icon}</span> ${action.name}`;
        btn.dataset.action = actionId;
        btn.addEventListener('click', () => handleAction(actionId));
        
        elements.actionsContainer.appendChild(btn);
    });
}

function getAvailableActions(room) {
    const actions = ['look'];
    
    // Add room-specific actions
    if (room.actions) {
        room.actions.forEach(actionId => {
            // Check if item is already taken (use takenItems for permanent tracking)
            if (actionId.startsWith('take_')) {
                const itemId = GAME_CONFIG.actions[actionId]?.item;
                // Skip if item was already taken (ever)
                if (itemId && gameState.takenItems.includes(itemId)) return;
                // Skip if item is not in this room's item list
                if (itemId && !room.items?.includes(itemId)) return;
            }
            actions.push(actionId);
        });
    }
    
    return [...new Set(actions)];
}

async function handleAction(actionId) {
    if (gameState.gameOver) return;
    
    const action = GAME_CONFIG.actions[actionId];
    if (!action) return;
    
    // Training mode
    if (elements.trainingMode.checked) {
        if (!gameState.lastIntent) {
            addLog("Consult the Dungeon Master first to set an intent!", 'danger');
            return;
        }
        await trainAgent(actionId);
    }
    
    // Execute action
    executeAction(actionId);
}

function executeAction(actionId) {
    const room = GAME_CONFIG.rooms[gameState.currentRoom];
    const action = GAME_CONFIG.actions[actionId];
    
    gameState.moveCount++;
    
    switch (actionId) {
        case 'look':
            addLog(room.description);
            if (room.items && room.items.length > 0) {
                // Filter out items that have ever been taken (not just in inventory)
                const visibleItems = room.items.filter(i => !gameState.takenItems.includes(i));
                if (visibleItems.length > 0) {
                    addLog(`You notice: ${visibleItems.map(i => GAME_CONFIG.items[i]?.name).join(', ')}`, 'important');
                }
            }
            break;
            
        case 'take_torch':
        case 'take_candle':
        case 'take_key':
        case 'take_golden_key':
        case 'take_skeleton_key':
        case 'take_dagger':
        case 'take_holy_water':
        case 'take_cross':
        case 'take_map':
        case 'take_scroll':
        case 'take_tome':
            const itemId = action.item;
            // Check if item was already taken (either in inventory or previously taken)
            if (itemId && !gameState.takenItems.includes(itemId)) {
                gameState.inventory.push(itemId);
                gameState.takenItems.push(itemId);  // Mark as permanently taken
                const item = GAME_CONFIG.items[itemId];
                addLog(`You picked up: ${item.icon} ${item.name}`, 'success');
                
                // Special handling for map
                if (itemId === 'old_map') {
                    addLog("The map reveals the layout of the dungeon!", 'important');
                    // Reveal all rooms on map
                    Object.keys(GAME_CONFIG.rooms).forEach(r => {
                        if (!gameState.visitedRooms.includes(r)) {
                            gameState.visitedRooms.push(r);
                        }
                    });
                }
            } else if (itemId && gameState.takenItems.includes(itemId)) {
                addLog("You've already taken this item.", 'danger');
            }
            break;
            
        case 'read_books':
            addLog("The ancient tomes speak of dark rituals and hidden passages...");
            addLog("'Light the candles from shortest to tallest to reveal secrets.'", 'important');
            break;
            
        case 'read_inscriptions':
            addLog("The weathered inscriptions read: LEFT, RIGHT, MIDDLE");
            addLog("Perhaps this is a sequence of some kind...", 'important');
            break;
            
        case 'read_tome':
            if (gameState.inventory.includes('ancient_tome')) {
                addLog("The Tome of Escape reveals the final ritual!", 'important');
                addLog("'Pour holy water upon the symbols, recite the scroll, then speak the words of power.'");
            } else {
                addLog("You need to take the tome first.");
            }
            break;
            
        case 'examine_portrait':
            addLog("The portrait shows a nobleman holding up fingers...");
            addLog("First hand: 3 fingers. Second: 7. Third: 4. Fourth: 1.", 'important');
            break;
            
        case 'open_safe':
            if (gameState.puzzlesSolved.includes('safe_combination')) {
                addLog("The safe is already open.");
            } else {
                // Simple puzzle - in full version would have input
                gameState.puzzlesSolved.push('safe_combination');
                if (!gameState.inventory.includes('golden_key')) {
                    gameState.inventory.push('golden_key');
                    addLog("You enter 3-7-4-1... Click! The safe opens!", 'success');
                    addLog("Inside you find a 🔑 Golden Key!", 'important');
                }
            }
            break;
            
        case 'light_candles':
            if (!gameState.puzzlesSolved.includes('candle_sequence')) {
                gameState.puzzlesSolved.push('candle_sequence');
                addLog("You light the candles from shortest to tallest...", 'success');
                addLog("A hidden compartment opens in the wall!", 'important');
                if (!gameState.inventory.includes('rusty_key')) {
                    gameState.inventory.push('rusty_key');
                    addLog("You find a 🗝️ Rusty Key inside!");
                }
            } else {
                addLog("The candles are already lit.");
            }
            break;
            
        case 'examine_mirror':
            addLog("You stare into the ornate mirror...");
            if (gameState.inventory.includes('dagger')) {
                if (!gameState.puzzlesSolved.includes('mirror_puzzle')) {
                    gameState.puzzlesSolved.push('mirror_puzzle');
                    addLog("You hold the dagger at midnight position. The mirror shimmers!", 'success');
                    addLog("A secret passage to the Chapel is revealed!", 'important');
                }
            } else {
                addLog("Your reflection seems to be waiting for something...", 'important');
            }
            break;
            
        case 'pull_lever':
            if (!gameState.puzzlesSolved.includes('lever_mechanism')) {
                // Check if all three keys are present
                const keysNeeded = ['rusty_key', 'golden_key', 'skeleton_key'];
                const hasAllKeys = keysNeeded.every(
                    k => gameState.inventory.includes(k)
                );
                if (hasAllKeys) {
                    gameState.puzzlesSolved.push('lever_mechanism');
                    gameState.unlockedDoors.push('dungeon-vault');
                    addLog("You insert all three keys and pull the lever...", 'success');
                    addLog("CLUNK! The passage to the Vault opens!", 'important');
                    
                    // Consume all three keys
                    keysNeeded.forEach(keyId => {
                        const index = gameState.inventory.indexOf(keyId);
                        if (index > -1) {
                            gameState.inventory.splice(index, 1);
                        }
                    });
                    addLog("🗝️ The three keys crumble to dust, their magic spent.", 'warning');
                } else {
                    addLog("The mechanism requires three keys to operate.", 'danger');
                }
            } else {
                addLog("The mechanism has already been activated.");
            }
            break;
            
        case 'perform_ritual':
            const ritualItems = ['holy_water', 'ancient_scroll', 'ancient_tome'];
            const hasRitualItems = ritualItems.every(
                i => gameState.inventory.includes(i)
            );
            if (hasRitualItems) {
                gameState.puzzlesSolved.push('final_ritual');
                addLog("You pour the holy water upon the symbols...", 'important');
                addLog("You recite the incantation from the scroll...");
                addLog("The tome glows with otherworldly light!", 'success');
                addLog("THE EXIT DOOR OPENS! ESCAPE NOW!", 'success');
                
                // Consume ritual items
                ritualItems.forEach(itemId => {
                    const index = gameState.inventory.indexOf(itemId);
                    if (index > -1) {
                        gameState.inventory.splice(index, 1);
                    }
                });
                addLog("✨ The ritual components dissolve into magical energy.", 'warning');
            } else {
                addLog("You need holy water, the ancient scroll, and the tome to perform the ritual.", 'danger');
            }
            break;
            
        case 'open_exit':
            if (gameState.puzzlesSolved.includes('final_ritual')) {
                handleVictory();
            } else {
                addLog("The exit is sealed by dark magic. You must complete the ritual first.", 'danger');
            }
            break;
            
        default:
            addLog(`You ${action.name.toLowerCase()}...`);
    }

    render();
}

// ============================================
// INVENTORY
// ============================================
function updateInventory() {
    const slots = elements.inventoryContainer.querySelectorAll('.inventory-slot');
    
    slots.forEach((slot, index) => {
        slot.innerHTML = '';
        slot.classList.remove('selected');
        slot.classList.add('empty');
        slot.title = '';
        
        if (index < gameState.inventory.length) {
            const itemId = gameState.inventory[index];
            const item = GAME_CONFIG.items[itemId];
            if (item) {
                slot.textContent = item.icon;
                slot.classList.remove('empty');
                slot.title = `${item.name}: ${item.description}`;
            }
        }
    });
}

// ============================================
// GAME LOG
// ============================================
function addLog(message, type = '') {
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('p');
    entry.className = `log-entry ${type}`;
    entry.textContent = message;
    entry.dataset.timestamp = timestamp;
    
    elements.gameLog.insertBefore(entry, elements.gameLog.firstChild);
    
    // Keep only last 200 entries for full session tracking
    while (elements.gameLog.children.length > 200) {
        elements.gameLog.removeChild(elements.gameLog.lastChild);
    }
}

// Export full log for analysis
function exportLog() {
    const entries = Array.from(elements.gameLog.querySelectorAll('.log-entry'));
    const logText = entries.reverse().map(entry => {
        const time = entry.dataset.timestamp || '';
        return `[${time}] ${entry.textContent}`;
    }).join('\n');
    
    // Add game state summary at the top
    const summary = `
=== GAME SESSION LOG ===
Time: ${new Date().toLocaleString()}
Room: ${gameState.currentRoom}
Rooms Visited: ${gameState.visitedRooms.join(', ')}
Inventory: ${gameState.inventory.join(', ') || 'Empty'}
Puzzles Solved: ${gameState.puzzlesSolved.join(', ') || 'None'}
Moves: ${gameState.moveCount}
Training Samples: ${gameState.training.totalCount}
========================

${logText}
`;
    
    // Copy to clipboard
    navigator.clipboard.writeText(summary).then(() => {
        addLog('[LOG EXPORTED TO CLIPBOARD]', 'success');
    }).catch(err => {
        console.log('Export log:', summary);
        addLog('[LOG PRINTED TO CONSOLE - Check F12]', 'important');
    });
    
    return summary;
}

// Make exportLog available globally for console access
window.exportLog = exportLog;

// ============================================
// AI CHATBOT
// ============================================
elements.askChatbot.addEventListener('click', async () => {
    const query = elements.chatInput.value.trim() || "What should I do next?";
    
    // Log the user's question
    addLog(`YOU: "${query}"`);
    addLog(`[Room: ${gameState.currentRoom}, Items: ${gameState.inventory.length}, Puzzles: ${gameState.puzzlesSolved.length}]`, '');
    
    elements.aiResponse.querySelector('.response-text').textContent = "The spirits are speaking...";
    elements.aiResponse.querySelector('.response-text').classList.add('active');
    
    try {
        const response = await fetch(`${API_URL}/chatbot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                step: gameState.visitedRooms.length,
                inventory: gameState.inventory,
                currentRoom: gameState.currentRoom,
                puzzlesSolved: gameState.puzzlesSolved,
                doorLocked: !gameState.puzzlesSolved.includes('final_ritual')
            })
        });
        
        const data = await response.json();
        elements.aiResponse.querySelector('.response-text').textContent = data.message;
        gameState.lastIntent = data.intent;
        elements.agentStatus.textContent = `Intent: ${data.intent}`;
        
        addLog(`DM: "${data.message}"`, 'important');
        addLog(`[Intent: ${data.intent}]`, '');
        elements.chatInput.value = '';
        
    } catch (error) {
        elements.aiResponse.querySelector('.response-text').textContent = "The spirits are silent...";
        elements.agentStatus.textContent = "Connection failed";
        addLog(`[ERROR: Connection failed]`, 'danger');
        console.error(error);
    }
});

// ============================================
// AI AGENT
// ============================================
elements.agentAct.addEventListener('click', async () => {
    if (!gameState.lastIntent) {
        addLog("Consult the Dungeon Master first!", 'danger');
        return;
    }
    
    addLog(`[AGENT REQUEST: Intent="${gameState.lastIntent}"]`, '');
    elements.agentStatus.textContent = "Agent thinking...";
    elements.agentStatus.classList.add('thinking');
    
    try {
        // Get available actions
        const room = GAME_CONFIG.rooms[gameState.currentRoom];
        const availableActions = getAvailableActions(room);
        const actionMask = GAME_CONFIG.intents.map(() => 1); // Simplified mask
        
        const response = await fetch(`${API_URL}/agent/act`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state: {
                    step: gameState.visitedRooms.length,
                    inventory: gameState.inventory,
                    doorLocked: !gameState.puzzlesSolved.includes('final_ritual')
                },
                intent: gameState.lastIntent,
                mask: actionMask
            })
        });
        
        const data = await response.json();
        elements.agentStatus.classList.remove('thinking');
        
        // Map agent action to game action
        const agentAction = mapAgentAction(data.action_id, availableActions);
        elements.agentStatus.textContent = `Agent chose: ${agentAction}`;
        
        // Store agent's prediction index (0-8) for accuracy tracking
        // Backend returns action_index (number) alongside action_id (string)
        gameState.lastAgentPrediction = data.action_index;
        
        // Log agent decision
        addLog(`[AGENT RESPONSE: action="${data.action_id}" → mapped="${agentAction}" seq_len=${data.sequence_length || '?'}]`, '');
        
        // Track prediction
        gameState.training.totalPredictions++;
        
        // Highlight the action button
        const actionBtn = document.querySelector(`[data-action="${agentAction}"]`);
        if (actionBtn) {
            actionBtn.classList.add('highlight');
            setTimeout(() => {
                actionBtn.classList.remove('highlight');
                executeAction(agentAction);
            }, 1000);
        } else {
            // If action not found, try navigation or default
            addLog(`Agent suggests: ${data.action_id}`);
        }
        
    } catch (error) {
        elements.agentStatus.classList.remove('thinking');
        elements.agentStatus.textContent = "Agent offline";
        console.error(error);
    }
});

function mapAgentAction(agentActionId, availableActions) {
    // Direct match - return immediately
    if (availableActions.includes(agentActionId)) {
        return agentActionId;
    }
    
    // Category-to-specific action mapping
    // Maps general action categories to patterns that match specific actions
    const categoryPatterns = {
        'look': ['look'],
        'read_books': ['read_', 'examine_', 'study_'],
        'take_key': ['take_key', 'take_skeleton', 'take_golden', 'take_dragon'],
        'use_keys': ['use_key', 'pull_lever', 'unlock_'],
        'open_exit': ['open_exit', 'exit', 'escape'],
        'navigate': ['go_', 'enter_', 'move_'],
        'take_item': ['take_'],  // Matches take_torch, take_dagger, etc.
        'solve_puzzle': ['solve_', 'light_', 'perform_', 'open_safe', 'examine_mirror'],
        'interact': ['pray', 'examine_', 'search_', 'open_']
    };
    
    const patterns = categoryPatterns[agentActionId];
    if (patterns) {
        for (const pattern of patterns) {
            const match = availableActions.find(action => action.startsWith(pattern) || action === pattern);
            if (match) {
                return match;
            }
        }
    }
    
    // Fallback: try to find any action containing the category keyword
    const keyword = agentActionId.replace('_', '');
    const partialMatch = availableActions.find(action => 
        action.includes(keyword) || keyword.includes(action.split('_')[0])
    );
    if (partialMatch) {
        return partialMatch;
    }
    
    // Last resort: return first available action
    return availableActions[0] || 'look';
}

// ============================================
// TRAINING
// ============================================
async function trainAgent(actionId) {
    const action = GAME_CONFIG.actions[actionId];
    if (!action) return;
    
    elements.agentStatus.textContent = "Training...";
    
    try {
        // Map action to LSTM agent's ACTION_MAP indices
        // ACTION_MAP = ["look", "read_books", "take_key", "use_keys", "open_exit", "navigate", "take_item", "solve_puzzle", "interact"]
        const actionTypeMap = {
            // Direct matches
            'look': 0,
            'read_books': 1,
            'read_inscriptions': 1,
            'read_tome': 1,
            'take_key': 2,
            'take_golden_key': 2,
            'take_skeleton_key': 2,
            'use_keys': 3,
            'open_exit': 4,
            // Navigation actions
            'go_north': 5,
            'go_south': 5,
            'go_east': 5,
            'go_west': 5,
            // Take item actions
            'take_torch': 6,
            'take_candle': 6,
            'take_dagger': 6,
            'take_holy_water': 6,
            'take_cross': 6,
            'take_map': 6,
            'take_scroll': 6,
            'take_tome': 6,
            // Puzzle actions
            'light_candles': 7,
            'open_safe': 7,
            'examine_mirror': 7,
            'pull_lever': 7,
            'perform_ritual': 7,
            // Interact actions
            'examine_portrait': 8,
            'examine_altar': 8,
            'pray': 8
        };
        
        const actionIndex = actionTypeMap[actionId] ?? 0;
        
        // Record move in game history for batch training
        const currentState = {
            step: gameState.visitedRooms.length,
            inventory: [...gameState.inventory],
            room: gameState.currentRoom,
            puzzlesSolved: [...gameState.puzzlesSolved],
            doorLocked: !gameState.puzzlesSolved.includes('final_ritual')
        };
        gameState.gameHistory.push({
            state: currentState,
            intent: gameState.lastIntent || 'inspect',
            actionIndex: actionIndex,
            actionId: actionId
        });
        
        // Update action distribution (with safety check)
        if (gameState.training.actionDistribution) {
            gameState.training.actionDistribution[actionIndex]++;
        }
        try {
            updateActionDistribution();
            updateHistoryCount();
        } catch (e) {
            console.warn('UI update warning:', e);
        }
        
        // Check if agent's prediction matches user's action (for accuracy tracking)
        if (gameState.lastAgentPrediction !== null) {
            if (gameState.lastAgentPrediction === actionIndex) {
                gameState.training.correctPredictions++;
                addLog(`[ACCURACY: Agent predicted correctly! (${actionIndex})]`, 'success');
            } else {
                addLog(`[ACCURACY: Agent predicted ${gameState.lastAgentPrediction}, user did ${actionIndex}]`, '');
            }
            gameState.lastAgentPrediction = null;  // Clear after comparison
        }
        
        const response = await fetch(`${API_URL}/agent/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state: {
                    step: gameState.visitedRooms.length,
                    inventory: gameState.inventory,
                    doorLocked: !gameState.puzzlesSolved.includes('final_ritual')
                },
                intent: gameState.lastIntent,
                correct_action_id: actionIndex
            })
        });
        
        const data = await response.json();
        
        // Handle different response statuses
        if (data.status === 'error') {
            elements.agentStatus.textContent = `Error: ${data.error}`;
            console.error('Training error:', data.error);
            return;
        }
        
        if (data.status === 'skipped') {
            elements.agentStatus.textContent = `Building history... (${data.sequence_length || 1} states)`;
            return;
        }
        
        // Update training metrics for successful training
        gameState.training.totalCount++;
        gameState.training.lossHistory.push(data.loss);
        gameState.training.samples.push({
            intent: gameState.lastIntent,
            action: actionId,
            loss: data.loss,
            sequenceLen: data.sequence_length || 1,
            timestamp: new Date().toLocaleTimeString()
        });
        
        // Keep only last 50 samples
        if (gameState.training.samples.length > 50) {
            gameState.training.samples.shift();
            gameState.training.lossHistory.shift();
        }
        
        elements.agentStatus.textContent = `LSTM Trained! Loss: ${data.loss.toFixed(4)} (${data.sequence_length || 1} states)`;
        addLog(`Agent learned: ${gameState.lastIntent} → ${action.name}`, 'success');
        
        updateTrainingDisplay();
        drawLossChart();
        
    } catch (error) {
        elements.agentStatus.textContent = `Training error: ${error.message}`;
        console.error('Training fetch error:', error);
        addLog(`[TRAIN ERROR] ${error.message}`, 'danger');
    }
}

function updateTrainingDisplay() {
    const training = gameState.training;
    
    // Update counts
    elements.trainingCount.textContent = training.totalCount;
    if (elements.gamesCompleted) {
        elements.gamesCompleted.textContent = gameState.gamesCompleted;
    }
    
    // Calculate average loss
    if (training.lossHistory.length > 0) {
        const avgLoss = training.lossHistory.reduce((a, b) => a + b, 0) / training.lossHistory.length;
        elements.avgLoss.textContent = avgLoss.toFixed(3);
    }
    
    // Calculate accuracy based on actual prediction correctness
    let accuracyPercent = 0;
    if (training.totalPredictions > 0) {
        accuracyPercent = (training.correctPredictions / training.totalPredictions) * 100;
        elements.accuracy.textContent = `${accuracyPercent.toFixed(0)}%`;
    } else {
        elements.accuracy.textContent = '-';
    }
    
    // Update progress bar
    const progress = Math.min(100, (training.totalCount / GAME_CONFIG.training.learningThreshold) * 100);
    elements.learningProgress.style.width = `${progress}%`;
    
    // Update status and model health
    updateModelHealth(training.totalCount, accuracyPercent, gameState.gamesCompleted);
    
    if (training.totalCount === 0) {
        elements.learningStatus.textContent = 'Untrained';
        elements.learningStatus.className = 'progress-status';
    } else if (training.totalCount < GAME_CONFIG.training.learningThreshold) {
        elements.learningStatus.textContent = 'Learning...';
        elements.learningStatus.className = 'progress-status learning';
    } else {
        elements.learningStatus.textContent = 'Well Trained!';
        elements.learningStatus.className = 'progress-status trained';
    }
    
    // Update history (if element exists)
    if (elements.trainingHistory && training.samples.length > 0) {
        elements.trainingHistory.innerHTML = training.samples.slice(-5).reverse().map(s => `
            <div class="history-item">
                <span>${s.intent} → ${s.action}</span>
                <span class="history-loss">${s.loss.toFixed(4)}</span>
            </div>
        `).join('');
    }
}

// Update model health indicator
function updateModelHealth(samples, accuracy, games) {
    if (!elements.modelHealth) return;
    
    let icon, text, className;
    
    if (games === 0 && samples < 20) {
        icon = '🔴';
        text = `Untrained - Play ${3 - games} more games to train`;
        className = 'model-health untrained';
    } else if (games < 3 || accuracy < 30) {
        icon = '🟡';
        text = `Learning - ${games} games, ${accuracy.toFixed(0)}% accuracy`;
        className = 'model-health learning';
    } else if (accuracy < 60) {
        icon = '🟡';
        text = `Needs more training - ${accuracy.toFixed(0)}% accuracy`;
        className = 'model-health learning';
    } else {
        icon = '🟢';
        text = `Well trained! ${accuracy.toFixed(0)}% accuracy`;
        className = 'model-health trained';
    }
    
    elements.modelHealth.innerHTML = `
        <span class="health-icon">${icon}</span>
        <span class="health-text">${text}</span>
    `;
    elements.modelHealth.className = className;
}

// Update action distribution bars
function updateActionDistribution() {
    if (!elements.actionDist) return;
    
    const dist = gameState.training.actionDistribution;
    const maxCount = Math.max(...dist, 1);
    
    const bars = elements.actionDist.querySelectorAll('.bar');
    bars.forEach((bar, i) => {
        const height = (dist[i] / maxCount) * 25;
        bar.style.height = `${Math.max(2, height)}px`;
        bar.title = `${dist[i]} actions`;
    });
}

// Update history count display
function updateHistoryCount() {
    // Re-query the element since it might be recreated
    const historyCountEl = document.getElementById('history-count');
    if (historyCountEl) {
        historyCountEl.textContent = gameState.gameHistory.length;
    }
}

// Auto-train toggle state
let autoTrainEnabled = false;

// Batch training function - trains on all recorded history
async function batchTrainOnHistory(epochs = 3) {
    if (gameState.gameHistory.length === 0) {
        addLog('[BATCH TRAIN] No history to train on!', 'danger');
        return;
    }
    
    addLog(`[BATCH TRAIN] Starting batch training on ${gameState.gameHistory.length} moves (${epochs} epochs)...`, 'important');
    
    // Store original content and show training state
    let originalBtnHTML = '';
    if (elements.batchTrainBtn) {
        elements.batchTrainBtn.disabled = true;
        originalBtnHTML = elements.batchTrainBtn.innerHTML;
        elements.batchTrainBtn.innerHTML = '<span class="btn-icon">⏳</span> Training...';
    }
    
    try {
        const response = await fetch(`${API_URL}/agent/batch_train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                history: gameState.gameHistory,
                epochs: epochs
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog(`[BATCH TRAIN] ✅ Completed! Final loss: ${data.final_loss.toFixed(4)}, trained on ${data.samples_trained} samples`, 'success');
            
            // Update training metrics
            gameState.training.epochsCompleted += epochs;
            gameState.training.lossHistory.push(data.final_loss);
            gameState.training.totalCount += data.samples_trained;
            
            updateTrainingDisplay();
        } else {
            addLog(`[BATCH TRAIN] ❌ Failed: ${data.error}`, 'danger');
        }
    } catch (error) {
        addLog(`[BATCH TRAIN] ❌ Error: ${error.message}`, 'danger');
        console.error('Batch training error:', error);
    }
    
    if (elements.batchTrainBtn) {
        elements.batchTrainBtn.disabled = false;
        // Restore button with updated count
        elements.batchTrainBtn.innerHTML = `<span class="btn-icon">🧠</span> Train on History (<span id="history-count">${gameState.gameHistory.length}</span> moves)`;
    }
}

// ============================================
// VICTORY
// ============================================
function handleVictory() {
    gameState.gameOver = true;
    gameState.gamesCompleted++;
    clearInterval(gameTimer);
    
    // Calculate final stats
    const elapsed = Math.floor((Date.now() - gameState.startTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const secs = (elapsed % 60).toString().padStart(2, '0');
    
    document.getElementById('final-time').textContent = `${mins}:${secs}`;
    document.getElementById('final-moves').textContent = gameState.moveCount;
    document.getElementById('final-items').textContent = gameState.inventory.length;
    
    elements.victoryModal.classList.remove('hidden');
    addLog("🎉 VICTORY! Quest complete! You are a legendary hero! 🎉", 'success');
    addLog(`[GAME ${gameState.gamesCompleted}] Completed with ${gameState.gameHistory.length} moves recorded`, 'important');
    
    // Auto-train if enabled
    if (autoTrainEnabled && gameState.gameHistory.length > 0) {
        addLog('[AUTO-TRAIN] Starting automatic batch training...', 'important');
        setTimeout(() => batchTrainOnHistory(3), 1000);
    } else if (gameState.gameHistory.length > 0) {
        addLog('[TIP] Click "Train on History" to train the model on this game!', '');
    }
    
    updateTrainingDisplay();
}

// ============================================
// RESTART
// ============================================
elements.restartBtn.addEventListener('click', () => {
    elements.restartModal.classList.remove('hidden');
});

elements.cancelRestart.addEventListener('click', () => {
    elements.restartModal.classList.add('hidden');
});

elements.confirmRestart.addEventListener('click', () => {
    restartGame();
});

elements.playAgain.addEventListener('click', () => {
    elements.victoryModal.classList.add('hidden');
    restartGame();
});

// Training button event listeners
if (elements.batchTrainBtn) {
    elements.batchTrainBtn.addEventListener('click', () => {
        batchTrainOnHistory(3);  // Train for 3 epochs
    });
}

if (elements.autoTrainBtn) {
    elements.autoTrainBtn.addEventListener('click', () => {
        autoTrainEnabled = !autoTrainEnabled;
        elements.autoTrainStatus.textContent = autoTrainEnabled ? 'ON' : 'OFF';
        elements.autoTrainBtn.classList.toggle('active', autoTrainEnabled);
        addLog(`[AUTO-TRAIN] ${autoTrainEnabled ? 'Enabled' : 'Disabled'} - Will train after each game`, autoTrainEnabled ? 'success' : '');
    });
}

function restartGame() {
    // Preserve training data and game history across restarts
    const trainingData = { ...gameState.training };
    const gameHistory = [...gameState.gameHistory];  // Keep history for batch training
    const gamesCompleted = gameState.gamesCompleted;
    
    Object.assign(gameState, {
        currentRoom: 'hall',
        visitedRooms: ['hall'],
        inventory: [],
        puzzlesSolved: [],
        unlockedDoors: [],
        takenItems: [],  // Reset taken items so they reappear on new game
        moveCount: 0,
        startTime: Date.now(),
        gameOver: false,
        lastIntent: null,
        lastAgentPrediction: null,
        gameHistory: gameHistory,  // Preserve for batch training
        gamesCompleted: gamesCompleted,
        training: trainingData
    });
    
    // Reset LSTM agent's state buffer (but not the model weights!)
    fetch(`${API_URL}/agent/reset`, { method: 'POST' })
        .catch(err => console.log('Agent reset:', err));
    
    // Update displays
    updateHistoryCount();
    
    // Clear log
    elements.gameLog.innerHTML = '<p class="log-entry">Your adventure begins anew! Roll for initiative...</p>';
    
    // Hide modals
    elements.restartModal.classList.add('hidden');
    elements.victoryModal.classList.add('hidden');
    
    // Reset AI display
    elements.aiResponse.querySelector('.response-text').textContent = "The Dungeon Master awaits your questions...";
    elements.agentStatus.textContent = "LSTM Agent ready";
    
    // Restart timer
    if (gameTimer) clearInterval(gameTimer);
    startGameTimer();
    
render();
    addLog("A new quest begins, brave adventurer!", 'important');
}

// ============================================
// KEYBOARD SHORTCUTS
// ============================================
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    
    switch (e.key) {
        case 'ArrowUp':
        case 'w':
            navigate('north');
            break;
        case 'ArrowDown':
        case 's':
            navigate('south');
            break;
        case 'ArrowRight':
        case 'd':
            navigate('east');
            break;
        case 'ArrowLeft':
        case 'a':
            navigate('west');
            break;
        case 'l':
            executeAction('look');
            break;
        case 'e':
            // Export log with 'E' key
            exportLog();
            break;
    }
});

// ============================================
// CHAT INPUT ENTER KEY
// ============================================
elements.chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        elements.askChatbot.click();
    }
});
