import asyncio
import json
import logging
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Import SAFER v3.0 components
from safer_v3.simulation.engine_sim import EngineSimulator
from safer_v3.decision.simplex import SimplexDecisionModule, SimplexConfig
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard.server")

app = FastAPI(title="SAFER v3.0 Dashboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="dashboard"), name="static")

# --- Simulation State ---

class SimulationManager:
    def __init__(self):
        self.simulator = EngineSimulator(total_cycles=300, seed=42)
        
        # Configure Simplex
        config = SimplexConfig(
            physics_threshold=0.1,
            divergence_threshold=30.0,
        )
        self.simplex = SimplexDecisionModule(config)
        
        # Configure Alert Manager
        self.alert_manager = AlertManager()
        self.alert_manager.add_rules(create_rul_alert_rules())
        
        self.trajectory = None
        self.current_cycle = 0
        self.is_running = False
        self.is_paused = False
        self.active_connections: List[WebSocket] = []

    def start_simulation(self):
        if not self.trajectory:
            logger.info("Generating new trajectory...")
            self.trajectory = self.simulator.generate_trajectory()
            self.current_cycle = 0
        self.is_running = True
        self.is_paused = False

    def stop_simulation(self):
        self.is_running = False

    def pause_simulation(self):
        self.is_paused = True
    
    def resume_simulation(self):
        self.is_paused = False

    def reset_simulation(self):
        self.is_running = False
        self.is_paused = False
        self.trajectory = None
        self.current_cycle = 0
        self.start_simulation()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass  # Handle disconnected clients gracefully

    async def run_loop(self):
        """Main simulation loop running in background."""
        while True:
            if self.is_running and not self.is_paused and self.trajectory:
                if self.current_cycle < len(self.trajectory['cycle']):
                    # 1. Get current cycle data
                    idx = self.current_cycle
                    sensor_data = self.trajectory['sensors'][idx] # Shape (14,)
                    true_rul = float(self.trajectory['rul'][idx])
                    
                    # 2. Simulate model outputs (adding some noise for realism demo)
                    # In a real system, these would come from the models.
                    # Here we simulate the models being slightly off.
                    baseline_rul = true_rul + np.random.normal(0, 5)
                    complex_rul = true_rul + np.random.normal(0, 2)
                    
                    # 3. Decision Logic (Simplex)
                    decision = self.simplex.decide(
                        complex_rul=complex_rul,
                        baseline_rul=baseline_rul,
                        rul_lower=complex_rul - 10,
                        rul_upper=complex_rul + 10,
                        physics_residual=0.05 + (0.001 * idx) # Increasing residual
                    )
                    
                    # 4. Alert Logic
                    alerts = self.alert_manager.process(rul_value=decision.rul)
                    
                    # 5. Prepare Payload
                    error = decision.rul - true_rul
                    accuracy_status = "Good" if abs(error) < 5 else "Deviating"
                    
                    payload = {
                        "cycle": int(self.trajectory['cycle'][idx]),
                        "sensors": sensor_data.tolist(),
                        "rul": {
                           "true": round(true_rul, 1),
                           "baseline": round(baseline_rul, 1),
                           "complex": round(complex_rul, 1),
                           "final": round(decision.rul, 1),
                           "error": round(error, 1),
                           "status": accuracy_status
                        },
                        "simplex_state": decision.state.name,
                        "alerts": [
                            {"level": a.level.name, "message": a.message} for a in alerts
                        ],
                        "status": "Running" if self.is_running and not self.is_paused else "Paused"
                    }
                    
                    # 6. Broadcast
                    await self.broadcast(payload)
                    
                    self.current_cycle += 1
                else:
                    logger.info("Simulation finished. Resetting...")
                    await asyncio.sleep(2)
                    self.reset_simulation()
            elif self.is_paused:
                 # Send status update even when paused so new clients know
                 await self.broadcast({"status": "Paused"})

            await asyncio.sleep(0.5)  # Update every 500ms

# Global Simulation Instance
sim_manager = SimulationManager()

@app.on_event("startup")
async def startup_event():
    sim_manager.start_simulation()
    asyncio.create_task(sim_manager.run_loop())

@app.get("/")
async def get_index():
    with open("dashboard/index.html", "r") as f:
        return HTMLResponse(content=f.read())

    async def run_loop(self):
        """Main simulation loop running in background."""
        while True:
            if self.is_running and not self.is_paused and self.trajectory:
                if self.current_cycle < len(self.trajectory['cycle']):
                    # 1. Get current cycle data
                    idx = self.current_cycle
                    sensor_data = self.trajectory['sensors'][idx] # Shape (14,)
                    true_rul = float(self.trajectory['rul'][idx])
                    
                    # 2. Simulate model outputs (adding some noise for realism demo)
                    # In a real system, these would come from the models.
                    # Here we simulate the models being slightly off.
                    baseline_rul = true_rul + np.random.normal(0, 5)
                    complex_rul = true_rul + np.random.normal(0, 2)
                    
                    # 3. Decision Logic (Simplex)
                    decision = self.simplex.decide(
                        complex_rul=complex_rul,
                        baseline_rul=baseline_rul,
                        rul_lower=complex_rul - 10,
                        rul_upper=complex_rul + 10,
                        physics_residual=0.05 + (0.001 * idx) # Increasing residual
                    )
                    
                    # 4. Alert Logic
                    alerts = self.alert_manager.process(rul_value=decision.rul)
                    
                    # 5. Prepare Payload
                    error = decision.rul - true_rul
                    accuracy_status = "Good" if abs(error) < 5 else "Deviating"
                    
                    payload = {
                        "cycle": int(self.trajectory['cycle'][idx]),
                        "sensors": sensor_data.tolist(),
                        "rul": {
                           "true": round(true_rul, 1),
                           "baseline": round(baseline_rul, 1),
                           "complex": round(complex_rul, 1),
                           "final": round(decision.rul, 1),
                           "error": round(error, 1),
                           "status": accuracy_status
                        },
                        "simplex_state": decision.state.name,
                        "alerts": [
                            {"level": a.level.name, "message": a.message} for a in alerts
                        ],
                        "status": "Running" if self.is_running and not self.is_paused else "Paused"
                    }
                    
                    # 6. Broadcast
                    await self.broadcast(payload)
                    
                    self.current_cycle += 1
                else:
                    logger.info("Simulation finished. Resetting...")
                    await asyncio.sleep(2)
                    self.reset_simulation()
            elif self.is_paused:
                 # Send status update even when paused so new clients know
                 await self.broadcast({"status": "Paused"})

            await asyncio.sleep(0.5)  # Update every 500ms

# Global Simulation Instance
sim_manager = SimulationManager()

@app.on_event("startup")
async def startup_event():
    sim_manager.start_simulation()
    asyncio.create_task(sim_manager.run_loop())

@app.get("/")
async def get_index():
    with open("dashboard/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await sim_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle commands from frontend
            if data == "reset":
                sim_manager.reset_simulation()
            elif data == "pause":
                sim_manager.pause_simulation()
            elif data == "resume":
                sim_manager.resume_simulation()
    except WebSocketDisconnect:
        sim_manager.disconnect(websocket)
