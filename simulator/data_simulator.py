"""
simulator/data_simulator.py
Synthetic manufacturing sensor data generator.
Streams realistic readings to the FastAPI /api/ingest endpoint.
Supports normal operation, gradual drift, and sudden fault injection modes.

Usage:
    python -m simulator.data_simulator
    python -m simulator.data_simulator --fault bearing_failure --machine M-002
"""

import asyncio
import argparse
import random
import json
from datetime import datetime
from typing import Optional

import httpx

from backend.config import settings

# ── Machine profiles ───────────────────────────────────────────────

MACHINES = ["M-001", "M-002", "M-003", "M-004"]

BASE_PROFILES = {
    "M-001": {
        "temp": 65.0,
        "pressure": 120.0,
        "vibration": 2.5,
        "defect_rate": 0.02,
        "speed": 45.0,
    },
    "M-002": {
        "temp": 70.0,
        "pressure": 130.0,
        "vibration": 3.0,
        "defect_rate": 0.025,
        "speed": 50.0,
    },
    "M-003": {
        "temp": 60.0,
        "pressure": 110.0,
        "vibration": 2.0,
        "defect_rate": 0.015,
        "speed": 40.0,
    },
    "M-004": {
        "temp": 75.0,
        "pressure": 140.0,
        "vibration": 3.5,
        "defect_rate": 0.03,
        "speed": 55.0,
    },
}

# ── Fault injection scenarios ──────────────────────────────────────

FAULT_SCENARIOS = {
    "bearing_failure": {
        "vibration_delta": 6.0,
        "temp_delta": 15.0,
        "pressure_delta": 0.0,
        "defect_delta": 0.04,
        "speed_delta": -10.0,
    },
    "coolant_loss": {
        "vibration_delta": 1.0,
        "temp_delta": 30.0,
        "pressure_delta": -20.0,
        "defect_delta": 0.06,
        "speed_delta": -5.0,
    },
    "pressure_spike": {
        "vibration_delta": 2.0,
        "temp_delta": 5.0,
        "pressure_delta": 50.0,
        "defect_delta": 0.03,
        "speed_delta": 0.0,
    },
    "tool_wear": {
        "vibration_delta": 0.5,
        "temp_delta": 3.0,
        "pressure_delta": 0.0,
        "defect_delta": 0.09,
        "speed_delta": -8.0,
    },
    "normal": {
        "vibration_delta": 0.0,
        "temp_delta": 0.0,
        "pressure_delta": 0.0,
        "defect_delta": 0.0,
        "speed_delta": 0.0,
    },
}


def _noise(sigma: float = 1.0) -> float:
    """Gaussian noise."""
    return random.gauss(0, sigma)


def simulate_sensor_reading(
    machine_id: str,
    fault: str = "normal",
    drift_step: int = 0,
) -> dict:
    """
    Generate one synthetic sensor reading.

    Args:
        machine_id: One of MACHINES.
        fault: Key into FAULT_SCENARIOS.
        drift_step: Gradually increases fault magnitude over time (simulates developing fault).
    """
    base = BASE_PROFILES.get(machine_id, BASE_PROFILES["M-001"])
    scenario = FAULT_SCENARIOS.get(fault, FAULT_SCENARIOS["normal"])

    # Drift factor: fault intensifies over 60 steps
    drift = min(drift_step / 60.0, 1.0)

    reading = {
        "machine_id": machine_id,
        "temperature": round(
            base["temp"] + scenario["temp_delta"] * drift + _noise(1.5), 2
        ),
        "pressure": round(
            max(0, base["pressure"] + scenario["pressure_delta"] * drift + _noise(3.0)),
            2,
        ),
        "vibration": round(
            max(
                0, base["vibration"] + scenario["vibration_delta"] * drift + _noise(0.3)
            ),
            3,
        ),
        "defect_rate": round(
            max(
                0,
                min(
                    1.0,
                    base["defect_rate"]
                    + scenario["defect_delta"] * drift
                    + _noise(0.005),
                ),
            ),
            4,
        ),
        "production_speed": round(
            max(0, base["speed"] + scenario["speed_delta"] * drift + _noise(1.0)), 2
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return reading


async def post_reading(client: httpx.AsyncClient, payload: dict) -> dict | None:
    """POST one reading to FastAPI ingest endpoint."""
    try:
        response = await client.post(
            f"{settings.FASTAPI_INGEST_URL}",
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"[Simulator] HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"[Simulator] Connection error: {e}")
    return None


async def run_simulation_loop(
    machine_id: str = "M-001",
    fault: str = "normal",
    interval: float | None = None,
    max_readings: int = 0,  # 0 = infinite
):
    """
    Main simulation loop. Streams readings until interrupted or max_readings reached.

    Args:
        machine_id: Target machine.
        fault: Fault scenario to simulate.
        interval: Seconds between readings (defaults to settings).
        max_readings: Stop after N readings (0 = run forever).
    """

    interval = interval or settings.SIMULATOR_INTERVAL_SECONDS
    print(
        f"[Simulator] Starting → machine={machine_id}, fault={fault}, interval={interval}s"
    )

    drift_step = 60
    count = 0

    async with httpx.AsyncClient() as client:
        while True:
            payload = simulate_sensor_reading(
                machine_id, fault=fault, drift_step=drift_step
            )
            result = await post_reading(client, payload)

            if result:
                anomaly_flag = "⚠ ANOMALY" if result.get("is_anomaly") else "✓ normal"
                print(
                    f"[{datetime.utcnow().strftime('%H:%M:%S')}] {machine_id} | "
                    f"T={payload['temperature']}°C "
                    f"P={payload['pressure']}kPa "
                    f"V={payload['vibration']}mm/s "
                    f"DR={payload['defect_rate'] * 100:.2f}% "
                    f"| {anomaly_flag} ({result.get('severity', '?')})"
                )
            else:
                # Print locally if API unavailable
                print(
                    f"[{datetime.utcnow().strftime('%H:%M:%S')}] {json.dumps(payload)}"
                )

            drift_step += 1
            count += 1
            if max_readings and count >= max_readings:
                print(f"[Simulator] Completed {count} readings. Stopping.")
                break

            await asyncio.sleep(interval)


async def run_multi_machine_simulation():
    """Simulate all machines concurrently with random fault injection."""
    tasks = []
    for machine_id in MACHINES:
        # Randomly assign a fault to one machine
        fault = random.choice(
            ["normal", "normal", "normal", "bearing_failure", "tool_wear"]
        )
        task = asyncio.create_task(
            run_simulation_loop(machine_id=machine_id, fault=fault)
        )
        tasks.append(task)
        await asyncio.sleep(0.3)  # Stagger starts

    await asyncio.gather(*tasks)


# ── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="AI QC Sensor Data Simulator")
    parser.add_argument("--machine", default="M-001", choices=MACHINES)
    parser.add_argument(
        "--fault", default="normal", choices=list(FAULT_SCENARIOS.keys())
    )
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument(
        "--all-machines", action="store_true", help="Simulate all machines"
    )
    parser.add_argument("--max", type=int, default=0, help="Max readings (0=infinite)")
    args = parser.parse_args()

    try:
        if args.all_machines:
            asyncio.run(run_multi_machine_simulation())
        else:
            asyncio.run(
                run_simulation_loop(
                    machine_id=args.machine,
                    fault=args.fault,
                    interval=args.interval,
                    max_readings=args.max,
                )
            )
    except KeyboardInterrupt:
        print("\n[Simulator] Stopped by user.")


if __name__ == "__main__":
    main()
