import asyncio
from model.simulator import create_simulator_state

async def test():
    sim = create_simulator_state()
    for i in range(100):
        await sim.tick()
        if len(sim.recent_anomalies) > 0:
            for a in sim.recent_anomalies:
                print(f"Anomaly at {a['timestamp']} Station: {a['station_id']}: Type={a['type']} Score={a['anomaly_score_pct']} Rules={a.get('rules_fired')} Model={a.get('model_confidence_pct')}")
    await sim.close()

if __name__ == "__main__":
    asyncio.run(test())
