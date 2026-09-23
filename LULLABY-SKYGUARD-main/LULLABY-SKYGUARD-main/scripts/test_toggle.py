import asyncio
from model.simulator import create_simulator_state

async def main():
    sim = create_simulator_state()
    print("Initial mode:", sim.mode)
    print("Tambaram status:", sim.manager.get_station_status("AWS-CHN-101")["status"])
    
    # Start replay
    sim.start_replay()
    print("Started replay. Mode:", sim.mode)
    
    # Force a station to offline
    sim.manager.buffers["AWS-CHN-101"].health.status = "OFFLINE"
    print("Tambaram status during replay:", sim.manager.get_station_status("AWS-CHN-101")["status"])
    
    # Stop replay
    sim.stop_replay()
    print("Stopped replay. Mode:", sim.mode)
    print("Tambaram status after stop:", sim.manager.get_station_status("AWS-CHN-101")["status"])

if __name__ == "__main__":
    asyncio.run(main())
