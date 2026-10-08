from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import asyncio
import omni.kit.asset_converter as ac

SRC = "/home/robot/Desktop/Tina/fanuc_sim/meshes/nastavak_za_simulaciju.stl"
DST = "/home/robot/Desktop/Tina/fanuc_sim/meshes/nastavak.usd"

async def go():
    task = ac.get_instance().create_converter_task(SRC, DST, None)
    ok = await task.wait_until_finished()
    print("KONVERZIJA:", "OK" if ok else "FAIL", "" if ok else task.get_error_message())

asyncio.get_event_loop().run_until_complete(go())
simulation_app.close()
