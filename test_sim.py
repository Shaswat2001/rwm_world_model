import time
import mujoco
import mujoco.viewer

XML_PATH = "/Users/shaswatgarg_mini/.cache/rwm/assets/robots/mujoco/booster_t1/booster_t1.xml"  # your MJCF file

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Reset (uses keyframe 0 if present)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            # real-time pacing
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
