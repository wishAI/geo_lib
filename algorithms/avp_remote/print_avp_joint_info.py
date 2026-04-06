import mujoco
from avp_stream import VisionProStreamer

from avp_config import AVP_IP, AVP_ROBOT_XML_PATH

avp_ip = AVP_IP
robot_path = str(AVP_ROBOT_XML_PATH)

model = mujoco.MjModel.from_xml_path(robot_path)
data = mujoco.MjData(model)

s = VisionProStreamer(ip=avp_ip)
s.configure_mujoco(robot_path, model, data, relative_to=[0, 0, 0.8, 90])
s.start_webrtc()


# printed format example:
# left hand wrist:  [[[ 0.95134634  0.28279546  0.12233841  0.34763023]
#   [-0.0390449  -0.2831994   0.95826596  0.12040544]
#   [ 0.30563942 -0.91641945 -0.25837907 -0.02239518]
#   [ 0.          0.          0.          1.        ]]]
# right hand wrist:  [[[-0.96688592 -0.24610409 -0.06756126  0.39047408]
#   [ 0.03187424 -0.37910667  0.92480391 -0.24184811]
#   [-0.25321096  0.89202631  0.37439731 -0.03548871]
#   [ 0.          0.          0.          1.        ]]]


while True:
    # Your control logic using hand tracking
    data = s.get_latest()
    print('left hand wrist: ', data.get('left_wrist'), " right hand wrist: ", data.get('right_wrist'))
    print(' -------- ')
    # ... update robot based on hand positions ...
    
    mujoco.mj_step(model, data)
    s.update_sim()  # Stream updated poses to Vision Pro
