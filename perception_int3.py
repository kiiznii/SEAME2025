#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perception_int.py — ORIGINAL-STYLE LKAS + ACC + Lead scenario
+ GO latch (decision trigger)
+ Auto cleanup of leftover actors
+ Chase (3rd-person) camera with TRUE projected radar overlay

Design goals (per user spec):
1) Keep original pipeline style: use LK_algo for lanes, robust radar tracker & lead scenario.
2) Wait for decision (any ctrl msg) -> GO latch. Before GO, ego/lead fully stopped.
3) Auto cleanup at startup.
4) Visuals: original front HUD + chase camera window; chase shows radar detections
   projected with camera pinhole model (not schematic).
5) Topics/CLI/payloads: same as original (demo/lk/* and demo/acc/*).
"""

import argparse, time, json, os, math, traceback
from typing import Optional
import numpy as np
import cv2
import carla
import zenoh
from collections import deque
from queue import Queue, Empty

# ---- Import LK_algo from original path (common/LK_algo.py), with fallback ----
try:
    from common.LK_algo import (
        detect_lanes_and_center, fuse_lanes_with_memory,
        lane_mid_x, lookahead_ratio, speed_of
    )
except Exception:
    from LK_algo import (
        detect_lanes_and_center, fuse_lanes_with_memory,
        lane_mid_x, lookahead_ratio, speed_of
    )

# ========= ORIGINAL radar parameters (조정) =========
MIN_VALID_DEPTH            = 0.5
MAX_VALID_DEPTH            = 120.0

BASE_MAX_AZIMUTH_DEG       = 3.0
MAX_AZIMUTH_DEG_LIMIT      = 6.0
MAX_DEPTH_JUMP_PER_SEC     = 40.0
MIN_DOPPLER_ABS            = 0.8

# ★ MOD: ‘차선 내부’ 코리더 폭(자차 기준 좌우 half-width) — in-lane 전용 게이팅 (2.0→1.8)
LANE_LATERAL_LIMIT_M       = 1.8  # ★ MOD: guardrail 억제 강화

EMA_ALPHA_DIST             = 0.30
EMA_ALPHA_RELSPEED         = 0.30
RELSPEED_EPS               = 0.30
TARGET_LOST_TIMEOUT_SEC    = 0.60
MIN_CONFIRM_FRAMES         = 2

# ---- Radar queue (as in original spirit) ----
_radar_q = deque(maxlen=2)
def _radar_callback(meas: carla.RadarMeasurement):
    _radar_q.append(meas)

class TargetTracker:
    def __init__(self, dt: float, target_lost_timeout_sec: float = TARGET_LOST_TIMEOUT_SEC):
        self.dt = dt
        self.ema_distance = None
        self.prev_distance = None
        self.ema_rel_speed = 0.0
        self.last_seen_time = -1e9
        self.current_distance = float('inf')
        self.target_lost_timeout = target_lost_timeout_sec
        self.last_update_ok = False

    def reset(self):
        self.ema_distance = None
        self.prev_distance = None
        self.ema_rel_speed = 0.0
        self.current_distance = float('inf')
        self.last_update_ok = False

    def update_from_radar(self, radar_measurement: carla.RadarMeasurement, sim_time: float,
                          max_az_deg: float) -> bool:
        best = None
        az_lim = np.radians(max_az_deg)
        for det in radar_measurement:
            depth = float(det.depth)
            if not (MIN_VALID_DEPTH <= depth <= MAX_VALID_DEPTH):
                continue
            az = float(det.azimuth)
            if abs(az) >= az_lim:
                continue

            # in-lane 게이팅
            lateral = abs(depth * np.sin(az))
            if lateral > LANE_LATERAL_LIMIT_M:
                continue

            # 정지물/가드레일 억제(중앙 제외 뺄셈)
            doppler = float(det.velocity)
            if abs(doppler) < MIN_DOPPLER_ABS and abs(az) > np.radians(1.0):
                continue

            # 정면에 가장 가까운(작은 |az|) 후보를 우선
            if (best is None) or (abs(az) < abs(float(best.azimuth))):
                best = det

        if best is None:
            self.current_distance = float('inf')
            self.last_update_ok = False
            return False

        depth = float(best.depth)
        if self.prev_distance is not None:
            max_jump = MAX_DEPTH_JUMP_PER_SEC * self.dt
            if abs(depth - self.prev_distance) > max_jump:
                self.last_update_ok = False
                return False

        if self.ema_distance is None:
            self.ema_distance = depth
        else:
            self.ema_distance = (1 - EMA_ALPHA_DIST) * self.ema_distance + EMA_ALPHA_DIST * depth

        v_radar = -float(best.velocity)  # + approaching
        v_depth = 0.0 if self.prev_distance is None else (self.prev_distance - depth) / self.dt
        if v_radar > 0 and v_depth > 0:
            rel_speed_inst = 0.6 * v_radar + 0.4 * v_depth
        elif (v_radar > 0) or (v_depth > 0):
            rel_speed_inst = max(0.0, min(v_radar, v_depth))
        else:
            rel_speed_inst = 0.0
        if rel_speed_inst < RELSPEED_EPS:
            rel_speed_inst = 0.0

        self.ema_rel_speed = (1 - EMA_ALPHA_RELSPEED) * self.ema_rel_speed + EMA_ALPHA_RELSPEED * rel_speed_inst
        self.prev_distance = depth
        self.current_distance = float(self.ema_distance)
        self.last_seen_time = sim_time
        self.last_update_ok = True
        return True

    def compute_ttc(self) -> float:
        if self.ema_rel_speed > RELSPEED_EPS and self.current_distance < float('inf'):
            return max(0.0, self.current_distance / self.ema_rel_speed)
        return float('inf')

    def is_lost(self, sim_time: float) -> bool:
        return (sim_time - self.last_seen_time) > self.target_lost_timeout

def _speed_of_actor(actor) -> float:
    v = actor.get_velocity()
    return float(np.linalg.norm([v.x, v.y, v.z]))

# ========= (ADDED) GO latch =========
_go = {'seen': False}

# ========= Transform helpers =========
def transform_to_matrix(T: carla.Transform) -> np.ndarray:
    r = T.rotation
    l = T.location
    cy, sy = math.cos(math.radians(r.yaw)),   math.sin(math.radians(r.yaw))
    cp, sp = math.cos(math.radians(r.pitch)), math.sin(math.radians(r.pitch))
    cr, sr = math.cos(math.radians(r.roll)),  math.sin(math.radians(r.roll))
    R = np.array([
        [ cp*cy,               cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ cp*sy,               sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,                  cp*sr,             cp*cr           ],
    ], dtype=np.float64)
    t = np.array([[l.x], [l.y], [l.z]], dtype=np.float64)
    M = np.vstack([np.hstack([R, t]), np.array([0,0,0,1], dtype=np.float64)])
    return M

def world_to_sensor(p_world: carla.Location, sensor_T: carla.Transform) -> np.ndarray:
    Mw = transform_to_matrix(sensor_T)
    Ms = np.linalg.inv(Mw)
    pw = np.array([p_world.x, p_world.y, p_world.z, 1.0], dtype=np.float64)
    ps = Ms @ pw
    return ps[:3]

# ========= Chase overlay: true projection =========
def draw_radar_on_chase_rays(
    img_bgr: np.ndarray,
    radar_meas: 'carla.RadarMeasurement',
    radar_actor: carla.Actor,
    chase_sensor: carla.Actor,
    fx: float, fy: float, cx: float, cy: float,
    thickness: int = 2,
    min_depth_m: float = 6.0,
    max_depth_m: float = 150.0,
    max_lat_m: float = 2.2,
    max_alt_deg: float = 6.0,
    require_approach: bool = True,
    start_t: float = 0.12
) -> np.ndarray:
    if img_bgr is None or radar_meas is None:
        return img_bgr
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    T_radar = radar_actor.get_transform()
    T_chase = chase_sensor.get_transform()

    ro_w = T_radar.location
    Xo, Yo, Zo = world_to_sensor(ro_w, T_chase)
    if Xo > 0.05:
        u0 = cx + fx * (Yo / Xo)
        v0 = cy - fy * (Zo / Xo)
        origin_ok = (0 <= u0 < W and 0 <= v0 < H)
    else:
        origin_ok = False
    if not origin_ok:
        u0, v0 = W // 2, int(H * 0.82)

    max_alt = math.radians(max_alt_deg)

    for d in radar_meas:
        depth = float(d.depth)
        if not (min_depth_m <= depth <= max_depth_m):
            continue
        if require_approach and not (d.velocity < 0):
            continue
        az  = float(d.azimuth)
        alt = float(d.altitude)
        if abs(alt) > max_alt:
            continue

        xr = depth * math.cos(alt) * math.cos(az)
        yr = depth * math.cos(alt) * math.sin(az)
        zr = depth * math.sin(alt)

        if abs(yr) > max_lat_m:
            continue

        pw = T_radar.transform(carla.Location(x=xr, y=yr, z=zr))
        X, Y, Z = world_to_sensor(pw, T_chase)
        if X <= 0.05:
            continue

        u = cx + fx * (Y / X)
        v = cy - fy * (Z / X)
        if not (0 <= u < W and 0 <= v < H):
            continue

        u_start = int(u0 + start_t * (u - u0))
        v_start = int(v0 + start_t * (v - v0))

        color = (0, 255, 255) if d.velocity < 0 else (200, 200, 200)
        cv2.line(
            vis,
            (u_start, v_start),
            (int(u), int(v)),
            color,
            thickness,
            lineType=cv2.LINE_AA
        )
    return vis


def main():
    ap = argparse.ArgumentParser()
    # ==== ORIGINAL CLI kept intact ====
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--fps',  type=int, default=20)
    ap.add_argument('--spawn_idx', type=int, default=328)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=580)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--display', type=int, default=1)
    ap.add_argument('--realtime', type=int, default=1)
    ap.add_argument('--record', type=str, default='')
    ap.add_argument('--record_mode', choices=['raw','vis','both'], default='vis')
    ap.add_argument('--canny_low', type=int, default=60)
    ap.add_argument('--canny_high', type=int, default=180)
    ap.add_argument('--hough_thresh', type=int, default=40)
    ap.add_argument('--hough_min_len', type=int, default=40)
    ap.add_argument('--hough_max_gap', type=int, default=120)
    ap.add_argument('--pub_jpeg', action='store_true')
    # ==== ADDED: GO latch + cleanup ====
    ap.add_argument('--wait_for_decision', type=int, default=1,
                    help='wait for first ctrl msg on demo/lk/ctrl before moving')
    ap.add_argument('--cleanup', choices=['none','ego_lead','all'], default='all',
                    help='cleanup leftover actors before spawn')

    # ★ MOD: 레이더 조향/범위/FOV 튜닝 파라미터 CLI 추가
    ap.add_argument('--radar_range', type=float, default=70.0, help='Radar range in meters (orig 120)')  # ★ MOD
    ap.add_argument('--radar_hfov', type=float, default=12.0, help='Radar horizontal FOV deg (orig 20)') # ★ MOD
    ap.add_argument('--radar_vfov', type=float, default=8.0,  help='Radar vertical FOV deg (orig 10)')  # ★ MOD
    ap.add_argument('--radar_yaw_gain_deg', type=float, default=15.0,
                    help='Max yaw offset at full steer=±1.0 (deg)')  # ★ MOD
    ap.add_argument('--radar_yaw_alpha', type=float, default=0.25,
                    help='LPF alpha for yaw command (0~1, higher=faster)')  # ★ MOD
    args = ap.parse_args()

    # Recorder (original style)
    vw_raw = vw_vis = None
    def _open_writer(path, fps, w, h):
        if not path: return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {path}")
        return vw

    if args.record:
        base = args.record
        root, ext = os.path.splitext(base)
        if ext == '':
            base = base + '.mp4'
            root, ext = base[:-4], '.mp4'
        if args.record_mode == 'vis':
            vw_vis = _open_writer(base, args.fps, args.width, args.height)
        elif args.record_mode == 'raw':
            vw_raw = _open_writer(base, args.fps, args.width, args.height)
        else:
            vw_raw = _open_writer(f"{root}_raw{ext}", args.fps, args.width, args.height)
            vw_vis = _open_writer(f"{root}_vis{ext}", args.fps, args.width, args.height)

    # Zenoh (original topics) + GO latch
    z = zenoh.open({})
    pub_lk_feat = z.declare_publisher('demo/lk/features')
    pub_jpg     = z.declare_publisher('demo/lk/frame') if args.pub_jpeg else None
    pub_acc_feat= z.declare_publisher('demo/acc/features')

    ctrl_q = Queue(maxsize=1)
    last_ctrl = {"throttle": 0.0, "brake": 0.0, "steer": 0.0}
    def _ctrl_cb(sample):
        try:
            if args.wait_for_decision and not _go['seen']:
                _go['seen'] = True  # latch GO on first message
            msg = json.loads(bytes(sample.payload).decode('utf-8'))
            while not ctrl_q.empty(): ctrl_q.get_nowait()
            ctrl_q.put_nowait(msg)
        except Exception:
            pass
    sub_ctrl = z.declare_subscriber('demo/lk/ctrl', _ctrl_cb)

    # CARLA
    client = carla.Client(args.host, args.port); client.set_timeout(5.0)
    world  = client.get_world()

    # Auto cleanup
    if args.cleanup != 'none':
        try:
            actors = world.get_actors()
            victims = []
            if args.cleanup == 'all':
                victims = list(actors.filter('vehicle.*')) + list(actors.filter('sensor.*')) + list(actors.filter('walker.*'))
            else:  # ego/lead only
                for a in actors.filter('vehicle.*'):
                    if a.attributes.get('role_name','') in ('ego','lead'):
                        victims.append(a)
                for s in actors.filter('sensor.*'):
                    p = s.parent
                    if p and p.attributes.get('role_name','') in ('ego','lead'):
                        victims.append(s)
            for a in victims:
                try: a.destroy()
                except: pass
            time.sleep(0.05); world.tick()
        except Exception as e:
            print('[WARN] cleanup failed:', e)

    original = world.get_settings()
    fixed_dt = 1.0 / max(1, args.fps)
    sync = carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=fixed_dt
    )
    world.apply_settings(sync)

    # Ego spawn
    bp = world.get_blueprint_library()
    ego_bp = (bp.filter('vehicle.*model3*') or bp.filter('vehicle.*'))[0]
    ego_bp.set_attribute('role_name', 'ego')
    sps = world.get_map().get_spawn_points()
    tf  = sps[min(max(0, args.spawn_idx), len(sps)-1)]
    ego = world.try_spawn_actor(ego_bp, tf)
    if not ego:
        for sp in sps:
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego: break
    assert ego, 'Failed to spawn ego vehicle'

    # Lead + TM (original spirit)
    lead = None
    tm = None
    tm_port = None
    try:
        lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
        lead_bp.set_attribute('role_name', 'lead')
        ego_wp = world.get_map().get_waypoint(tf.location)
        lead_wp = ego_wp.next(30.0)[0]
        lead_tf = lead_wp.transform
        lead_tf.location.z = tf.location.z
        lead = world.try_spawn_actor(lead_bp, lead_tf)
        if lead:
            for port in range(8000, 8010):
                try:
                    tm = client.get_trafficmanager(port); tm.set_synchronous_mode(True); tm_port = port; break
                except RuntimeError:
                    tm = None
            if tm is not None:
                tm.auto_lane_change(lead, False)
                lead.set_autopilot(True, tm_port)
                print(f"[INFO] Lead autopilot ON via TM:{tm_port}")
            else:
                lead.set_autopilot(False)
                lead.apply_control(carla.VehicleControl(throttle=0.20))
                print("[WARN] No TM port free, lead moves with constant low throttle.")
    except Exception as e:
        print(f"[WARN] Lead vehicle spawn failed: {e}")

    # Sensors (front camera + radar)
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(fixed_dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam    = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    radar_bp = bp.find('sensor.other.radar')
    # ★ MOD: 레이더 Range/FOV 축소
    radar_bp.set_attribute('range', str(int(args.radar_range)))       # ★ MOD
    radar_bp.set_attribute('horizontal_fov', str(float(args.radar_hfov)))  # ★ MOD
    radar_bp.set_attribute('vertical_fov', str(float(args.radar_vfov)))    # ★ MOD
    radar_bp.set_attribute('sensor_tick', str(fixed_dt))
    # ★ MOD: 레이더 기본 위치/자세 (yaw=0에서 시작, 이후 프레임마다 steer 기반으로 yaw 보정)
    radar_tf_base = carla.Transform(carla.Location(x=2.8, z=1.0), carla.Rotation(yaw=0.0))  # ★ MOD
    radar         = world.spawn_actor(radar_bp, radar_tf_base, attach_to=ego)
    radar.listen(_radar_callback)

    # (ADDED) Chase camera
    chase = None
    latest_chase = {'img': None}
    try:
        chase_bp = bp.find('sensor.camera.rgb')
        chase_bp.set_attribute('image_size_x', str(args.width))
        chase_bp.set_attribute('image_size_y', str(args.height))
        chase_bp.set_attribute('fov', '70')
        chase_bp.set_attribute('sensor_tick', str(fixed_dt))
        chase_tf = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-10.0))
        chase = world.spawn_actor(chase_bp, chase_tf, attach_to=ego)
        chase.listen(lambda img:
            latest_chase.__setitem__('img',
                np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, :3].copy()))
        # Intrinsics for chase
        chase_fov_deg = float(chase_bp.get_attribute('fov').as_float())
        W, H = args.width, args.height
        fx = W / (2.0 * math.tan(math.radians(chase_fov_deg) / 2.0))
        fy = fx
        cx, cy = W / 2.0, H / 2.0
    except Exception as e:
        print(f"[WARN] Chase camera spawn failed: {e}")
        traceback.print_exc()
        chase = None
        fx = fy = cx = cy = None

    latest = {'bgr': None, 'frame_id': None}
    def on_image(img: carla.Image):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        latest['bgr'] = arr[:, :, :3].copy()
        latest['frame_id'] = img.frame
    cam.listen(on_image)

    lane_mem = {"left":None, "right":None, "t_left":None, "t_right":None, "center_x":None}
    tracker = TargetTracker(dt=fixed_dt)
    confirm_streak = 0

    # Lead speed profile (original timing, but only after GO)
    LEAD_SET_KPH = 45.0
    def _apply_tm_speed(veh, desired_kph: float):
        nonlocal tm
        if tm is None: return
        try:
            if hasattr(tm, "set_desired_speed"):
                tm.set_desired_speed(veh, float(desired_kph)); return
        except Exception:
            pass
        try:
            limit_kph = float(veh.get_speed_limit())
            if limit_kph > 1.0:
                perc = 100.0 * (1.0 - (desired_kph / limit_kph))
                perc = float(np.clip(perc, 0.0, 90.0))
                tm.vehicle_percentage_speed_difference(veh, perc)
        except Exception:
            try: tm.vehicle_percentage_speed_difference(veh, 20.0)
            except Exception: pass

    lead_phase = None
    last_phase = None
    last_tm_speed_update = -1e9

    if args.display:
        cv2.namedWindow('lane_follow', cv2.WINDOW_NORMAL)
        cv2.namedWindow('chase', cv2.WINDOW_NORMAL)

    frame_period = fixed_dt
    next_t = time.perf_counter()
    frame_count = 0

    # ★ MOD: 레이더 yaw 동적 보정 상태
    radar_yaw_cmd = 0.0  # deg  # ★ MOD

    print('[INFO] Perception running... Press q to quit.')
    try:
        # Hold vehicles until GO
        def _hold(v):
            try:
                v.set_autopilot(False)
                v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            except: pass
        _hold(ego)
        if lead is not None: _hold(lead)

        while True:
            if args.realtime:
                now = time.perf_counter()
                if now < next_t: time.sleep(next_t - now)
                next_t += frame_period

            world.tick()
            frame_count += 1
            sim_time = frame_count * fixed_dt
            if latest['bgr'] is None: continue

            # Preview radar for chase overlay
            radar_for_viz = _radar_q[-1] if _radar_q else None

            # GO gating
            if not _go['seen']:
                _hold(ego)
                if lead is not None: _hold(lead)
            else:
                if ego.get_control().hand_brake:
                    try: ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                    except: pass
                if lead is not None and lead.get_control().hand_brake:
                    try: lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                    except: pass

            # Lead phase timing (only after GO)
            if _go['seen']:
                if sim_time < 10.0:
                    lead_phase = 'CRUISE1'
                elif sim_time < 25.0:
                    lead_phase = 'STOP'
                else:
                    lead_phase = 'CRUISE2'

                if lead and (lead_phase != last_phase):
                    if lead_phase in ('CRUISE1', 'CRUISE2'):
                        if tm is not None:
                            try: lead.set_autopilot(True, tm_port)
                            except Exception: pass
                            _apply_tm_speed(lead, LEAD_SET_KPH)
                            last_tm_speed_update = sim_time
                        else:
                            try:
                                lead.set_autopilot(False)
                                lead.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0, hand_brake=False))
                            except Exception: pass
                    elif lead_phase == 'STOP':
                        try:
                            lead.set_autopilot(False)
                            lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                        except Exception: pass
                    last_phase = lead_phase

                if lead and tm and lead_phase in ('CRUISE1', 'CRUISE2'):
                    if (sim_time - last_tm_speed_update) > 1.0:
                        _apply_tm_speed(lead, LEAD_SET_KPH)
                        last_tm_speed_update = sim_time
                    try:
                        lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                    except Exception:
                        pass

            # Last ctrl (for HUD + 레이더 yaw 제어)
            try: last_ctrl = ctrl_q.get_nowait()
            except Empty: pass
            thr = float(last_ctrl.get("throttle", 0.0))
            brk = float(last_ctrl.get("brake", 0.0))
            st  = float(last_ctrl.get("steer", 0.0))

            # ★ MOD: 조향에 따라 레이더 허용각 + 레이더 자체 yaw 오프셋 적용
            max_az_deg = float(np.clip(BASE_MAX_AZIMUTH_DEG + 6.0 * abs(st),
                                       BASE_MAX_AZIMUTH_DEG, MAX_AZIMUTH_DEG_LIMIT))
            # 레이더 자체 yaw 명령(±radar_yaw_gain_deg * steer)
            desired_yaw = float(np.clip(args.radar_yaw_gain_deg * st, -args.radar_yaw_gain_deg, args.radar_yaw_gain_deg))  # deg  # ★ MOD
            radar_yaw_cmd = (1.0 - args.radar_yaw_alpha) * radar_yaw_cmd + args.radar_yaw_alpha * desired_yaw            # ★ MOD
            # 센서 상대 Transform 갱신 (부모=ego 기준)
            try:
                radar.set_transform(carla.Transform(radar_tf_base.location,
                                    carla.Rotation(yaw=radar_yaw_cmd)))  # ★ MOD
            except:
                pass

            bgr = latest['bgr']; h, w = bgr.shape[:2]

            # ===== LKAS =====
            vis = bgr.copy()
            lanes = detect_lanes_and_center(
                bgr, roi_vertices=None,
                canny_low=args.canny_low, canny_high=args.canny_high,
                hough_thresh=args.hough_thresh,
                hough_min_len=args.hough_min_len,
                hough_max_gap=args.hough_max_gap,
            )
            used_left, used_right, center_line, _ = fuse_lanes_with_memory(
                lanes, latest['frame_id'], lane_mem, ttl_frames=60
            )

            v_kmh = speed_of(ego) * 3.6
            y_anchor = int(lookahead_ratio(v_kmh) * h)

            x_lane_mid = None
            if used_left and used_right:
                x_lane_mid, _, _ = lane_mid_x(used_left, used_right, y_anchor)
                lane_mem['center_x'] = x_lane_mid
            else:
                x_lane_mid = lane_mem.get('center_x', None)

            overlay = vis.copy()
            if 'roi_vertices' in lanes and lanes['roi_vertices'] is not None:
                cv2.fillPoly(overlay, lanes['roi_vertices'], color=(0, 0, 255))
                vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0.0)
            for x1, y1, x2, y2 in lanes.get('line_segs', []):
                cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1, cv2.LINE_AA)
            if used_left:
                lx1, ly1, lx2, ly2 = used_left
                cv2.line(vis, (lx1, ly1), (lx2, ly2), (0, 0, 255), 5, cv2.LINE_AA)
            if used_right:
                rx1, ry1, rx2, ry2 = used_right
                cv2.line(vis, (rx1, ry1), (rx2, ry2), (0, 0, 255), 5, cv2.LINE_AA)
            Xmid = w // 2
            cv2.line(vis, (Xmid, 0), (Xmid, h-1), (0, 255, 255), 1, cv2.LINE_AA)
            if x_lane_mid is not None:
                cv2.line(vis, (x_lane_mid, 0), (x_lane_mid, h-1), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(vis, (0, y_anchor), (w-1, y_anchor), (0, 255, 0), 1)

            # ===== Radar tracking =====
            update_ok = False
            if _radar_q:
                meas_for_tracker = _radar_q.popleft()
                update_ok = tracker.update_from_radar(meas_for_tracker, sim_time, max_az_deg=max_az_deg)
            if tracker.is_lost(sim_time):
                tracker.reset()
                update_ok = False

            confirm_streak = (confirm_streak + 1) if update_ok else 0
            has_target_confirmed = (confirm_streak >= MIN_CONFIRM_FRAMES)

            dist  = float(tracker.current_distance)
            rel_v = float(tracker.ema_rel_speed)
            ttc   = float(tracker.compute_ttc())
            ego_v = float(speed_of(ego))
            lead_v_est = max(0.0, ego_v - rel_v) if (np.isfinite(dist) and has_target_confirmed) else None

            # ACC publish (original payload 유지)
            acc_payload = {
                'frame_id': latest['frame_id'],
                'sim_time': sim_time,
                'distance': dist if (np.isfinite(dist) and has_target_confirmed) else None,
                'rel_speed': rel_v if has_target_confirmed else 0.0,
                'ttc': ttc if (np.isfinite(ttc) and has_target_confirmed) else None,
                'ego_speed': ego_v,
                'lead_speed_est': lead_v_est,
                'has_target': bool(has_target_confirmed)
            }
            pub_acc_feat.put(json.dumps(acc_payload).encode('utf-8'))

            # LKAS publish (original)
            lk_payload = {
                'frame_id': latest['frame_id'],
                'w': w, 'h': h,
                'y_anchor': y_anchor,
                'x_lane_mid': x_lane_mid
            }
            pub_lk_feat.put(json.dumps(lk_payload).encode('utf-8'))

            # HUD
            hud_txt = f"v={v_kmh:5.1f} km/h  thr={thr:.2f} brk={brk:.2f} str={st:+.2f}"
            cv2.putText(vis, hud_txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            ttc_disp = ttc if (np.isfinite(ttc) and has_target_confirmed) else -1.0
            lead_speed_disp = -1.0
            if lead:
                try: lead_speed_disp = _speed_of_actor(lead) * 3.6
                except: pass
            cv2.putText(vis, f"ACC dist={dist if (np.isfinite(dist) and has_target_confirmed) else -1:5.1f}m  "
                              f"rel={rel_v:+4.1f}m/s  TTC={ttc_disp:4.1f}s  "
                              f"tgt={int(has_target_confirmed)}  az={max_az_deg:.1f}deg  "
                              f"radYaw={radar_yaw_cmd:+4.1f}°",  # ★ MOD: 디버그
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"LEAD phase={lead_phase or '-':>8s}  v_lead={lead_speed_disp:5.1f} km/h",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,240,180), 2, cv2.LINE_AA)

            # JPEG publish
            if pub_jpg is not None:
                ok, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok: pub_jpg.put(buf.tobytes())

            # === Display ===
            if args.display:
                cv2.imshow("lane_follow", vis)
                if chase is not None and latest_chase['img'] is not None and fx is not None:
                    chase_img = latest_chase['img']
                    if radar_for_viz is not None:
                        chase_img = draw_radar_on_chase_rays(
                            chase_img, radar_for_viz,
                            radar_actor=radar, chase_sensor=chase,
                            fx=fx, fy=fy, cx=cx, cy=cy,
                            thickness=1
                        )
                    cv2.imshow('chase', chase_img)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    finally:
        try: cam.stop()
        except: pass
        try: radar.stop()
        except: pass
        try:
            if chase is not None: chase.stop()
        except: pass
        try: cam.destroy()
        except: pass
        try: radar.destroy()
        except: pass
        try:
            if chase is not None: chase.destroy()
        except: pass
        try:
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except: pass
        try:
            ego.destroy()
        except: pass
        try:
            if lead:
                try: lead.apply_control(carla.VehicleControl(hand_brake=False))
                except: pass
                lead.destroy()
        except: pass
        if args.display:
            try: cv2.destroyAllWindows()
            except: pass
        try:
            world.apply_settings(original); time.sleep(0.05); world.tick()
        except: pass
        try:
            sub_ctrl.undeclare()
        except: pass
        try: pub_lk_feat.undeclare()
        except: pass
        try:
            if pub_jpg is not None: pub_jpg.undeclare()
        except: pass
        try: pub_acc_feat.undeclare()
        except: pass
        try: z.close()
        except: pass
        print('[INFO] Perception stopped.')

if __name__ == '__main__':
    main()

