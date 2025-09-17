#!/usr/bin/env python3
# perception.py — LKAS + ACC + Lead vehicle
# - 카메라 LKAS 시각화(원래 스타일)
# - Radar → 거리/상대속도/TTC 퍼블리시 (+has_target=연속2프레임 확인)
# - 오탐 방지: FOV 축소, 방위각 제한, 도플러 컷오프, 점프필터, EMA,
#              차로-코리더 게이팅(|lateral|≤2.0m)
# - Lead: 평균 45km/h 목표, 시나리오 = [0~10s 45km/h] → [10~25s 정지] → [25s~ 45km/h]
#         TM 있을 때: 속도는 TM로 명령, 정지는 TM off + 브레이크/핸드브레이크

import argparse, time, json, os
from typing import Optional
import numpy as np
import cv2
import carla
import zenoh
from collections import deque
from queue import Queue, Empty

# === LKAS helpers ===
from common.LK_algo import (
    detect_lanes_and_center, fuse_lanes_with_memory,
    lane_mid_x, lookahead_ratio, speed_of
)

# === Radar 파라미터 ===
MIN_VALID_DEPTH            = 0.5
MAX_VALID_DEPTH            = 120.0

BASE_MAX_AZIMUTH_DEG       = 3.0     # 기본 허용 방위각
MAX_AZIMUTH_DEG_LIMIT      = 6.0     # 커브 시 상한
MAX_DEPTH_JUMP_PER_SEC     = 40.0
MIN_DOPPLER_ABS            = 0.8     # |doppler| < 0.8 → 정지물로 간주

LANE_LATERAL_LIMIT_M       = 2.0     # 차로-코리더 게이팅(|y|<=2.0m)

EMA_ALPHA_DIST             = 0.30
EMA_ALPHA_RELSPEED         = 0.30
RELSPEED_EPS               = 0.30
TARGET_LOST_TIMEOUT_SEC    = 0.60
MIN_CONFIRM_FRAMES         = 2       # 연속 확인 후만 has_target=True

# ---- Radar queue ----
_radar_q = deque(maxlen=1)
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
        """레이더 측정에서 최적 타깃 1개 선택 (코리더/도플러/방위각/점프 필터)."""
        best = None
        az_lim = np.radians(max_az_deg)
        for det in radar_measurement:
            depth = float(det.depth)
            if not (MIN_VALID_DEPTH <= depth <= MAX_VALID_DEPTH):
                continue
            az = float(det.azimuth)
            if abs(az) >= az_lim:
                continue

            # 차로-코리더 게이팅: 횡방향 |y| = depth * sin(azimuth)
            lateral = abs(depth * np.sin(az))
            if lateral > LANE_LATERAL_LIMIT_M:
                continue

            # 도플러 컷오프: 거의 정지물은 배제(중앙 부근 제외시 false positive 억제)
            doppler = float(det.velocity)
            if abs(doppler) < MIN_DOPPLER_ABS and abs(az) > np.radians(1.0):
                continue

            # 중심에 더 가까운 타깃 우선
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

        # 거리 EMA
        if self.ema_distance is None:
            self.ema_distance = depth
        else:
            self.ema_distance = (1 - EMA_ALPHA_DIST) * self.ema_distance + EMA_ALPHA_DIST * depth

        # 상대속도: 레이더 도플러(+면 접근으로 사용)와 거리미분 융합(보수적)
        v_radar = -float(best.velocity)  # + 접근
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
    """m/s"""
    v = actor.get_velocity()
    return float(np.linalg.norm([v.x, v.y, v.z]))


def main():
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args()

    # === Recorder ===
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

    # === Zenoh ===
    z = zenoh.open({})
    pub_lk_feat = z.declare_publisher('demo/lk/features')
    pub_jpg     = z.declare_publisher('demo/lk/frame') if args.pub_jpeg else None
    pub_acc_feat= z.declare_publisher('demo/acc/features')

    # Decision 제어값 구독 (조향으로 방위각 완화)
    ctrl_q = Queue(maxsize=1)
    last_ctrl = {"throttle": 0.0, "brake": 0.0, "steer": 0.0}
    def _ctrl_cb(sample):
        try:
            msg = json.loads(bytes(sample.payload).decode('utf-8'))
            while not ctrl_q.empty(): ctrl_q.get_nowait()
            ctrl_q.put_nowait(msg)
        except: pass
    sub_ctrl = z.declare_subscriber('demo/lk/ctrl', _ctrl_cb)

    # === CARLA ===
    client = carla.Client(args.host, args.port); client.set_timeout(5.0)
    world  = client.get_world()

    original = world.get_settings()
    fixed_dt = 1.0 / max(1, args.fps)
    sync = carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=fixed_dt
    )
    world.apply_settings(sync)

    # Ego
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

    # Lead + TM
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
                    tm = client.get_trafficmanager(port)
                    tm.set_synchronous_mode(True)
                    tm_port = port
                    break
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

    # Sensors
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(fixed_dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam    = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    radar_bp = bp.find('sensor.other.radar')
    radar_bp.set_attribute('range', '120')
    radar_bp.set_attribute('horizontal_fov', '20')   # FOV 축소
    radar_bp.set_attribute('vertical_fov', '10')
    radar_bp.set_attribute('sensor_tick', str(fixed_dt))
    radar_tf = carla.Transform(carla.Location(x=2.8, z=1.0))
    radar    = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)
    radar.listen(_radar_callback)

    latest = {'bgr': None, 'frame_id': None}
    def on_image(img: carla.Image):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        latest['bgr'] = arr[:, :, :3].copy()
        latest['frame_id'] = img.frame
    cam.listen(on_image)

    lane_mem = {"left":None, "right":None, "t_left":None, "t_right":None, "center_x":None}
    tracker = TargetTracker(dt=fixed_dt)
    confirm_streak = 0

    # ---- Lead speed profile state machine ----
    # phase: CRUISE1(0~10s, 45km/h) → STOP(10~25s) → CRUISE2(25s~, 45km/h)
    LEAD_SET_KPH = 45.0
    def _apply_tm_speed(veh, desired_kph: float):
        """TM이 있을 때 목표속도로 설정. set_desired_speed가 없으면 퍼센트 감속으로 근사."""
        nonlocal tm
        if tm is None: return
        try:
            # 일부 버전에서 제공됨 (km/h)
            if hasattr(tm, "set_desired_speed"):
                tm.set_desired_speed(veh, float(desired_kph))
                return
        except Exception:
            pass
        # fallback: 현재 제한속도를 읽어 근사 퍼센티지 설정
        try:
            limit_kph = float(veh.get_speed_limit())  # km/h
            if limit_kph > 1.0:
                perc = 100.0 * (1.0 - (desired_kph / limit_kph))
                perc = float(np.clip(perc, 0.0, 90.0))
                tm.vehicle_percentage_speed_difference(veh, perc)
        except Exception:
            # 마지막 안전망: 아주 작은 감속이라도 적용
            try: tm.vehicle_percentage_speed_difference(veh, 20.0)
            except Exception: pass

    lead_phase = None  # None→초기, 이후 'CRUISE1'/'STOP'/'CRUISE2'
    last_phase = None
    last_tm_speed_update = -1e9  # 주기적 갱신용

    if args.display:
        cv2.namedWindow('lane_follow', cv2.WINDOW_NORMAL)

    frame_period = fixed_dt
    next_t = time.perf_counter()
    frame_count = 0

    print('[INFO] Perception running... Press q to quit.')
    try:
        while True:
            if args.realtime:
                now = time.perf_counter()
                if now < next_t: time.sleep(next_t - now)
                next_t += frame_period

            world.tick()
            frame_count += 1
            sim_time = frame_count * fixed_dt
            if latest['bgr'] is None: continue

            # ---- Lead phase selection by time ----
            if sim_time < 10.0:
                lead_phase = 'CRUISE1'
            elif sim_time < 25.0:
                lead_phase = 'STOP'
            else:
                lead_phase = 'CRUISE2'

            # On phase change: enforce control mode/speed immediately
            if lead and (lead_phase != last_phase):
                if lead_phase in ('CRUISE1', 'CRUISE2'):
                    # Run with TM autopilot + desired 45 km/h
                    if tm is not None:
                        try: lead.set_autopilot(True, tm_port)
                        except Exception: pass
                        _apply_tm_speed(lead, LEAD_SET_KPH)
                        last_tm_speed_update = sim_time
                    else:
                        # no TM → 수동 저스로틀 주행
                        try:
                            lead.set_autopilot(False)
                            lead.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0, hand_brake=False))
                        except Exception: pass
                elif lead_phase == 'STOP':
                    # Ensure full stop: disable TM, hard brake + handbrake
                    try:
                        lead.set_autopilot(False)
                        lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                    except Exception: pass
                last_phase = lead_phase

            # While cruising with TM, refresh target speed every ~1s (limit changes에 대응)
            if lead and tm and lead_phase in ('CRUISE1', 'CRUISE2'):
                if (sim_time - last_tm_speed_update) > 1.0:
                    _apply_tm_speed(lead, LEAD_SET_KPH)
                    last_tm_speed_update = sim_time
                # 혹시 STOP phase 직후 잔여 제동 해제
                try:
                    lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                except Exception:
                    pass

            # ---- 최신 제어값 (ego HUD 표시용) ----
            try: last_ctrl = ctrl_q.get_nowait()
            except Empty: pass
            thr = float(last_ctrl.get("throttle", 0.0))
            brk = float(last_ctrl.get("brake", 0.0))
            st  = float(last_ctrl.get("steer", 0.0))

            # 커브 시 방위각 완화 (3° → 최대 6°)
            max_az_deg = float(np.clip(BASE_MAX_AZIMUTH_DEG + 6.0 * abs(st),
                                       BASE_MAX_AZIMUTH_DEG, MAX_AZIMUTH_DEG_LIMIT))

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

            # 시각화
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

            # ===== Radar 추적 =====
            update_ok = False
            if _radar_q:
                meas = _radar_q.popleft()
                update_ok = tracker.update_from_radar(meas, sim_time, max_az_deg=max_az_deg)
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

            # ACC 피처 퍼블리시
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

            # LKAS 피처 퍼블리시
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
                              f"tgt={int(has_target_confirmed)}  az={max_az_deg:.1f}deg",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"LEAD phase={lead_phase or '-':>8s}  v_lead={lead_speed_disp:5.1f} km/h",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,240,180), 2, cv2.LINE_AA)

            # JPEG 퍼블리시(옵션)
            if pub_jpg is not None:
                ok, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    pub_jpg.put(buf.tobytes())

            if args.display:
                cv2.imshow("lane_follow", vis)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    finally:
        try: cam.stop()
        except: pass
        try: radar.stop()
        except: pass
        try: cam.destroy()
        except: pass
        try: radar.destroy()
        except: pass
        try:
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except: pass
        try: ego.destroy()
        except: pass
        try:
            if lead:
                # 정지 상태에서 핸드브레이크 해제 후 제거
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
        try: sub_ctrl.undeclare()
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

