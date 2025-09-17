#!/usr/bin/env python3
# decision.py — LKAS + ACC
# - 앞차 없으면 80km/h 확실히 순항 (크루즈 부스트)
# - FOLLOW 진입: has_target 연속≥2 + TTC 유한 & TTC<FOLLOW_ENTER_TTC + 거리<MAX  # ★ 변경
# - AEB는 타깃 확정 시에만

import argparse, json, time
from queue import Queue, Empty
import numpy as np
import carla
import zenoh

from common.LK_algo import gains_for_speed, speed_of

# ---- ACC 파라미터 ----
TARGET_SPEED_KPH             = 80.0
TARGET_SPEED_MPS             = TARGET_SPEED_KPH / 3.6
TIME_HEADWAY                 = 1.8
MIN_GAP                      = 5.0

KP_GAP                       = 0.30
KV_SPEED                     = 0.22
MAX_ACCEL                    = 1.2
MAX_DECEL                    = 5.0
THR_GAIN                     = 0.40
BRK_GAIN                     = 0.35

# ★ MOD: 타이트한 TTC
AEB_ENTER_TTC                = 2.5   # ★ MOD (3.0→2.5)
AEB_EXIT_TTC                 = 4.0   # ★ MOD (4.5→4.0)
FOLLOW_ENTER_TTC             = 7.0   # ★ MOD (8.0→6.0)
FOLLOW_EXIT_TTC              = 9.0   # ★ MOD (10.0→8.0)
AEB_ACTIVATION_SPEED_THRESHOLD = 2.8
AEB_MIN_HOLD_SEC             = 0.8
MODE_COOLDOWN_SEC            = 0.4

MIN_CONFIRM_FRAMES           = 2
FOLLOW_MAX_DIST_M            = 60.0  # ★ MOD (80→60)

# 레이트 리미터(20Hz)
THR_RATE_UP                  = 0.08
THR_RATE_DOWN                = 0.25
BRK_RATE_UP                  = 0.25
BRK_RATE_DOWN                = 0.12

# 접근 바이어스
APPROACH_REL_GAIN            = 0.06
TTC_SLOW_START               = 7.0
TTC_BIAS_GAIN                = 0.05

def acc_longitudinal_control(ego_speed, lead_speed, distance, target_speed_mps):
    desired_gap = MIN_GAP + TIME_HEADWAY * ego_speed
    gap_error = (distance if distance is not None else float('inf')) - desired_gap
    if lead_speed is not None:
        speed_err = (lead_speed - ego_speed)
    else:
        speed_err = (target_speed_mps - ego_speed)
    accel_cmd = KP_GAP * gap_error + KV_SPEED * speed_err
    accel_cmd = max(-MAX_DECEL, min(MAX_ACCEL, float(accel_cmd)))
    if accel_cmd >= 0:
        throttle, brake = min(1.0, THR_GAIN * accel_cmd), 0.0
    else:
        throttle, brake = 0.0, min(1.0, BRK_GAIN * abs(accel_cmd))
    return float(throttle), float(brake)

def rate_limit(prev, desired, max_up, max_down):
    delta = desired - prev
    if delta >  max_up:   desired = prev + max_up
    if delta < -max_down: desired = prev - max_down
    return float(np.clip(desired, 0.0, 1.0))

def find_by_role(world, role='ego'):
    for v in world.get_actors().filter('vehicle.*'):
        if v.attributes.get('role_name','') == role:
            return v
    return None

def find_vehicle_from_camera_parent(world):
    for s in world.get_actors().filter('sensor.camera.rgb'):
        p = s.parent
        if p is not None and 'vehicle.' in p.type_id:
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--role', default='ego')
    ap.add_argument('--deadband', type=int, default=8)
    ap.add_argument('--apply_hz', type=float, default=20.0)
    # ★ MOD: 코너 보수성 파라미터
    ap.add_argument('--corner_ttc_bias', type=float, default=1.0,
                    help='|steer|=1.0일 때 FOLLOW_ENTER_TTC를 추가로 낮출 값(초)')  # ★ MOD
    ap.add_argument('--corner_dist_scale', type=float, default=0.8,
                    help='|steer|=1.0일 때 FOLLOW_MAX_DIST_M 배수를 적용 (<=1.0)')   # ★ MOD
    ap.add_argument('--relv_min_for_follow', type=float, default=0.5,
                    help='FOLLOW 진입 시 요구되는 최소 접근 속도(m/s)')               # ★ MOD
    args = ap.parse_args()

    # ---- CARLA ----
    client = carla.Client(args.host, args.port); client.set_timeout(5.0)
    world  = client.get_world()

    ego = None
    t_end = time.perf_counter() + 10.0
    while ego is None and time.perf_counter() < t_end:
        ego = find_by_role(world, args.role)
        if ego: break
        ego = find_vehicle_from_camera_parent(world)
        if ego: break
        time.sleep(0.2)
    if ego is None:
        vehicles = [(v.id, v.type_id, v.attributes.get('role_name','')) for v in world.get_actors().filter('vehicle.*')]
        raise RuntimeError(
            "[ERR] ego 차량을 찾지 못했습니다. perception 실행/서버 확인 필요.\n"
            f"현재 vehicles: {vehicles}"
        )
    try:
        ego.set_autopilot(False)
    except: pass

    # ---- Zenoh ----
    z = zenoh.open({})
    q_lk = Queue(maxsize=200)
    def _on_lk(s):
        try: q_lk.put_nowait(s)
        except: pass
    sub_lk = z.declare_subscriber('demo/lk/features', _on_lk)

    q_acc = Queue(maxsize=200)
    def _on_acc(s):
        try: q_acc.put_nowait(s)
        except: pass
    sub_acc = z.declare_subscriber('demo/acc/features', _on_acc)

    pub_ctrl = z.declare_publisher('demo/lk/ctrl')

    # ---- 상태 ----
    last_w = None
    last_x_mid = None

    acc_dist = None
    acc_rel_v = 0.0
    acc_ttc = float('inf')
    acc_lead_speed_est = None
    acc_has_target = False
    target_stable_frames = 0

    current_mode = "CRUISE"
    last_mode_change_time = -1e9
    safe_stop_locked = False

    last_thr = 0.0
    last_brk = 0.0

    period = 1.0 / max(1e-3, args.apply_hz)
    next_t = time.perf_counter() + period
    log_next = time.perf_counter() + 1.0
    sim_time = 0.0

    print(f"[INFO] Decision controlling id={ego.id}, role={ego.attributes.get('role_name','')}")
    print("[INFO] Decision running... Ctrl+C to stop.")

    try:
        while True:
            # 최신 피처 수신
            try:
                while True:
                    s = q_lk.get_nowait()
                    feat = json.loads(bytes(s.payload).decode('utf-8'))
                    last_w = feat.get('w', last_w)
                    last_x_mid = feat.get('x_lane_mid', last_x_mid)
            except Empty:
                pass
            try:
                while True:
                    s = q_acc.get_nowait()
                    feat = json.loads(bytes(s.payload).decode('utf-8'))
                    d   = feat.get('distance', None)
                    acc_dist = float(d) if d is not None else None
                    acc_rel_v = float(feat.get('rel_speed', acc_rel_v))
                    ttc = feat.get('ttc', None)
                    acc_ttc = float(ttc) if ttc is not None else float('inf')
                    ls = feat.get('lead_speed_est', None)
                    acc_lead_speed_est = float(ls) if ls is not None else None
                    acc_has_target = bool(feat.get('has_target', False))
            except Empty:
                pass

            # 연속 확인
            if acc_has_target and (acc_dist is not None) and np.isfinite(acc_dist):
                target_stable_frames = min(1000, target_stable_frames + 1)
            else:
                target_stable_frames = 0

            # 페이싱
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period
            sim_time += period

            # ---- LKAS ----
            v_mps = speed_of(ego)
            kp, clip = gains_for_speed(v_mps)
            steer = 0.0
            if (last_w is not None) and (last_x_mid is not None):
                err_px = (last_w / 2.0) - float(last_x_mid)
                if abs(err_px) < args.deadband:
                    err_px = 0.0
                steer = float(np.clip(kp * -err_px, -clip, clip))

            # ---- ACC 상태기계 ----
            throttle_des = 0.0
            brake_des = 0.0
            distance = float(acc_dist) if (acc_dist is not None) else float('inf')
            ttc      = float(acc_ttc) if np.isfinite(acc_ttc) else float('inf')

            # ★ MOD: 코너 보수성 - 조향 비율(|steer| 0~1)
            steer_abs = float(min(1.0, abs(steer)))
            follow_enter_ttc_dyn = max(0.5, FOLLOW_ENTER_TTC - args.corner_ttc_bias * steer_abs)  # ★ MOD
            follow_exit_ttc_dyn  = max(follow_enter_ttc_dyn + 1.0, FOLLOW_EXIT_TTC - 0.5 * steer_abs)  # ★ MOD
            follow_max_dist_dyn  = FOLLOW_MAX_DIST_M * (1.0 - (1.0 - args.corner_dist_scale) * steer_abs)  # ★ MOD

            # ★ MOD: FOLLOW 표적 게이팅 강화 (접근 중 + 거리/ttc 기준)
            target_ready = (
                (target_stable_frames >= MIN_CONFIRM_FRAMES)
                and (distance < follow_max_dist_dyn)
                and np.isfinite(acc_ttc)
                and (acc_ttc < follow_enter_ttc_dyn)
                and (acc_rel_v > args.relv_min_for_follow)  # 접근 중일 때만 FOLLOW 진입
            )

            tsc = sim_time - last_mode_change_time
            if current_mode == "CRUISE":
                if target_ready and (tsc > MODE_COOLDOWN_SEC):
                    current_mode, last_mode_change_time = "FOLLOW", sim_time
                if target_ready and (ttc < AEB_ENTER_TTC) and (v_mps > AEB_ACTIVATION_SPEED_THRESHOLD) and (tsc > MODE_COOLDOWN_SEC):
                    current_mode, last_mode_change_time, safe_stop_locked = "AEB", sim_time, False

            elif current_mode == "FOLLOW":
                if target_ready and (ttc < AEB_ENTER_TTC) and (v_mps > AEB_ACTIVATION_SPEED_THRESHOLD) and (tsc > MODE_COOLDOWN_SEC):
                    current_mode, last_mode_change_time, safe_stop_locked = "AEB", sim_time, False
                # 유지 조건(히스테리시스 + 거리 유지), target_ready 해제 시 CRUISE 복귀
                elif ((not target_ready) or (ttc >= follow_exit_ttc_dyn) or (distance == float('inf'))) and (tsc > MODE_COOLDOWN_SEC):
                    current_mode, last_mode_change_time = "CRUISE", sim_time

            elif current_mode == "AEB":
                if (not safe_stop_locked) and (distance <= 6.0):
                    safe_stop_locked = True
                if safe_stop_locked:
                    throttle_des, brake_des = 0.0, 1.0
                else:
                    throttle_des, brake_des = 0.0, min(1.0, max(0.0, distance / max(1e-3, 6.0)))
                if (sim_time - last_mode_change_time) >= AEB_MIN_HOLD_SEC and (ttc > AEB_EXIT_TTC):
                    current_mode = "FOLLOW" if target_ready else "CRUISE"
                    last_mode_change_time = sim_time
                    safe_stop_locked = False

            # FOLLOW/CRUISE 제어
            if current_mode != "AEB":
                lead_speed_for_acc = None
                if target_ready:
                    # rel_v(+접근) → lead 절대속도 = ego - rel_v
                    lead_speed_for_acc = max(0.0, v_mps - float(acc_rel_v))
                throttle_des, brake_des = acc_longitudinal_control(
                    v_mps, lead_speed_for_acc, distance, TARGET_SPEED_MPS
                )

                # 접근 바이어스(FOLLOW일 때만)
                if current_mode == "FOLLOW":
                    if acc_rel_v > 0.0:
                        brake_des    = min(1.0, brake_des + APPROACH_REL_GAIN * acc_rel_v)
                        throttle_des = max(0.0, throttle_des - APPROACH_REL_GAIN * acc_rel_v)
                    if np.isfinite(ttc) and ttc < TTC_SLOW_START:
                        bias = TTC_BIAS_GAIN * (TTC_SLOW_START - ttc)
                        throttle_des = max(0.0, throttle_des - bias)
                        brake_des    = min(1.0, brake_des + bias)

                # 크루즈 부스트: 타깃 없으면 80km/h로 적극 가속
                if (not target_ready):
                    speed_gap = max(0.0, TARGET_SPEED_MPS - v_mps)
                    boost = np.clip(0.12 + 0.06 * speed_gap, 0.12, 0.60)
                    throttle_des = max(throttle_des, float(boost))
                    brake_des = 0.0

            # 레이트 리미팅
            throttle_cmd = rate_limit(last_thr, throttle_des, THR_RATE_UP, THR_RATE_DOWN)
            brake_cmd    = rate_limit(last_brk, brake_des, BRK_RATE_UP, BRK_RATE_DOWN)

            # 적용
            ego.apply_control(carla.VehicleControl(
                throttle=float(np.clip(throttle_cmd, 0.0, 1.0)),
                brake=float(np.clip(brake_cmd, 0.0, 1.0)),
                steer=float(np.clip(steer, -1.0, 1.0)),
                hand_brake=False, manual_gear_shift=False
            ))
            last_thr, last_brk = throttle_cmd, brake_cmd

            # HUD (+ 디버그)
            ctrl_msg = {
                "throttle": float(throttle_cmd),
                "brake": float(brake_cmd),
                "steer": float(steer),
                "mode": current_mode,
                "ts": time.time(),
                # ★ MOD: 디버그 주입
                "dbg_follow_enter_ttc": follow_enter_ttc_dyn,
                "dbg_follow_exit_ttc": follow_exit_ttc_dyn,
                "dbg_follow_max_dist": follow_max_dist_dyn,
                "dbg_relv": acc_rel_v
            }
            pub_ctrl.put(json.dumps(ctrl_msg).encode('utf-8'))

            if now >= log_next:
                print(f"[DBG] mode={current_mode:6s}  v={v_mps*3.6:5.1f} km/h  "
                      f"tgt_ready={int(target_ready)}  dist={(distance if np.isfinite(distance) else -1):5.1f}  "
                      f"rel={float(acc_rel_v):+4.1f}  ttc={(ttc if np.isfinite(ttc) else -1):4.1f}  "
                      f"thr={throttle_cmd:.2f}  brk={brake_cmd:.2f}  steer={steer:+.3f}  "
                      f"fe_ttc={follow_enter_ttc_dyn:.1f} fx_ttc={follow_exit_ttc_dyn:.1f} "
                      f"fmax={follow_max_dist_dyn:.1f}")
                log_next = now + 1.0

    finally:
        try: sub_lk.undeclare()
        except: pass
        try: sub_acc.undeclare()
        except: pass
        try: pub_ctrl.undeclare()
        except: pass
        try: z.close()
        except: pass
        print('[INFO] Decision stopped.')

if __name__ == '__main__':
    main()

