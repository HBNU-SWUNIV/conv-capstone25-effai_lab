#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU anomaly-detection GUI for Raspberry Pi (Streamlined & Enhanced Version)
"""
import os
import time
import threading
import queue
import serial
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from collections import deque
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime

# ─── 설정 ─────────────────────────────────────────────────────────
SERIAL_PORT      = '/dev/ttyACM0'
BAUD_RATE        = 9600
COLLECTION_SEC   = 60
DATA_FILE        = 'training_data.csv'
MODEL_FILE       = 'autoencoder.pth'
SCALER_FILE      = 'scaler.joblib'
THRESHOLD_FILE   = 'threshold.txt'
EPOCHS           = 100
BATCH_SIZE       = 32
LR               = 0.001
INPUT_DIM        = 6
SEQ_LEN          = 20

# ─── LSTM Autoencoder 정의 ────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=16, latent_dim=8):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.enc_fc       = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.dec_fc       = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.seq_len = seq_len

    def forward(self, x):
        _, (h, _) = self.encoder_lstm(x)
        z      = self.enc_fc(h[-1])
        d_in   = self.dec_fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(d_in)
        return out

# ─── 시계열 시퀀스 생성 ────────────────────────────────────────────
def create_sequences(arr, seq_len):
    return np.array([arr[i:i+seq_len] for i in range(len(arr)-seq_len+1)])

# ─── 백엔드 작업 스레드 함수들 ─────────────────────────────────────
def collect_data(progress_cb, log_cb):
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    log_cb(f"[{timestamp}][수집] {COLLECTION_SEC}초간 데이터 수집 시작")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        log_cb(f"[수집 오류] 시리얼 연결 실패: {e}")
        return False

    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    with open(DATA_FILE, 'w') as f:
        start = time.time()
        header = "timestamp_ms,ax,ay,az,gx,gy,gz"
        f.write(header + '\n')
        ser.readline()

        while True:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                if line and len(line.split(',')) >= 7:
                    f.write(line + '\n')
            elapsed = time.time() - start
            pct = int((elapsed / COLLECTION_SEC) * 100)
            progress_cb(min(pct, 100))
            if elapsed >= COLLECTION_SEC:
                break
    ser.close()
    progress_cb(100)
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    log_cb(f"[{timestamp}][수집] 완료")
    return True

def train_model(progress_cb, log_cb):
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    log_cb(f"\n[{timestamp}][학습] CSV 로드 및 학습 시작")
    if not os.path.exists(DATA_FILE):
        log_cb(f"[학습 오류] 데이터 파일({DATA_FILE})이 없습니다.")
        return False

    df = pd.read_csv(DATA_FILE)
    if df.shape[0] < SEQ_LEN:
        log_cb(f"[학습 오류] 학습 데이터가 너무 적습니다 ({df.shape[0]}개). 최소 {SEQ_LEN}개가 필요합니다.")
        return False
        
    X  = df.drop('timestamp_ms', axis=1)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    seqs = create_sequences(Xs, SEQ_LEN)
    ds   = TensorDataset(torch.FloatTensor(seqs))
    dl   = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = LSTMAutoencoder(INPUT_DIM, SEQ_LEN)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total = 0
        for (batch,) in dl:
            out   = model(batch)
            loss  = crit(out, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        pct = int(((epoch+1)/EPOCHS)*100)
        progress_cb(pct)
        if (epoch+1) % 10 == 0:
            log_cb(f"[학습] Epoch {epoch+1}/{EPOCHS}, Loss {total/len(dl):.6f}")
    torch.save(model.state_dict(), MODEL_FILE)
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    log_cb(f"[{timestamp}][학습] 모델 저장")

    model.eval()
    recon_err = []
    with torch.no_grad():
        for (batch,) in DataLoader(ds, batch_size=BATCH_SIZE):
            rec = model(batch)
            err = torch.mean((rec-batch)**2, dim=[1,2]).numpy()
            recon_err.extend(err)
    thr = np.mean(recon_err) + 3*np.std(recon_err)
    with open(THRESHOLD_FILE, 'w') as f:
        f.write(str(thr))
    progress_cb(100)
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    log_cb(f"[{timestamp}][학습] 임계값={thr:.6f} 저장, 완료")
    return True

# [수정] detect_realtime 함수: status_cb 콜백 추가 및 로깅 로직 변경
def detect_realtime(stop_evt, log_cb, status_cb):
    try:
        model = LSTMAutoencoder(INPUT_DIM, SEQ_LEN)
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        scaler = joblib.load(SCALER_FILE)
        with open(THRESHOLD_FILE) as f:
            thr = float(f.read())
    except FileNotFoundError as e:
        log_cb(f"\n[탐지 오류] 모델/스케일러/임계값 파일 로드 실패: {e}")
        status_cb('standby') # [추가] 오류 시 상태를 '대기'로 변경
        return
        
    log_cb("\n[탐지] 모델·스케일러·임계값 로드 완료. 탐지를 시작합니다.")
    status_cb('normal') # [추가] 탐지 시작 시 '정상' 상태로 변경

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        ser.flushInput(); ser.readline()
    except serial.SerialException as e:
        log_cb(f"[탐지 오류] 시리얼 연결 실패: {e}")
        status_cb('standby') # [추가] 오류 시 상태를 '대기'로 변경
        return

    buf = deque(maxlen=SEQ_LEN)
    last_anomaly_log_time = 0 # [추가] 이상 로그 시간 제어를 위한 변수
    anomaly_counter = 0 # [추가] 이상 신호 순서 카운터

    while not stop_evt.is_set():
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                row = {
                    'ax': float(parts[1]), 'ay': float(parts[2]), 'az': float(parts[3]),
                    'gx': float(parts[4]), 'gy': float(parts[5]), 'gz': float(parts[6])
                }
            except (ValueError, IndexError):
                continue
            
            scaled = scaler.transform(pd.DataFrame([row])).flatten()
            buf.append(scaled)

            if len(buf) == SEQ_LEN:
                seq = torch.FloatTensor(np.array(buf)).unsqueeze(0)
                with torch.no_grad():
                    rec = model(seq)
                    err = torch.mean((rec-seq)**2).item()
                
                # [수정] 상태 콜백 호출 및 로그 출력 조건 변경
                if err > thr:
                    status_cb('anomaly') # GUI에 '이상' 상태 전송
                    current_time = time.time()
                    # 2초 간격으로 로그 출력
                    if current_time - last_anomaly_log_time > 2:
                        anomaly_counter += 1
                        timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
                        log_cb(f"\n▶{anomaly_counter} [이상신호 발생] {timestamp}\n"
                            f"   - Error     : {err:.6f}\n"
                            f"   - Threshold : {thr:.6f}\n"
                            f"{'-'*35}")
                        last_anomaly_log_time = current_time
                else:
                    status_cb('normal') # GUI에 '정상' 상태 전송
        
        time.sleep(0.01)
        
    ser.close()
    log_cb("[탐지] 중단됨")

# ─── GUI 클래스 ──────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IMU Sensor Anomaly Detection")
        self.option_add("*Font", ("TkFixedFont", 10))
        self.geometry("550x450") # [추가] 창 크기 고정

        # ── 위젯 ─────────────────────────────
        # [추가] 상태 표시 라벨
        self.status_indicator = tk.Label(self, text="", bg="gray", fg="white",
                                         font=("TkFixedFont", 12, "bold"), pady=5)
        self.status_indicator.pack(fill="x", padx=5, pady=5)

        frm_btn = tk.Frame(self)
        self.btn_start = ttk.Button(frm_btn, text="Start",
                                    command=self.start_process_clicked)
        self.btn_stop = ttk.Button(frm_btn, text="Stop",
                                   command=self.stop_process_clicked, state="disabled")
        
        self.btn_start.pack(side="left", padx=5, pady=5)
        self.btn_stop.pack(side="left", padx=5, pady=5)
        frm_btn.pack()

        self.progress = ttk.Progressbar(self, length=400, mode="determinate")
        self.progress.pack(pady=(0,5))
        self.lbl_pct   = tk.Label(self, text="0 %")
        self.lbl_pct.pack()

        self.txt_log = scrolledtext.ScrolledText(self, width=70, height=18, state="disabled")
        self.txt_log.pack(padx=5, pady=5)

        # ── 내부 상태 ────────────────────────
        self.queue = queue.Queue()
        self.after(100, self.poll_queue) # [수정] poll_log -> poll_queue
        self.process_thread = None
        self.stop_evt      = threading.Event()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ── 콜백 헬퍼 ───────────────────────────
    # [수정] 큐에 넣는 메시지를 튜플 형태로 변경 (타입, 내용)
    def log(self, msg):
        self.queue.put(('log', msg))

    def status_update_cb(self, status):
        self.queue.put(('status', status))

    def set_progress(self, pct):
        self.progress['value'] = pct
        self.lbl_pct.config(text=f"{pct} %")
        self.update_idletasks()
    
    def reset_progress(self):
        self.set_progress(0)

    # [수정] 큐에서 메시지를 꺼내 타입에 따라 처리하는 함수
    def poll_queue(self):
        while not self.queue.empty():
            msg_type, content = self.queue.get_nowait()
            if msg_type == 'log':
                self.txt_log.config(state="normal")
                self.txt_log.insert("end", content + "\n")
                self.txt_log.see("end")
                self.txt_log.config(state="disabled")
            elif msg_type == 'status':
                self.update_status_indicator(content)
        self.after(100, self.poll_queue)

    # [추가] 상태 표시 라벨의 색상과 텍스트를 변경하는 함수
    def update_status_indicator(self, status):
        if status == 'normal':
            self.status_indicator.config(text="NORMAL", bg="#4CAF50") # 초록색
        elif status == 'anomaly':
            self.status_indicator.config(text="ABNORMAL", bg="#F44336") # 빨간색
        else: # 'standby' 또는 기타
            self.status_indicator.config(text="", bg="gray")


    # ── 버튼 핸들러 ──────────────────
    def start_process_clicked(self):
        if self.process_thread and self.process_thread.is_alive():
            return
            
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.stop_evt.clear()
        
        self.process_thread = threading.Thread(
            target=self._run_full_process,
            daemon=True)
        self.process_thread.start()

    def stop_process_clicked(self):
        if self.process_thread and self.process_thread.is_alive():
            self.log("[중단] 사용자가 중단 요청")
            self.stop_evt.set()
        self.btn_stop.config(state="disabled")

    def _run_full_process(self):
        self.after(0, self.update_status_indicator, 'standby') # [추가] 시작 시 회색으로
        
        self.reset_progress()
        if not collect_data(self.set_progress, self.log):
            self.log("[오류] 데이터 수집 실패. 프로세스를 중단합니다.")
            self.after(0, self.reset_ui)
            return
        
        self.reset_progress()
        if not train_model(self.set_progress, self.log):
            self.log("[오류] 모델 학습 실패. 프로세스를 중단합니다.")
            self.after(0, self.reset_ui)
            return

        self.reset_progress()
        # [수정] detect_realtime 호출 시 status_update_cb 콜백 전달
        detect_realtime(self.stop_evt, self.log, self.status_update_cb)
        
        self.after(0, self.reset_ui)

    def reset_ui(self):
        self.log("\n[알림] 프로세스가 종료되었습니다. 다시 시작할 수 있습니다.")
        self.reset_progress()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.update_status_indicator('standby') # [추가] UI 리셋 시 상태 표시도 리셋

    def on_closing(self):
        if self.process_thread and self.process_thread.is_alive():
            self.stop_evt.set()
            self.process_thread.join(timeout=2)
        self.destroy()

# ─── 메인 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()