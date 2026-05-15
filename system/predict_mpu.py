import serial
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from model import model

ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(2)

_model = model()


def update_chart(data_history, ax1, ax2, canvas):
    ax1.clear()
    ax2.clear()

    if not data_history:
        return np.zeros((700, 1200, 3), dtype=np.uint8)

    scores = np.array([d["score"] for d in data_history])
    preds  = np.array([d["pred"]  for d in data_history])
    rms    = np.array([d["rms"]   for d in data_history])
    x      = np.arange(len(scores))

    sehat   = preds == 1
    anomali = preds == -1

    # ── Panel 1: Anomaly Score ────────────────────────────────────────────────
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Anomaly boundary")

    # Coloured background fills between consecutive windows
    for i in range(1, len(scores)):
        color = "#e74c3c" if preds[i] == -1 else "#2ecc71"
        ax1.fill_between(x[i - 1:i + 1], scores[i - 1:i + 1], 0,
                         color=color, alpha=0.15)

    # Line segments coloured by prediction
    for i in range(1, len(scores)):
        seg_color = "#e74c3c" if preds[i] == -1 else "#27ae60"
        ax1.plot(x[i - 1:i + 1], scores[i - 1:i + 1],
                 color=seg_color, linewidth=2.5)

    ax1.scatter(x[sehat],   scores[sehat],   s=50,  color="#27ae60",
                edgecolors="white", linewidth=1, zorder=5, label="Healthy")
    ax1.scatter(x[anomali], scores[anomali], s=80,  color="#e74c3c",
                marker="X", edgecolors="white", linewidth=1, zorder=5, label="Anomaly")

    status_color = "#e74c3c" if preds[-1] == -1 else "#27ae60"
    status_text  = "⚠ ANOMALY DETECTED" if preds[-1] == -1 else "✔ MOTOR HEALTHY"

    ax1.set_title(f"Anomaly Score  —  {status_text}",
                  fontsize=12, fontweight="bold", color=status_color)
    ax1.set_ylabel("Anomaly Score")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # ── Panel 2: RMS ──────────────────────────────────────────────────────────
    ax2.plot(x, rms, linewidth=1.2, alpha=0.6, color="#3498db", label="RMS vibration")

    ax2.scatter(x[sehat],   rms[sehat],   s=50, color="#27ae60",
                edgecolors="white", linewidth=1, zorder=5, label="Healthy")
    ax2.scatter(x[anomali], rms[anomali], s=80, color="#e74c3c",
                marker="X", edgecolors="white", linewidth=1, zorder=5, label="Anomaly")

    ax2.set_title("RMS Vibration per Window", fontsize=12)
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("RMS")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.4)

    canvas.draw()
    img_rgba = np.array(canvas.buffer_rgba())
    return cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)


# ── State ─────────────────────────────────────────────────────────────────────
temp      = []   # raw sensor readings for the current 1000-sample window
data_list = []   # rolling history of prediction results (max 100 windows)
counter   = 0

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=100, sharex=True)
fig.suptitle("Predictive Maintenance — Motor Vibration Monitor",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
canvas = FigureCanvasAgg(fig)

cv2.namedWindow("Monitor", cv2.WINDOW_AUTOSIZE)

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        mpu_line = ser.readline().decode(errors="ignore").strip()
        try:
            data = json.loads(mpu_line)

            if len(temp) < 1001:
                temp.append(data)
                counter += 1
                print(counter, data)

            else:
                # ── Extract features & predict ────────────────────────────
                df    = pd.DataFrame(temp)
                feats = _model.extract_feature(df)
                print("Features:\n", feats)

                X     = _model.scaler.transform(feats)
                pred  = _model.model.predict(X)
                score = _model.model.decision_function(X)
                
                print(pred)
                
                # RMS from feature column index 2 (adjust if needed)
                rms_value = float(X[:, 2].mean())
                
                for p,s in zip(pred,score):
                    result = {
                        "pred":  int(p),
                        "score": float(s),
                        "rms":   rms_value,
                    }
                    print(f"pred: {result['pred']}  score: {result['score']:.4f}"
                        f"  rms: {result['rms']:.4f}")

                    data_list.append(result)
                    if len(data_list) > 100:
                        data_list.pop(0)

                # ── Reset window ──────────────────────────────────────────
                counter = 0
                temp.clear()

                # ── Render chart ──────────────────────────────────────────
                chart = update_chart(data_list, ax1, ax2, canvas)
                cv2.imshow("Monitor", chart)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        except json.JSONDecodeError:
            pass

except KeyboardInterrupt:
    pass

finally:
    ser.close()
    cv2.destroyAllWindows()
    plt.close(fig)