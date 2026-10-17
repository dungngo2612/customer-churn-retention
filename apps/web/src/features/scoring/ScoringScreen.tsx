import { useEffect, useMemo, useState } from "react";

import { fetchScoringRun } from "./api";
import { demoScoringRun } from "./demo";
import type { PredictionBatch } from "./types";
import "./scoring.css";

export interface ScoringScreenProps {
  runId: string;
  useDemoData?: boolean;
}

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; batch: PredictionBatch; isDemo: boolean };

const probabilityFormatter = new Intl.NumberFormat("vi-VN", {
  style: "percent",
  maximumFractionDigits: 1,
});

const dateFormatter = new Intl.DateTimeFormat("vi-VN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  timeZoneName: "short",
});

export function ScoringScreen({
  runId,
  useDemoData = false,
}: ScoringScreenProps) {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [minimumPercent, setMinimumPercent] = useState(0);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [reloadCount, setReloadCount] = useState(0);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setState({ status: "loading" });

    async function loadRun(): Promise<void> {
      try {
        const batch = useDemoData
          ? demoScoringRun
          : await fetchScoringRun(runId, controller.signal);
        if (active) setState({ status: "ready", batch, isDemo: useDemoData });
      } catch (error) {
        if (active && !controller.signal.aborted) {
          setState({
            status: "error",
            message:
              error instanceof Error
                ? error.message
                : "Không tải được kết quả chấm điểm.",
          });
        }
      }
    }

    void loadRun();
    return () => {
      active = false;
      controller.abort();
    };
  }, [runId, useDemoData, reloadCount]);

  const visiblePredictions = useMemo(() => {
    if (state.status !== "ready") return [];
    return state.batch.predictions
      .filter((prediction) => prediction.churn_probability >= minimumPercent / 100)
      .sort((left, right) =>
        sortOrder === "desc"
          ? right.churn_probability - left.churn_probability
          : left.churn_probability - right.churn_probability,
      );
  }, [state, minimumPercent, sortOrder]);

  const isDemo = state.status === "ready" ? state.isDemo : useDemoData;

  return (
    <section className="scoring-screen" aria-busy={state.status === "loading"}>
      <header className="scoring-screen__header">
        <h1>Danh sách chấm điểm churn</h1>
        <p>Xem nguy cơ khách hàng rời bỏ và lọc theo ngưỡng xác suất.</p>
      </header>

      {isDemo && (
        <p className="scoring-screen__demo">
          <strong>DEMO</strong> — Dữ liệu giả, không phải kết quả dự đoán thực tế.
        </p>
      )}

      {state.status === "loading" && (
        <p className="scoring-screen__state" role="status">
          Đang tải kết quả chấm điểm…
        </p>
      )}

      {state.status === "error" && (
        <div className="scoring-screen__state scoring-screen__state--error" role="alert">
          <p>{state.message}</p>
          <button type="button" onClick={() => setReloadCount((count) => count + 1)}>
            Thử lại
          </button>
        </div>
      )}

      {state.status === "ready" && (
        <>
          <p className="scoring-screen__run">Lần chấm điểm: {state.batch.scoring_run_id}</p>

          <div className="scoring-screen__controls">
            <label>
              Sắp xếp xác suất
              <select
                value={sortOrder}
                onChange={(event) =>
                  setSortOrder(event.currentTarget.value === "asc" ? "asc" : "desc")
                }
                disabled={state.batch.predictions.length === 0}
              >
                <option value="desc">Cao → thấp</option>
                <option value="asc">Thấp → cao</option>
              </select>
            </label>
            <label>
              Ngưỡng tối thiểu: {minimumPercent}%
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={minimumPercent}
                onChange={(event) => setMinimumPercent(Number(event.currentTarget.value))}
                aria-valuetext={`${minimumPercent}%`}
                disabled={state.batch.predictions.length === 0}
              />
            </label>
          </div>

          <p className="scoring-screen__count" aria-live="polite">
            Hiển thị {visiblePredictions.length} / {state.batch.predictions.length} khách hàng
          </p>

          {visiblePredictions.length === 0 ? (
            <p className="scoring-screen__state" role="status">
              {state.batch.predictions.length === 0
                ? "Lần chấm điểm này chưa có khách hàng."
                : `Không có khách hàng đạt ngưỡng ${minimumPercent}%. Hạ ngưỡng để xem thêm.`}
            </p>
          ) : (
            <div className="scoring-screen__table-wrap">
              <table>
                <caption>Thời gian hiển thị theo múi giờ của thiết bị.</caption>
                <thead>
                  <tr>
                    <th scope="col">Mã khách hàng</th>
                    <th scope="col" aria-sort={sortOrder === "desc" ? "descending" : "ascending"}>
                      Xác suất churn
                    </th>
                    <th scope="col">Phiên bản mô hình</th>
                    <th scope="col">Thời điểm chấm điểm</th>
                  </tr>
                </thead>
                <tbody>
                  {visiblePredictions.map((prediction) => (
                    <tr key={prediction.customer_id}>
                      <td>{prediction.customer_id}</td>
                      <td className="scoring-screen__probability">
                        {probabilityFormatter.format(prediction.churn_probability)}
                      </td>
                      <td>{prediction.model_version}</td>
                      <td>
                        <time dateTime={prediction.scored_at}>
                          {dateFormatter.format(new Date(prediction.scored_at))}
                        </time>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
