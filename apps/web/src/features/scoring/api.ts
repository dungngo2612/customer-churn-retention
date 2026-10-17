import type { PredictionBatch } from "./types";

export async function fetchScoringRun(
  id: string,
  signal?: AbortSignal,
): Promise<PredictionBatch> {
  if (!id.trim()) {
    throw new Error("Thiếu mã lần chấm điểm.");
  }

  let response: Response;
  try {
    response = await fetch(
      `/scoring/runs/${encodeURIComponent(id)}`,
      signal ? { signal } : {},
    );
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error("Không kết nối được đến API chấm điểm.");
  }

  if (!response.ok) {
    throw new Error(
      response.status === 404
        ? `Không tìm thấy lần chấm điểm "${id}".`
        : "Không tải được kết quả chấm điểm. Vui lòng thử lại.",
    );
  }

  try {
    return (await response.json()) as PredictionBatch;
  } catch {
    throw new Error("Dữ liệu trả về không đọc được. Vui lòng thử lại.");
  }
}
