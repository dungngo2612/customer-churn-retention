from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from churn_retention.adapters.database.dataset_repository import DatasetRepository
from churn_retention.adapters.database.session import get_db
from churn_retention.modules.datasets import use_cases

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", status_code=status.HTTP_201_CREATED,)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db),) -> Dict[str, Any]:
    """
    Endpoint tải lên file CSV dataset.
    - Trả về 201 nếu file hợp lệ và gán dataset_id thành công.
    - Trả về 422 kèm danh sách chi tiết (row, column, reason) nếu vi phạm quy tắc dữ liệu.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hệ thống chỉ hỗ trợ file định dạng .csv",
        )

    # Đọc toàn bộ nội dung file tải lên dưới dạng bytes
    content = await file.read()
    repo = DatasetRepository(db)
    result = use_cases.process_dataset_upload(file_bytes=content, filename=file.filename, dataset_repo=repo)

    # Nếu dữ liệu vi phạm quy tắc kiểm định
    if result["status"] == "invalid":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Dữ liệu không vượt qua kiểm định",
                "total_rows": result["total_rows"],
                "errors": result["errors"],
            },
        )

    # Nếu lỗi do định dạng file CSV hỏng không đọc được
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )

    return result


@router.get("", response_model=List[Dict[str, Any]])
def list_all_datasets():
    """Lấy danh sách tóm tắt toàn bộ các dataset đã upload."""
    return use_cases.list_datasets()


@router.get("/{dataset_id}")
def get_dataset_by_id(dataset_id: str):
    """Tra cứu chi tiết một dataset theo dataset_id."""
    data = use_cases.get_dataset_detail(dataset_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy dataset với mã '{dataset_id}'",
        )
    return data
