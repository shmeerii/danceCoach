"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import (
    ALLOWED_LIVE_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_LIVE_FRAME_UPLOAD_BYTES,
    MAX_LIVE_FRAME_UPLOAD_MB,
)
from app.services.pose_extractor import extract_pose_from_video
from app.services.reference_catalog import list_saved_references
from app.services.single_frame_pose_extractor import extract_pose_from_image_bytes
from app.services.video_validator import validate_video
from app.utils.file_utils import (
    build_reference_video_filename,
    build_upload_path,
    ensure_runtime_directories,
    media_type_for_video_suffix,
    resolve_output_file,
    resolve_reference_media_file,
    save_reference_video_copy,
    write_bytes,
)
from app.utils.reference_metadata import attach_reference_media_to_output_json

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


_configure_logging()

app = FastAPI(
    title="Dance Pose Extraction API",
    description="Local backend for reference pose landmark extraction from dance videos.",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_runtime_directories()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Reference video extraction (stable). Do not modify this for live camera mode
# unless updating shared pose schema intentionally.
@app.post("/extract-pose")
async def extract_pose(file: UploadFile = File(...)) -> JSONResponse:
    """
    Extract pose landmarks from an uploaded dance video.

    Request:
      - multipart/form-data
      - one file field named `file`
    """
    upload_path: Path | None = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file was uploaded.")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video type '{suffix}'. Allowed: {allowed}.",
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {max_mb} MB.",
            )

        upload_path = build_upload_path(file.filename)
        write_bytes(upload_path, content)

        # Validate before running pose extraction.
        validation = validate_video(
            upload_path,
            file_size_bytes=len(content),
        )
        if not validation.is_valid:
            # User-friendly + structured validation details.
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Video validation failed. Please upload a clearer reference video.",
                    **validation.to_dict(),
                },
            )

        reference_id = uuid.uuid4().hex
        output_filename = f"{reference_id}_poses.json"
        extraction = extract_pose_from_video(
            upload_path,
            output_basename=output_filename,
        )

        if not extraction.success or not extraction.output_path:
            # Log technical errors, but keep the user-facing message simple.
            if extraction.errors:
                logger.error("Pose extraction errors: %s", extraction.errors)
            raise HTTPException(
                status_code=500,
                detail="Pose extraction failed for this video. Try a different recording.",
            )

        video_metadata = extraction.metadata.get("video_metadata", {}) or {}
        quality = extraction.quality_metrics or {}
        duration_seconds = float(video_metadata.get("duration_seconds") or 0.0)

        try:
            reference_video_filename = build_reference_video_filename(
                reference_id,
                suffix,
            )
            save_reference_video_copy(upload_path, reference_video_filename)
            attach_reference_media_to_output_json(
                extraction.output_path,
                reference_id=reference_id,
                reference_video_filename=reference_video_filename,
            )
        except Exception as exc:
            logger.exception(
                "Could not save reference video for reference_id=%s",
                reference_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Pose extraction succeeded but saving the reference video failed.",
            ) from exc

        summary = {
            "processed_frames": int(video_metadata.get("processed_frames", 0)),
            "pose_detected_frames": int(
                video_metadata.get("pose_detected_frames", 0)
            ),
            "pose_detection_percentage": float(
                quality.get("pose_detection_percentage", 0.0)
            ),
            "reference_quality": str(quality.get("reference_quality", "unknown")),
        }

        # Keep `output_path` as a stable API URL for the frontend to fetch.
        output_path = f"/outputs/{output_filename}"
        reference_video_url = f"/reference-media/{reference_video_filename}"

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "output_filename": output_filename,
                "output_path": output_path,
                "video_metadata": video_metadata,
                "quality": quality,
                "summary": summary,
                "warnings": extraction.warnings,
                "reference_id": reference_id,
                "reference_video_filename": reference_video_filename,
                "reference_video_url": reference_video_url,
                "duration_seconds": duration_seconds,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Technical error in POST /extract-pose")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your video. Please try again.",
        ) from exc
    finally:
        # Always delete temporary uploaded videos.
        if upload_path is not None:
            try:
                if upload_path.is_file():
                    upload_path.unlink()
            except Exception:
                logger.exception("Could not delete temporary upload: %s", upload_path)


@app.get("/references")
def list_references() -> dict[str, list[dict[str, object]]]:
    """
    List extracted references from saved pose JSON files in outputs/.

    Older JSON files without reference_media metadata are included with
    reference_video_url set to null.
    """
    return {"references": list_saved_references()}


@app.get("/outputs/{filename}")
def get_output(filename: str) -> FileResponse:
    """
    Serve generated JSON outputs from backend/outputs/.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(filename).suffix.lower()
    if suffix != ".json":
        raise HTTPException(status_code=400, detail="Only JSON files are allowed.")

    try:
        output_path = resolve_output_file(filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")

    return FileResponse(
        path=output_path,
        media_type="application/json",
        filename=output_path.name,
    )


@app.post("/extract-live-pose-frame")
async def extract_live_pose_frame(file: UploadFile = File(...)) -> JSONResponse:
    """
    Extract pose landmarks from a single camera snapshot (Practice Mode).

    Request:
      - multipart/form-data
      - one file field named `file` (image/jpeg or image/png)

    Does not save the image to disk.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file was uploaded.")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_LIVE_IMAGE_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_LIVE_IMAGE_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type '{suffix}'. Allowed: {allowed}.",
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > MAX_LIVE_FRAME_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Live frame exceeds maximum size of "
                    f"{MAX_LIVE_FRAME_UPLOAD_MB} MB."
                ),
            )

        result = extract_pose_from_image_bytes(content)
        return JSONResponse(status_code=200, content=result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Technical error in POST /extract-live-pose-frame")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing the camera frame.",
        ) from exc


@app.get("/reference-media/{filename}")
def get_reference_media(filename: str) -> FileResponse:
    """
    Serve saved reference videos from backend/reference_media/ for playback.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    try:
        media_path = resolve_reference_media_file(filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="Reference video not found.")

    suffix = media_path.suffix.lower()
    return FileResponse(
        path=media_path,
        media_type=media_type_for_video_suffix(suffix),
        filename=media_path.name,
    )
