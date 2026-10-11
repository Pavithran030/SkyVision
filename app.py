"""SkyVision Modern Web Application and REST API Server."""

import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import config
from src.inference import PredictionService

app = FastAPI(
    title="SkyVision API",
    description="Sky Cloud Recognition & Rainfall Nowcasting Server",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = PredictionService()


@app.get("/api/status")
def get_status():
    """Return live model status."""
    return {
        "cloud_model_loaded": service.cloud_engine is not None,
        "rainfall_model_loaded": service.rainfall_classifier is not None,
        "backbone": "MobileNetV3-Small",
        "label_source": service.rainfall_metadata.get("label_source", "open_meteo_proxy_reanalysis"),
        "metadata": service.rainfall_metadata,
    }


@app.get("/api/samples")
def list_samples():
    """List sample benchmark sky images."""
    if not config.SAMPLES_DIR.exists():
        return {"samples": []}
    samples = []
    for cdir in sorted(config.SAMPLES_DIR.iterdir()):
        if cdir.is_dir():
            for f in sorted(cdir.glob("*.jpg"))[:6]:
                samples.append({
                    "filename": f.name,
                    "class_name": cdir.name,
                    "relative_path": f"{cdir.name}/{f.name}",
                    "url": f"/api/sample-image/{cdir.name}/{f.name}",
                })
    return {"samples": samples}


@app.get("/api/sample-image/{class_name}/{filename}")
def get_sample_image(class_name: str, filename: str):
    """Serve a sample image."""
    img_path = config.SAMPLES_DIR / class_name / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))


@app.post("/api/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_relative_path: Optional[str] = Form(None),
    temperature_2m: Optional[float] = Form(28.0),
    relative_humidity_2m: Optional[float] = Form(75.0),
    surface_pressure: Optional[float] = Form(1008.0),
    wind_speed_10m: Optional[float] = Form(14.0),
):
    """Run computer vision and precipitation nowcasting on an image."""
    image_bytes = None
    if file and file.filename:
        image_bytes = await file.read()
    elif sample_relative_path:
        fpath = config.SAMPLES_DIR / sample_relative_path
        if not fpath.exists():
            raise HTTPException(status_code=400, detail="Sample file not found")
        image_bytes = fpath.read_bytes()
    else:
        raise HTTPException(status_code=400, detail="No image provided")

    weather_context = {
        "temperature_2m": temperature_2m or 28.0,
        "relative_humidity_2m": relative_humidity_2m or 75.0,
        "surface_pressure": surface_pressure or 1008.0,
        "wind_speed_10m": wind_speed_10m or 14.0,
    }

    result = service.analyze_sky_image(image_bytes, weather_features=weather_context)
    val = result.get("validation")
    val_info = {
        "is_valid": val.is_valid if val else False,
        "width": val.width if val else 0,
        "height": val.height if val else 0,
        "mean_intensity": round(val.mean_intensity, 1) if val else 0.0,
        "warnings": val.warnings if val else [],
    }

    return {
        "success": result["success"],
        "error": result.get("error"),
        "validation": val_info,
        "cloud_result": result.get("cloud_result"),
        "rainfall_result": result.get("rainfall_result"),
    }


@app.get("/reports/{filename}")
def get_report_file(filename: str):
    """Serve figures from reports folder."""
    p = config.REPORTS_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p))


# Mount modern static UI directly at root
if config.STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")


def main():
    print("\n" + "=" * 60)
    print(" ⛅ SkyVision: Ground Sky Recognition & Rainfall Nowcasting")
    print("=" * 60)
    print(" Web Dashboard:     http://127.0.0.1:8000")
    print(" Interactive Docs:  http://127.0.0.1:8000/docs")
    print(" Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
