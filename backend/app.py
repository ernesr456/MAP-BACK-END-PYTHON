# backend/app.py
import asyncio
import io
import gzip
import time
from pathlib import Path
from typing import Tuple

import aiohttp
import numpy as np
import xarray as xr
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import mercantile
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import Affine

import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
MRMS_BASE = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
LATEST_URL = MRMS_BASE + "MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "latest.grib2"

CACHE_TTL = 60  # seconds
dataset_cache = {"ts": 0, "ds": None}

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="MRMS RALA Tile Server")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helper functions
# ---------------------------
async def download_latest_if_needed() -> Path:
    now = time.time()
    if CACHE_FILE.exists() and (now - CACHE_FILE.stat().st_mtime) < CACHE_TTL:
        return CACHE_FILE

    async with aiohttp.ClientSession() as sess:
        async with sess.get(LATEST_URL) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to download latest MRMS ({resp.status})")
            content = await resp.read()

    # decompress gzip
    with gzip.open(io.BytesIO(content), "rb") as gz:
        data = gz.read()
    CACHE_FILE.write_bytes(data)
    return CACHE_FILE

def open_dataset(grib_path: Path) -> xr.Dataset:
    ds = xr.open_dataset(str(grib_path), engine="cfgrib")
    return ds

async def get_cached_dataset() -> xr.Dataset:
    now = time.time()
    if dataset_cache["ds"] is None or (now - dataset_cache["ts"]) > CACHE_TTL:
        grib_path = await download_latest_if_needed()
        dataset_cache["ds"] = open_dataset(grib_path)
        dataset_cache["ts"] = now
    return dataset_cache["ds"]

def get_var_and_coords(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].values

    if "latitude" in ds.coords and "longitude" in ds.coords:
        lats = ds["latitude"].values
        lons = ds["longitude"].values
    elif "lat" in ds.coords and "lon" in ds.coords:
        lats = ds["lat"].values
        lons = ds["lon"].values
    elif "y" in ds.coords and "x" in ds.coords:
        y = ds["y"].values
        x = ds["x"].values
        lons, lats = np.meshgrid(x, y)
    else:
        raise RuntimeError("Could not find lat/lon coords in dataset")
    return data, lats, lons

def render_tile_from_grid(data: np.ndarray, lats: np.ndarray, lons: np.ndarray, z: int, x: int, y: int, tile_size=256) -> bytes:
    tile = mercantile.Tile(x=x, y=y, z=z)
    w, s, e, n = mercantile.bounds(tile)

    # create transform for source grid
    ny, nx = data.shape
    lon0, lon1 = lons[0,0], lons[0,-1]
    lat0, lat1 = lats[0,0], lats[-1,0]
    resx = (lon1 - lon0) / max(nx-1, 1)
    resy = (lat1 - lat0) / max(ny-1, 1)
    src_transform = Affine.translation(lon0 - resx/2, lat0 - resy/2) * Affine.scale(resx, resy)

    # destination tile bounds in WebMercator
    wm_w, wm_s, wm_e, wm_n = transform_bounds("EPSG:4326", "EPSG:3857", w, s, e, n, densify_pts=21)
    dst_transform = Affine((wm_e - wm_w) / tile_size, 0, wm_w, 0, -(wm_n - wm_s) / tile_size, wm_n)
    dst = np.full((tile_size, tile_size), np.nan, dtype=np.float32)

    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        resampling=Resampling.bilinear,
        num_threads=2,
    )

    # Map reflectivity to RGBA using colormap
    cmap = plt.get_cmap("turbo")
    nan_mask = np.isnan(dst)
    vmin, vmax = -10.0, 75.0
    normed = (np.clip(dst, vmin, vmax) - vmin) / (vmax - vmin)
    rgba = cmap(normed)
    rgba[nan_mask, :] = (0,0,0,0)
    img = (rgba * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

# ---------------------------
# API endpoint
# ---------------------------
@app.get("/tiles/{z}/{x}/{y}.png")
async def tile(z: int, x: int, y: int):
    try:
        ds = await get_cached_dataset()
        data, lats, lons = get_var_and_coords(ds)
        png = render_tile_from_grid(data, lats, lons, z, x, y)
        return Response(content=png, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
