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
from rasterio.transform import from_bounds

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

# Correct CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://map-front-end-nextjs.vercel.app",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
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
    # Get tile bounds in lat/lon
    tile = mercantile.Tile(x=x, y=y, z=z)
    west, south, east, north = mercantile.bounds(tile)

    # Source bounds
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Destination array
    dst = np.full((tile_size, tile_size), np.nan, dtype=np.float32)

    # Reproject from lat/lon to WebMercator
    src_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, data.shape[1], data.shape[0])
    dst_transform = from_bounds(west, south, east, north, tile_size, tile_size)

    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",  # keep in lat/lon for Leaflet overlay
        resampling=Resampling.bilinear,
        num_threads=2,
    )

    # Map reflectivity to RGBA
    colors = [
        "#000000", "#00FFFF", "#0000FF", "#00FF00", "#FFFF00",
        "#FFA500", "#FF0000", "#800000"
    ]
    bounds = [0, 5, 10, 20, 30, 40, 50, 60, 70]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    nan_mask = np.isnan(dst)
    rgba = cmap(norm(np.nan_to_num(dst, nan=0)))
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
        print("Tile error:", e)
        raise HTTPException(status_code=502, detail=str(e))
