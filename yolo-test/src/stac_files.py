from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import rasterio
from rasterio.warp import transform_bounds
from pystac import Asset, Catalog, Collection, Extent, Item
from pystac.catalog import CatalogType
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension, RasterBand


# ---------- parsing ----------

@dataclass(frozen=True)
class Record:
    cog_path: Path
    epoch: int
    dt_utc: datetime
    station: str
    local_time: str
    user: Optional[str]
    product: Optional[str]

def parse_tail(tail: str):
    tokens = tail.split("_")

    is_cog = tokens[-1] == "cog"
    if is_cog:
        tokens = tokens[:-1]

    if len(tokens) >= 2:
        user = "_".join(tokens[:-1])
        product = tokens[-1]
    elif tokens:
        user = tokens[0]
        product = None
    else:
        user = None
        product = None

    return user, product, is_cog

def parse_name(path: Path, products: Sequence[str] = ("plan",)) -> Record:
    parts = path.name.split(".")
    epoch = int(parts[0])
    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)

    local_hms = parts[3].replace("_", ":")
    local_time = f"{parts[1]} {parts[2]} {local_hms} {parts[4]} {parts[5]}"

    station = parts[6]



    tail = parts[-2]  
    user, product, is_cog = parse_tail(tail)
    

    return Record(
        cog_path=path,
        epoch=epoch,
        dt_utc=dt_utc,
        station=station,
        local_time=local_time,
        user=user,
        product=product,
    )


# ---------- STAC creation ----------

def polygon_from_bounds(left: float, bottom: float, right: float, top: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [left, bottom],
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]]
    }

def make_item_id(rec: Record) -> str:
    # stable, unique and human-readable
    return f"{rec.station}_{rec.epoch}"

def create_item_from_cog(rec: Record, asset_href: Optional[str] = None) -> Item:
    with rasterio.open(rec.cog_path) as ds:
        # Convert raster bounds from native CRS (eg EPSG:28355) to EPSG:4326 for STAC
        left, bottom, right, top = transform_bounds(
            ds.crs, "EPSG:4326",
            ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top,
            densify_pts=21,  # helps for non-linear edges
        )

        geom = {
            "type": "Polygon",
            "coordinates": [[
                [left, bottom],
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ]]
        }
        bbox = [left, bottom, right, top]

        item = Item(
            id=make_item_id(rec),
            geometry=geom,
            bbox=bbox,
            datetime=rec.dt_utc,
            properties={
                "station": rec.station,
                "user": rec.user,
                "product": rec.product,
                "local_time": rec.local_time,
            },
        )

        href = asset_href if asset_href is not None else rec.cog_path.name
        item.add_asset(
            "image",
            Asset(
                href=href,
                media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                roles=["data"],
            ),
        )

        # PROJ stays in native CRS ✅
        proj = ProjectionExtension.ext(item, add_if_missing=True)
        proj.epsg = ds.crs.to_epsg() if ds.crs else None
        proj.shape = [ds.height, ds.width]
        proj.transform = [ds.transform.a, ds.transform.b, ds.transform.c,
                          ds.transform.d, ds.transform.e, ds.transform.f]

        # RASTER applies to asset ✅
        img_asset = item.assets["image"]
        rast = RasterExtension.ext(img_asset, add_if_missing=True)
        rast.bands = [RasterBand.create(nodata=ds.nodata) for _ in range(ds.count)]

    return item


# ---------- write static STAC ----------

def build_static_stac(
    cogs: list[Path],
    out_dir: Path = Path("ard"),
    catalog_id: str = "coastal_imaging",
    collection_id: str = "coastsnap",
) -> Catalog:
    assets_dir = out_dir / "assets"
    items_dir = out_dir / "items"

    assets_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)

    items: list[Item] = []

    for cog in cogs:
        rec = parse_name(cog)

        # If you are publishing, you usually copy or sync assets into a known folder
        # Here we assume assets already live at assets_dir / cog.name
        asset_href = f"../assets/{cog.name}"

        item = create_item_from_cog(rec, asset_href=asset_href)

        item_path = items_dir / f"{item.id}.json"
        item.set_self_href(str(item_path))
        item.make_asset_hrefs_relative()  # makes asset href relative to item location

        item.save_object()
        items.append(item)

    collection = Collection(
        id=collection_id,
        description="STAC collection containing CoastSnap images",
        extent=Extent.from_items(items),
    )
    for it in items:
        collection.add_item(it)

    catalog = Catalog(
        id=catalog_id,
        description="STAC catalogue containing coastal imaging data",
    )
    catalog.add_child(collection)

    catalog.set_self_href(str(out_dir / "catalog.json"))
    collection.set_self_href(str(out_dir / "collection.json"))

    catalog.save(catalog_type=CatalogType.SELF_CONTAINED)
    return catalog

if __name__ == "__main__":
    

    # index only COGs
    cogs = sorted(Path(r"C:\Users\z3551540\Code\Github\CoastSnap-Object-Detection\yolo-test\stac_cog\assets").glob("*_cog.tif"))

    # build STAC JSONs under ./ard
    catalog = build_static_stac(cogs, out_dir=Path("stac_cog"))
    print(catalog.describe())