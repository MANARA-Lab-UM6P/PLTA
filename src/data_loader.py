# =============================================================================
# src/data_loader.py — Gowalla dataset loading and coordinate conversion
# =============================================================================

import math
import pickle
import pandas as pd


def latlong_to_flat_earth(lat, lon, ref_lat, ref_lon):
    """
    Convert (lat, lon) to flat-Earth Cartesian (x, y) in metres.

    Uses a simple equirectangular projection centred on (ref_lat, ref_lon).
    """
    y = (lat - ref_lat) * 111_320
    x = (lon - ref_lon) * math.cos(math.radians(ref_lat)) * 111_320
    return [x, y]


def load_gowalla(raw_file, cache_file,
                 lon_min=-122.52, lon_max=-122.36,
                 lat_min=37.70,   lat_max=37.84,
                 ref_lat=37.7749, ref_lon=-122.4194):
    """
    Load the Gowalla check-in dataset, filter it to a geographic bounding box,
    split into *workers* (multi-checkin users) and *tasks* (unique locations),
    and convert all coordinates to flat-Earth metres.

    A pickle cache is written on first call and reused on subsequent calls.

    Parameters
    ----------
    raw_file  : path to loc-gowalla_totalCheckins.txt  (tab-separated)
    cache_file: path where the preprocessed data is cached (.pkl)

    Returns
    -------
    workers_coords : list of [x, y]
    tasks_coords   : list of [x, y]
    """
    try:
        with open(cache_file, "rb") as f:
            workers_coords, tasks_coords = pickle.load(f)
        return workers_coords, tasks_coords
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        pass

    df = pd.read_csv(raw_file, sep="\t", header=None)
    df.columns = ["user", "check-in_time", "latitude", "longitude", "location_id"]
    df.sort_values(by=["user", "check-in_time"], inplace=True)

    # Filter to the San Francisco area
    df = df[
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) &
        (df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max)
    ]

    # Workers: users with more than one check-in, deduplicated by location
    single_checkin = df.drop_duplicates(subset=["user"], keep=False)
    multi_checkin  = df.drop(single_checkin.index)
    workers_dedup  = multi_checkin.drop_duplicates(subset=["latitude", "longitude"])
    workers        = workers_dedup.drop_duplicates(subset=["user"], keep="last")

    # Tasks: remaining unique locations not occupied by any worker
    tasks_raw = df.drop(workers.index).drop_duplicates(subset=["latitude", "longitude"])
    # Remove task locations that coincide with worker locations
    tasks = pd.merge(tasks_raw, workers,
                     on=["latitude", "longitude"],
                     how="left", indicator=True)
    tasks = tasks[tasks["_merge"] == "left_only"].drop(columns=["_merge"])

    workers_coords = [
        latlong_to_flat_earth(row["latitude"], row["longitude"], ref_lat, ref_lon)
        for _, row in workers.iterrows()
    ]
    tasks_coords = [
        latlong_to_flat_earth(row["latitude"], row["longitude"], ref_lat, ref_lon)
        for _, row in tasks.iterrows()
    ]

    with open(cache_file, "wb") as f:
        pickle.dump((workers_coords, tasks_coords), f)

    return workers_coords, tasks_coords
