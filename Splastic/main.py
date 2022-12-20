from fastapi import FastAPI
from dataclasses import dataclass
from typing import List
from fastapi import Response


from parcels import FieldSet, Field, ParticleSet, Variable, JITParticle
import numpy as np
from parcels import AdvectionRK4, plotTrajectoriesFile, ErrorCode

import math
import json
import xarray as xr
from datetime import timedelta

app = FastAPI()
field_set = None
data_name = './NC_files/hk_cleaned.nc'
temp_file_name = './NC_files/temporary.nc'

@dataclass
class Particles:
    # [[lon, lat], ...]
    particles: List[List[float]]

def DeleteParticle(particle, fieldset, time):
    particle.delete()

@app.on_event("startup")
async def startup_event():
    global field_set
    # Set up fieldset
    variables = {'U': 'u',
                 'V': 'v',
                 'depth': 'w'
                 }

    dimensions = {'lat': 'lat_rho',
                  'lon': 'lon_rho',
                  }

    field_set = FieldSet.from_c_grid_dataset(data_name, variables, dimensions, allow_time_extrapolation=True)

@app.get("/")
async def root():
    return {"message": "Welcome to Splastic!"}


@app.post("/execute")
async def execute(input_particles: Particles, forward: bool, time_duration: int, time_delta: int, output_delta: int, return_trajectory: bool):
    # input_particles: Particles; 2D array of initial positions of trash objects in lon and lat
    # forward: Boolean; true for stepping forward in time, false for stepping backwards in time
    # time_duration: Integer; number of days to run algorithm eg. 30
    # time_delta: Integer; number of minutes per step eg. 10
    # output_delta: Integer: number of hours to record each step
    # return_trajectory: Boolean; whether to record the trajectory of individual particles

    #Set up particle set
    num_particles = len(input_particles.particles)
    if num_particles == 0:
        return None
    particles_list = np.array(input_particles.particles)
    lon = particles_list[:, 0]
    lat = particles_list[:, 1]
    pset = ParticleSet(fieldset=field_set, pclass=JITParticle, lat= lat, lon=lon)

    output_file = pset.ParticleFile(name=temp_file_name,
                                outputdt=timedelta(hours=output_delta))
    
    #Set negative dt for running in reverse
    if not forward:
        time_delta = -time_delta
    
    #Advect using Runge-Kutta 4th order
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=time_duration),
                 dt=timedelta(minutes=time_delta),
                 output_file=output_file,
                 # Delete particle when out of bounds
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                 )

    output_file.close()

    #Read
    parcels = xr.open_dataset(temp_file_name)

    lons = parcels["lon"].to_numpy()
    lats = parcels["lat"].to_numpy()

    if return_trajectory:
        #Get list of trajectories
        trajectories = []
        for i in range(num_particles):
            trajectories.append(list(zip(lons[i], lats[i])))

        # print(trajectories)

        json_data = json.dumps({'trajectories': trajectories})
        return Response(content=json_data, media_type="application/json")
    else:
        #Pair list of start, end positions
        #Still named trajectories for convenience
        trajectories = []
        for i in range(num_particles):
            start = [lons[i][0], lats[i][0]]
            end = [lons[i][-1], lats[i][-1]]
            trajectories.append([start, end])
        json_data = json.dumps({'trajectories': trajectories})
        return Response(content=json_data, media_type="application/json")


