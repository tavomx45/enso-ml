import os
import cdsapi

# 1. Forzar a la librería a buscar el archivo de configuración en tu carpeta de proyecto
project_dir = "/home/gustavo-paredes/Documents/Developer/Yachay/enso-ml/"
os.environ["CDSAPI_RC"] = os.path.join(project_dir, ".cdsapirc")

# 2. Definir y crear la ruta de destino (data/raw) relativa a la raíz del proyecto
output_dir = os.path.join(project_dir, "data", "raw")
os.makedirs(output_dir, exist_ok=True)

output_filename = "era5_land_monthly.nc"
output_path = os.path.join(output_dir, output_filename)

# 3. Configurar la petición a la API de Copernicus
dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_dewpoint_temperature", 
        "2m_temperature", 
        "volumetric_soil_water_layer_1", 
        "volumetric_soil_water_layer_2", 
        "volumetric_soil_water_layer_3", 
        "volumetric_soil_water_layer_4", 
        "runoff", 
        "total_evaporation", 
        "10m_u_component_of_wind", 
        "10m_v_component_of_wind", 
        "surface_pressure", 
        "total_precipitation"
    ],
    "year": [str(y) for y in range(1950, 2026)],
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [1.5, -81.5, -5, -78.5]
}

# 4. Ahora el cliente leerá correctamente el archivo desde la ruta que le indicamos
client = cdsapi.Client()
client.retrieve(dataset, request).download(output_path)
