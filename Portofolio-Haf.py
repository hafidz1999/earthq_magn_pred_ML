import streamlit as st

# Home Section
st.title("Welcome to My Portfolio")
st.markdown("""
Hi, my name is Hafidz Abdurrafi, a Machine Learning enthusiast. I am rather new at machine learning, and this portfolio showcases one of my work in Machine Learning.
""")

# Projects Section
st.header("Model Prediksi Magnitudo Gempa dengan Random Tree Regression dan Bayes Optimization")
st.write("Proyek ini melibatkan prediksi magnitudo gempa berdasarkan data seismik. Tujuannya adalah untuk mengembangkan model yang dapat memperkirakan magnitudo gempa dengan akurat menggunakan Random Tree Regression yang dioptimalkan dengan teknik Bayesian.")

st.subheader("Deskripsi Data:")
st.write("Dataset diambil dari Kaggle, diunggah oleh **Agapitus Keyka Vigiliant**, dengan judul **'Earthquakes in Indonesia'**.")
st.write("Mengutip secara verbatim: Dataset ini diambil dari Earthquake Repository yang dikelola oleh BMKG (Badan Meteorologi, Klimatologi, dan Geofisika Indonesia).")
st.write("Selain itu: Dataset baru yang saya gunakan dalam model ini (katalog_gempa_v2.tsv) diambil dari Preliminary Earthquake Catalog yang mencakup data mekanisme fokus (jika ada). Dataset ini berisi data kejadian gempa dari 1 Nov 2008 hingga 9 Apr 2024, tetapi mungkin tidak akurat untuk beberapa kejadian gempa terakhir yang tercatat.")
st.write("Terdapat 38 variabel dalam dataset ini, namun saya memilih untuk hanya menggunakan 8:")
st.markdown("- latitude\n- longitude\n- magnitude\n- mag_type\n- depth\n- phasecount\n- azimuth_gap\n- location\n")

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
# The file ID from your Google Drive link
file_id = '1oeIA6imfMm649Gs_F5LiGmSkXReQ_YMc'
# Creating the download URL
download_url = f'https://drive.google.com/uc?id={file_id}'
# Load the dataset
df = pd.read_csv(download_url, sep='\t')
#Missing Value
df.isnull().sum()
selected_columns = [
    'latitude', 'longitude', 'magnitude',
    'mag_type', 'depth', 'phasecount',
    'azimuth_gap', 'location'
]
df_selected = df[selected_columns]
#Drop missing value
df_cleaned = df_selected.dropna()

# Create a GeoDataFrame for spatial plotting
gdf = gpd.GeoDataFrame(df_cleaned, geometry=gpd.points_from_xy(df_cleaned.longitude, df_cleaned.latitude))
# Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
gdf.crs = "EPSG:4326"
# Reproject to Web Mercator (EPSG:3857) for compatibility with base maps
gdf = gdf.to_crs(epsg=3857)
# Plot the earthquake data on a map of Indonesia
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, marker='o', color='red', alpha=0.5, markersize=5)
# Add basemap using contextily
ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
# Add labels and title
ax.set_title('Earthquake Events di Indonesia', fontsize=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

df_preview = df_cleaned.head()
st.write("Berikut preview dari dataset:")
st.dataframe(df_preview)
st.write("Berikut plot dari dataset:")
st.pyplot(plt.gcf())

location_to_group = {
    'Sumatra': [
        'Southern Sumatra, Indonesia', 'Northern Sumatra, Indonesia',
        'Southwest of Sumatra, Indonesia', 'Off West Coast of Northern Sumatra',
        'Sunda Strait, Indonesia'
    ],
    'Sunda Strait and Java': [
        'Java, Indonesia', 'South of Java, Indonesia', 'Sumba Region, Indonesia',
        'Sumbawa Region, Indonesia', 'South of Sumbawa, Indonesia',
        'South of Bali, Indonesia', 'Bali Region, Indonesia'
    ],
    'Lesser Sunda Islands': [
        'Tanimbar Islands Reg., Indonesia', 'Tanimbar Islands Region, Indonesia',
        'Timor Sea', 'Aru Islands Region, Indonesia',
        'Flores Region, Indonesia', 'Bali Sea', 'Flores Sea'
    ],
    'Banda Sea': [
        'Banda Sea', 'Seram, Indonesia', 'Ceram Sea', 'Buru, Indonesia'
    ],
    'Sulawesi and Sangihe Islands': [
        'Talaud Islands, Indonesia', 'Minahassa Peninsula, Sulawesi',
        'Sulawesi, Indonesia', 'Celebes Sea'
    ],
    'Halmahera': [
        'Halmahera, Indonesia', 'Northern Molucca Sea', 'Southern Molucca Sea',
        'North of Halmahera, Indonesia'
    ]
}
# Function to categorize each location
def categorize_location(location):
    for group, locations in location_to_group.items():
        if location in locations:
            return group
    return 'Other'  # For any location that doesn't fit the predefined groups
# Apply the function to create the new column
df_cleaned['volcanic_group'] = df_cleaned['location'].apply(categorize_location)
df_cleaned = df_cleaned.drop('location', axis=1)
df_preview = df_cleaned.head()

# Re-create the GeoDataFrame to include the 'volcanic_group' column
gdf = gpd.GeoDataFrame(df_cleaned, geometry=gpd.points_from_xy(df_cleaned.longitude, df_cleaned.latitude))
# Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
gdf.crs = "EPSG:4326"
# Reproject to Web Mercator (EPSG:3857) for compatibility with base maps
gdf = gdf.to_crs(epsg=3857)
# Check the GeoDataFrame
gdf.head()
color_map = {
    'Sumatra': 'red',
    'Sunda Strait and Java': 'green',
    'Lesser Sunda Islands': 'blue',
    'Banda Sea': 'purple',
    'Sulawesi and Sangihe Islands': 'orange',
    'Halmahera': 'cyan',
    'Other': 'gray'
}
# Plot the earthquake data on a map of Indonesia
fig, ax = plt.subplots(figsize=(10, 10))
# Plot each volcanic group with a distinct color
for group, color in color_map.items():
    gdf[gdf['volcanic_group'] == group].plot(ax=ax, marker='o', color=color, alpha=0.5, markersize=5, label=group)
# Add basemap using contextily
ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
# Add labels and title
ax.set_title('Earthquake Events di Indonesia dibagi Volcanic Geographical Groups', fontsize=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# Add legend
ax.legend()

st.write("Variabel lokasi terlalu banyak variasi, dengan terlalu banyak nilai unik. Oleh karena itu, saya mengelompokkan mereka berdasarkan wilayah vulkanik dan menggantinya:")
st.markdown("- Sumatra\n- Selat Sunda dan Jawa\n- Kepulauan Sunda Kecil\n- Laut Banda\n- Sulawesi dan Kepulauan Sangihe\n- Halmahera\n")
st.write("Berikut ini adalah pratinjau dataset baru:")
st.dataframe(df_preview)
st.write("Berikut ini adalah plot dari dataset yang telah dikategorikan:")
st.pyplot(plt.gcf())

st.markdown(" ")

st.header("Data Modelling:")
st.write("Untuk proyek ini, saya memilih **Random Tree Regression** karena kemampuannya yang kuat dalam menangani hubungan non-linear. Model ini kemudian dioptimalkan lebih lanjut menggunakan **Bayesian Optimization** untuk menyempurnakan hyperparameter.")

st.subheader("Feature Engineering:")
st.write("- One-hot encoding untuk fitur kategorikal")

st.markdown(" ")

st.header("Performance Metrics:")
st.markdown("- Mean Squared Error:  0.22569345072520083\n- Skor R^2:  0.6611528389009467\n")
st.write("Dari pemahaman saya, MSE yang lebih rendah menunjukkan kesesuaian model yang lebih baik terhadap data. Dalam kasus ini, MSE sekitar **~0.226** menunjukkan bahwa prediksi model cukup dekat dengan magnitudo gempa yang sebenarnya, secara rata-rata. Skor R² sekitar ~0.661 menunjukkan bahwa model ini menjelaskan sekitar **66%** variansi dalam data.")

st.subheader("Feature Importance:")
st.markdown("- Phasecount (0.307)\n- Longitude (0.233) & Latitude (0.185)\n- Azimuth Gap (0.109) & Depth (0.074)\n- Mag_type (kurang dari ~0.02) dan Volcanic Group (kurang dari ~0.018)\n")
st.write("Skor pentingnya fitur menunjukkan seberapa besar kontribusi setiap fitur terhadap prediksi magnitudo gempa. Mag_type dan Volcanic Group memiliki skor penting yang relatif rendah, sementara Phasecount, longitude, dan latitude adalah prediktor yang paling signifikan.")

st.subheader("Parameter Terbaik:")
st.write("Proses optimasi Bayesian telah mengidentifikasi set hyperparameter berikut sebagai yang terbaik untuk model:")
st.markdown("- max_depth = 30\n- max_features = 'sqrt'\n- min_samples_leaf = 1\n- min_samples_split = 4\n- n_estimators = 468\n")

st.markdown(" ")

st.header("Ringkasan:")
st.write("Skor MSE dan R² mencerminkan model yang cukup akurat tetapi tidak sempurna. Skor R² sekitar **~0.661** menunjukkan bahwa meskipun model ini baik, masih ada **34%** variansi dalam magnitudo gempa yang tidak dijelaskan, yang bisa disebabkan oleh keterbatasan fitur input, noise dalam data, atau faktor lainnya.")

st.subheader("Batasan:")
st.markdown("- Kategorisasi variabel lokasi:\n")
st.write("Mungkin ada cara yang lebih baik untuk memasukkan dan mengkategorikan variabel 'lokasi' selain yang digunakan dalam model, tetapi karena keterbatasan pengetahuan saya dalam bidang geografi, geologi, atau bidang studi terkait lainnya, ada beberapa keterbatasan dalam implementasi saya.")
st.markdown("- Mag_type dan Volcanic Group:\n")
st.write("Pengalaman saya yang terbatas dalam menggunakan algoritma Random Tree Regression membuat model ini tidak sepenuhnya menangkap hubungan kompleks yang mungkin dimiliki oleh kategori ini dengan magnitudo gempa, atau bahwa fitur-fitur ini mungkin tidak memiliki hubungan langsung yang kuat dengan magnitudo dalam dataset.")
st.markdown("- **34%** Variansi yang Tidak Dijelaskan:\n")
st.write("Masih ada variansi yang tidak dijelaskan, yang berarti mungkin ada faktor lain yang belum diperhitungkan atau kompleksitas yang tidak dapat ditangkap oleh model.")
st.markdown("- Model lain:\n")
st.write("Ada model-model lain yang mungkin lebih baik dalam menangani data geografis, seperti Gradient Boosting Machines, Support Vector Regression, atau bahkan Neural Networks.")
st.markdown(" ")

# Links Section
st.subheader("Links:")
# Kaggle dataset link
kaggle = "https://www.kaggle.com/datasets/kekavigi/earthquakes-in-indonesia/data?select=katalog_gempa_v2.tsv"
st.markdown(f"- [Kaggle Dataset: Earthquakes in Indonesia]({kaggle})")
# Project folder link
proj = "https://drive.google.com/drive/u/0/folders/1nSdvmZQdQpgz8b7RhPtdCNEBx1cxlQyE"
st.markdown(f"- [Google Drive: Project Folder]({proj})")
# LinkedIn profile link
linkedin = "https://www.linkedin.com/in/hafidz-abdurrafi-2409/"
st.markdown(f"- [LinkedIn: Hafidz Abdurrafi]({linkedin})")
# GitHub profile link
github = "https://github.com/hafidz1999"
st.markdown(f"- [GitHub: hafidz1999]({github})")

st.markdown(" ")

# Contact Section
st.header("Contact")
st.markdown("Feel free to reach out to me at hafidzabdurrafi1999@gmail.com or connect with me on LinkedIn (hafidz-abdurrafi-2409).")