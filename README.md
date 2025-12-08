# IBB Traffic Graph Dataset

This repository shares the dataset used in the paper **IBB Traffic Graph Data: Benchmarking and Road Traffic Prediction**.

You can access the full presentation file [here](https://drive.google.com/file/d/1NZ2Jpnc2JyasNjdPwZ3UyRHgcX94s0XD/view?usp=drive_link).

## Dataset Overview

The IBB Road Traffic Graph dataset is a **spatio-temporal** dataset combining a road network (graph) and traffic flow features collected from Istanbul Metropolitan Municipality (IBB) traffic sensors.

* **Data Scope:** 2 years of **pre-processed** traffic data.
* **Temporal Resolution:** Calculated in **1-hour intervals** (24 records per sensor per day).
* **Graph Structure:** The road network is provided as a NetworkX graph (`graph.gexf`).


### Graph Statistics

| Metric | Value |
| :--- | :--- |
| **Number of Nodes (Sensors)** | 2464 |
| **Number of Edges** | 3695 |

---

## Node Features (Sensor Attributes)

Each **node** in the graph represents a traffic sensor and includes the following static attributes:

* **`lat`**: Latitude coordinate.
* **`long`**: Longitude coordinate.
* **`district`**: The administrative district.
* **`continent`**: Specifies landmass (**Asia** or **Europe**).
* **`highway`**: Classification (**highway** or **urban area**).
* **`population`**: Population density in the area.
* **`label`**: Unique identifier (geohash).

---

## Traffic Flow Features (Time-Series)

Four traffic flow features are provided **per hour, per node**:

| Feature Name | Description |
| :--- | :--- |
| **Minimum speed** | Minimum recorded car speed (km/h). |
| **Average speed** | Average recorded car speed (km/h). |
| **Maximum speed** | Maximum recorded car speed (km/h). |
| **Number of vehicles** | Count of cars detected. |

---

## 🔗 Usage and Raw Data

For simple usage examples and further exploration:

* **Usage Demo:** See `'intro.ipynb'`.

The original, **raw data** (containing 4 years of unprocessed data) is available directly from the IBB website:

* **Source:** [https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set](https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set)

---

## 📝 Citation

If you use this dataset in your research, please cite our work:

```bibtex
@misc{olug2024ibbtrafficgraphdata,
  title={IBB Traffic Graph Data: Benchmarking and Road Traffic Prediction Model}, 
  author={Eren Olug and Kiymet Kaya and Resul Tugay and Sule Gunduz Oguducu},
  year={2024},
  eprint={2408.01016},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={[https://arxiv.org/abs/2408.01016](https://arxiv.org/abs/2408.01016)}, 
}