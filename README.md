# CityLink Analyzer

*An interactive tool for analyzing and optimizing city-to-city travel routes across India.*

**CityLink Analyzer** is a Python-based GUI application developed as part of the **Design and Analysis of Algorithms** course. It leverages graph algorithms and real-time traffic data to provide insights for travel planning and urban network analysis, processing a dataset of 330+ city connections across India.

## Project Overview

Built under the guidance of **Deepika Prakash Mam**, this project combines advanced graph algorithms with a user-friendly interface to solve practical problems in route optimization and network connectivity. It integrates real-time traffic data via the GraphHopper API and offers interactive visualizations, making it a powerful tool for travelers, urban planners, and algorithm enthusiasts.

### Key Features

- **Shortest Path Finder**: Computes optimal routes using **Dijkstra’s** and **Floyd-Warshall** algorithms.
- **Traveling Salesman Problem (TSP)**: Solves multi-city tour optimization with dynamic programming.
- **Centrality Analysis**: Identifies most/least connected cities using degree centrality and diversity scores.
- **Critical Roads Detection**: Finds critical connections (bridges) via **DFS**.
- **Minimum Spanning Tree (MST)**: Generates MST using **Prim’s** algorithm.
- **Real-Time Traffic Updates**: Fetches live traffic data with caching for efficiency.
- **Interactive Visualizations**: Displays graphs (NetworkX, Matplotlib) and maps (Folium).
- **Data Export**: Saves results as CSV or PDF for easy sharing.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Tkinter, Pandas, NetworkX, Matplotlib, Folium, Requests
- **API**: GraphHopper (for live traffic data)
- **Data**: CSV dataset (`Dataset.csv`) with 330+ city routes
- **Dependencies**: JSON, ReportLab (for PDF export)

## Installation

Follow these steps to set up and run the CityLink Analyzer locally:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/[your-username]/citylink-analyzer.git
   cd citylink-analyzer
   ```

2. **Install Dependencies**: Create a virtual environment (optional) and install required packages:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add GraphHopper API Key**:

   - Sign up for a free API key at GraphHopper.
   - In `CityLinkAnalyzer.py`, replace the placeholder API key (`"5eafb0ad-f5fd-475f-ba76-2845ea30ee55"`) with your key.

4. **Place the Dataset**: Ensure `Dataset.csv` is in the project directory (included in the repository).

5. **Run the Application**:

   ```bash
   python CityLinkAnalyzer.py
   ```

## Usage

1. **Launch the GUI**: Run the script to open the CityLink Analyzer interface.
2. **Select Features**:
   - Choose options like "Find Shortest Path," "Get Direct Distance," or "Simulate Road Closure."
   - Input cities or roads using dropdowns or listboxes.
3. **View Results**:
   - Results display in the GUI with details like distances, paths, or critical connections.
   - Visualize routes on interactive maps or graphs.
4. **Export Data**:
   - Save results as CSV or PDF via the "Export" buttons.

### Example

- **Find Shortest Path**: Select "Gwalior" to "Agra" to get the shortest route (e.g., 133.67 km).
- **Simulate Road Closure**: Close the "Mumbai-Surat" road and analyze the new MST and proposed links.
- **Visualize**: Generate a Folium map highlighting a TSP route across multiple cities.

## Screenshots

| GUI Interface | Folium Map | NetworkX Graph |
| --- | --- | --- |
|  |  |  |

*Note*: Replace placeholder images with actual screenshots by running the visualization functions (e.g., `graph.visualize_map()` or `graph.visualize_graph()`).

## Project Structure

```
citylink-analyzer/
├── CityLinkAnalyzer.py                    # Main application script
├── Dataset.csv        # Dataset of city connections
├── requirements.txt        # Dependencies
├── traffic_cache.json      # Cached traffic data
├── updated_edges.json      # Cached edge updates
├── city_map.html           # Generated Folium map
└── README.md               # This file
```

## Challenges and Solutions

- **Data Quality**: Addressed unrealistic distances (e.g., Surat-Nagpur at 3.32 km) by integrating live traffic data and caching results.
- **API Rate Limits**: Implemented retry logic and throttling to handle GraphHopper API constraints.
- **Algorithm Efficiency**: Optimized TSP with bitmask DP and used heap-based Dijkstra’s for scalability.

## Future Improvements

- Validate and clean CSV data to correct outliers.
- Add support for multiple travel modes (e.g., train, flight).
- Enhance visualizations with real-time traffic overlays.
- Deploy as a web app using Flask or Django.

## Credits

- **Team**:
  - Aman Kaushal
  - Manan Rathi
  - Prakhar Agrawal

## Contact

For questions or collaboration, reach out via LinkedIn or email.

Happy exploring with CityLink Analyzer! 🚗🗺️
