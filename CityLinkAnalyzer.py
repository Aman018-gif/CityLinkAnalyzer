import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
import heapq
import sys
import networkx as nx
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import requests
import folium
import json
import time
import threading

# City coordinates (covers all cities in the dataset)
CITY_COORDINATES = {
    'Gwalior': (26.2183, 78.1828), 'Agra': (27.1767, 77.9750),
    'Hyderabad': (17.3850, 78.4867), 'Bangalore': (12.9716, 77.5946),
    'Surat': (21.1702, 72.8311), 'Nagpur': (21.1458, 79.0882),
    'Bhopal': (23.2599, 77.4126), 'Thiruvananthapuram': (8.5241, 76.9366),
    'Mumbai': (19.0760, 72.8777), 'Amritsar': (31.6340, 74.8723),
    'Delhi': (28.7041, 77.1025), 'Bareilly': (28.3670, 79.4304),
    'Tirupati': (13.6288, 79.4192), 'Allahabad': (25.4358, 81.8463),
    'Ajmer': (26.4499, 74.6399), 'Jammu': (32.7266, 74.8570),
    'Raipur': (21.2514, 81.6296), 'Ranchi': (23.3441, 85.3096),
    'Nashik': (19.9975, 73.7898), 'Rourkela': (22.2604, 84.8536),
    'Visakhapatnam': (17.6868, 83.2185), 'Chandigarh': (30.7333, 76.7794),
    'Vadodara': (22.3072, 73.1812), 'Shimla': (31.1048, 77.1734),
    'Dhanbad': (23.7957, 86.4304), 'Jodhpur': (26.2389, 73.0243),
    'Bikaner': (28.0229, 73.3119), 'Pune': (18.5204, 73.8567),
    'Madurai': (9.9252, 78.1198), 'Udaipur': (24.5854, 73.7125),
    'Kanpur': (26.4499, 80.3319), 'Lucknow': (26.8467, 80.9462),
    'Mysore': (12.2958, 76.6394), 'Patna': (25.5941, 85.1376),
    'Varanasi': (25.3176, 82.9739), 'Ludhiana': (30.9000, 75.8573),
    'Coimbatore': (11.0168, 76.9558), 'Indore': (22.7196, 75.8577),
    'Jaipur': (26.9124, 75.7873), 'Dehradun': (30.3165, 78.0322),
    'Kolkata': (22.5726, 88.3639), 'Jaisalmer': (26.9157, 70.9083),
    'Guwahati': (26.1445, 91.7362), 'Bhubaneswar': (20.2961, 85.8245),
    'Ahmedabad': (23.0225, 72.5714), 'Meerut': (28.9845, 77.7064),
    'Chennai': (13.0827, 80.2707)
}

class WeightedGraph:
    def __init__(self, loading_label=None, skip_live_update=False):
        self.adjacency_list = {}
        self.timer = 0
        self.coordinates = CITY_COORDINATES
        self.traffic_cache = {}
        self.cache_file = "traffic_cache.json"
        self.csv_file = "Dataset.csv"
        self.updated_edges_file = "updated_edges.json"
        self.loading_label = loading_label
        self.after_ids = []
        self.skip_live_update = skip_live_update  # Flag to skip live updates for "Get Direct Distance"
        self.load_cache()
        if not self.skip_live_update:
            self.update_csv_with_live_distances()
        self.load_graph_from_csv()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    loaded_cache = json.load(f)
                    self.traffic_cache = {tuple(k.split('_')): v for k, v in loaded_cache.items()}
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.traffic_cache = {}

    def save_cache(self):
        try:
            cache_to_save = {f"{k[0]}_{k[1]}": v for k, v in self.traffic_cache.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(cache_to_save, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_updated_edges(self):
        if os.path.exists(self.updated_edges_file):
            try:
                with open(self.updated_edges_file, 'r') as f:
                    data = json.load(f)
                    return data.get('last_updated', 0), data.get('processed', [])
            except Exception as e:
                print(f"Error loading updated edges: {e}")
        return 0, []

    def save_updated_edges(self, last_updated, processed):
        try:
            with open(self.updated_edges_file, 'w') as f:
                json.dump({'last_updated': last_updated, 'processed': processed}, f)
        except Exception as e:
            print(f"Error saving updated edges: {e}")

    def log_error(self, message, result_text=None):
        print(message)
        if result_text:
            root.after(0, lambda: result_text.insert(tk.END, f"Error: {message}\n"))

    def update_csv_with_live_distances(self):
        try:
            df = pd.read_csv(self.csv_file)
            print(f"CSV columns in update_csv_with_live_distances: {df.columns.tolist()}")
            required_columns = ['Source', 'Destination', 'Distance (km)']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"CSV missing required columns: {missing}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        last_updated, processed = self.load_updated_edges()
        processed_set = set(processed)
        batch_size = 100
        updated = False
        total_edges = len(df)
        i = 0
        api_calls = 0
        rate_limit = 60  # API calls per minute

        for idx, row in df.iterrows():
            src = row['Source']
            dest = row['Destination']
            edge_key = f"{src}_{dest}"
            if edge_key in processed_set:
                continue

            # Check API rate limit
            if api_calls >= rate_limit:
                print(f"Rate limit of {rate_limit} calls reached. Waiting for 60 seconds...")
                time.sleep(60)  # Wait for 1 minute to reset the rate limit
                api_calls = 0  # Reset the counter

            print(f"API call {i+1}/{min(batch_size, total_edges)} for {src} to {dest}")
            if self.loading_label:
                after_id = root.after(0, lambda: self.loading_label.config(text=f"Updating {i+1}/{min(batch_size, total_edges)} edges..."))
                self.after_ids.append(after_id)
            
            distance = self.get_live_traffic_weight(src, dest)[0]
            if distance != float('inf') and distance != row['Distance (km)']:
                df.at[idx, 'Distance (km)'] = distance
                updated = True
            processed.append(edge_key)
            i += 1
            api_calls += 1
            if i >= batch_size:
                break
            time.sleep(0.5)  # Throttle to avoid hitting the rate limit too quickly

        if updated:
            df.to_csv(self.csv_file, index=False)
            print(f"Updated {self.csv_file} with live distances")
        self.save_updated_edges(time.time(), processed)

        for after_id in self.after_ids:
            root.after_cancel(after_id)
        self.after_ids.clear()

    def load_graph_from_csv(self):
        self.adjacency_list = {}
        try:
            df = pd.read_csv(self.csv_file)
            print(f"CSV columns in load_graph_from_csv: {df.columns.tolist()}")
            required_columns = ['Source', 'Destination', 'Distance (km)']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"CSV missing required columns: {missing}")

            for _, row in df.iterrows():
                self.add_undirected_edge(row['Source'], row['Destination'], row['Distance (km)'])
        except Exception as e:
            print(f"Error loading graph from CSV: {e}")
            raise

    def add_undirected_edge(self, source, destination, weight):
        if source not in self.adjacency_list:
            self.adjacency_list[source] = []
        if destination not in self.adjacency_list:
            self.adjacency_list[destination] = []
        self.adjacency_list[source].append((destination, weight))
        self.adjacency_list[destination].append((source, weight))

    def remove_undirected_edge(self, source, destination):
        if source in self.adjacency_list:
            self.adjacency_list[source] = [(d, w) for d, w in self.adjacency_list[source] if d != destination]
            if not self.adjacency_list[source]:
                del self.adjacency_list[source]
        if destination in self.adjacency_list:
            self.adjacency_list[destination] = [(s, w) for s, w in self.adjacency_list[destination] if s != source]
            if not self.adjacency_list[destination]:
                del self.adjacency_list[destination]

    def get_live_traffic_weight(self, src, dest, result_text=None):
        key = (src, dest)
        if key in self.traffic_cache:
            return self.traffic_cache[key]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                src_coords = self.coordinates.get(src)
                dest_coords = self.coordinates.get(dest)
                if not src_coords or not dest_coords:
                    self.log_error(f"Missing coordinates: {src} or {dest}", result_text)
                    for neighbor, weight in self.adjacency_list.get(src, []):
                        if neighbor == dest:
                            result = (weight, weight / 60)
                            self.traffic_cache[key] = result
                            self.save_cache()
                            return result
                    return float('inf'), float('inf')

                url = "https://graphhopper.com/api/1/route"
                params = {
                    "point": [f"{src_coords[0]},{src_coords[1]}", f"{dest_coords[0]},{dest_coords[1]}"],
                    "vehicle": "car",
                    "key": "5eafb0ad-f5fd-475f-ba76-2845ea30ee55"
                }
                response = requests.get(url, params=params, timeout=5)

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    self.log_error(f"GraphHopper rate limit exceeded. Retry {attempt+1}/{max_retries} after {retry_after} seconds.", result_text)
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                data = response.json()

                if 'paths' in data and data['paths']:
                    distance = data['paths'][0]['distance'] / 1000
                    duration = data['paths'][0]['time'] / 3600000
                    result = (distance, duration * 1000)
                    self.traffic_cache[key] = result
                    self.save_cache()
                    return result
                else:
                    self.log_error(f"No route found for {src} to {dest}", result_text)
                    for neighbor, weight in self.adjacency_list.get(src, []):
                        if neighbor == dest:
                            result = (weight, weight / 60)
                            self.traffic_cache[key] = result
                            self.save_cache()
                            return result
                    return float('inf'), float('inf')
            except requests.exceptions.RequestException as e:
                self.log_error(f"GraphHopper API error for {src} to {dest}: {e}", result_text)
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                for neighbor, weight in self.adjacency_list.get(src, []):
                    if neighbor == dest:
                        result = (weight, weight / 60)
                        self.traffic_cache[key] = result
                        self.save_cache()
                        return result
                return float('inf'), float('inf')

    def dijkstra(self, src, dest):
        inf = sys.maxsize
        node_data = {vertex: {'cost': inf, 'pred': []} for vertex in self.adjacency_list}
        node_data[src]['cost'] = 0
        visited = []
        min_heap = [(0, src)]
        
        while min_heap:
            cost, temp = heapq.heappop(min_heap)
            if temp in visited:
                continue
            visited.append(temp)
            
            if temp == dest:
                break
            
            if temp in self.adjacency_list:
                for neighbor, weight in self.adjacency_list[temp]:
                    if neighbor not in visited:
                        new_cost = node_data[temp]['cost'] + weight
                        if new_cost < node_data[neighbor]['cost']:
                            node_data[neighbor]['cost'] = new_cost
                            node_data[neighbor]['pred'] = node_data[temp]['pred'] + [temp]
                            heapq.heappush(min_heap, (new_cost, neighbor))
        
        if node_data[dest]['cost'] == inf:
            return inf, "No path"
        return node_data[dest]['cost'], " -> ".join(node_data[dest]['pred'] + [dest])

    def degree_centrality(self):
        max_deg = 0
        min_deg = float('inf')
        vertex_max = vertex_min = None
        max_diversity = 0
        min_diversity = float('inf')
        
        degree_map = {vertex: len(edges) for vertex, edges in self.adjacency_list.items()}
        
        for vertex, edges in self.adjacency_list.items():
            degree = len(edges)
            diversity_score = sum(degree_map.get(neighbor, 0) for neighbor, _ in edges)
            
            if degree > max_deg:
                max_deg = degree
                vertex_max = vertex
                max_diversity = diversity_score
            elif degree == max_deg and diversity_score > max_diversity:
                vertex_max = vertex
                max_diversity = diversity_score
            
            if degree < min_deg:
                min_deg = degree
                vertex_min = vertex
                min_diversity = diversity_score
            elif degree == min_deg and diversity_score < min_diversity:
                vertex_min = vertex
            elif degree == min_deg and diversity_score == min_diversity and vertex < vertex_min:
                vertex_min = vertex
        
        return vertex_max, max_deg, max_diversity, vertex_min, min_deg, min_diversity

    def compute_city_rankings(self):
        city_data = []
        degree_map = {vertex: len(edges) for vertex, edges in self.adjacency_list.items()}
        
        for vertex, edges in self.adjacency_list.items():
            degree = len(edges)
            diversity_score = sum(degree_map.get(neighbor, 0) for neighbor, _ in edges)
            city_data.append((vertex, degree, diversity_score))
        
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            return merge(left, right)
        
        def merge(left, right):
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if (left[i][1] > right[j][1]) or (left[i][1] == right[j][1] and left[i][2] >= right[j][2]):
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        sorted_cities = merge_sort(city_data)
        return sorted_cities

    def tsp_bitmask_dp(self, selected_cities):
        if len(selected_cities) < 2:
            return float('inf'), []
        
        n = len(selected_cities)
        city_to_index = {city: i for i, city in enumerate(selected_cities)}
        
        cost = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            cost[i][i] = 0
        for city1 in selected_cities:
            for neighbor, weight in self.adjacency_list.get(city1, []):
                if neighbor in selected_cities:
                    i = city_to_index[city1]
                    j = city_to_index[neighbor]
                    cost[i][j] = weight
        
        dp = [[-1] * (1 << n) for _ in range(n)]
        parent = [[None] * (1 << n) for _ in range(n)]
        
        def tsp_dp(pos, visited):
            if visited == (1 << n) - 1:
                return cost[pos][0] if cost[pos][0] != float('inf') else float('inf')
            if dp[pos][visited] != -1:
                return dp[pos][visited]
            
            min_cost = float('inf')
            min_city = None
            for city in range(n):
                if visited & (1 << city) == 0:
                    new_cost = cost[pos][city] + tsp_dp(city, visited | (1 << city))
                    if new_cost < min_cost:
                        min_cost = new_cost
                        min_city = city
            
            dp[pos][visited] = min_cost
            parent[pos][visited] = min_city
            return min_cost
        
        total_cost = tsp_dp(0, 1)
        
        path = [selected_cities[0]]
        pos = 0
        visited = 1
        while visited != (1 << n) - 1:
            next_city = parent[pos][visited]
            if next_city is None:
                return float('inf'), []
            path.append(selected_cities[next_city])
            visited |= (1 << next_city)
            pos = next_city
        path.append(selected_cities[0])
        
        return total_cost, path

    def prim(self, start_vertex):
        visited = set()
        output_list = []
        min_heap = [(0, start_vertex, None)]
        
        while min_heap:
            wt, v, parent = heapq.heappop(min_heap)
            if v not in visited:
                visited.add(v)
                if parent is not None:
                    output_list.append((parent, v, wt))
                for neighbor, weight in self.adjacency_list.get(v, []):
                    if neighbor not in visited:
                        heapq.heappush(min_heap, (weight, neighbor, v))
        
        return output_list

    def dfs(self, node, parent, vis, tin, low, bridges):
        vis[node] = 1
        tin[node] = low[node] = self.timer
        self.timer += 1
        
        for neighbor, _ in self.adjacency_list.get(node, []):
            if neighbor == parent:
                continue
            if vis[neighbor] == 0:
                self.dfs(neighbor, node, vis, tin, low, bridges)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > tin[node]:
                    bridges.append([node, neighbor])
            else:
                low[node] = min(low[node], tin[neighbor])

    def critical_connections(self):
        vis = {vertex: 0 for vertex in self.adjacency_list}
        tin = {vertex: -1 for vertex in self.adjacency_list}
        low = {vertex: -1 for vertex in self.adjacency_list}
        bridges = []
        
        for vertex in self.adjacency_list:
            if not vis[vertex]:
                self.dfs(vertex, -1, vis, tin, low, bridges)
        
        return bridges

    def floyd_warshall(self):
        vertices = list(self.adjacency_list.keys())
        n = len(vertices)
        
        inf = float('inf')
        dist = [[inf] * n for _ in range(n)]
        next_vertex = [[None] * n for _ in range(n)]
        
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        for i in range(n):
            dist[i][i] = 0
        
        for source in self.adjacency_list:
            i = vertex_to_index[source]
            for dest, weight in self.adjacency_list[source]:
                j = vertex_to_index[dest]
                dist[i][j] = weight
                next_vertex[i][j] = dest
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] != inf and dist[k][j] != inf:
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
                            next_vertex[i][j] = next_vertex[i][k]
        
        result = {}
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] != inf:
                    path = []
                    start = vertices[i]
                    end = vertices[j]
                    if next_vertex[i][j] is not None:
                        path.append(start)
                        curr = start
                        while curr != end:
                            curr_idx = vertex_to_index[curr]
                            curr = next_vertex[curr_idx][j]
                            path.append(curr)
                    result[(start, end)] = (dist[i][j], path)
        
        return result

    def find_connected_components(self):
        visited = {vertex: False for vertex in self.adjacency_list}
        components = []
        
        def dfs_component(node, component):
            visited[node] = True
            component.append(node)
            for neighbor, _ in self.adjacency_list.get(node, []):
                if not visited[neighbor]:
                    dfs_component(neighbor, component)
        
        for vertex in self.adjacency_list:
            if not visited[vertex]:
                component = []
                dfs_component(vertex, component)
                components.append(component)
        
        return components

    def propose_new_links(self, components):
        if len(components) <= 1:
            return []
        new_links = []
        all_weights = [w for v in self.adjacency_list.values() for _, w in v]
        avg_weight = sum(all_weights) / len(all_weights) if all_weights else 1000
        
        for i in range(len(components) - 1):
            src = components[i][0]
            dest = components[i + 1][0]
            new_links.append((src, dest, avg_weight))
        
        return new_links

    def simulate_road_closure(self, closed_roads):
        original_adj = {k: v[:] for k, v in self.adjacency_list.items()}
        for src, dest in closed_roads:
            self.remove_undirected_edge(src, dest)
        mst = self.prim(next(iter(self.adjacency_list)) if self.adjacency_list else None)
        components = self.find_connected_components()
        new_links = self.propose_new_links(components)
        self.adjacency_list = {k: v[:] for k, v in original_adj.items()}
        
        return mst, components, new_links

    def visualize_graph(self):
        G = nx.Graph()
        for node, edges in self.adjacency_list.items():
            for neighbor, weight in edges:
                G.add_edge(node, neighbor, weight=weight)
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=100)
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue")
        edges = G.edges(data=True)
        weights = [edge[2]['weight'] / 100 for edge in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color="gray")
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        edge_labels = {(u, v): f"{d['weight']} km" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red")
        plt.title("Complex City Graph Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_mst(self, mst_edges):
        G = nx.Graph()
        for parent, child, weight in mst_edges:
            G.add_edge(parent, child, weight=weight)
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightgreen")
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Minimum Spanning Tree Visualization")
        plt.show()

    def visualize_map(self, highlight_path=None):
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="OpenStreetMap")
        for city, (lat, lon) in self.coordinates.items():
            folium.Marker([lat, lon], popup=city, tooltip=city).add_to(m)
        for src, edges in self.adjacency_list.items():
            src_lat, src_lon = self.coordinates.get(src, (0, 0))
            for dest, weight in edges:
                dest_lat, dest_lon = self.coordinates.get(dest, (0, 0))
                if src_lat and dest_lat:
                    folium.PolyLine(
                        locations=[[src_lat, src_lon], [dest_lat, dest_lon]],
                        color="blue",
                        weight=2,
                        opacity=0.5,
                        popup=f"{src} -> {dest}: {weight} km"
                    ).add_to(m)
        if highlight_path:
            path_coords = []
            for city in highlight_path:
                if city in self.coordinates:
                    path_coords.append(self.coordinates[city])
            if path_coords:
                folium.PolyLine(
                    locations=path_coords,
                    color="red",
                    weight=4,
                    opacity=0.8,
                    popup="Highlighted Path"
                ).add_to(m)
        map_file = "city_map.html"
        m.save(map_file)
        os.system(f"start {map_file}")

def export_to_pdf(title, content, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(content.replace('\n', '<br />'), styles['BodyText']))
    doc.build(story)
    messagebox.showinfo("Export", f"PDF saved as:\n{os.path.abspath(filename)}")

def export_to_csv(command, result_text, filename):
    content = result_text.get(1.0, tk.END).strip()
    if not content:
        messagebox.showwarning("Export", "No results to export!")
        return
    if command in [find_shortest_path, find_distance]:
        df = pd.DataFrame([content.split('\n')], columns=["Result"])
    elif command == find_central_hub:
        lines = content.split('\n')
        data = []
        for line in lines:
            if line:
                parts = line.split(': ')
                city_info = parts[1].split(' (')
                city = city_info[0]
                degree = int(city_info[1].split(' cities, ')[0])
                diversity = int(city_info[1].split('Diversity Score = ')[1].strip(')'))
                data.append((city, degree, diversity))
        df = pd.DataFrame(data, columns=["City", "Degree", "Diversity Score"])
    elif command == find_mst:
        lines = content.split("\n")[1:]
        data = [tuple(line.split(" -> ") + [line.split("(")[-1].strip(" km\n)")]) for line in lines if line]
        df = pd.DataFrame(data, columns=["Source", "Destination", "Weight"])
    elif command == find_critical_roads:
        bridges = eval(content.split(": ")[1])
        data = [[x[0], x[1], "N/A"] for x in bridges]
        df = pd.DataFrame(data, columns=["Source", "Destination", "Weight"])
    elif command == find_all_shortest_paths:
        lines = content.split("\n")[1:]
        data = []
        for i in range(0, len(lines)-1, 4):
            src_dest = lines[i].split("From ")[1].split(" to ")
            distance = float(lines[i+1].split(": ")[1].split(" km")[0])
            path = lines[i+2].split(": ")[1]
            data.append((src_dest[0], src_dest[1], distance, path))
        df = pd.DataFrame(data, columns=["Source", "Destination", "Distance", "Path"])
    elif command == rank_cities_by_degree:
        lines = content.split("\n")[1:]
        data = []
        for line in lines:
            if line:
                parts = line.split(": Degree = ")
                city = parts[0].split(". ")[1]
                degree = int(parts[1].split(", Diversity Score = ")[0])
                diversity = int(parts[1].split(", Diversity Score = ")[1])
                data.append((city, degree, diversity))
        df = pd.DataFrame(data, columns=["City", "Degree", "Diversity Score"])
    elif command == find_tsp_route:
        lines = content.split("\n")
        total_distance = float(lines[0].split(": ")[1].split(" km")[0])
        path = lines[1].split(": ")[1]
        df = pd.DataFrame([[total_distance, path]], columns=["Total Distance (km)", "Route"])
    elif command == simulate_road_closure:
        lines = content.split("\n")
        mst_data = []
        components_data = []
        new_links_data = []
        current_section = None
        for line in lines:
            if line.startswith("Minimum Spanning Tree Edges:"):
                current_section = "mst"
                continue
            elif line.startswith("Connected Components:"):
                current_section = "components"
                continue
            elif line.startswith("Proposed New Links:"):
                current_section = "new_links"
                continue
            if line and current_section == "mst":
                parts = line.split(" -> ")
                weight = line.split("(")[-1].strip(" km\n)")
                mst_data.append((parts[0], parts[1], weight))
            elif line and current_section == "components":
                component = line.strip("[]").split(", ")
                components_data.append(component)
            elif line and current_section == "new_links":
                parts = line.split(" -> ")
                weight = line.split("(")[-1].strip(" km\n)")
                new_links_data.append((parts[0], parts[1], weight))
        df_mst = pd.DataFrame(mst_data, columns=["Source", "Destination", "Weight"]) if mst_data else pd.DataFrame()
        df_components = pd.DataFrame({"Component": [", ".join(c) for c in components_data]}) if components_data else pd.DataFrame()
        df_new_links = pd.DataFrame(new_links_data, columns=["Source", "Destination", "Weight"]) if new_links_data else pd.DataFrame()
        with pd.ExcelWriter(filename.replace(".csv", ".xlsx")) as writer:
            df_mst.to_excel(writer, sheet_name="MST", index=False)
            df_components.to_excel(writer, sheet_name="Components", index=False)
            df_new_links.to_excel(writer, sheet_name="New Links", index=False)
        messagebox.showinfo("Export", f"Excel saved as:\n{os.path.abspath(filename.replace('.csv', '.xlsx'))}")
        return
    else:
        df = pd.DataFrame([content.split('\n')], columns=["Result"])
    df.to_csv(filename, index=False)
    messagebox.showinfo("Export", f"CSV saved as:\n{os.path.abspath(filename)}")

def export_to_pdf_with_dialog(title, result_text):
    content = result_text.get(1.0, tk.END).strip()
    if not content:
        messagebox.showwarning("Export", "No results to export!")
        return
    filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if filename:
        export_to_pdf(title, content, filename)

def export_to_csv_with_dialog(command, result_text):
    content = result_text.get(1.0, tk.END).strip()
    if not content:
        messagebox.showwarning("Export", "No results to export!")
        return
    filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if filename:
        export_to_csv(command, result_text, filename)

def open_window(title, command):
    window = tk.Toplevel(root)
    window.title(title)
    window.geometry("500x600")
    window.configure(bg='#f5f5f5')
    content_frame = ttk.Frame(window, padding="20", relief="groove", borderwidth=2)
    content_frame.pack(expand=True, fill="both", padx=10, pady=10)
    ttk.Label(content_frame, text=title, font=('Helvetica', 16, 'bold')).pack(pady=10)
    
    if command in [find_shortest_path, find_distance]:
        ttk.Label(content_frame, text="Source City:").pack(pady=5)
        src_combo = ttk.Combobox(content_frame, values=city_list, width=35)
        src_combo.pack(pady=5)
        src_combo.set(city_list[0])
        ttk.Label(content_frame, text="Destination City:").pack(pady=5)
        dest_combo = ttk.Combobox(content_frame, values=city_list, width=35)
        dest_combo.pack(pady=5)
        dest_combo.set(city_list[0])
        
        if command == find_distance:
            use_traffic_var = tk.BooleanVar()
            ttk.Checkbutton(content_frame, text="Use Live Traffic Data", variable=use_traffic_var).pack(pady=5)
        
        result_text = scrolledtext.ScrolledText(content_frame, width=40, height=10, wrap=tk.WORD)
        result_text.pack(pady=10, fill="both", expand=True)
        
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=10)
        
        if command == find_shortest_path:
            ttk.Button(button_frame, text="Calculate",
                      command=lambda: command(src_combo, dest_combo, result_text)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Visualize on Map",
                      command=lambda: graph.visualize_map(highlight_path=result_text.get(1.0, tk.END).split("Path: ")[-1].strip().split(" -> ") if "Path" in result_text.get(1.0, tk.END) else None)).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Button(button_frame, text="Calculate",
                      command=lambda: command(src_combo, dest_combo, result_text, use_traffic_var)).pack(side=tk.LEFT, padx=5)
    elif command == find_tsp_route:
        ttk.Label(content_frame, text="Select Cities to Visit:").pack(pady=5)
        listbox = tk.Listbox(content_frame, selectmode=tk.MULTIPLE, width=40, height=10)
        for city in city_list:
            listbox.insert(tk.END, city)
        listbox.pack(pady=5)
        result_text = scrolledtext.ScrolledText(content_frame, width=40, height=8, wrap=tk.WORD)
        result_text.pack(pady=10, fill="both", expand=True)
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Calculate",
                  command=lambda: command(listbox, result_text)).pack(side=tk.LEFT, padx=5)
    elif command == simulate_road_closure:
        ttk.Label(content_frame, text="Select Roads to Close:").pack(pady=5)
        listbox = tk.Listbox(content_frame, selectmode=tk.MULTIPLE, width=40, height=10)
        for src, dest in edge_list:
            listbox.insert(tk.END, f"{src} -> {dest}")
        listbox.pack(pady=5)
        result_text = scrolledtext.ScrolledText(content_frame, width=40, height=8, wrap=tk.WORD)
        result_text.pack(pady=10, fill="both", expand=True)
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Simulate",
                  command=lambda: command(listbox, result_text)).pack(side=tk.LEFT, padx=5)
    else:
        result_text = scrolledtext.ScrolledText(content_frame, width=40, height=10, wrap=tk.WORD)
        result_text.pack(pady=10, fill="both", expand=True)
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Calculate",
                  command=lambda: command(result_text)).pack(side=tk.LEFT, padx=5)
        if command == find_mst:
            ttk.Button(button_frame, text="Visualize on Map",
                      command=lambda: graph.visualize_map()).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="Export PDF",
              command=lambda: export_to_pdf_with_dialog(title, result_text)).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Export CSV",
              command=lambda: export_to_csv_with_dialog(command, result_text)).pack(side=tk.LEFT, padx=5)

def find_shortest_path(src_combo, dest_combo, result_text):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Calculating...\n")
    root.config(cursor="wait")
    result_text.config(state='disabled')
    
    def calculate():
        src = src_combo.get()
        dest = dest_combo.get()
        if src in graph.adjacency_list and dest in graph.adjacency_list:
            distance, path = graph.dijkstra(src, dest)
            root.after(0, lambda: update_result(distance, path))
        else:
            root.after(0, lambda: update_result(None, "Error: Invalid city names"))
    
    def update_result(distance, path):
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        if distance is None:
            result_text.insert(tk.END, path)
        elif distance == float('inf'):
            result_text.insert(tk.END, "No path exists between the cities")
        else:
            result_text.insert(tk.END, f"Distance: {distance} km\nPath: {path}")
        root.config(cursor="")
        result_text.config(state='normal')
    
    threading.Thread(target=calculate, daemon=True).start()

def find_distance(src_combo, dest_combo, result_text, use_traffic_var):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Calculating...\n")
    root.config(cursor="wait")
    result_text.config(state='disabled')
    
    def calculate():
        src = src_combo.get()
        dest = dest_combo.get()
        if src in graph.adjacency_list and dest in graph.adjacency_list:
            if use_traffic_var.get():
                distance, _ = graph.get_live_traffic_weight(src, dest, result_text)
                if distance == float('inf'):
                    root.after(0, lambda: update_result(None, "No direct route found with live data"))
                else:
                    root.after(0, lambda: update_result(distance, f"Live Distance: {distance} km"))
            else:
                for edge in graph.adjacency_list.get(src, []):
                    if edge[0] == dest:
                        root.after(0, lambda: update_result(edge[1], f"Direct Distance: {edge[1]} km"))
                        return
                root.after(0, lambda: update_result(None, "No direct road found"))
        else:
            root.after(0, lambda: update_result(None, "Error: Invalid city names"))
    
    def update_result(distance, message):
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, message)
        root.config(cursor="")
        result_text.config(state='normal')
    
    threading.Thread(target=calculate, daemon=True).start()

def find_central_hub(result_text):
    result_text.delete(1.0, tk.END)
    max_hub, max_deg, max_diversity, min_hub, min_deg, min_diversity = graph.degree_centrality()
    result_text.insert(tk.END, f"Most Connected: {max_hub} ({max_deg} cities, Diversity Score = {max_diversity})\n"
                              f"Least Connected: {min_hub} ({min_deg} cities, Diversity Score = {min_diversity})")

def find_critical_roads(result_text):
    result_text.delete(1.0, tk.END)
    bridges = graph.critical_connections()
    result_text.insert(tk.END, f"Critical Connections: {bridges}")

def find_mst(result_text):
    result_text.delete(1.0, tk.END)
    mst = graph.prim(next(iter(graph.adjacency_list)))
    result_text.insert(tk.END, "Minimum Spanning Tree Edges:\n")
    for parent, child, weight in mst:
        result_text.insert(tk.END, f"{parent} -> {child} ({weight} km)\n")
    graph.visualize_mst(mst)

def find_all_shortest_paths(result_text):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Calculating all shortest paths...\n\n")
    fw_result = graph.floyd_warshall()
    for (src, dest), (distance, path) in fw_result.items():
        result_text.insert(tk.END, f"From {src} to {dest}:\n")
        result_text.insert(tk.END, f"Distance: {distance} km\n")
        result_text.insert(tk.END, f"Path: {' -> '.join(path)}\n\n")

def rank_cities_by_degree(result_text):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "City Rankings by Degree:\n")
    sorted_cities = graph.compute_city_rankings()
    for i, (city, degree, diversity) in enumerate(sorted_cities, 1):
        result_text.insert(tk.END, f"{i}. {city}: Degree = {degree}, Diversity Score = {diversity}\n")

def find_tsp_route(listbox, result_text):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Calculating...\n")
    root.config(cursor="wait")
    result_text.config(state='disabled')
    
    def calculate():
        selected_indices = listbox.curselection()
        selected_cities = [listbox.get(i) for i in selected_indices]
        if len(selected_cities) < 2:
            root.after(0, lambda: update_result(None, "Error: Select at least 2 cities"))
            return
        total_cost, path = graph.tsp_bitmask_dp(selected_cities)
        if total_cost == float('inf'):
            root.after(0, lambda: update_result(None, "Error: No valid route exists"))
        else:
            root.after(0, lambda: update_result(total_cost, f"Total Distance: {total_cost} km\nRoute: {' -> '.join(path)}"))
    
    def update_result(total_cost, message):
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, message)
        root.config(cursor="")
        result_text.config(state='normal')
    
    threading.Thread(target=calculate, daemon=True).start()

def simulate_road_closure(listbox, result_text):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Calculating...\n")
    root.config(cursor="wait")
    result_text.config(state='disabled')
    
    def calculate():
        selected_indices = listbox.curselection()
        selected_roads = [listbox.get(i).split(" -> ") for i in selected_indices]
        closed_roads = [(src, dest) for src, dest in selected_roads]
        if not closed_roads:
            root.after(0, lambda: update_result(None, "Error: Select at least one road to close"))
            return
        mst, components, new_links = graph.simulate_road_closure(closed_roads)
        output = "Minimum Spanning Tree Edges:\n"
        for parent, child, weight in mst:
            output += f"{parent} -> {child} ({weight} km)\n"
        output += "\nConnected Components:\n"
        for i, component in enumerate(components, 1):
            output += f"Component {i}: {component}\n"
        output += "\nProposed New Links:\n"
        if new_links:
            for src, dest, weight in new_links:
                output += f"{src} -> {dest} ({weight} km)\n"
        else:
            output += "No new links needed (graph remains connected)\n"
        root.after(0, lambda: update_result(mst, output))
    
    def update_result(mst, message):
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, message)
        graph.visualize_mst(mst)
        root.config(cursor="")
        result_text.config(state='normal')
    
    threading.Thread(target=calculate, daemon=True).start()

def show_graph():
    graph.visualize_graph()

# Main GUI
root = tk.Tk()
root.title("CityLink Analyzer")
root.geometry("700x600")
root.configure(bg='#e0e7ff')
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('Accent.TButton', font=('Helvetica', 12, 'bold'), padding=10, background='#4CAF50')
style.map('Accent.TButton', background=[('active', '#45a049')])
style.configure('TLabel', background='#e0e7ff')
style.configure('TCombobox', font=('Helvetica', 10))

# Show loading message during CSV update
header_frame = ttk.Frame(root, padding="20", relief="raised", borderwidth=2)
header_frame.pack(fill="x")
loading_label = ttk.Label(header_frame, text="Loading... Updating distances with live data...", font=('Helvetica', 12))
loading_label.pack()

# Initialize graph and update CSV in a separate thread
graph = None

def initialize_graph():
    global graph
    # Skip live update for initial graph load if "Get Direct Distance" will handle it separately
    graph = WeightedGraph(loading_label, skip_live_update=False)
    root.after(0, lambda: finalize_gui())

def finalize_gui():
    loading_label.destroy()
    ttk.Label(header_frame, text="CityLink Analyzer", font=('Helvetica', 26, 'bold')).pack()
    ttk.Label(header_frame, text="Explore India's City Connections", font=('Helvetica', 12, 'italic')).pack()

    global city_list, edge_list
    city_list = sorted(list(graph.adjacency_list.keys()))
    edge_list = [(src, dest) for src in graph.adjacency_list for dest, _ in graph.adjacency_list[src] if src < dest]

    button_frame = ttk.Frame(root, padding="30")
    button_frame.pack(expand=True)
    buttons = [
        ("Find Shortest Path", lambda: open_window("Shortest Path Finder", find_shortest_path)),
        ("Get Direct Distance", lambda: open_window("Distance Calculator", find_distance)),
        ("Find Central Hubs", lambda: open_window("Centrality Analyzer", find_central_hub)),
        ("Identify Critical Roads", lambda: open_window("Critical Roads", find_critical_roads)),
        ("Calculate MST", lambda: open_window("Minimum Spanning Tree", find_mst)),
        ("All Shortest Paths", lambda: open_window("All Pairs Shortest Paths", find_all_shortest_paths)),
        ("Rank Cities by Degree", lambda: open_window("City Degree Rankings", rank_cities_by_degree)),
        ("Traveling Salesman Route", lambda: open_window("TSP Route Finder", find_tsp_route)),
        ("Simulate Road Closure", lambda: open_window("Road Closure Simulation", simulate_road_closure)),
        ("Visualize Network", show_graph)
    ]
    for i, (text, cmd) in enumerate(buttons):
        ttk.Button(button_frame, text=text, command=cmd, style='Accent.TButton', width=25).grid(row=i//2, column=i%2, padx=10, pady=10)
    
    footer_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=1)
    footer_frame.pack(side=tk.BOTTOM, fill="x")
    ttk.Label(footer_frame, text=" AMAN KAUSHAL | MANAN RATHI | PRAKHAR AGRAWAL | CITY LINK ANALYZER", font=('Helvetica', 10, 'italic')).pack()

threading.Thread(target=initialize_graph, daemon=True).start()

root.mainloop()