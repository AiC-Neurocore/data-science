# Установка необходимых библиотек
!pip install networkx pyvis ipywidgets community plotly

import networkx as nx
from pyvis.network import Network
from google.colab import files
import io
import json
import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np
from community import community_louvain
import pandas as pd
import plotly.express as px
import pickle
import os

# Загрузка файла GraphML
uploaded = files.upload()
graphml_file = list(uploaded.keys())[0]

# Чтение графа из файла GraphML
G = nx.read_graphml(graphml_file)

# Кэширование кластеризации
cache_file = "partition_cache.pkl"
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        partition = pickle.load(f)
else:
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    partition = community_louvain.best_partition(G_undirected)
    with open(cache_file, 'wb') as f:
        pickle.dump(partition, f)

# Экспоненциальное масштабирование размеров узлов
def get_node_size(size, min_size=10, max_size=100, exponent=5):
    if size is None or float(size) <= 0:
        return min_size
    sizes = [float(G.nodes[n].get('size', 10)) for n in G.nodes if float(G.nodes[n].get('size', 10)) > 0]
    min_size_val, max_size_val = min(sizes, default=1), max(sizes, default=1)
    if max_size_val == min_size_val:
        return min_size
    normalized = (float(size) - min_size_val) / (max_size_val - min_size_val)
    scaled = normalized ** exponent
    return min_size + (max_size - min_size) * scaled

# Нормализация толщины рёбер
def get_edge_width(weight, min_width=0.5, max_width=5):
    if weight is None or float(weight) <= 0:
        return min_width
    weights = [float(G.edges[e].get('weight', 0)) for e in G.edges if float(G.edges[e].get('weight', 0)) > 0]
    min_weight, max_weight = min(weights, default=1), max(weights, default=1)
    if max_weight == min_weight:
        return min_width
    normalized = (float(weight) - min_weight) / (max_weight - min_weight)
    return min_width + (max_width - min_width) * normalized

# Виджеты для настройки
min_size_filter = widgets.FloatSlider(value=10, min=10, max=max([float(G.nodes[n].get('size', 10)) for n in G.nodes], default=10), step=0.1, description='Min Wallet Size:')
min_weight_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.edges[e].get('weight', 0)) for e in G.edges], default=1), step=0.1, description='Min Tx Weight:')
min_color_filter = widgets.FloatSlider(value=0, min=0, max=max([float(G.nodes[n].get('color', 0)) for n in G.nodes], default=1), step=0.1, description='Min Color:')
node_size_scale = widgets.IntSlider(value=300, min=20, max=600, step=10, description='Max Node Size:')
size_exponent = widgets.FloatSlider(value=5.0, min=1.0, max=8.0, step=0.1, description='Size Exponent:')
edge_width_scale = widgets.FloatSlider(value=5.0, min=0.5, max=15.0, step=0.5, description='Max Edge Width:')
font_size = widgets.IntSlider(value=14, min=8, max=24, step=2, description='Font Size:')
bg_color = widgets.ColorPicker(value='#0a0a23', description='Background:')
font_color = widgets.ColorPicker(value='#ffffff', description='Font Color:')
physics_enabled = widgets.Checkbox(value=True, description='Enable Physics')
gravity = widgets.FloatSlider(value=-60, min=-100, max=0, step=5, description='Gravity:')
highlight_cluster = widgets.Dropdown(options=['All'] + list(set(partition.values())), value='All', description='Highlight Cluster:')
show_top_wallets = widgets.Checkbox(value=True, description='Highlight Top Wallets')
top_wallets_count = widgets.IntSlider(value=5, min=1, max=20, step=1, description='Top Wallets:')
color_mode = widgets.Dropdown(options=['Cluster', 'Color Attribute', 'Degree'], value='Color Attribute', description='Color Mode:')
focus_wallet = widgets.Text(value='', description='Focus Wallet:', placeholder='Enter wallet address')
render_mode = widgets.Dropdown(options=['2D', '3D'], value='2D', description='Render Mode:')
rotate_top_wallets = widgets.Checkbox(value=True, description='Rotate Top Wallets')
high_quality = widgets.Checkbox(value=True, description='High Quality Render')
theme_mode = widgets.Dropdown(options=['Dark', 'Light'], value='Dark', description='Theme:')

# Функция для 2D-визуализации
def render_2d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees):
    net = Network(notebook=True, height="800px", width="100%", 
                  bgcolor=bg_color.value, font_color=font_color.value, cdn_resources='remote')

    for node in filtered_nodes:
        size_val = float(G.nodes[node].get('size', 10))
        label = G.nodes[node].get('label', node)[:8] if node not in top_wallets else G.nodes[node].get('label', node)[:16]
        color_val = float(G.nodes[node].get('color', 0))
        cluster = partition.get(node, 0)
        node_size = get_node_size(size_val, max_size=node_size_scale.value, exponent=size_exponent.value)

        if color_mode.value == 'Cluster':
            color = {'background': f'hsl({(cluster * 60) % 360}, 70%, 50%)',
                     'border': '#ffffff',
                     'highlight': {'background': f'hsl({(cluster * 60) % 360}, 70%, 60%)', 'border': '#ffffff'}}
        elif color_mode.value == 'Color Attribute':
            color = {'background': f'hsl({color_val % 360}, 70%, 50%)',
                     'border': '#ffffff',
                     'highlight': {'background': f'hsl({color_val % 360}, 70%, 60%)', 'border': '#ffffff'}}
        else:
            degree = degrees[node]
            max_degree = max(degrees.values(), default=1)
            color = {'background': f'hsl({(degree/max_degree) * 240}, 70%, 50%)',
                     'border': '#ffffff',
                     'highlight': {'background': f'hsl({(degree/max_degree) * 240}, 70%, 60%)', 'border': '#ffffff'}}
        
        if highlight_cluster.value != 'All' and int(highlight_cluster.value) != cluster:
            color = {'background': '#444444', 'border': '#666666'}
            opacity = 0.3
        else:
            opacity = 1.0
        if node in top_wallets:
            color = {'background': 'radial-gradient(circle, #ff4d4d, #cc0000)',
                     'border': '#ffffff',
                     'highlight': {'background': '#ff8080', 'border': '#ffffff'}}
            node_size *= 1.7

        title = f"Address: {G.nodes[node].get('label', node)}<br>Size: {size_val:.2f}<br>Color: {color_val:.2f}<br>Cluster: {cluster}<br>Degree: {degrees[node]}"
        net.add_node(node, label=label, size=node_size, title=title, color=color, opacity=opacity,
                     borderWidth=2, borderWidthSelected=5, mass=size_val/10)

    top_edges = sorted(filtered_edges, key=lambda e: float(G.edges[e].get('weight', 1)), reverse=True)[:min(10, len(filtered_edges))]
    for u, v in filtered_edges:
        weight = float(G.edges[u, v].get('weight', 1))
        width = get_edge_width(weight, max_width=edge_width_scale.value)
        opacity = 1.0 if highlight_cluster.value == 'All' or (partition.get(u) == partition.get(v) == int(highlight_cluster.value)) else 0.3
        edge_class = 'top-edge' if (u, v) in top_edges and high_quality.value else ''
        net.add_edge(u, v, width=width, title=f"Tx Weight: {weight:.2f}", 
                     color={'color': '#aaaaaa', 'opacity': opacity, 'highlight': '#ffffff'},
                     arrows='to', custom_class=edge_class)

    options = {
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 10,
                "max": node_size_scale.value
            },
            "font": {
                "size": font_size.value,
                "face": "arial",
                "color": font_color.value
            },
            "shadow": {"enabled": high_quality.value, "size": 10, "x": 5, "y": 5},
            "borderWidth": 2
        },
        "edges": {
            "color": {"inherit": False},
            "smooth": {"type": "dynamic", "roundness": 0.5},
            "shadow": {"enabled": high_quality.value},
            "hoverWidth": 3,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}
        },
        "physics": {
            "enabled": physics_enabled.value,
            "forceAtlas2Based": {
                "gravitationalConstant": gravity.value,
                "centralGravity": 0.01,
                "springLength": 100,
                "avoidOverlap": 0.5
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "zoomView": True,
            "dragView": True,
            "dragNodes": True,
            "hoverConnectedEdges": True,
            "selectConnectedEdges": True
        }
    }
    net.set_options(json.dumps(options))
    return net

# Функция для 3D-визуализации
def render_3d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees):
    nodes_data = []
    for node in filtered_nodes:
        size_val = float(G.nodes[node].get('size', 10))
        color_val = float(G.nodes[node].get('color', 0))
        cluster = partition.get(node, 0)
        node_size = get_node_size(size_val, max_size=node_size_scale.value/20, exponent=size_exponent.value)
        if color_mode.value == 'Cluster':
            color = f'hsl({(cluster * 60) % 360}, 70%, 50%)'
        elif color_mode.value == 'Color Attribute':
            color = f'hsl({color_val % 360}, 70%, 50%)'
        else:
            degree = degrees[node]
            max_degree = max(degrees.values(), default=1)
            color = f'hsl({(degree/max_degree) * 240}, 70%, 50%)'
        if node in top_wallets:
            color = '#ff4d4d'
        nodes_data.append({
            'id': node,
            'size': node_size,
            'color': color,
            'label': G.nodes[node].get('label', node)[:16] if node in top_wallets else G.nodes[node].get('label', node)[:8]
        })

    links_data = []
    for u, v in filtered_edges:
        weight = float(G.edges[u, v].get('weight', 1))
        links_data.append({
            'source': u,
            'target': v,
            'value': get_edge_width(weight, max_width=edge_width_scale.value/2)
        })

    html_content = f"""
    <html>
    <head>
        <script src="https://unpkg.com/3d-force-graph@1.70.10/dist/3d-force-graph.min.js"></script>
        <style>
            body {{ margin: 0; background: {bg_color.value}; }}
            #graph {{ width: 100%; height: 800px; }}
        </style>
    </head>
    <body>
        <div id="graph"></div>
        <script>
            const nodes = {json.dumps(nodes_data)};
            const links = {json.dumps(links_data)};
            const Graph = ForceGraph3D()
                (document.getElementById('graph'))
                .graphData({{ nodes: nodes, links: links }})
                .nodeLabel('label')
                .nodeVal('size')
                .nodeColor('color')
                .linkWidth('value')
                .linkDirectionalArrowLength(3)
                .linkDirectionalArrowRelPos(1)
                .backgroundColor('{bg_color.value}')
                .nodeOpacity(0.9)
                .linkOpacity(0.7)
                .d3Force('charge', d3.forceManyBody().strength(-100));
        </script>
    </body>
    </html>
    """
    with open("graph_3d.html", "w") as f:
        f.write(html_content)
    return html_content

# Функция для создания легенды
def create_legend(filtered_nodes, filtered_edges, top_wallet, render_mode):
    total_nodes = len(filtered_nodes)
    total_edges = len(filtered_edges)
    max_size = max([float(G.nodes[n].get('size', 10)) for n in filtered_nodes], default=10)
    cluster_count = len(set(partition[n] for n in filtered_nodes))
    return f"""
    <div style='background: rgba(0,0,0,0.85); color: white; padding: 15px; border-radius: 10px; position: absolute; top: 10px; left: 10px; z-index: 1000; font-family: Arial;'>
        <b>Tokenomics Visualization ({render_mode})</b><br>
        - <b>Nodes</b>: {total_nodes}<br>
        - <b>Edges</b>: {total_edges}<br>
        - <b>Max Wallet Size</b>: {max_size:.2f} (Address: {top_wallet[:8]})<br>
        - <b>Clusters</b>: {cluster_count}<br>
        - <b>Node Size</b>: Exponentially scaled (adjust with Size Exponent)<br>
        - <b>Edge Width</b>: Transaction weight<br>
        - <b>Node Color</b>: {color_mode.value}<br>
        - <b>Tooltip</b>: Hover for details<br>
        - <b>Top Wallets</b>: Red nodes with full labels
    </div>
    """

# Функция для аналитики
def render_analytics(filtered_nodes):
    sizes = [float(G.nodes[n].get('size', 10)) for n in filtered_nodes]
    fig = px.histogram(x=sizes, nbins=20, title="Wallet Size Distribution", labels={'x': 'Wallet Size'})
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    fig.show()

# Основная функция обновления
def update_visualization(change=None):
    display(HTML("<p style='color: white;'>Loading graph...</p>"))

    # Адаптация темы
    if theme_mode.value == 'Light':
        bg_color.value, font_color.value = '#f0f0f0', '#000000'
    else:
        bg_color.value, font_color.value = '#0a0a23', '#ffffff'

    # Фильтрация узлов и рёбер
    filtered_nodes = [n for n in G.nodes if float(G.nodes[n].get('size', 10)) >= min_size_filter.value and float(G.nodes[n].get('color', 0)) >= min_color_filter.value]
    if focus_wallet.value.strip():
        try:
            focus_node = next(n for n in filtered_nodes if G.nodes[n].get('label', n) == focus_wallet.value.strip())
            neighbors = set(nx.ego_graph(G, focus_node, radius=2).nodes())
            filtered_nodes = [n for n in filtered_nodes if n in neighbors]
        except StopIteration:
            pass
    filtered_edges = [(u, v) for u, v in G.edges if u in filtered_nodes and v in filtered_nodes and float(G.edges[u, v].get('weight', 0)) >= min_weight_filter.value]

    # Определение топ-кошельков
    if show_top_wallets.value:
        sizes = [(n, float(G.nodes[n].get('size', 10))) for n in filtered_nodes]
        top_wallets = [n for n, _ in sorted(sizes, key=lambda x: x[1], reverse=True)[:top_wallets_count.value]]
    else:
        top_wallets = []

    # Вычисление степеней узлов
    degrees = dict(G.degree(filtered_nodes))

    # Рендеринг
    if render_mode.value == '2D':
        net = render_2d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees)
        net.show("graph.html")
        with open("graph.html", "r") as f:
            html_content = f.read()
    else:
        html_content = render_3d_visualization(filtered_nodes, filtered_edges, top_wallets, degrees)

    # Стили и анимации
    rotate_animation = 'rotate 20s linear infinite' if rotate_top_wallets.value else 'none'
    custom_css = f"""
    <style>
        .vis-network canvas {{
            background: linear-gradient(135deg, {bg_color.value} 0%, {'#e0e0e0' if theme_mode.value == 'Light' else '#16213e'} 100%);
            transition: background 0.5s;
        }}
        .vis-network .vis-node {{
            transition: all 0.3s;
        }}
        .vis-network .vis-node:hover {{
            animation: pulse 1.2s infinite;
        }}
        .vis-network .vis-node.top-wallet {{
            animation: wave 2s infinite, {rotate_animation};
        }}
        .vis-network .vis-edge.top-edge {{
            animation: pulse-edge 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.15); }}
            100% {{ transform: scale(1); }}
        }}
        @keyframes wave {{
            0% {{ box-shadow: 0 0 10px #ff4d4d; }}
            50% {{ box-shadow: 0 0 25px #ff4d4d; }}
            100% {{ box-shadow: 0 0 10px #ff4d4d; }}
        }}
        @keyframes rotate {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes pulse-edge {{
            0% {{ stroke-opacity: 0.7; }}
            50% {{ stroke-opacity: 1; }}
            100% {{ stroke-opacity: 0.7; }}
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <script>
        if ({'true' if render_mode.value == '2D' and high_quality.value else 'false'}) {{
            particlesJS('particles-js', {{
                "particles": {{
                    "number": {{ "value": 60, "density": {{ "enable": true, "value_area": 800 }} }},
                    "color": {{ "value": "{'#000000' if theme_mode.value == 'Light' else '#ffffff'}" }},
                    "shape": {{ "type": "circle" }},
                    "opacity": {{ "value": 0.5, "random": true }},
                    "size": {{ "value": 3, "random": true }},
                    "line_linked": {{ "enable": false }},
                    "move": {{ "enable": true, "speed": 2, "direction": "none", "random": true }}
                }},
                "interactivity": {{ "detect_on": "canvas", "events": {{ "onhover": {{ "enable": true, "mode": "repulse" }} }} }}
            }});
        }}
    </script>
    """

    # Легенда
    top_wallet = max([(n, float(G.nodes[n].get('size', 10))) for n in filtered_nodes], key=lambda x: x[1], default=('None', 0))[0]
    legend_html = create_legend(filtered_nodes, filtered_edges, top_wallet, render_mode.value)

    # Сохранение скриншота
    screenshot_button = widgets.Button(description="Save Screenshot")
    def on_screenshot_button_clicked(b):
        display(HTML("""
        <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
        <script>
            html2canvas(document.querySelector(".vis-network") || document.querySelector("#graph")).then(canvas => {{
                var link = document.createElement('a');
                link.download = 'graph_screenshot.png';
                link.href = canvas.toDataURL();
                link.click();
            }});
        </script>
        """))
    screenshot_button.on_click(on_screenshot_button_clicked)

    # Экспорт данных
    export_button = widgets.Button(description="Export CSV")
    def on_export_button_clicked(b):
        df = pd.DataFrame([(n, G.nodes[n].get('size', 10), G.nodes[n].get('color', 0), partition.get(n, 0), degrees[n]) for n in filtered_nodes],
                          columns=['Address', 'Size', 'Color', 'Cluster', 'Degree'])
        df.to_csv("graph_data.csv", index=False)
        files.download("graph_data.csv")
    export_button.on_click(on_export_button_clicked)

    # Отображение
    display(HTML(f"<div id='particles-js' style='position: absolute; width: 100%; height: 800px;'></div>" + custom_css + legend_html + html_content))
    display(widgets.HBox([screenshot_button, export_button]))

    # Аналитика
    render_analytics(filtered_nodes)

# Привязка виджетов
min_size_filter.observe(update_visualization, names='value')
min_weight_filter.observe(update_visualization, names='value')
min_color_filter.observe(update_visualization, names='value')
node_size_scale.observe(update_visualization, names='value')
size_exponent.observe(update_visualization, names='value')
edge_width_scale.observe(update_visualization, names='value')
font_size.observe(update_visualization, names='value')
bg_color.observe(update_visualization, names='value')
font_color.observe(update_visualization, names='value')
physics_enabled.observe(update_visualization, names='value')
gravity.observe(update_visualization, names='value')
highlight_cluster.observe(update_visualization, names='value')
show_top_wallets.observe(update_visualization, names='value')
top_wallets_count.observe(update_visualization, names='value')
color_mode.observe(update_visualization, names='value')
focus_wallet.observe(update_visualization, names='value')
render_mode.observe(update_visualization, names='value')
rotate_top_wallets.observe(update_visualization, names='value')
high_quality.observe(update_visualization, names='value')
theme_mode.observe(update_visualization, names='value')

# Отображение виджетов
display(widgets.VBox([
    min_size_filter, min_weight_filter, min_color_filter, node_size_scale, size_exponent,
    edge_width_scale, font_size, bg_color, font_color, physics_enabled, gravity,
    highlight_cluster, show_top_wallets, top_wallets_count, color_mode, focus_wallet,
    render_mode, rotate_top_wallets, high_quality, theme_mode
]))

# Кнопка для скачивания HTML
download_button = widgets.Button(description="Download HTML")
def on_download_button_clicked(b):
    files.download("graph.html" if render_mode.value == '2D' else "graph_3d.html")
download_button.on_click(on_download_button_clicked)
display(download_button)

# Первоначальная визуализация
update_visualization()
