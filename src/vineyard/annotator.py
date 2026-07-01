"""Interactive acoustic data annotator built with Dash.

Run via console script:
    annotator

Or directly:
    python -m vineyard.annotator

Then open http://localhost:8050 in a browser.

Click on any spectrogram to log the time, sensor, channel, and frequency to
data/acoustic/annotations.csv. A green X marks each logged point.
"""

import csv
import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dotenv
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, ctx, dcc, html
from plotly.subplots import make_subplots
from scipy import signal
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream

dotenv.load_dotenv()

ROOT = Path(__file__).parent.parent.parent  # vineyard_wind/
INVENTORY_PATH = ROOT / "data" / "acoustic"
ANNOTATION_CSV = ROOT / "data" / "acoustic" / "annotations.csv"

SEGMENT_S = 60
DEFAULT_START = "2023-12-01T21:00:00"

STFT_PARAMS = {"nperseg": 128, "hop": 8, "nfft": 2**12}  # ~5% overlap

SENSORS: dict[str, dict] = {
    "3dvha": {
        "channel_names": [
            "Front Hydrophone",
            "Right Hydrophone",
            "Left Hydrophone",
            "Back Hydrophone",
            "Particle Motion X",
            "Particle Motion Y",
            "Particle Motion Z",
            "Omni Hydrophone",
        ],
    },
    "vla1": {
        "channel_names": ["Channel 1", "Channel 2", "Channel 3", "Channel 4"],
    },
    "vla2": {
        "channel_names": ["Channel 1", "Channel 2", "Channel 3", "Channel 4"],
    },
}

ANNOTATION_FIELDS = [
    "time_utc",
    "sensor",
    "channel",
    "channel_name",
    "frequency_hz",
    "segment_start",
]


def all_channel_options() -> list[dict]:
    """Flat list of all sensor:channel options for the checklist."""
    options = []
    for sensor, info in SENSORS.items():
        for i, name in enumerate(info["channel_names"]):
            options.append(
                {
                    "label": f"{sensor.upper()} – {name}",
                    "value": f"{sensor}:{i}",
                }
            )
    return options


def parse_selections(values: list[str]) -> list[tuple[str, int]]:
    """Parse ['sensor:ch', ...] checklist values into (sensor, channel_idx) pairs."""
    result = []
    for v in values or []:
        sensor, ch_str = v.split(":")
        result.append((sensor, int(ch_str)))
    return result


def load_segment(
    sensor: str,
    seg_start: np.datetime64,
    seg_end: np.datetime64,
    channels: list[int],
    target_fs: float,
    filt_freq: list[float],
) -> DataStream | None:
    inventory = INVENTORY_PATH / f"inventory_{sensor}.csv"
    try:
        ds = read_inventory(inventory, seg_start, seg_end, channels=channels)
    except Exception as exc:
        logging.error("read_inventory failed for %s: %s", sensor, exc)
        return None

    dec = int(np.round(ds.stats.sampling_rate / target_fs))
    if dec > 1:
        ds.decimate(dec)

    if len(filt_freq) == 2:
        ds.filter(filt_type="bandpass", freq=filt_freq)
    elif len(filt_freq) == 1:
        ds.filter(filt_type="highpass", freq=filt_freq[0])

    if sensor == "3dvha":
        ds_out = ds.copy()
        for i, ch in enumerate(channels):
            if ch < 4:
                ds_out.data[i] = -ds.data[i]
        return ds_out

    return ds


def load_all_sensors(
    selections: list[tuple[str, int]],
    seg_start: np.datetime64,
    seg_end: np.datetime64,
    target_fs: float,
    filt_freq: list[float],
) -> tuple[dict[str, DataStream], dict[str, list[int]]]:
    """Load data for every unique sensor appearing in selections.

    Returns:
        ds_map:   sensor → DataStream
        ch_map:   sensor → sorted list of loaded channel indices
    """
    sensor_channels: dict[str, list[int]] = {}
    for sensor, ch in selections:
        sensor_channels.setdefault(sensor, []).append(ch)

    ds_map, ch_map = {}, {}
    for sensor, chs in sensor_channels.items():
        chs_sorted = sorted(set(chs))
        ds = load_segment(sensor, seg_start, seg_end, chs_sorted, target_fs, filt_freq)
        if ds is not None:
            ds_map[sensor] = ds
            ch_map[sensor] = chs_sorted

    return ds_map, ch_map


def spectrogram_db(
    data: np.ndarray,
    fs: float,
    nperseg: int,
    hop: int,
    nfft: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = signal.windows.hann(nperseg)
    stft = signal.ShortTimeFFT(win, hop, fs, mfft=nfft, scale_to="psd")
    Sxx = stft.spectrogram(data)
    f = stft.f
    t = stft.t(len(data))

    mask = (f >= fmin) & (f <= fmax)
    f, Sxx = f[mask], Sxx[mask, :]
    Sxx_db = 10 * np.log10(Sxx / (Sxx.max() + 1e-30))
    return t, f, Sxx_db


def build_figure(
    selections: list[tuple[str, int]],
    ds_map: dict[str, DataStream],
    ch_map: dict[str, list[int]],
    seg_start: np.datetime64,
    fmin: float,
    fmax: float,
    vmin: float,
    vmax: float,
    annotations: list[dict],
) -> go.Figure:
    # Only keep selections whose sensor loaded successfully
    valid = [(s, c) for s, c in selections if s in ds_map]
    n = len(valid)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No data available for this time range.", template="plotly_dark"
        )
        return fig

    seg_end = seg_start + np.timedelta64(int(SEGMENT_S * 1e6), "us")

    fig = make_subplots(
        rows=2 * n,
        cols=1,
        shared_xaxes=True,
        row_heights=[h for _ in range(n) for h in [1, 3]],
        vertical_spacing=0.02,
    )

    x_range = None

    for i, (sensor, ch) in enumerate(valid):
        ts_row = 2 * i + 1
        sp_row = 2 * i + 2
        ds = ds_map[sensor]
        data_idx = ch_map[sensor].index(ch)
        label = f"{sensor.upper()} – {SENSORS[sensor]['channel_names'][ch]}"
        fs = ds.stats.sampling_rate

        # Time series
        t_s = np.arange(ds.num_samples) / fs
        amp = ds.data[data_idx]
        peak = np.max(np.abs(amp)) or 1.0

        fig.add_trace(
            go.Scatter(
                x=t_s,
                y=amp / peak,
                mode="lines",
                line=dict(color="#5b8dd9", width=0.7),
                name=label,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=ts_row,
            col=1,
        )

        # Spectrogram heatmap
        t_stft, f_stft, Sxx_db = spectrogram_db(
            ds.data[data_idx],
            fs,
            STFT_PARAMS["nperseg"],
            STFT_PARAMS["hop"],
            STFT_PARAMS["nfft"],
            fmin,
            fmax,
        )
        if x_range is None:
            x_range = [t_stft[0], t_stft[-1]]

        fig.add_trace(
            go.Heatmap(
                x=t_stft,
                y=f_stft,
                z=Sxx_db,
                colorscale="inferno",
                zmin=vmin,
                zmax=vmax,
                showscale=(i == n - 1),
                colorbar=dict(
                    title=dict(text="dB (norm.)", side="right"),
                    thickness=14,
                    len=0.4,
                    y=0.2,
                ),
                hovertemplate="t=%{x:.2f} s<br>f=%{y:.1f} Hz<br>%{z:.1f} dB<extra></extra>",
            ),
            row=sp_row,
            col=1,
        )

        # Annotation markers
        ann_t, ann_f = [], []
        for ann in annotations:
            if ann.get("sensor") != sensor:
                continue
            if int(ann.get("channel", -1)) != ch:
                continue
            try:
                ann_time = np.datetime64(ann["time_utc"], "us")
            except (KeyError, ValueError):
                continue
            if seg_start <= ann_time < seg_end:
                t_off = float((ann_time - seg_start).astype(np.int64)) / 1e6
                ann_t.append(t_off)
                ann_f.append(float(ann.get("frequency_hz", (fmin + fmax) / 2)))

        fig.add_trace(
            go.Scatter(
                x=ann_t,
                y=ann_f,
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=14,
                    color="lime",
                    line=dict(width=2.5, color="lime"),
                ),
                name="Annotations" if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate="t=%{x:.2f} s<br>f=%{y:.1f} Hz<extra>Annotation</extra>",
            ),
            row=sp_row,
            col=1,
        )

        # Axis styling
        fig.update_yaxes(
            title_text=label,
            title_font=dict(size=9),
            range=[-1.1, 1.1],
            showticklabels=False,
            row=ts_row,
            col=1,
        )
        fig.update_yaxes(
            title_text="Freq (Hz)",
            title_font=dict(size=9),
            row=sp_row,
            col=1,
        )
        fig.update_xaxes(
            title_text="Time offset (s)" if i == n - 1 else "",
            row=sp_row,
            col=1,
        )

    if x_range:
        fig.update_xaxes(range=x_range)

    seg_str = str(seg_start).replace("T", " ").split(".")[0]
    fig.update_layout(
        title=dict(text=f"{seg_str} UTC  (+{SEGMENT_S} s)", font=dict(size=13)),
        height=max(160 * 2 * n, 300),
        template="plotly_dark",
        margin=dict(l=15, r=80, t=55, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig


def ensure_csv() -> None:
    if not ANNOTATION_CSV.exists():
        ANNOTATION_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(ANNOTATION_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS).writeheader()


def append_csv(row: dict) -> None:
    ensure_csv()
    with open(ANNOTATION_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS).writerow(row)


def load_csv() -> list[dict]:
    if not ANNOTATION_CSV.exists():
        return []
    with open(ANNOTATION_CSV, newline="") as f:
        return list(csv.DictReader(f))


def create_app() -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.title = "Acoustic Annotator"

    app.layout = dbc.Container(
        fluid=True,
        children=[
            dcc.Store(id="seg-start-store"),
            dcc.Store(id="annotations-store", data=load_csv()),
            dbc.Row(
                dbc.Col(
                    html.H5("Acoustic Data Annotator", className="text-white py-2 mb-0")
                )
            ),
            dbc.Row(
                [
                    # Controls
                    dbc.Col(
                        width=1,
                        children=[
                            dbc.Card(
                                body=True,
                                className="mb-2 p-2",
                                style={"background": "#1e2130"},
                                children=[
                                    html.Label(
                                        "Channels", className="fw-bold text-white mb-1"
                                    ),
                                    dcc.Checklist(
                                        id="channel-cl",
                                        options=all_channel_options(),
                                        value=["3dvha:7", "vla1:2", "vla2:0"],
                                        className="small",
                                        inputStyle={"marginRight": "6px"},
                                        labelStyle={
                                            "display": "block",
                                            "lineHeight": "1.8",
                                            "color": "white",
                                        },
                                    ),
                                ],
                            ),
                            dbc.Card(
                                body=True,
                                className="mb-2 p-2",
                                style={"background": "#1e2130"},
                                children=[
                                    html.Label(
                                        "Segment start (UTC)",
                                        className="fw-bold text-white mb-1",
                                    ),
                                    dbc.Input(
                                        id="start-input",
                                        value=DEFAULT_START,
                                        type="text",
                                        size="sm",
                                        className="mb-2 font-monospace",
                                        style={"fontSize": "0.8rem"},
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button(
                                                    "◀ Prev",
                                                    id="prev-btn",
                                                    color="secondary",
                                                    size="sm",
                                                    className="w-100",
                                                )
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "Next ▶",
                                                    id="next-btn",
                                                    color="secondary",
                                                    size="sm",
                                                    className="w-100",
                                                )
                                            ),
                                        ],
                                        className="mb-2 g-1",
                                    ),
                                    dbc.Button(
                                        "Load",
                                        id="load-btn",
                                        color="primary",
                                        size="sm",
                                        className="w-100",
                                    ),
                                ],
                            ),
                            dbc.Card(
                                body=True,
                                className="mb-2 p-2",
                                style={"background": "#1e2130"},
                                children=[
                                    html.Label(
                                        "DSP", className="fw-bold text-white mb-1"
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Small(
                                                        "Target fs (Hz)",
                                                        className="text-muted",
                                                    ),
                                                    dbc.Input(
                                                        id="target-fs",
                                                        value=200,
                                                        type="number",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="mb-1 g-1",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Small(
                                                        "fmin (Hz)",
                                                        className="text-muted",
                                                    ),
                                                    dbc.Input(
                                                        id="fmin",
                                                        value=10,
                                                        type="number",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Small(
                                                        "fmax (Hz)",
                                                        className="text-muted",
                                                    ),
                                                    dbc.Input(
                                                        id="fmax",
                                                        value=50,
                                                        type="number",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="mb-1 g-1",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Small(
                                                        "vmin (dB)",
                                                        className="text-muted",
                                                    ),
                                                    dbc.Input(
                                                        id="vmin",
                                                        value=-60,
                                                        type="number",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Small(
                                                        "vmax (dB)",
                                                        className="text-muted",
                                                    ),
                                                    dbc.Input(
                                                        id="vmax",
                                                        value=0,
                                                        type="number",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="g-1",
                                    ),
                                ],
                            ),
                            dbc.Card(
                                body=True,
                                className="mb-2 p-2",
                                style={"background": "#1e2130"},
                                children=[
                                    html.Label(
                                        "Annotations",
                                        className="fw-bold text-white mb-1",
                                    ),
                                    html.Div(
                                        id="ann-count", className="text-info small mb-2"
                                    ),
                                    dbc.Button(
                                        "Clear session",
                                        id="clear-btn",
                                        color="danger",
                                        size="sm",
                                        outline=True,
                                        className="w-100 mb-1",
                                    ),
                                    html.Small(
                                        f"CSV: {ANNOTATION_CSV.relative_to(ROOT)}",
                                        className="text-muted",
                                        style={"fontSize": "0.7rem"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Main plot
                    dbc.Col(
                        width=11,
                        children=[
                            dcc.Loading(
                                dcc.Graph(
                                    id="main-graph",
                                    config={
                                        "scrollZoom": True,
                                        "displayModeBar": True,
                                        "modeBarButtonsToRemove": [
                                            "select2d",
                                            "lasso2d",
                                        ],
                                    },
                                    style={"minHeight": "400px"},
                                ),
                                type="circle",
                                color="#5b8dd9",
                            ),
                            html.Div(
                                id="click-feedback", className="text-success small mt-1"
                            ),
                        ],
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    [
                        html.H6(
                            "Recent annotations (this session)",
                            className="text-white mt-3",
                        ),
                        html.Div(id="ann-table"),
                    ]
                )
            ),
        ],
    )

    _register_callbacks(app)
    return app


def _register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("seg-start-store", "data"),
        Output("start-input", "value"),
        Input("load-btn", "n_clicks"),
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        State("start-input", "value"),
        State("seg-start-store", "data"),
    )
    def update_segment(_load, _prev, _next, start_input: str, current: str | None):
        dt = np.timedelta64(int(SEGMENT_S * 1e6), "us")
        triggered = ctx.triggered_id

        if triggered == "prev-btn" and current:
            t = np.datetime64(current, "us") - dt
        elif triggered == "next-btn" and current:
            t = np.datetime64(current, "us") + dt
        else:
            try:
                t = np.datetime64(start_input.strip(), "us")
            except Exception:
                t = np.datetime64(DEFAULT_START, "us")

        t_str = str(t)
        return t_str, t_str.split(".")[0]

    @app.callback(
        Output("main-graph", "figure"),
        Input("seg-start-store", "data"),
        Input("channel-cl", "value"),
        Input("target-fs", "value"),
        Input("fmin", "value"),
        Input("fmax", "value"),
        Input("vmin", "value"),
        Input("vmax", "value"),
        State("annotations-store", "data"),
    )
    def update_figure(
        seg_start_str, selected, target_fs, fmin, fmax, vmin, vmax, annotations
    ):
        if not seg_start_str or not selected:
            return go.Figure(layout=dict(template="plotly_dark"))

        selections = parse_selections(selected)
        seg_start = np.datetime64(seg_start_str, "us")
        seg_end = seg_start + np.timedelta64(int(SEGMENT_S * 1e6), "us")

        ds_map, ch_map = load_all_sensors(
            selections,
            seg_start,
            seg_end,
            target_fs=float(target_fs or 1000),
            filt_freq=[float(fmin or 15), float(fmax or 100)],
        )

        return build_figure(
            selections,
            ds_map,
            ch_map,
            seg_start,
            float(fmin or 15),
            float(fmax or 100),
            float(vmin or -60),
            float(vmax or 0),
            annotations or [],
        )

    @app.callback(
        Output("annotations-store", "data"),
        Output("click-feedback", "children"),
        Output("main-graph", "figure", allow_duplicate=True),
        Input("main-graph", "clickData"),
        Input("clear-btn", "n_clicks"),
        State("annotations-store", "data"),
        State("seg-start-store", "data"),
        State("channel-cl", "value"),
        prevent_initial_call=True,
    )
    def handle_click(click_data, _clear, annotations, seg_start_str, selected):
        annotations = annotations or []
        triggered = ctx.triggered_id

        if triggered == "clear-btn":
            return [], "Session annotations cleared (CSV unchanged).", dash.no_update

        if triggered != "main-graph" or not click_data or not seg_start_str:
            return annotations, dash.no_update, dash.no_update

        point = click_data["points"][0]
        curve_num = point.get("curveNumber", -1)

        selections = parse_selections(selected or [])
        n = len(selections)

        # Heatmap traces sit at indices 3*i + 1
        heatmap_indices = [3 * i + 1 for i in range(n)]
        if curve_num not in heatmap_indices:
            return annotations, "Click on a spectrogram to annotate.", dash.no_update

        selection_idx = heatmap_indices.index(curve_num)
        sensor, ch = selections[selection_idx]
        t_offset_s = point.get("x")
        freq = point.get("y")
        if t_offset_s is None or freq is None:
            return annotations, dash.no_update, dash.no_update

        seg_start = np.datetime64(seg_start_str, "us")
        abs_time = seg_start + np.timedelta64(int(float(t_offset_s) * 1e6), "us")
        abs_time_str = str(abs_time)
        ch_name = SENSORS[sensor]["channel_names"][ch]

        row = {
            "time_utc": abs_time_str,
            "sensor": sensor,
            "channel": ch,
            "channel_name": ch_name,
            "frequency_hz": round(float(freq), 2),
            "segment_start": seg_start_str,
        }
        annotations = annotations + [row]
        append_csv(row)

        # Patch just the annotation scatter trace — no data reload needed.
        # Annotation markers for selection i sit at trace index 3*i + 2.
        patched = Patch()
        ann_trace = 3 * selection_idx + 2
        patched["data"][ann_trace]["x"].append(float(t_offset_s))
        patched["data"][ann_trace]["y"].append(float(freq))

        feedback = (
            f"✓ Logged: {sensor.upper()} ch{ch} ({ch_name}) "
            f"@ {abs_time_str.split('.')[0]} UTC  |  {float(freq):.1f} Hz"
        )
        return annotations, feedback, patched

    @app.callback(
        Output("ann-table", "children"),
        Output("ann-count", "children"),
        Input("annotations-store", "data"),
    )
    def update_ann_table(annotations):
        annotations = annotations or []
        csv_count = len(load_csv())
        count_text = f"{csv_count} total in CSV · {len(annotations)} this session"

        if not annotations:
            return html.P(
                "No annotations yet.", className="text-muted small"
            ), count_text

        rows = [
            html.Tr(
                [
                    html.Td(
                        a.get("time_utc", "").split(".")[0],
                        className="text-white small font-monospace",
                    ),
                    html.Td(
                        str(a.get("sensor", "")).upper(), className="text-white small"
                    ),
                    html.Td(str(a.get("channel", "")), className="text-white small"),
                    html.Td(a.get("channel_name", ""), className="text-white small"),
                    html.Td(
                        f"{float(a.get('frequency_hz', 0)):.1f} Hz",
                        className="text-white small",
                    ),
                ]
            )
            for a in reversed(annotations[-30:])
        ]

        table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(h)
                            for h in ["Time (UTC)", "Sensor", "Ch", "Name", "Frequency"]
                        ],
                        className="text-white",
                    )
                ),
                html.Tbody(rows),
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            size="sm",
            color="dark",
        )
        return table, count_text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_csv()
    app = create_app()
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
