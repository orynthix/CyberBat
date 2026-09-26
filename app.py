"""Offline Streamlit dashboard for world-model inference."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DEFAULT_CONFIG, MITRE_STAGES
from src.dataset.feature_extractor import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    FeatureNormalizer,
    TrafficFeatureExtractor,
)
from src.dataset.ingestion import load_events
from src.explainability.explainer import explain_features
from src.forecasting import (
    forecast_from_models,
    normalize_cached_features,
    replay_forecasts,
    resolve_label_semantics,
)


def _load_artifact(path: str | Path) -> tuple[Any, Any, Any, Any, Any, str, str]:
    try:
        import torch
        from src.models.attack_head import AttackHead
        from src.models.encoder import StateAutoencoder
        from src.models.world_model import WorldModel
    except ImportError as exc:
        raise RuntimeError("Inference requires PyTorch; install requirements.txt") from exc
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    config = artifact.get("config", DEFAULT_CONFIG)
    encoder = StateAutoencoder(len(FEATURE_NAMES), config.model.latent_dim, config.model.hidden_dim)
    world_model = WorldModel(
        config.model.latent_dim,
        config.model.hidden_dim,
        config.model.num_layers,
        config.model.num_heads,
        config.model.dropout,
        config.model.backbone,
    )
    attack_head = AttackHead(config.model.latent_dim, config.model.hidden_dim)
    encoder.load_state_dict(artifact["encoder"])
    world_model.load_state_dict(artifact["world_model"])
    attack_head.load_state_dict(artifact["attack_head"])
    encoder.eval()
    world_model.eval()
    attack_head.eval()
    return (
        encoder,
        world_model,
        attack_head,
        FeatureNormalizer.from_state_dict(artifact["normalizer"]),
        config,
        artifact.get("label_semantics", "unknown"),
        artifact.get("attack_supervision", "unknown"),
    )


def forecast_file(
    data_path: str | Path,
    checkpoint_path: str | Path,
    rollout_steps: int,
) -> dict[str, Any]:
    """Run local model inference and return dashboard-ready arrays."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Inference requires PyTorch; install requirements.txt") from exc
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    records = TrafficFeatureExtractor().aggregate_records(load_events(data_path))
    if not records:
        raise ValueError("the uploaded file contains no supported traffic events")
    encoder, world_model, attack_head, normalizer, config, label_semantics, attack_supervision = _load_artifact(checkpoint_path)
    matrix = np.vstack([record.values for record in records])
    normalized = normalizer.transform(matrix)
    sequence_length = config.windows.sequence_length
    if len(normalized) < sequence_length:
        raise ValueError(f"at least {sequence_length} traffic windows are required for inference")
    inputs = np.stack(
        [normalized[index : index + sequence_length] for index in range(len(normalized) - sequence_length + 1)]
    )
    with torch.no_grad():
        latent = encoder.encode(torch.from_numpy(inputs))
        trajectories = world_model.forward_rollout(latent, rollout_steps)
        risk, stage_logits = attack_head(trajectories)
        stage_probabilities = torch.softmax(stage_logits, dim=-1)
    risk_values = risk.numpy()
    stage_values = stage_probabilities.argmax(dim=-1).numpy()
    return {
        "risk": risk_values,
        "stage": stage_values,
        "stage_probabilities": stage_probabilities.numpy(),
        "records": records,
        "matrix": matrix,
        "feature_names": FEATURE_NAMES,
        "normalized": normalized,
        "encoder": encoder,
        "world_model": world_model,
        "attack_head": attack_head,
        "checkpoint_normalizer": normalizer,
        "sequence_length": sequence_length,
        "stage_supported": label_semantics in {"source_stage", "mitre_stage"},
        "label_semantics": label_semantics,
        "attack_supervision": attack_supervision,
        "temporal_order_verified": False,
    }


def _discover_local_sources() -> tuple[dict[str, Path], Path | None]:
    root = Path(__file__).resolve().parent
    caches = {
        path.name: path
        for path in sorted((root / "data" / "cache").glob("*"))
        if (path / "metadata.json").exists()
    }
    checkpoints = sorted((root / "artifacts").glob("**/world_model.pt"))
    return caches, checkpoints[0] if checkpoints else None


def _load_latest_metrics(checkpoint_path: str | Path) -> dict[str, Any] | None:
    root = Path(__file__).resolve().parent
    reports = sorted((root / "artifacts").glob("**/metrics.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    import json

    for report in reports:
        try:
            metrics = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        protocol = metrics.get("evaluation_protocol", {})
        if protocol.get("schema") != "cyberbat.heldout-evaluation.v2":
            continue
        if (
            protocol.get("horizon_supervision_valid") is not True
            or protocol.get("temporal_order_verified") is not True
        ):
            continue
        if Path(metrics.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
            continue
        return metrics
    return None


def _load_lead_time_report(source_name: str, cache_path: Path) -> dict[str, Any] | None:
    import json

    metadata_path = cache_path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    windowing = metadata.get("windowing", {})
    if (
        windowing.get("timestamp_used") is not True
        or windowing.get("timestamp_order_verified") is not True
        or metadata.get("label_semantics") not in {"source_stage", "mitre_stage"}
    ):
        return None
    report = Path(__file__).resolve().parent / "artifacts" / "unsw_nb15_quick" / "forecast_lead_time_unsw.json"
    if source_name != "unsw_nb15" or not report.exists():
        return None
    try:
        import json

        result = json.loads(report.read_text(encoding="utf-8"))
        return result if "forecast_lead_time" in result else None
    except (OSError, ValueError):
        return None


def _stage_name(value: int) -> str:
    """Return a safe stage label for model outputs and legacy checkpoints."""

    return MITRE_STAGES.get(int(value), "Unmapped forecast stage")


def forecast_cache(
    cache_path: str | Path,
    checkpoint_path: str | Path,
    rollout_steps: int,
    context_windows: int = 128,
) -> dict[str, Any]:
    """Forecast the next K states from the latest locally cached traffic."""

    import json
    import torch

    metadata = json.loads((Path(cache_path) / "metadata.json").read_text(encoding="utf-8"))
    count = int(metadata["window_count"])
    features = np.memmap(
        Path(cache_path) / metadata["features_file"],
        dtype=metadata.get("features_dtype", "float16"),
        mode="r",
        shape=(count, int(metadata["feature_count"])),
    )
    encoder, world_model, attack_head, checkpoint_normalizer, config, label_semantics, attack_supervision = _load_artifact(checkpoint_path)
    sequence_length = config.windows.sequence_length
    required = max(sequence_length, context_windows)
    if count < sequence_length:
        raise ValueError(f"cache requires at least {sequence_length} windows, found {count}")
    cache_values = np.asarray(features[max(0, count - required) : count], dtype=np.float32)
    raw_values = (
        FeatureNormalizer.from_state_dict(metadata["normalizer"]).inverse_transform(cache_values)
        if metadata.get("features_normalized", False)
        else cache_values
    )
    normalized = normalize_cached_features(cache_values, metadata, checkpoint_normalizer)
    context = normalized[-sequence_length:]
    with torch.no_grad():
        latent = encoder.encode(torch.from_numpy(context[None, ...]))
        current_risk_tensor, current_stage_logits = attack_head(latent)
        current_stage_probabilities = torch.softmax(current_stage_logits, dim=-1)[0].numpy()
        risks: list[float] = []
        stages: list[int] = []
        probabilities: list[np.ndarray] = []
        for step in range(1, rollout_steps + 1):
            trajectory = world_model.forward_rollout(latent, step)
            risk, stage_logits = attack_head(trajectory)
            stage_probability = torch.softmax(stage_logits, dim=-1)[0].numpy()
            risks.append(float(risk[0]))
            stages.append(int(stage_probability.argmax()))
            probabilities.append(stage_probability)
    forecast = forecast_from_models(context, encoder, world_model, attack_head, rollout_steps)
    return {
        "risk": np.asarray(risks),
        "stage": np.asarray(stages),
        "stage_probabilities": np.asarray(probabilities),
        "records": [],
        "matrix": raw_values[-context_windows:],
        "feature_names": FEATURE_NAMES,
        "normalized": normalized,
        "encoder": encoder,
        "world_model": world_model,
        "attack_head": attack_head,
        "checkpoint_normalizer": checkpoint_normalizer,
        "sequence_length": sequence_length,
        "source": Path(cache_path).name,
        "observed_windows": count,
        "current_risk": float(current_risk_tensor[0]),
        "current_stage": int(current_stage_probabilities.argmax()),
        "current_stage_probabilities": current_stage_probabilities,
        "trend": forecast.trend,
        "trend_slope_per_step": forecast.trend_slope_per_step,
        "stage_supported": label_semantics in {"source_stage", "mitre_stage"},
        "label_semantics": label_semantics,
        "attack_supervision": attack_supervision,
        "temporal_order_verified": (
            metadata.get("windowing", {}).get("timestamp_used") is True
            and metadata.get("windowing", {}).get("timestamp_order_verified") is True
        ),
        "windowing": metadata.get("windowing", {}),
    }


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("The dashboard requires the optional 'streamlit' package.") from exc
    st.set_page_config(page_title="CyberBat | World Model", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        :root { --ink:#edf5ff; --muted:#9aaccb; --panel:rgba(14,27,51,.82); --line:rgba(133,164,219,.19); --cyan:#69dcff; --mint:#65e4b2; }
        .stApp { background:radial-gradient(ellipse at 76% -8%,rgba(35,101,153,.36),transparent 39%),radial-gradient(ellipse at 8% 40%,rgba(17,46,86,.28),transparent 40%),#060c18; color:var(--ink); }
        .stApp:before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.16;
          background-image:linear-gradient(rgba(107,156,230,.075) 1px,transparent 1px),linear-gradient(90deg,rgba(107,156,230,.075) 1px,transparent 1px);
          background-size:48px 48px; mask-image:linear-gradient(to bottom,black,transparent 86%); }
        [data-testid="stMainBlockContainer"] { max-width:1440px; padding-top:2rem; padding-bottom:4rem; }
        h1,h2,h3 { font-family:ui-sans-serif,system-ui,sans-serif !important; letter-spacing:-.035em; }
        p, label, .stMarkdown, .stDataFrame { font-family:ui-sans-serif,system-ui,sans-serif; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(7,16,32,.98),rgba(5,11,23,.96)); border-right:1px solid var(--line); }
        [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(18,35,64,.93),rgba(11,22,42,.78)); border:1px solid var(--line); border-radius:20px; padding:18px 20px; box-shadow:0 18px 50px rgba(0,0,0,.19); transition:transform .2s ease,border-color .2s ease; }
        [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:rgba(105,220,255,.38); }
        [data-testid="stMetricValue"] { font-family:ui-sans-serif,system-ui,sans-serif; }
        .hero { position:relative; overflow:hidden; padding:42px 46px; margin:4px 0 28px; border:1px solid rgba(111,185,240,.22);
          border-radius:30px; background:linear-gradient(112deg,rgba(18,45,83,.96),rgba(12,26,50,.9) 53%,rgba(9,19,37,.78)); box-shadow:0 28px 90px rgba(0,0,0,.27); }
        .hero:after { content:""; position:absolute; width:310px; height:310px; right:-75px; top:-138px; border-radius:50%;
          border:1px solid rgba(106,193,255,.28); box-shadow:0 0 0 24px rgba(106,193,255,.045),0 0 0 50px rgba(106,193,255,.027),inset 0 0 55px rgba(105,220,255,.08); animation:pulse 6s ease-in-out infinite; }
        .eyebrow { color:var(--cyan); font-size:11px; font-weight:750; letter-spacing:.19em; text-transform:uppercase; }
        .hero h1 { margin:12px 0 11px; font-size:clamp(34px,4vw,54px); line-height:1.04; max-width:850px; }
        .hero p { color:#afc1de; max-width:760px; margin:0; font-size:16px; line-height:1.7; }
        .status { display:inline-flex; align-items:center; gap:8px; color:#a8badb; font-size:13px; margin-top:18px; }
        .dot { width:8px; height:8px; border-radius:50%; background:var(--mint); box-shadow:0 0 14px var(--mint); }
        .stage { padding:19px 22px; border-radius:19px; background:linear-gradient(100deg,rgba(33,72,112,.62),rgba(15,29,52,.76)); border:1px solid rgba(105,220,255,.22); }
        .stage strong { color:#a9eaff; font-size:20px; }
        .brand { display:flex; align-items:center; gap:12px; padding:8px 0 18px; }
        .brand-mark { width:44px; height:34px; filter:drop-shadow(0 0 10px rgba(99,213,255,.55)); }
        .brand-name { font-size:22px; font-weight:800; letter-spacing:-.04em; color:#eef5ff; }
        .brand-sub { color:#7891b9; font-size:10px; letter-spacing:.13em; text-transform:uppercase; }
        .future { color:var(--cyan); font-weight:700; }
        .step-card { height:100%; min-height:198px; padding:24px; border:1px solid var(--line); border-radius:22px; background:linear-gradient(145deg,rgba(15,30,55,.9),rgba(10,19,37,.86)); box-shadow:0 18px 45px rgba(0,0,0,.15); transition:transform .22s ease,border-color .22s ease; }
        .step-card:hover { transform:translateY(-4px); border-color:rgba(105,220,255,.42); }
        .step-index { display:inline-grid; place-items:center; width:35px; height:35px; border-radius:12px; color:#9deaff; background:rgba(80,185,225,.12); border:1px solid rgba(105,220,255,.2); font-size:13px; font-weight:800; }
        .step-card h3 { font-size:18px; margin:18px 0 8px; }
        .step-card p { color:var(--muted); font-size:14px; line-height:1.65; margin:0; }
        .overview-panel { padding:24px 26px; height:100%; border:1px solid var(--line); border-radius:22px; background:rgba(12,24,45,.78); }
        .overview-panel h3 { margin-top:0; font-size:19px; }
        .overview-panel p, .overview-panel li { color:var(--muted); line-height:1.7; font-size:14px; }
        .signal-pill { display:inline-block; border:1px solid rgba(105,220,255,.22); color:#a9eaff; background:rgba(42,112,151,.12); padding:7px 11px; border-radius:99px; font-size:12px; margin:4px 5px 4px 0; }
        .section-kicker { color:#78dfff; font-size:11px; text-transform:uppercase; letter-spacing:.16em; font-weight:750; margin-bottom:7px; }
        .soft-callout { padding:15px 18px; border-left:3px solid #63d5ff; border-radius:0 14px 14px 0; background:rgba(36,83,120,.18); color:#b9c9e7; line-height:1.65; }
        .stButton>button { min-height:46px; border-radius:14px; border:1px solid rgba(99,213,255,.35); background:linear-gradient(135deg,rgba(39,105,149,.55),rgba(25,58,98,.55)); color:#eff9ff; font-weight:650; transition:all .2s ease; }
        .stButton>button:hover { border-color:var(--cyan); box-shadow:0 0 24px rgba(99,213,255,.2); transform:translateY(-1px); }
        [data-testid="stExpander"] { border-color:var(--line); border-radius:16px; background:rgba(12,24,45,.52); }
        @keyframes pulse { 50% { transform:scale(1.08); opacity:.65; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="brand"><svg class="brand-mark" viewBox="0 0 100 70" aria-label="CyberBat logo">'
        '<path fill="#e8f5ff" d="M50 19 39 8 34 25 18 15 23 35 4 33 22 47 12 61 36 52 50 68 64 52 88 61 78 47 96 33 77 35 82 15 66 25 61 8Z"/>'
        '<path fill="#071226" d="M50 28 43 37 50 34 57 37Z"/></svg>'
        '<div><div class="brand-name">CyberBat</div><div class="brand-sub">Network world model</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Offline network intelligence console")
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "Forecast dashboard", "Upload & investigate"],
        key="workspace_page",
    )
    local_caches, discovered_checkpoint = _discover_local_sources()
    default_checkpoint = str(discovered_checkpoint.relative_to(Path(__file__).resolve().parent)) if discovered_checkpoint else "artifacts/world_model.pt"
    checkpoint = default_checkpoint
    horizon = 5
    threshold = 0.7
    cache_name = None
    if page != "Overview":
        st.sidebar.markdown("#### Forecast controls")
        checkpoint = st.sidebar.text_input("Model checkpoint", default_checkpoint)
        horizon = st.sidebar.slider("Forecast horizon (K steps)", min_value=1, max_value=20, value=5)
        threshold = st.sidebar.slider("Alert threshold", min_value=0.0, max_value=1.0, value=0.7)
    if page == "Forecast dashboard" and local_caches:
        cache_name = st.sidebar.selectbox(
            "Local traffic source",
            options=list(local_caches),
            index=list(local_caches).index("unsw_nb15") if "unsw_nb15" in local_caches else 0,
            help="Choose a locally prepared cache; nothing is sent to a cloud service.",
        )
    upload = st.sidebar.file_uploader("Drop traffic data", type=["csv", "tsv", "parquet", "pcap", "pcapng"]) if page == "Upload & investigate" else None
    st.sidebar.divider()
    st.sidebar.caption("All processing runs locally. No traffic leaves this machine.")
    hero_title = (
        "Understand the model. Then inspect what it sees next."
        if page == "Overview"
        else "Follow the risk trajectory."
        if page == "Forecast dashboard"
        else "Investigate a traffic capture."
    )
    hero_copy = (
        "CyberBat does more than label a traffic window: it compresses recent network behavior into a state, learns how states evolve, and rolls that state forward to estimate future risk."
        if page == "Overview"
        else "Start from the latest locally prepared traffic history, then inspect the model’s simulated risk at each future step."
        if page == "Forecast dashboard"
        else "Run an isolated, local what-if analysis on a traffic file. This is separate from the prepared-cache forecast."
    )
    st.markdown(
        f'<section class="hero"><div class="eyebrow">Temporal threat intelligence · local inference</div>'
        f'<h1>{hero_title}</h1><p>{hero_copy}</p>'
        '<div class="status"><span class="dot"></span> Offline engine · local inference · no cloud dependency</div></section>',
        unsafe_allow_html=True,
    )
    if page == "Overview":
        cache_summary = "No prepared local cache found"
        if local_caches:
            selected_cache_name = "unsw_nb15" if "unsw_nb15" in local_caches else next(iter(local_caches))
            try:
                import json

                selected_metadata = json.loads(
                    (local_caches[selected_cache_name] / "metadata.json").read_text(encoding="utf-8")
                )
                cache_summary = (
                    f"{selected_cache_name} · {int(selected_metadata['window_count']):,} prepared windows"
                )
            except (OSError, ValueError, KeyError, TypeError):
                cache_summary = f"{selected_cache_name} · cache metadata unavailable"
        checkpoint_summary = (
            f"Found · {discovered_checkpoint.relative_to(Path(__file__).resolve().parent)}"
            if discovered_checkpoint
            else "Checkpoint not found"
        )
        st.markdown('<div class="section-kicker">The idea in one minute</div>', unsafe_allow_html=True)
        st.subheader("A model that simulates what could happen next")
        st.markdown(
            '<p style="color:#9aaccb;max-width:850px;line-height:1.75;margin-top:-8px">'
            'Traditional detection asks, “What is this traffic right now?” CyberBat asks a different question: '
            '“Given the recent sequence of network states, what risk pattern does the learned model project next?” '
            'The result is a forecast to investigate—not proof that an attack will occur.</p>',
            unsafe_allow_html=True,
        )
        step_cards = [
            ("01", "Observe", "Traffic is grouped into ordered windows and summarized with flow and packet statistics."),
            ("02", "Encode", "An encoder compresses each window into a compact latent network state."),
            ("03", "Simulate", "A temporal transition model recursively predicts future latent states T+1 through T+K."),
            ("04", "Interpret", "An attack head scores the simulated trajectory; explanations and labels need evidence checks."),
        ]
        for start in (0, 2):
            step_columns = st.columns(2, gap="medium")
            for column, (number, title, description) in zip(
                step_columns, step_cards[start : start + 2]
            ):
                column.markdown(
                    f'<div class="step-card"><span class="step-index">{number}</span>'
                    f'<h3>{title}</h3><p>{description}</p></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1.12, 0.88], gap="large")
        with left:
            st.markdown(
                '<div class="overview-panel"><div class="section-kicker">How to read the forecast</div>'
                '<h3>Model outputs, not incident facts</h3>'
                '<p><b>Risk score</b> is a 0–1 sigmoid output from the saved attack head. It is not calibrated as a real-world probability.</p>'
                '<p><b>T+1, T+2…</b> are successive model rollout steps. They are not seconds or minutes unless the data has verified timestamps and the window duration is known.</p>'
                '<p><b>Threshold crossings</b> count forecast steps whose score reaches your chosen display threshold. They are not confirmed alerts or attacks.</p>'
                '<p><b>Stage labels</b> are shown only when the checkpoint’s label semantics support them; binary benign/attack labels cannot validate six MITRE stages.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                '<div class="overview-panel"><div class="section-kicker">Current local setup</div>'
                '<h3>Ready to explore</h3>'
                f'<p><span class="signal-pill">{checkpoint_summary}</span></p>'
                f'<p><span class="signal-pill">{cache_summary}</span></p>'
                '<p>Everything runs on this machine. The selected cache and checkpoint stay local.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        caution, action = st.columns([1.35, 0.65], gap="large", vertical_alignment="center")
        with caution:
            st.markdown(
                '<div class="soft-callout"><b>Important validation note</b><br>'
                'The available UNSW cache is row-ordered, not verified chronological traffic. Its replay scores cannot establish real-time early warning or chronological lead time. The currently saved checkpoint also predates horizon-aligned attack supervision.</div>',
                unsafe_allow_html=True,
            )
        with action:
            st.button(
                "Open the forecast dashboard  →",
                type="primary",
                use_container_width=True,
                on_click=lambda: st.session_state.update(
                    {"workspace_page": "Forecast dashboard"}
                ),
            )
            st.button(
                "Investigate a traffic file",
                use_container_width=True,
                on_click=lambda: st.session_state.update(
                    {"workspace_page": "Upload & investigate"}
                ),
            )
        with st.expander("What CyberBat does not do"):
            st.markdown(
                "- It does not prove that an intrusion has happened or guarantee a future attack.\n"
                "- It does not identify an attacker, inspect host processes or decrypt payloads.\n"
                "- It does not provide validated MITRE ATT&CK stages for binary-labeled caches.\n"
                "- It is not a live network sensor, calibrated production detector, or automated response system."
            )
        return
    if page == "Upload & investigate" and upload is None:
        st.info("Choose a CSV, TSV, Parquet, PCAP, or PCAPNG file in the sidebar to run an isolated investigation.")
        return
    temporary_path: Path | None = None
    try:
        if page == "Upload & investigate" and upload is not None:
            suffix = Path(upload.name).suffix.lower()
            with tempfile.NamedTemporaryFile(prefix="cyberbat_", suffix=suffix, delete=False) as handle:
                handle.write(upload.getvalue())
                temporary_path = Path(handle.name)
            result = forecast_file(temporary_path, checkpoint, horizon)
            source_label = upload.name
            result["observed_windows"] = len(result["records"])
        elif page == "Forecast dashboard" and cache_name is not None:
            result = forecast_cache(local_caches[cache_name], checkpoint, horizon)
            source_label = f"local cache: {cache_name}"
        else:
            st.error("No local cache was found. Run prepare_dataset.py first.")
            return
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        return
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    risk = result["risk"]
    stages = result["stage"]
    latest_stage = int(stages[-1])
    stage_supported = bool(result.get("stage_supported", False))
    current_risk = float(np.clip(result.get("current_risk", risk[0]), 0.0, 1.0))
    latest_risk = float(np.clip(risk[-1], 0.0, 1.0))
    flagged = [index for index, value in enumerate(risk) if value >= threshold]
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Observed-state score",
        f"{current_risk:.1%}",
        delta=f"{latest_risk - current_risk:+.1%} by T+{horizon}",
        help="The attack head's sigmoid output applied to the observed latent history. This saved checkpoint has no separate current-state supervision or probability calibration.",
    )
    metric_columns[0].caption("Model score on the latest observed context; not an incident probability.")
    if result.get("attack_supervision") != "horizon_aligned_rollout_prefixes_v1":
        st.warning(
            "Checkpoint training caveat: this model predates horizon-aligned attack supervision. "
            "T+2 and later risk outputs are exploratory and are not validated by the checkpoint's training targets."
        )
    metric_columns[1].metric(
        "Terminal forecast",
        f"{latest_risk:.1%}",
        help="The risk score at the final requested rollout step T+K. This is a model output, not an observed outcome.",
    )
    metric_columns[1].caption(f"Final step T+{horizon}; intermediate steps appear in the chart.")
    metric_columns[2].metric(
        "Steps above threshold",
        f"{len(flagged)} / {len(risk)}",
        help="Count of future rollout steps whose risk score is at or above the display threshold. These are not confirmed alerts.",
    )
    metric_columns[2].caption(f"Display threshold: {threshold:.0%}; crossing does not confirm an attack.")
    metric_columns[3].metric(
        "History windows",
        f"{result['sequence_length']:,}",
        help="Number of most recent traffic windows passed to the encoder/world model as observed context.",
    )
    metric_columns[3].caption(f"From {result.get('observed_windows', result['sequence_length']):,} available source windows.")
    st.markdown(
            f'<div class="stage">Model stage output at terminal horizon · T+{horizon}<br><strong>{_stage_name(latest_stage) if stage_supported else "Unavailable — checkpoint stage labels unverified"}</strong>'
            f'<span style="float:right;color:#b7c5df">model risk score: {latest_risk:.1%}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This is the attack head’s top class, not a verified attack phase. "
        "The active checkpoint and binary dataset labels do not support validated six-stage ATT&CK predictions."
        if not stage_supported
        else "Top-scoring stage class from the model; treat as a hypothesis, not analyst-verified ground truth."
    )
    st.markdown('<div class="section-kicker">The forecast</div>', unsafe_allow_html=True)
    st.subheader("Observed state → simulated future")
    st.markdown(
        '<p style="color:#9aaccb;line-height:1.65;margin-top:-8px">'
        'Each point is the model’s score for the observed context or a recursively predicted future state. '
        'The dotted line is your chosen display threshold; it is not a calibrated decision boundary.</p>',
        unsafe_allow_html=True,
    )
    try:
        import plotly.graph_objects as go
        figure = go.Figure()
        chart_risk = np.concatenate(([current_risk], np.asarray(risk, dtype=np.float64)))
        chart_steps = ["Current"] + [f"T+{index}" for index in range(1, len(risk) + 1)]
        figure.add_trace(go.Scatter(x=chart_steps, y=chart_risk, mode="lines+markers", name="Risk", line={"color":"#63d5ff","width":3}, fill="tozeroy", fillcolor="rgba(99,213,255,.10)", hovertemplate="%{x}<br>Risk %{y:.1%}<extra></extra>"))
        figure.add_hline(y=threshold, line_dash="dot", line_color="#ff9cac", annotation_text=f"Alert {threshold:.0%}")
        figure.update_layout(height=360, margin={"l":10,"r":10,"t":10,"b":10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color":"#b9c9e7"}, xaxis_title="State", yaxis_title="Risk", hovermode="x unified")
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    except ImportError:
        st.line_chart(risk)
    st.caption(f"Source: {source_label} · {result.get('observed_windows', len(risk)):,} observed windows · {len(flagged)} future steps exceed the configured alert threshold.")
    if not result.get("temporal_order_verified", False):
        st.warning(
            "Temporal-order caveat: these forecast steps follow the cache or file's supplied window order, "
            "which is not verified chronological time. Treat them as model rollouts, not validated "
            "real-time early-warning evidence."
        )
    if result.get("trend"):
        direction = {"increasing": "↑ increasing", "decreasing": "↓ decreasing", "stable": "→ stable"}.get(result["trend"], result["trend"])
        st.info(
            f"Forecast trend: **{direction}** · average fitted slope "
            f"{float(result.get('trend_slope_per_step', 0.0)):+.4f} score units per rollout step. "
            "This summarizes direction across the requested horizon; it is not confidence or proof of escalation."
        )
    lead_report = _load_lead_time_report(cache_name or "", local_caches[cache_name] if cache_name else Path())
    with st.expander("Early warning evidence"):
        if lead_report is None:
            st.info("No valid lead-time result: this cache lacks verified chronological timestamps and suitable stage labels.")
        else:
            lead = lead_report["forecast_lead_time"]
            st.metric("Mean lead time", f"{lead['mean_lead_time_seconds']:.2f} temporal windows")
            st.caption(
                f"{lead['events_with_advance_warning']} of {lead['events']} labeled attack onsets had a prior alert "
                f"({lead['advance_warning_rate']:.1%}). Wall-clock interpretation is unavailable because this cache uses synthetic row order."
            )
    st.subheader("Stage forecast")
    stage_rows = [
        {"state": "Current", "risk": current_risk, "stage": _stage_name(int(result.get("current_stage", latest_stage))) if stage_supported else "Unavailable — checkpoint stage labels unverified"},
        *[
            {"state": f"T+{index + 1}", "risk": float(np.clip(risk[index], 0.0, 1.0)), "stage": _stage_name(int(stages[index])) if stage_supported else "Unavailable — checkpoint stage labels unverified"}
            for index in range(len(risk))
        ],
    ]
    st.dataframe(stage_rows, hide_index=True, use_container_width=True)
    if flagged:
        st.dataframe(
            [
                {
                    "window": f"T+{index + 1}",
                    "risk": float(np.clip(risk[index], 0.0, 1.0)),
                    "stage": _stage_name(int(stages[index])) if stage_supported else "Unavailable",
                }
                for index in flagged
            ],
            use_container_width=True,
        )
    evaluation = _load_latest_metrics(checkpoint)
    with st.expander("Measured model vs baseline"):
        if evaluation is None:
            st.info(
                "No validated held-out comparison is available for this checkpoint. "
                "Reports are withheld unless timestamp order and horizon-specific supervision are verified."
            )
        else:
            world = evaluation.get("world_model", {})
            baseline = evaluation.get("logistic_regression", {})
            protocol = evaluation.get("evaluation_protocol", {})
            if protocol.get("horizon_supervision_valid") is not True:
                st.warning("This checkpoint predates horizon-aligned attack supervision; its forecast scores are exploratory.")
            if protocol.get("temporal_order_verified") is not True:
                st.warning("The evaluation windows do not have verified chronological timestamps; metrics are not evidence of real-time early-warning performance.")
            st.caption("Measured on the saved held-out evaluation report for this checkpoint; values are not inferred from the live forecast.")
            st.dataframe(
                [
                    {"metric": label, "temporal_world_model": world.get(key), "logistic_current_state": baseline.get(key)}
                    for label, key in (("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"), ("FPR", "fpr"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"))
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "F1 balances precision and recall; precision is the share of predicted attacks that match labels; "
                "recall is the share of labeled attacks detected; FPR is the share of benign rows incorrectly flagged; "
                "ROC-AUC and PR-AUC summarize ranking across thresholds."
            )
            st.caption(f"Evaluation samples: {evaluation.get('samples', 'unavailable')} · checkpoint: {Path(evaluation.get('checkpoint', 'unknown')).name}")
    if page == "Forecast dashboard" and cache_name is not None:
        with st.expander("Forecast vs reality replay"):
            st.markdown(
                "This optional check compares model scores at sampled historical origins with the labels at the corresponding later row positions. "
                "It is meaningful as a chronological forecast test only when row order is verified time order and the checkpoint was trained for matching horizons."
            )
            run_replay = st.checkbox("Run deterministic replay comparison", value=False)
            if run_replay:
                try:
                    import json

                    metadata = json.loads((local_caches[cache_name] / "metadata.json").read_text(encoding="utf-8"))
                    count = int(metadata["window_count"])
                    replay_features = np.memmap(
                        local_caches[cache_name] / metadata["features_file"],
                        dtype=metadata.get("features_dtype", "float16"),
                        mode="r",
                        shape=(count, int(metadata["feature_count"])),
                    )
                    replay_labels = np.memmap(
                        local_caches[cache_name] / metadata["labels_file"],
                        dtype="int8",
                        mode="r",
                        shape=(count,),
                    )
                    normalized_replay_features = normalize_cached_features(
                        replay_features,
                        metadata,
                        result["checkpoint_normalizer"],
                    )
                    label_semantics = resolve_label_semantics(
                        metadata.get("label_semantics"),
                        result.get("label_semantics"),
                        replay_labels,
                    )
                    replay = replay_forecasts(
                        normalized_replay_features,
                        replay_labels,
                        result["encoder"],
                        result["world_model"],
                        result["attack_head"],
                        result["sequence_length"],
                        horizon,
                        threshold,
                        max_origins=20,
                        label_semantics=label_semantics,
                    )
                    replay_rows = [
                        {
                            "horizon": f"T+{item['horizon']}",
                            "Compared rows": item["samples"],
                            "F1 score": item["f1"],
                            "Precision": item["precision"],
                            "Recall": item["recall"],
                            "PR-AUC": item["pr_auc"],
                            "Stage accuracy": item["stage_accuracy"],
                        }
                        for item in replay["forecast_vs_actual"]
                    ]
                    st.dataframe(replay_rows, hide_index=True, use_container_width=True)
                    st.caption(
                        f"{replay['replay']['origins_evaluated']} sampled origins. "
                        "Each row compares forecast scores to labels at matching future offsets; it does not describe the live forecast above."
                    )
                    with st.expander("What do these replay statistics mean?"):
                        st.markdown(
                            "- **Compared rows:** number of sampled forecast/label pairs used at that horizon.\n"
                            "- **Precision:** among rows the model scores above threshold, the share labeled attack.\n"
                            "- **Recall:** among attack-labeled rows, the share whose model score crosses threshold.\n"
                            "- **F1:** harmonic balance of precision and recall at the selected threshold.\n"
                            "- **PR-AUC:** area under the precision–recall curve across score thresholds; useful when classes are imbalanced.\n"
                            "- **Stage accuracy:** exact stage matches; unavailable for binary benign/attack labels."
                        )
                    windowing = metadata.get("windowing", {})
                    if (
                        windowing.get("timestamp_used") is not True
                        or windowing.get("timestamp_order_verified") is not True
                    ):
                        st.warning("This cache has no verified chronological timestamps. Treat these as row-order label comparisons, not chronological forecasting or real-time early-warning evidence.")
                    if result.get("attack_supervision") != "horizon_aligned_rollout_prefixes_v1":
                        st.warning("This checkpoint predates horizon-aligned attack supervision. The replay numbers are exploratory comparisons, not validation of T+2/T+3 forecasts.")
                    if label_semantics == "binary_attack":
                        st.caption("Stage accuracy is unavailable: this cache stores binary benign/attack labels, not MITRE-aligned stage ground truth.")
                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    st.warning(f"Replay unavailable: {exc}")
    flagged_flows = []
    if result["records"]:
        for index in flagged:
            record_index = min(index + result["sequence_length"] - 1, len(result["records"]) - 1)
            for event in result["records"][record_index].events:
                flagged_flows.append(
                    {
                        "window": index,
                        "source": event.values.get("src_ip", event.values.get("src", "")),
                        "destination": event.values.get("dst_ip", event.values.get("dst", "")),
                        "source_port": event.values.get("src_port", ""),
                        "destination_port": event.values.get("dst_port", ""),
                    }
                )
    if flagged_flows:
        st.subheader("Flagged flows")
        st.caption(
            "These are source records associated with above-threshold forecast steps in the uploaded capture. "
            "They are context for analyst review, not confirmed malicious flows."
        )
        st.dataframe(flagged_flows, use_container_width=True)
    try:
        import torch

        def risk_predictor(samples: np.ndarray) -> np.ndarray:
            repeated = np.repeat(
                result["normalized"][-result["sequence_length"] :][None, :, :],
                len(samples),
                axis=0,
            )
            repeated[:, -1, :] = samples
            with torch.no_grad():
                latent = result["encoder"].encode(torch.from_numpy(repeated.astype(np.float32)))
                trajectory = result["world_model"].forward_rollout(latent, horizon)
                values, _ = result["attack_head"](trajectory)
            return values.numpy()

        explanation = explain_features(
            risk_predictor,
            result["normalized"][-1:].astype(np.float32),
            result["feature_names"],
        )[0]["features"]
        st.subheader("Driving feature attribution")
        st.caption(
            "SHAP attribution estimates how changing each normalized input feature moves this model’s local forecast score. "
            "It describes model behavior, not causation or proof that a feature caused an attack."
        )
        try:
            import plotly.express as px

            chart_data = explanation[:10]
            figure = px.bar(
                chart_data,
                x="contribution",
                y="name",
                orientation="h",
                title="Top SHAP contributions",
            )
            figure.update_layout(height=360, margin={"l":10,"r":10,"t":45,"b":10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color":"#b9c9e7"})
            st.plotly_chart(figure, use_container_width=True)
        except ImportError:
            st.dataframe(explanation[:10], use_container_width=True)
    except (RuntimeError, ValueError) as exc:
        st.info(str(exc))
    with st.expander("Inspect aggregated feature matrix"):
        st.caption(
            "Rows are traffic windows and columns are the 24 aggregated input features. "
            "Values below are restored to the source feature scale; the model receives checkpoint-normalized values. "
            "A zero can mean the input format did not provide that feature."
        )
        latest_features = result["matrix"][-1]
        st.dataframe(
            [
                {
                    "feature": name,
                    "latest_window_value": float(latest_features[index]),
                    "what_it_measures": FEATURE_DESCRIPTIONS[name],
                }
                for index, name in enumerate(result["feature_names"])
            ],
            hide_index=True,
            use_container_width=True,
        )
        if st.checkbox("Show recent feature matrix", value=False):
            st.dataframe(result["matrix"], use_container_width=True)
    with st.expander("Metric guide: what each number means"):
        st.markdown(
            """
            - **Observed-state score:** sigmoid score from the attack head applied to the encoded observed history. This checkpoint has no dedicated current-state training target, so do not read it as calibrated current incident probability.
            - **Terminal forecast:** the same head’s score on the final predicted rollout prefix, at T+K. It is the furthest simulated step requested.
            - **Steps above threshold:** how many of the K future scores meet or exceed the slider threshold. The threshold is a display choice, not a validated operational alert policy.
            - **History windows:** number of recent aggregated windows given to the model as context. This is not the total cache size or forecast duration.
            - **Risk change / trend:** score difference and fitted slope across rollout steps. These are model-score changes per step, not statistical confidence or elapsed time.
            - **Stage output:** highest-scoring class among six labels. It is hidden as unavailable when checkpoint labels cannot substantiate those classes.
            - **Precision / recall / F1 / PR-AUC:** replay or held-out metrics only when an evaluation is shown. They compare model scores with labeled targets; they are not calculated from the live terminal forecast.

            **Important:** the available UNSW cache has no verified chronological timestamps, and the saved checkpoint predates horizon-aligned training. The current forecast is therefore a model demonstration, not evidence of real-time early warning. A valid evaluation needs ordered timestamped windows, sequential labels, and a checkpoint trained with matching future-horizon targets.
            """
        )
    with st.expander("How to interpret this model"):
        st.markdown(
            """
            **What it does**
            - Converts traffic into fixed windows with flow and packet statistics.
            - Encodes each window into a latent network state.
            - Learns temporal transitions and recursively simulates the next K states.
            - Maps the simulated trajectory to a 0–1 learned risk score and a MITRE-style stage.

            **What the percentage means**
            The displayed value is the attack head's sigmoid score for the simulated
            trajectory. A value such as **62%** means the model's learned output is
            `0.62` for this forecast, not that an incident is guaranteed to happen
            or that it is a calibrated real-world probability. Its reliability depends
            on the training data, labels, and similarity of the current traffic to
            those data.

            **What it does not do**
            - It does not prove compromise, identify an attacker, or replace a SOC analyst.
            - It does not inspect host processes, files, identities, or encrypted payloads.
            - It does not provide analyst-validated ATT&CK ground truth; several dataset
            mappings are heuristic.
            - Timestamp-less datasets are ordered by row and do not represent real time.
            - The current quick checkpoint is a pipeline/demo model, not a production-calibrated detector.
            """,
        )
    with st.expander("Why this is forecasting, not conventional IDS detection"):
        st.markdown(
            """
            **Conventional IDS:** observed traffic → classify the current activity.

            **CyberBat:** observed traffic → encode the current network state →
            learn temporal state evolution → recursively simulate future states →
            forecast future risk and MITRE-aligned stage progression.

            The forecast boundary is the **Current → T+1...T+K** boundary in the
            chart. Forecast values are not observed ground truth. A sequential
            forecast-vs-reality comparison is only valid when the selected dataset
            provides subsequent labeled windows; otherwise the result must be
            reported as unavailable.
            """,
        )
    st.caption("CyberBat runs fully offline. External media is intentionally not embedded so the dashboard remains usable without internet access.")


if __name__ == "__main__":
    main()
