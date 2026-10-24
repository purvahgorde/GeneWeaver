
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import process_genome, run_alignment_pipeline

st.set_page_config(
    page_title="GeneWeaver | CRISPR Alignment Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1250px;
}
h1, h2, h3 { margin-bottom: 0.3rem; }
div[data-testid="stMetric"] {
    background: #172536;
    border: 1px solid #30455c;
    padding: 12px 16px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] { color: #b8cce0; }
div[data-testid="stMetricValue"] { color: #f2f8ff; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 GeneWeaver")
st.caption("CRISPR Alignment Engine · DNA Sequence Analysis")

# ---------- Input ----------
with st.container(border=True):
    st.subheader("Sequence Configuration")

    left, right = st.columns(2)

    with left:
        source = st.radio(
            "Input Dataset",
            ["Demo Dataset", "Upload FASTA"],
            horizontal=True,
        )

        uploaded_file = None
        if source == "Upload FASTA":
            uploaded_file = st.file_uploader(
                "Choose a FASTA file",
                type=["fasta", "fa", "fna", "txt"],
                help="Upload a text file containing DNA sequences in FASTA format.",
            )
        else:
            st.caption("Using built-in data/mock_genome.fasta")

        target = st.text_input(
            "Target DNA Sequence",
            value="ATGCGTACGT",
        ).strip().upper()

    with right:
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=1,
            value=10,
            step=1,
        )

        backend = st.selectbox(
            "Execution Backend",
            ["cpu", "dask"],
            format_func=lambda x: x.upper(),
        )

        st.caption(
            "CPU runs locally. Dask uses distributed task scheduling."
        )

    run_button = st.button(
        "▶ Run Alignment",
        type="primary",
        use_container_width=True,
    )


# ---------- Alignment ----------
if run_button:
    if source == "Upload FASTA" and uploaded_file is None:
        st.warning("Please upload a FASTA file.")
    elif not target or any(base not in "ACGT" for base in target):
        st.error("Target DNA must contain only A, C, G, and T.")
    elif int(chunk_size) != len(target):
        st.error("Chunk size must equal target sequence length.")
    else:
        temp_path = None

        try:
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".fasta",
                    delete=False,
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = Path(tmp.name)

                fasta_path = temp_path
            else:
                fasta_path = ROOT / "data" / "mock_genome.fasta"

            with st.spinner("Analyzing DNA sequences..."):
                start = time.perf_counter()

                sequences, chunks = process_genome(
                    fasta_path, int(chunk_size)
                )

                if backend == "dask":
                    from src.distributed.dask_processor import (
                        process_fasta_dask,
                    )

                    results = process_fasta_dask(
                        fasta_path=str(fasta_path),
                        target=target,
                        chunk_size=int(chunk_size),
                        skip_unequal=True,
                    )
                else:
                    results = run_alignment_pipeline(
                        fasta_path,
                        target,
                        int(chunk_size),
                        backend="cpu",
                        skip_unequal=True,
                    )

                elapsed_ms = (time.perf_counter() - start) * 1000

            aligned_chunks = [
                chunk for chunk in chunks
                if len(chunk) == len(target)
            ]

            if len(aligned_chunks) != len(results):
                raise ValueError(
                    "Alignment results do not match the input chunks."
                )

            total_matches = sum(r.matches for r in results)
            total_mismatches = sum(r.mismatches for r in results)
            total_bases = sum(r.alignment_length for r in results)

            similarity = (
                100 * total_matches / total_bases
                if total_bases else 0.0
            )

            st.session_state["alignment"] = {
                "sequences": len(sequences),
                "chunks": len(chunks),
                "aligned_chunks": aligned_chunks,
                "results": results,
                "matches": total_matches,
                "mismatches": total_mismatches,
                "similarity": similarity,
                "elapsed_ms": elapsed_ms,
                "target": target,
                "backend": backend,
            }

        except Exception as exc:
            st.session_state.pop("alignment", None)
            st.error(f"Alignment failed: {exc}")

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


# ---------- Results ----------
if "alignment" in st.session_state:
    data = st.session_state["alignment"]
    results = data["results"]
    chunks = data["aligned_chunks"]

    st.success(
        f"Alignment completed successfully using "
        f"{data['backend'].upper()}."
    )

    st.subheader("Alignment Overview")

    a, b, c, d, e, f = st.columns(6)

    a.metric("Sequences", data["sequences"])
    b.metric("Chunks", data["chunks"])
    c.metric("Aligned", len(results))
    d.metric("Matches", data["matches"])
    e.metric("Mismatches", data["mismatches"])
    f.metric("Similarity", f"{data['similarity']:.2f}%")

    st.caption(
        f"Execution time: {data['elapsed_ms']:.2f} ms "
        f"· Backend: {data['backend'].upper()}"
    )

    st.subheader("DNA Alignment Explorer")

    if results:
        selected = st.selectbox(
            "Select a DNA chunk",
            range(len(results)),
            format_func=lambda i: f"Chunk {i + 1}",
        )

        result = results[selected]
        chunk = chunks[selected]
        target = data["target"]

        st.write(
            f"**Matches:** {result.matches}  |  "
            f"**Mismatches:** {result.mismatches}  |  "
            f"**Similarity:** {result.score * 100:.2f}%"
        )

        st.caption("Green = Match · Red = Mismatch")

        cells = []
        for i, base in enumerate(chunk):
            matched = base.upper() == target[i]
            color = "#20c997" if matched else "#ff6b6b"
            cells.append(
                f'<span style="color:{color};'
                f'font-weight:700;">{base.upper()}</span>'
            )

        st.markdown(
            "<div style='background:#172536;padding:14px;"
            "border-radius:10px;font-family:monospace;"
            "font-size:22px;letter-spacing:7px;"
            "overflow-x:auto;'>"
            f"<div style='color:#cbd5e1;'>Target: {target}</div>"
            f"<div>Chunk:  {''.join(cells)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        positions = [p + 1 for p in result.mismatch_positions]

        if positions:
            st.caption(
                "Mismatch positions (1-based): "
                + ", ".join(map(str, positions))
            )
        else:
            st.caption("Perfect match — no mismatches.")

    else:
        st.info("No equal-length DNA chunks were available for alignment.")

    with st.expander("Biological Scoring Summary"):
        gc_content = (
            100 * sum(base in "GC" for base in data["target"])
            / len(data["target"])
        )

        average_score = (
            sum(r.score for r in results) / len(results)
            if results else 0.0
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("Target GC Content", f"{gc_content:.2f}%")
        s2.metric("Average Score", f"{average_score:.4f}")
        s3.metric("Aligned Chunks", len(results))

st.caption("GeneWeaver · CRISPR Alignment Engine")
