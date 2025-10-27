import streamlit as st
import subprocess
import os
import time
from pathlib import Path

# Page config
st.set_page_config(page_title="HunyuanWorld Generator", page_icon="🎨", layout="wide")

# Initialize session state
if 'panorama_generated' not in st.session_state:
    st.session_state.panorama_generated = False
if 'panorama_path' not in st.session_state:
    st.session_state.panorama_path = None
if 'scene_generated' not in st.session_state:
    st.session_state.scene_generated = False
if 'ply_files' not in st.session_state:
    st.session_state.ply_files = []
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None

st.title("🎨 HunyuanWorld 3D Scene Generator")
st.markdown("Generate panoramic images and convert them to 3D scenes")

# Step 1: Panorama Generation
st.header("Step 1: Generate Panorama")

col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.text_area("Enter your prompt:", 
                          value="At the moment of glacier collapse, giant ice walls collapse and create waves, with no wildlife, captured in a disaster documentary",
                          height=100)
with col2:
    output_name = st.text_input("Output name:", value="case1")

if st.button("🎨 Generate Panorama", type="primary", disabled=st.session_state.panorama_generated):
    output_dir = f"test_results/{output_name}"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    
    status_placeholder.info("🔄 Generating panorama... This may take 2-3 minutes.")
    
    # Escape quotes in prompt for shell command
    escaped_prompt = prompt.replace('"', '\\"')
    
    # Run panorama generation
    cmd = f"""
    source ~/.bashrc && \
    conda activate HunyuanWorld && \
    CUDA_VISIBLE_DEVICES=1 python3 demo_panogen.py \
        --prompt "{escaped_prompt}" \
        --output_path {output_dir}
    """
    
    # Show command for debugging
    with st.expander("🔍 Debug: View Command"):
        st.code(cmd, language="bash")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        logs = []
        for line in process.stdout:
            line = line.strip()
            if line:
                logs.append(line)
                # Show all logs in real-time
                log_placeholder.code('\n'.join(logs[-30:]))  # Show last 30 lines
        
        process.wait()
        
        # Show full logs on error
        if process.returncode != 0:
            with st.expander("📋 Full Error Logs", expanded=True):
                st.code('\n'.join(logs), language="bash")
        
        if process.returncode == 0:
            panorama_path = f"{output_dir}/panorama.png"
            if os.path.exists(panorama_path):
                st.session_state.panorama_generated = True
                st.session_state.panorama_path = panorama_path
                st.session_state.output_dir = output_dir
                status_placeholder.success("✅ Panorama generated successfully!")
                st.rerun()
            else:
                status_placeholder.error("❌ Panorama file not found!")
        else:
            status_placeholder.error(f"❌ Generation failed with code {process.returncode}")
            
    except Exception as e:
        status_placeholder.error(f"❌ Error: {str(e)}")

# Show panorama if generated
if st.session_state.panorama_generated and st.session_state.panorama_path:
    st.success("✅ Panorama Ready!")
    
    # Display panorama
    st.image(st.session_state.panorama_path, caption="Generated Panorama", use_container_width=True)
    
    # Download button
    with open(st.session_state.panorama_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Panorama (PNG)",
            data=file,
            file_name=f"{output_name}_panorama.png",
            mime="image/png"
        )
    
    st.markdown("---")
    
    # Step 2: Scene Generation
    st.header("Step 2: Generate 3D Scene")
    
    st.info(f"Using panorama: `{st.session_state.panorama_path}`")
    
    # Classes selector for scenegen only
    classes = st.selectbox("Scene Type:", ["outdoor", "indoor"])
    
    if st.button("🏔️ Generate 3D Scene", type="primary", disabled=st.session_state.scene_generated):
        status_placeholder2 = st.empty()
        log_placeholder2 = st.empty()
        
        status_placeholder2.info("🔄 Generating 3D scene... This may take 5-10 minutes.")
        
        # Run scene generation
        cmd = f"""
        source ~/.bashrc && \
        conda activate HunyuanWorld && \
        CUDA_VISIBLE_DEVICES=1 python3 demo_scenegen.py \
            --image_path {st.session_state.panorama_path} \
            --classes {classes} \
            --output_path {st.session_state.output_dir}
        """
        
        # Show command for debugging
        with st.expander("🔍 Debug: View Command"):
            st.code(cmd, language="bash")
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logs = []
            for line in process.stdout:
                line = line.strip()
                if line:
                    logs.append(line)
                    # Show all logs in real-time
                    log_placeholder2.code('\n'.join(logs[-30:]))
            
            process.wait()
            
            # Show full logs on error
            if process.returncode != 0:
                with st.expander("📋 Full Error Logs", expanded=True):
                    st.code('\n'.join(logs), language="bash")
            
            if process.returncode == 0:
                # Find generated PLY files
                ply_files = list(Path(st.session_state.output_dir).glob("*.ply"))
                if ply_files:
                    st.session_state.scene_generated = True
                    st.session_state.ply_files = [str(f) for f in sorted(ply_files)]
                    status_placeholder2.success("✅ 3D scene generated successfully!")
                    st.rerun()
                else:
                    status_placeholder2.error("❌ PLY files not found!")
            else:
                status_placeholder2.error(f"❌ Generation failed with code {process.returncode}")
                
        except Exception as e:
            status_placeholder2.error(f"❌ Error: {str(e)}")
    
    # Show download buttons if scene generated
    if st.session_state.scene_generated and hasattr(st.session_state, 'ply_files'):
        st.success("✅ 3D Scene Ready!")
        
        st.markdown("**Download Scene Layers:**")
        cols = st.columns(len(st.session_state.ply_files))
        
        for idx, ply_path in enumerate(st.session_state.ply_files):
            layer_name = Path(ply_path).stem  # e.g., "mesh_layer0"
            with cols[idx]:
                with open(ply_path, "rb") as file:
                    st.download_button(
                        label=f"⬇️ {layer_name}",
                        data=file,
                        file_name=f"{output_name}_{layer_name}.ply",
                        mime="application/octet-stream",
                        key=f"download_{layer_name}"
                    )
        
        st.info("💡 You can view PLY files using `modelviewer.html` or upload to https://3dviewer.net/")

# Reset button
if st.session_state.panorama_generated or st.session_state.scene_generated:
    st.markdown("---")
    if st.button("🔄 Start New Generation"):
        st.session_state.panorama_generated = False
        st.session_state.panorama_path = None
        st.session_state.scene_generated = False
        st.session_state.ply_files = []
        st.session_state.output_dir = None
        st.rerun()