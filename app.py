"""LaneGuard AI — Dashcam Lane Departure Warning Web Application.

Built with Gradio and OpenCV for local hackathon demonstration.
Follows API Contract specified in PROJECT_OVERVIEW.md and TASK_TRIEU.md.
"""

import os
import uuid
from typing import Tuple, Any, Dict, Optional
import gradio as gr

from src.video import process_video


DEFAULT_MODEL_PATH = "models/ufld_tusimple.onnx"


def analyze_video_callback(
    video_path: Optional[str],
    camera_center_ratio: float,
    model_path: str,
    max_frames_input: float,
    process_all_frames: bool,
    road_crop_ratio: Optional[float] = None,
    progress=gr.Progress(),
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str], str]:
    """Execute video pipeline and return processed video, metrics, and CSV file.

    Args:
        video_path: Path to the uploaded MP4 file.
        camera_center_ratio: Camera horizontal offset ratio (0.4 to 0.6).
        model_path: Path to ONNX model weights.
        max_frames_input: Max frames slider value.
        process_all_frames: If True, ignore max_frames limit and process all.
        road_crop_ratio: None or -1 for auto; otherwise fraction cropped from top.
        progress: Gradio progress tracker.

    Returns:
        (output_video_path, summary_dict, csv_file_path, status_message)
    """
    if not video_path or not os.path.exists(video_path):
        return (
            None,
            None,
            None,
            "⚠️ **Vui lòng tải lên một video MP4 hợp lệ** trước khi bấm 'Bắt đầu Phân tích'.",
        )

    # Validate file extension
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        return (
            None,
            None,
            None,
            f"⚠️ Định dạng file `{ext}` không được khuyến nghị. Vui lòng tải lên video `.mp4`.",
        )

    # Determine max frames
    limit_frames = None if process_all_frames else int(max_frames_input)

    # Create unique run directory
    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join("outputs", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    output_video_path = os.path.join(run_dir, "annotated_lane.mp4")

    # Friendly check for model weights existence
    if not os.path.exists(model_path):
        return (
            None,
            None,
            None,
            (
                f"❌ **Không tìm thấy model weights tại: `{model_path}`**\n\n"
                "**Hướng dẫn khắc phục:**\n"
                "1. Tải file ONNX pretrained UFLD (theo hướng dẫn trong `tasks/PROJECT_OVERVIEW.md` mục 3).\n"
                f"2. Đổi tên và lưu vào thư mục: `{model_path}`.\n"
                "3. Nhánh này đang chạy độc lập trước khi merge model của Quang. "
                "Sau khi merge `origin/quang`, hệ thống sẽ kích hoạt inference AI tự động."
            ),
        )

    try:
        def update_progress(ratio: float, desc: str):
            progress(ratio, desc=desc)

        result = process_video(
            input_path=video_path,
            output_path=output_video_path,
            model_path=model_path,
            camera_center_ratio=float(camera_center_ratio),
            max_frames=limit_frames,
            progress_callback=update_progress,
            road_crop_ratio=None if road_crop_ratio == -1 else road_crop_ratio,
        )

        summary = result["summary"]
        csv_path = result["csv_path"]
        final_video = result["output_video"]

        status_msg = (
            f"✅ **Phân tích hoàn tất thành công!**\n\n"
            f"- **Đã xử lý:** {summary['frames_processed']} frames "
            f"({summary['processing_seconds']}s @ **{summary['processing_fps']} FPS** thực đo)\n"
            f"- **FPS nguồn:** {summary['video_fps']} FPS | "
            f"**Inference TB:** {summary['avg_inference_ms']} ms/frame\n"
            f"- **Frame có đủ hai biên để tính độ lệch:** {summary['valid_lane_frames']}/{summary['frames_processed']} "
            f"({(summary['valid_lane_frames'] / max(1, summary['frames_processed']) * 100):.1f}%)\n"
            f"- **Frame cảnh báo (WARNING):** {summary['warning_frames']} frames\n"
            f"- **File CSV:** `{os.path.basename(csv_path)}` sẵn sàng tải về."
        )
        status_msg += "\n\nTỷ lệ đủ hai biên không phải độ chính xác nhận diện. UNKNOWN = chưa đủ bằng chứng để tính độ lệch."
        if summary['failed_frames']:
            status_msg = status_msg.replace("✅ **Phân tích hoàn tất thành công!**", "⚠️ **Đã xử lý video nhưng có frame lỗi.**")
            status_msg += f"\n\n⚠️ **{summary['failed_frames']} frame lỗi xử lý.** Xem `frame_errors` trong thống kê."

        return final_video, summary, csv_path, status_msg

    except Exception as exc:
        return (
            None,
            None,
            None,
            f"❌ **Lỗi trong quá trình xử lý pipeline:**\n```\n{str(exc)}\n```",
        )


def build_app() -> gr.Blocks:
    """Build and return the Gradio interface for LaneGuard AI."""
    custom_css = """
    .main-title { text-align: center; margin-bottom: 4px; }
    .sub-title { text-align: center; color: #666; margin-bottom: 16px; }
    .alert-banner {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        margin-bottom: 14px;
        border-radius: 4px;
        font-size: 0.9em;
        color: #92400e;
    }
    """

    with gr.Blocks(title="LaneGuard AI — LDW Dashboard") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        gr.Markdown(
            """
            <div class="main-title"><h1>🛡️ LaneGuard AI — Dashcam Lane Departure Warning</h1></div>
            <div class="sub-title"><b>MVP Hackathon</b> — Phân tích video camera hành trình, phát hiện & theo dõi làn đường, ước lượng tâm và cảnh báo lệch làn</div>
            <div class="alert-banner">
                ⚠️ <b>Lưu ý quan trọng:</b> Đây là hệ thống demo thử nghiệm trong tọa độ ảnh (image coordinates). Không phải hệ thống an toàn đạt chuẩn ADAS trên ô tô và không dùng để điều khiển lái xe thực tế.
            </div>
            """
        )

        with gr.Row():
            # Left column: Controls & Upload
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Dữ liệu & Cấu hình đầu vào")
                input_video = gr.Video(
                    label="Upload Dashcam Video (.mp4)",
                    sources=["upload"],
                    format="mp4",
                )

                with gr.Accordion("⚙️ Cấu hình nâng cao & Cân chỉnh", open=True):
                    camera_center_ratio = gr.Slider(
                        minimum=0.40,
                        maximum=0.60,
                        value=0.50,
                        step=0.01,
                        label="Camera Center Ratio (Tâm xe / camera)",
                        info="0.5 = camera đặt chính giữa xe; <0.5 lệch trái; >0.5 lệch phải.",
                    )
                    model_path_input = gr.Textbox(
                        label="Đường dẫn ONNX Model Weights",
                        value=DEFAULT_MODEL_PATH,
                        info="Model pretrained UFLD (mặc định: models/ufld_tusimple.onnx)",
                    )
                    road_crop_input = gr.Dropdown(
                        choices=[("Tự động theo góc camera", -1.0),
                                 ("Toàn ảnh (video mẫu)", 0.0),
                                 ("Bỏ 20% phía trên", 0.2),
                                 ("Bỏ 35% phía trên", 0.35),
                                 ("Bỏ 45% phía trên (camera nhiều trời)", 0.45)],
                        value=-1.0,
                        label="Vùng ảnh nhận diện đường",
                        info="Nếu đường vẽ lệch lên trời, thử bỏ 35% hoặc 45% phía trên ảnh.",
                    )
                    with gr.Row():
                        max_frames_slider = gr.Slider(
                            minimum=30,
                            maximum=900,
                            value=300,
                            step=30,
                            label="Giới hạn Frame (Cho demo nhanh)",
                        )
                        process_all_checkbox = gr.Checkbox(
                            label="Xử lý toàn bộ video",
                            value=False,
                            info="Bỏ giới hạn frame",
                        )

                analyze_btn = gr.Button(
                    "🚀 Bắt đầu Phân tích (Analyze Video)",
                    variant="primary",
                    size="lg",
                )

                status_box = gr.Markdown(
                    "💡 *Tải lên video MP4 rồi nhấn 'Bắt đầu Phân tích' để chạy pipeline.*"
                )

            # Right column: Annotated Output & Metrics
            with gr.Column(scale=6):
                gr.Markdown("### 📊 2. Kết quả Phân tích & Trực quan hóa")
                output_video = gr.Video(
                    label="Video đã xử lý (Annotated Lane & Departure HUD)",
                    interactive=False,
                    autoplay=True,
                )

                with gr.Row():
                    csv_output = gr.File(
                        label="📥 Tải về Telemetry CSV",
                    )

                with gr.Accordion("📈 Thống kê chi tiết (Performance & Metrics)", open=True):
                    summary_json = gr.JSON(
                        label="Summary Metrics Schema",
                    )

        # Event handling
        analyze_btn.click(
            fn=analyze_video_callback,
            inputs=[
                input_video,
                camera_center_ratio,
                model_path_input,
                max_frames_slider,
                process_all_checkbox,
                road_crop_input,
            ],
            outputs=[
                output_video,
                summary_json,
                csv_output,
                status_box,
            ],
        )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #888; font-size: 0.85em;">
                LaneGuard AI Team • Track 4 Capstone • Triệu (Web & Video Pipeline)
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
