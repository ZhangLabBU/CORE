# core/workflows.py
WORKFLOWS = {
  "wf_video_keyframe_report": {
    "task_type": "video",
    "steps": [
      {"id": "s1", "agent": "KeyframeAgent", "name": "extract_keyframes"},
      {"id": "s2", "agent": "VideoAgent", "name": "analyze_video"},
      {"id": "s3", "agent": "ReportAgent", "name": "write_report"},
    ],
    "final_step": "s3",
  },
  "wf_video_report_fast": {
    "task_type": "video",
    "steps": [
      {"id": "s1", "agent": "VideoAgent", "name": "analyze_video"},
      {"id": "s2", "agent": "ReportAgent", "name": "write_report"},
    ],
    "final_step": "s2",
  },
  "wf_map_meteo_report": {
    "task_type": "map_meteo",
    "steps": [
      {"id": "s1", "agent": "MapAgent", "name": "geo_route"},
      {"id": "s2", "agent": "MeteorologyAgent", "name": "weather"},
      {"id": "s3", "agent": "ReportAgent", "name": "write_advice"},
    ],
    "final_step": "s3",
  },
  "wf_meteo_report": {
    "task_type": "meteo",
    "steps": [
      {"id": "s1", "agent": "MeteorologyAgent", "name": "weather"},
      {"id": "s2", "agent": "ReportAgent", "name": "write_advice"},
    ],
    "final_step": "s2",
  },
  "wf_map_report": {
    "task_type": "map",
    "steps": [
      {"id": "s1", "agent": "MapAgent", "name": "geo_route"},
      {"id": "s2", "agent": "ReportAgent", "name": "write_advice"},
    ],
    "final_step": "s2",
  },
}
