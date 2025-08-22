import os
from streamlit.web.bootstrap import run

if __name__ == "__main__":
    script = os.path.join(os.path.dirname(__file__), "weather_app.py")
    run(script, "", [], {})   # 如需换端口可改为 ["--server.port=8502"]
