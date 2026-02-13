
import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.path.abspath("."), path)

if __name__ == "__main__":
    # Point to the bundled app.py
    app_path = resolve_path("app.py")
    
    # Set arguments for streamlit run
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
    ]
    
    # Launch
    sys.exit(stcli.main())
